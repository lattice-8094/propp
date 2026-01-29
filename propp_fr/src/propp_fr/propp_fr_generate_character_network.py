from .propp_fr_load_save_functions import load_book_file, load_entities_df

"""
character_network.py - Character Network Visualization for propp outputs

Builds co-occurrence-based character networks from propp .entities and .book files.
Outputs: PNG visualization, interactive HTML plot, CSV with centrality metrics.

Functions:
    generate_character_network(filename, ...) - Process a single book
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
from itertools import combinations
import os

import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
import plotly.graph_objects as go


# ======================================================================
# Configuration defaults
# ======================================================================
DEFAULT_TOP_N = 10
MIN_EDGE_WEIGHT = 1
RESTRICT_PERSONS = True

# ======================================================================
# Parsing .book and extracting main characters
# ======================================================================

def extract_topN_characters(
    book_characters: List[dict],
    top_N: int = DEFAULT_TOP_N,
) -> dict:
    """
    Select the top-N characters by mention count from .book data.

    Returns:
        dict:
            key: character id (str)
            value: {
                count: int,
                character_name: str
            }
    """

    top_characters = sorted(
        book_characters,
        key=lambda c: c["count"]["occurrence"],
        reverse=True
    )[:top_N]

    top_characters = {
        str(c["id"]): {
            "count": c["count"]["occurrence"],
            "mentions": c["mentions"],
            "number": c["number"]["argmax"],
        }
        for c in top_characters
    }

    for char_id, character in top_characters.items():
        mentions = character["mentions"]

        if mentions.get("proper"):
            character["character_name"] = mentions["proper"][0]["n"].title()
        elif mentions.get("common"):
            character["character_name"] = mentions["common"][0]["n"].title()
        else:
            character["character_name"] = char_id

        # remove mentions after extracting the name
        character.pop("mentions", None)

    return top_characters

# ======================================================================
# Graph building
# ======================================================================

def build_graph_from_entities(
    top_characters: dict,
    entities_df: pd.DataFrame,
    restrict_persons: bool = RESTRICT_PERSONS,
    min_edge_weight: int = MIN_EDGE_WEIGHT,
) -> nx.Graph:
    """
    Build a co-occurrence graph restricted to top-N characters.

    - nodes = top_ids
    - edges between two chars if they co-occur in at least one sentence_ID
      (weight = number of sentences where they co-occur)
    """

    # Filter persons if requested
    if restrict_persons:
        entities_df = entities_df[entities_df["cat"] == "PER"]

    # Drop rows with missing COREF (but keep 0)
    entities_df = entities_df[entities_df["COREF"].notna()]

    # Convert COREF to string to match top_ids
    entities_df = entities_df.copy()
    entities_df["COREF"] = entities_df["COREF"].astype(int).astype(str)

    top_ids = set(top_characters.keys())
    # Keep only mentions of our top-N characters
    ents_top = entities_df[entities_df["COREF"].isin(top_ids)].copy()

    if ents_top.empty:
        print(f"[WARN] No entity mentions for top-N characters in .entities file.")
        # Build graph with isolated nodes
        G = nx.Graph()
        for char_id in top_ids:
            G.add_node(char_id)
            G.nodes[char_id]["label"] = top_characters[char_id]["character_name"]
            G.nodes[char_id]["mention_count"] = top_characters[char_id]["count"]
        return G

    # Count co-occurrences by sentence
    print(f"[INFO] Computing co-occurrences for top-{len(top_ids)} characters...")
    cooc = Counter()

    for sent_id, group in tqdm(
        ents_top.groupby("sentence_ID"),
        desc="Processing sentences",
        leave=False,
    ):
        chars = sorted(set(group["COREF"]))
        if len(chars) < 2:
            continue
        for u, v in combinations(chars, 2):
            if u == v:
                continue
            u, v = sorted((u, v))
            cooc[(u, v)] += 1

    # Build graph
    G = nx.Graph()
    for char_id in top_ids:
        G.add_node(char_id)
        G.nodes[char_id]["label"] = top_characters[char_id]["character_name"]
        G.nodes[char_id]["mention_count"] = top_characters[char_id]["count"]

    for (u, v), w in cooc.items():
        if w >= min_edge_weight:
            G.add_edge(u, v, weight=int(w))

    print(
        f"[INFO] Graph built with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges."
    )
    return G

# ======================================================================
# Metrics computation
# ======================================================================

def compute_metrics(
    G: nx.Graph,
    top_characters: dict,
) -> pd.DataFrame:
    """
    Compute centrality metrics on the graph.

    Returns a DataFrame with: id, label, degree, pagerank, betweenness, closeness
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["id", "label", "degree", "pagerank",
                                     "betweenness", "closeness"])

    # Weighted degree
    degree_w = dict(G.degree(weight="weight"))

    # PageRank, betweenness, closeness
    pagerank = nx.pagerank(G, weight="weight") if G.number_of_edges() > 0 else {}
    betweens = (
        nx.betweenness_centrality(G, weight="weight", normalized=True)
        if G.number_of_edges() > 0
        else {}
    )
    closeness = (
        nx.closeness_centrality(G) if G.number_of_nodes() > 1 else {n: 0.0 for n in G.nodes()}
    )

    rows = []
    top_ids = set(top_characters.keys())
    # Iterate in order of top_ids (frequency order)
    for char_id in top_ids:
        if char_id not in G:
            continue
        rows.append({
            "id": char_id,
            "count": top_characters[char_id]["count"],
            "number": top_characters[char_id]["number"],
            "label": top_characters[char_id]["character_name"],
            "degree": degree_w.get(char_id, 0.0),
            "pagerank": pagerank.get(char_id, 0.0),
            "betweenness": betweens.get(char_id, 0.0),
            "closeness": closeness.get(char_id, 0.0),
        })

    return pd.DataFrame(rows)

# ======================================================================
# Visualization
# ======================================================================

def plot_network(
    G: nx.Graph,
    metrics_df: pd.DataFrame,
    title: str = "Character Network",
    spring_layout_K = 2,
    min_node_size: float = 300.0,
    max_node_size: float = 1000.0,
    min_edge_width: float = 1.0,
    max_edge_width: float = 10.0,
) -> go.Figure:
    """
    Create an interactive Plotly network visualization.

    - Edge thickness ~ edge weight (co-occurrences)
    - Node size ~ mention_count
    """
    if G.number_of_nodes() == 0:
        print("[WARN] Empty graph, nothing to plot.")
        return go.Figure()

    pos = nx.spring_layout(G, weight="weight", seed=42, k=spring_layout_K, iterations=100)

    # -------------------------
    # Edges
    # -------------------------
    def scale_edge_width(w, min_width=1.0, max_width=8.0):
        if max_w == min_w:
            return (min_width + max_width)/2
        return min_width + (max_width - min_width) * ((w - min_w) / (max_w - min_w))

    edge_weights = [data.get("weight", 1) for u, v, data in G.edges(data=True)]
    max_w = max(edge_weights) if edge_weights else 1
    min_w = min(edge_weights) if edge_weights else 1

    edge_traces = []
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        width = scale_edge_width(w, min_edge_width, max_edge_width)

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    width=width,
                    color=f"rgba(70,70,70,0.5)",  # semi-transparent
                ),
                hoverinfo="text",
                text=[f"{G.nodes[u].get('label', u)} — {G.nodes[v].get('label', v)}<br>Co-occurrences = {w}"],
                showlegend=False,
            )
        )

    # -------------------------
    # Nodes
    # -------------------------
    metrics_indexed = metrics_df.set_index("id")

    mention_counts = [G.nodes[n].get("mention_count", 0.0) for n in G.nodes()]
    max_mention = max(mention_counts) if mention_counts else 1.0
    min_mention = min(mention_counts) if mention_counts else 0.0

    def scale_node_size(m):
        return min_node_size + (max_node_size - min_node_size) * ((m - min_mention)/(max_mention - min_mention))**0.5

    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        mc = G.nodes[node].get("mention_count", 0.0)

        if node in metrics_indexed.index:
            row = metrics_indexed.loc[node]
            label = row["label"]
            pr = float(row["pagerank"])
            deg = float(row["degree"])
            btwn = float(row["betweenness"])
            close = float(row["closeness"])
        else:
            label, pr, deg, btwn, close = node, 0.0, 0.0, 0.0, 0.0

        node_color.append("rgba(70,130,180,1)")  # explicitly 1 opacity

        node_text.append(
            f"<b>{label}</b>"
            f"<br>Mentions: {mc:.0f}"
            f"<br>Degree: {deg:.1f}"
            f"<br>PageRank: {pr:.3f}"
            f"<br>Betweenness: {btwn:.3f}"
            f"<br>Closeness: {close:.3f}"
        )
        node_size.append(scale_node_size(mc))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[metrics_indexed.loc[n]["label"] if n in metrics_indexed.index else n for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=14, color="black", family="Arial Black"),
        marker=dict(
            size=node_size,
            color=node_color,
            sizemode="area",
            opacity=1,
            # line=dict(width=2, color="white"),
        ),
        hovertext=node_text,
        hoverinfo="text",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="rgba(250,250,250,1)",
        margin=dict(l=0, r=0, t=60, b=0),
        width=1000,
        height=800,
    )
    return fig


# ======================================================================
# Main API functions
# ======================================================================

def generate_character_network(
    file_name: Optional[str] = None,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    book_characters: Optional[List] = None,
    entities_df: Optional[pd.DataFrame] = None,
    top_n: Optional[int] = DEFAULT_TOP_N,
    keep_only_singular: bool = True,
    save_outputs: bool = False,
    show_plot: bool = True,
    spring_layout_K = 2,
    min_node_size: float = 300.0,
    max_node_size: float = 1000.0,
    min_edge_width: float = 1.0,
    max_edge_width: float = 10.0,
) -> Tuple[pd.DataFrame, nx.Graph, go.Figure]:
    """
    Process a single book and generate network visualization.

    Args:
        filename: Base name of the book (e.g., "1887_Guy-de-Maupassant_Le-Horla")
                  without .book or .entities extension.
                  Can also be a full path to .book or .entities file.
        input_dir: Directory containing .book and .entities files (default: current dir)
        output_dir: Directory for output files (default: same as input_dir)
        top_n: Number of top characters to include in network
        save_outputs: If True, save PNG/HTML/CSV files
        show_plot: If True, display the interactive plot (for notebooks)

    Returns:
        Tuple of (metrics_df, plotly_figure, networkx_graph)

    Outputs (if save_outputs=True):
        {filename}_network.png - Static network image
        {filename}_network.html - Interactive Plotly HTML
        {filename}_metrics.csv - Centrality metrics CSV

    Example (notebook):
        >>> from character_network import generate_character_network
        >>> metrics, fig, G = generate_character_network("1887_Guy-de-Maupassant_Le-Horla")
        >>> fig.show()
    """

    if not (book_characters and (entities_df is not None and not entities_df.empty)):
        if not file_name:
            raise ValueError("Pass valide file_name or book_characters + entities_df")
        if file_name.endswith(".book") or file_name.endswith(".entities"):
            file_name = file_name.rsplit(".")[0]
        if os.path.isfile(file_name + ".book"):
            input_dir = os.path.dirname(file_name)
            file_name = os.path.basename(file_name)
        elif os.path.isdir(input_dir):
            if os.path.isfile(os.path.join(input_dir,  file_name + ".book")):
                pass
            else:
                raise ValueError(f"Input file {os.path.join(input_dir,  file_name)} does not exist")

        book_characters = load_book_file(file_name, input_dir)
        entities_df = load_entities_df(file_name, input_dir)

    if keep_only_singular:
        book_characters = [character for character in book_characters if character["number"]["argmax"] == "Singular"]
    top_characters = extract_topN_characters(book_characters, top_n)
    G = build_graph_from_entities(top_characters, entities_df)
    network_metrics_df = compute_metrics(G, top_characters)
    graph_title = os.path.basename(file_name).replace("-", " ").replace("_", " - ") if file_name else "Character Network"
    graph_title += f" - [{top_n} Characters]"
    fig = plot_network(G, network_metrics_df,
                       title=graph_title,
                       spring_layout_K = spring_layout_K,
                       min_node_size = min_node_size,
                       max_node_size = max_node_size,
                       min_edge_width = min_edge_width,
                       max_edge_width = max_edge_width)

    # Save outputs if requested
    if save_outputs:
        if file_name.endswith(".book") or file_name.endswith(".entities"):
            file_name = file_name.rsplit(".")[0]
        if os.path.isfile(file_name + ".book"):
            input_dir = os.path.dirname(file_name)
            file_name = os.path.basename(file_name)
        if not output_dir:
            if not input_dir:
                raise ValueError("Pass valide output_dir or input_dir")
            output_dir = input_dir
        if not file_name:
            raise ValueError("Pass valid file_name to save output files")

        os.makedirs(output_dir, exist_ok=True)

        png_path = os.path.join(output_dir, f"{file_name}_network.png")
        html_path = os.path.join(output_dir, f"{file_name}_network.html")
        csv_path = os.path.join(output_dir, f"{file_name}_network_metrics.csv")

        fig.write_html(str(html_path))
        print(f"[INFO] HTML saved to {html_path}")

        try:
            fig.write_image(str(png_path), scale=4)
            print(f"[INFO] PNG saved to {png_path}")
        except Exception as e:
            print(f"[WARN] Could not save PNG (kaleido may not be installed): {e}")

        network_metrics_df.to_csv(csv_path, index=False)
        print(f"[INFO] Metrics CSV saved to {csv_path}")

    # Show plot in notebook if requested
    if show_plot:
        fig.show()

    return network_metrics_df.sort_values("count", ascending=False).reset_index(drop=True), G, fig