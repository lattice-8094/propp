"""
character_network.py - Character Network Visualization for propp outputs

Builds co-occurrence-based character networks from propp .entities and .book files.
Outputs: PNG visualization, interactive HTML plot, CSV with centrality metrics.

Functions:
    generate_character_network(filename, ...) - Process a single book
    generate_all_character_networks(folder_path, ...) - Batch process all books in folder
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
from itertools import combinations
import json
import ast

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
# Helpers: parsing .book file
# ======================================================================

def parse_book_file(book_path: Path) -> Dict:
    """
    Parse the .book file which contains character metadata.
    
    Supports two formats:
    - OLD: Python dict literals, one per line
    - NEW: JSON array of character objects
    
    Each character has: id, count (with occurrence), mentions (proper/common/pronoun), etc.
    
    Returns a dict mapping char_id (str) -> character info dict
    """
    characters = {}
    
    with open(book_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Try NEW format first (JSON array)
    if content.startswith('['):
        try:
            char_list = json.loads(content)
            for char_data in char_list:
                char_id = str(char_data.get('id', ''))
                characters[char_id] = char_data
            return characters
        except json.JSONDecodeError:
            pass  # Fall through to old format
    
    # OLD format: Python dict literals, one per line
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            char_data = ast.literal_eval(line)
            char_id = str(char_data.get('id', ''))
            characters[char_id] = char_data
        except (SyntaxError, ValueError):
            continue
    
    return characters


def extract_char_info(characters: Dict) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    Extract name and mention count from parsed .book data.
    
    Returns:
        name_map: char_id -> best character name
        mention_map: char_id -> occurrence count
    """
    name_map = {}
    mention_map = {}
    
    for char_id, data in characters.items():
        # Get occurrence count
        count_data = data.get('count', {})
        occurrence = count_data.get('occurrence', 0)
        mention_map[char_id] = float(occurrence)
        
        # Get best name from mentions
        mentions = data.get('mentions', {})
        proper = mentions.get('proper', [])
        common = mentions.get('common', [])
        
        # Prefer proper name with highest count
        best_name = char_id
        if proper:
            # proper is a list of dicts with 'n' (name) and 'c' (count)
            best_proper = max(proper, key=lambda x: x.get('c', 0))
            best_name = best_proper.get('n', char_id)
        elif common:
            best_common = max(common, key=lambda x: x.get('c', 0))
            best_name = best_common.get('n', char_id)
        
        name_map[char_id] = best_name
    
    return name_map, mention_map


def choose_topN_characters(
    characters: Dict,
    top_n: int = DEFAULT_TOP_N,
) -> Tuple[List[str], Dict[str, str], Dict[str, float]]:
    """
    Select the top-N characters by mention count from .book data.
    
    Returns:
        top_ids: list of character ids (as strings)
        name_map: char_id -> char_name for top-N
        mention_map: char_id -> mention_count for top-N
    """
    name_map, mention_map = extract_char_info(characters)
    
    # Sort by mention count descending
    sorted_chars = sorted(mention_map.items(), key=lambda x: x[1], reverse=True)
    top_chars = sorted_chars[:top_n]
    
    top_ids = [char_id for char_id, _ in top_chars]
    
    # Filter maps to top-N only
    top_name_map = {cid: name_map.get(cid, cid) for cid in top_ids}
    top_mention_map = {cid: mention_map.get(cid, 0.0) for cid in top_ids}
    
    return top_ids, top_name_map, top_mention_map


# ======================================================================
# Graph building
# ======================================================================

def build_graph_from_entities(
    entities_path: Path,
    top_ids: List[str],
    mention_map: Dict[str, float],
    name_map: Dict[str, str],
    restrict_persons: bool = RESTRICT_PERSONS,
    min_edge_weight: int = MIN_EDGE_WEIGHT,
) -> nx.Graph:
    """
    Build a co-occurrence graph restricted to top-N characters.
    
    - nodes = top_ids
    - edges between two chars if they co-occur in at least one sentence_ID
      (weight = number of sentences where they co-occur)
    """
    print(f"[INFO] Reading entities from {entities_path}")
    ents = pd.read_csv(
        entities_path,
        sep="\t",
        usecols=["sentence_ID", "cat", "COREF"],
        dtype={"sentence_ID": "int64", "cat": "string", "COREF": "Int64"},
        low_memory=False,
    )
    
    # Filter persons if requested
    if restrict_persons:
        ents = ents[ents["cat"] == "PER"]
    
    # Drop rows with missing COREF (but keep 0)
    ents = ents[ents["COREF"].notna()]
    
    # Convert COREF to string to match top_ids
    ents["COREF"] = ents["COREF"].astype(int).astype(str)
    
    top_ids_set = set(top_ids)
    # Keep only mentions of our top-N characters
    ents_top = ents[ents["COREF"].isin(top_ids_set)].copy()
    
    if ents_top.empty:
        print(f"[WARN] No entity mentions for top-N characters in .entities file.")
        # Build graph with isolated nodes
        G = nx.Graph()
        for cid in top_ids:
            G.add_node(cid)
            G.nodes[cid]["label"] = name_map.get(cid, cid)
            G.nodes[cid]["mention_count"] = mention_map.get(cid, 0.0)
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
            if u > v:
                u, v = v, u
            cooc[(u, v)] += 1
    
    # Build graph
    G = nx.Graph()
    for cid in top_ids:
        G.add_node(cid)
        G.nodes[cid]["label"] = name_map.get(cid, cid)
        G.nodes[cid]["mention_count"] = mention_map.get(cid, 0.0)
    
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
    top_ids: List[str],
    name_map: Dict[str, str],
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
    # Iterate in order of top_ids (frequency order)
    for cid in top_ids:
        if cid not in G:
            continue
        rows.append({
            "id": cid,
            "label": name_map.get(cid, cid),
            "degree": degree_w.get(cid, 0.0),
            "pagerank": pagerank.get(cid, 0.0),
            "betweenness": betweens.get(cid, 0.0),
            "closeness": closeness.get(cid, 0.0),
        })
    
    return pd.DataFrame(rows)


# ======================================================================
# Visualization
# ======================================================================

def plot_network(
    G: nx.Graph,
    metrics_df: pd.DataFrame,
    title: str = "Character Network",
    edge_width_divisor: float = 5.0,
    min_node_size: float = 300.0,
    max_node_size: float = 1000.0,
) -> go.Figure:
    """
    Create an interactive Plotly network visualization.
    
    - Edge thickness ~ edge weight (co-occurrences)
    - Node size ~ mention_count
    """
    if G.number_of_nodes() == 0:
        print("[WARN] Empty graph, nothing to plot.")
        return go.Figure()
    
    pos = nx.spring_layout(G, weight="weight", seed=42, k=2, iterations=50)
    
    # -------------------------
    # Edges
    # -------------------------
    edge_traces = []
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        width = max(1.5, w / edge_width_divisor)
        
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    width=width,
                    color="rgba(100,100,100,0.4)",
                ),
                hoverinfo="text",
                text=[f"{G.nodes[u].get('label', u)} — {G.nodes[v].get('label', v)}<br>co-occurrences = {w}"],
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
        if max_mention == min_mention:
            return (min_node_size + max_node_size) / 2.0
        return min_node_size + (max_node_size - min_node_size) * (
            (m - min_mention) / (max_mention - min_mention)
        )
    
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
        
        node_color.append("rgb(70, 130, 180)")  # Steel blue
        
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
        textfont=dict(size=11, color="black", family="Arial Black"),
        marker=dict(
            size=node_size,
            color=node_color,
            sizemode="area",
            line=dict(width=2, color="white"),
        ),
        hovertext=node_text,
        hoverinfo="text",
    )
    
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="rgba(250,250,250,1)",
        margin=dict(l=20, r=20, t=60, b=20),
        width=1000,
        height=800,
    )
    return fig


# ======================================================================
# Main API functions
# ======================================================================

def generate_character_network(
    filename: str,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    top_n: int = DEFAULT_TOP_N,
    save_outputs: bool = True,
    show_plot: bool = True,
) -> Tuple[pd.DataFrame, go.Figure, nx.Graph]:
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
    # Handle full path input
    filename_path = Path(filename)
    if filename_path.suffix in ['.book', '.entities']:
        input_dir = str(filename_path.parent)
        filename = filename_path.stem
    
    input_path = Path(input_dir) if input_dir else Path.cwd()
    output_path = Path(output_dir) if output_dir else input_path
    
    book_path = input_path / f"{filename}.book"
    entities_path = input_path / f"{filename}.entities"
    
    # Validate files exist
    if not book_path.exists():
        raise FileNotFoundError(f"Missing .book file: {book_path}")
    if not entities_path.exists():
        raise FileNotFoundError(f"Missing .entities file: {entities_path}")
    
    print(f"\n{'='*60}")
    print(f"[BOOK] {filename}")
    print(f"{'='*60}")
    
    # Parse .book file and select top-N characters
    characters = parse_book_file(book_path)
    top_ids, name_map, mention_map = choose_topN_characters(characters, top_n)
    
    print(f"[INFO] Top-{top_n} character ids: {top_ids}")
    print(f"[INFO] Names: {[name_map.get(cid, cid) for cid in top_ids]}")
    
    # Build graph from .entities
    G = build_graph_from_entities(
        entities_path,
        top_ids,
        mention_map,
        name_map,
    )
    
    # Compute metrics
    metrics_df = compute_metrics(G, top_ids, name_map)
    
    print(f"[INFO] Metrics for top-{top_n} characters:")
    print(metrics_df.to_string(index=False))
    
    # Create visualization
    fig = plot_network(
        G,
        metrics_df,
        title=f"{filename} — top-{top_n} character network",
    )
    
    # Show plot in notebook if requested
    if show_plot:
        fig.show()
    
    # Save outputs if requested
    if save_outputs:
        output_path.mkdir(parents=True, exist_ok=True)
        
        png_path = output_path / f"{filename}_network.png"
        html_path = output_path / f"{filename}_network.html"
        csv_path = output_path / f"{filename}_metrics.csv"
        
        fig.write_html(str(html_path))
        print(f"[INFO] HTML saved to {html_path}")
        
        try:
            fig.write_image(str(png_path), scale=2)
            print(f"[INFO] PNG saved to {png_path}")
        except Exception as e:
            print(f"[WARN] Could not save PNG (kaleido may not be installed): {e}")
        
        metrics_df.to_csv(csv_path, index=False)
        print(f"[INFO] Metrics CSV saved to {csv_path}")
    
    return metrics_df, fig, G


def generate_all_character_networks(
    folder_path: str,
    output_dir: Optional[str] = None,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """
    Batch process all books in a folder.
    
    Finds all file pairs where both {filename}.book and {filename}.entities exist,
    then processes each one.
    
    Args:
        folder_path: Directory containing .book and .entities files
        output_dir: Directory for output files (default: same as folder_path)
        top_n: Number of top characters to include in each network
    
    Returns:
        Combined DataFrame with metrics for all books (includes 'book_id' column)
    """
    input_path = Path(folder_path)
    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .book files
    book_files = list(input_path.glob("*.book"))
    
    # Filter to those that also have matching .entities
    valid_books = []
    for book_file in book_files:
        entities_file = book_file.with_suffix(".entities")
        if entities_file.exists():
            valid_books.append(book_file.stem)
    
    if not valid_books:
        print(f"[WARN] No valid book pairs found in {folder_path}")
        return pd.DataFrame()
    
    print(f"[INFO] Found {len(valid_books)} books to process")
    
    all_metrics = []
    
    for filename in tqdm(valid_books, desc="Processing books"):
        try:
            metrics_df, _, _ = generate_character_network(
                filename,
                input_dir=str(input_path),
                output_dir=str(output_path),
                top_n=top_n,
                show_plot=False,  # Don't show plots in batch mode
            )
            metrics_df["book_id"] = filename
            all_metrics.append(metrics_df)
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
            continue
    
    if all_metrics:
        combined_df = pd.concat(all_metrics, ignore_index=True)
        combined_path = output_path / "all_books_metrics.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"\n[INFO] Combined metrics saved to {combined_path}")
        return combined_df
    
    return pd.DataFrame()


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate character network visualizations from propp outputs"
    )
    parser.add_argument(
        "input",
        help="Book filename (without extension) for single mode, or folder path for batch mode"
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Process all books in the input folder"
    )
    parser.add_argument(
        "--input-dir", "-i",
        default=None,
        help="Input directory (for single book mode)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--top-n", "-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top characters to include (default: {DEFAULT_TOP_N})"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        generate_all_character_networks(args.input, args.output_dir, args.top_n)
    else:
        generate_character_network(args.input, args.input_dir, args.output_dir, args.top_n)
