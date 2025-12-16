from pathlib import Path

BASE_URL = "https://lattice-8094.github.io/propp"
SITE_DIR = Path("site")          # Zensical output directory
OUTPUT_DIR = Path("docs")        # Where sitemap.xml should be saved
OUTPUT_DIR.mkdir(exist_ok=True)

urls = []

for html in SITE_DIR.rglob("index.html"):
    rel = html.parent.relative_to(SITE_DIR)

    # Root page
    if rel == Path("."):
        url = f"{BASE_URL}/"
    else:
        url = f"{BASE_URL}/{rel.as_posix()}/"

    urls.append(url)

sitemap_path = OUTPUT_DIR / "sitemap.xml"

with sitemap_path.open("w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
    for url in sorted(urls):
        f.write(f"  <url><loc>{url}</loc></url>\n")
    f.write('</urlset>\n')

print(f"Sitemap written to {sitemap_path}")

