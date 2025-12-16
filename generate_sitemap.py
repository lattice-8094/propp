from pathlib import Path

BASE_URL = "https://lattice-8094.github.io/propp"
SITE_DIR = Path("site")  # zensical output dir

urls = []

for html in SITE_DIR.rglob("index.html"):
    rel = html.parent.relative_to(SITE_DIR)
    url = f"{BASE_URL}/{rel}/".replace("//", "/")
    urls.append(url)

with open("sitemap.xml", "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
    for url in urls:
        f.write(f"  <url><loc>{url}</loc></url>\n")
    f.write('</urlset>\n')
