#!/usr/bin/env python3
"""Build the midway report PDF from report/midway_report.md.

Lays the Markdown out like the course template (two columns, Arial, title
block across the top) and prints it to PDF with headless Chrome. Figures
come from report/make_figures.py; run that first. Inline math written as
$...$ is converted to MathML, so no network or LaTeX install is needed.

  python report/build_report.py                  # writes report/midway_report.pdf
  python report/build_report.py --chrome "C:/Program Files/Google/Chrome/Application/chrome.exe"

Needs: pip install markdown latex2mathml, and Google Chrome or Chromium.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown
from latex2mathml.converter import convert as latex_to_mathml

HERE = Path(__file__).resolve().parent
CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    "google-chrome", "chromium", "chromium-browser", "chrome",
]

CSS = """
@page { size: letter; margin: 0.6in 0.65in 0.6in 0.65in; }
body { font-family: Arial, Helvetica, sans-serif; font-size: 9.6pt; line-height: 1.28;
       color: #000; margin: 0; }
.header { font-style: italic; color: #888; font-size: 10pt; margin-bottom: 14pt; }
h1 { font-weight: normal; font-size: 17pt; text-align: center; margin: 0 0 6pt; }
.authors { text-align: center; font-style: italic; font-size: 11pt; }
.emails { text-align: center; font-style: italic; font-size: 10pt; color: #1155cc;
          margin-bottom: 14pt; }
.cols { column-count: 2; column-gap: 0.3in; text-align: justify; }
h2 { font-size: 10.5pt; margin: 9pt 0 2pt; break-after: avoid; }
p { margin: 0 0 5pt; }
ul, ol { margin: 0 0 5pt; padding-left: 14pt; }
li { margin-bottom: 2pt; }
figure { margin: 2pt 0 8pt; break-inside: avoid; text-align: center; }
figure img { width: 100%; }
figcaption { font-size: 8.6pt; text-align: center; margin-top: 2pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.2pt; margin: 3pt 0 2pt;
        break-inside: avoid; }
th, td { border-top: 0.5pt solid #000; border-bottom: 0.5pt solid #000; padding: 1.5pt 2.5pt;
         text-align: center; }
th:first-child, td:first-child { text-align: left; }
th { font-weight: bold; }
p:has(> em:only-child) { font-size: 8.6pt; text-align: center; }
math { font-size: 1.05em; }
"""


def front_matter(text: str) -> tuple[dict[str, str], str]:
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not match:
        raise SystemExit("midway_report.md must start with a --- front matter block")
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')
    return meta, text[match.end():]


def figures(body: str, fig_dir: Path) -> str:
    """<!-- figure:NAME.png\\ncaption\\n--> becomes a captioned image."""
    def replace(m: re.Match) -> str:
        path = fig_dir / m.group(1)
        if not path.exists():
            raise SystemExit(f"missing {path}; run report/make_figures.py first")
        caption = html.escape(" ".join(m.group(2).split()))
        return (f'<figure><img src="{path.as_uri()}">'
                f"<figcaption>{caption}</figcaption></figure>")
    return re.sub(r"<!--\s*figure:(\S+)\s*\n(.*?)-->", replace, body, flags=re.S)


def math(body: str) -> tuple[str, list[str]]:
    """Swap each $...$ for a token Markdown leaves alone; return the MathML for each."""
    pieces = []

    def replace(m: re.Match) -> str:
        pieces.append(latex_to_mathml(m.group(1)))
        return f"MATHTOKEN{len(pieces) - 1}X"
    return re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", replace, body), pieces


def find_chrome(given: str | None) -> str:
    for candidate in ([given] if given else CHROME_PATHS):
        found = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if found:
            return found
    raise SystemExit("Chrome not found; pass --chrome PATH")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=HERE / "midway_report.md")
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    ap.add_argument("--output", type=Path, default=HERE / "midway_report.pdf")
    ap.add_argument("--chrome")
    args = ap.parse_args()

    meta, body = front_matter(args.source.read_text(encoding="utf-8"))
    body = re.sub(r"<!--\s*table\d*\s*-->", "", figures(body, args.figures.resolve()))
    body, pieces = math(body)
    content = markdown.markdown(body, extensions=["tables", "sane_lists"])
    content = re.sub(r"MATHTOKEN(\d+)X", lambda m: pieces[int(m.group(1))], content)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body><div class="header">{html.escape(meta.get('header', ''))}</div>
<h1>{html.escape(meta.get('title', ''))}</h1>
<div class="authors">{html.escape(meta.get('authors', ''))}</div>
<div class="emails">{html.escape(meta.get('emails', ''))}</div>
<div class="cols">{content}</div></body></html>"""

    with tempfile.TemporaryDirectory() as tmp:
        page_path = Path(tmp) / "report.html"
        page_path.write_text(page, encoding="utf-8")
        subprocess.run([find_chrome(args.chrome), "--headless", "--disable-gpu",
                        "--no-pdf-header-footer", "--allow-file-access-from-files",
                        f"--print-to-pdf={args.output.resolve()}", page_path.as_uri()],
                       check=True, capture_output=True)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
