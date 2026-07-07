"""build_paper_v3.py — merge paper_v3 sections into ONE markdown, then convert
to PDF + DOCX. Adapted from build_paper_merged.py for paper_v3.

Differences from the v2 builder:
  - No main.tex: the title + abstract are parsed from 01_introduction.md
    (which opens with `# <title>`, then `## Abstract`, a `---` rule, then
    `# 1. Introduction`).
  - Uses paper_v3/references.bib.
  - Section order has no appendix.
  - Citations use pandoc-native [@key] markdown (resolved by --citeproc).

Output: paper_v3/build/paper_full.{md,pdf,docx}
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import build_paper_merged as base  # reuse table-fit + width-rebalance helpers

ROOT = Path(__file__).resolve().parent.parent
SEC = ROOT / "paper_v3" / "sections"
BIB = ROOT / "paper_v3" / "references.bib"
OUT = ROOT / "paper_v3" / "build" / "paper_full.md"
REPO_URL = "https://github.com/cch-ai922/tutor-train"

ORDER = [
    "01_introduction", "02_related_work", "03_method", "04_experimental_setup",
    "05_results", "06_discussion", "07_conclusion", "08_appendix_repro",
]

_UNICODE_MATH = {
    "≤": r"$\le$", "≥": r"$\ge$", "≈": r"$\approx$",
    "±": r"$\pm$", "→": r"$\rightarrow$", "×": r"$\times$",
    "−": "-", "∼": r"$\sim$", "≃": r"$\simeq$",
    "≅": r"$\cong$", "≠": r"$\ne$", "≪": r"$\ll$", "≫": r"$\gg$",
    "∈": r"$\in$", "∉": r"$\notin$", "∪": r"$\cup$", "∩": r"$\cap$",
    "α": r"$\alpha$", "β": r"$\beta$", "Δ": r"$\Delta$", "≡": r"$\equiv$",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
}


def normalize_unicode_math(md: str) -> str:
    for u, tex in _UNICODE_MATH.items():
        md = md.replace(u, tex)
    return md


def parse_title_abstract_body(md: str) -> tuple[str, str, str]:
    """01_introduction.md: `# <title>`, `## Abstract` ... `---`, then body."""
    title_m = re.match(r"#\s+(.+)", md)
    title = title_m.group(1).strip() if title_m else "Untitled"
    abs_m = re.search(r"##\s+Abstract\s*\n+(.*?)\n+---", md, re.S)
    abstract = abs_m.group(1).strip() if abs_m else ""
    body_m = re.search(r"(#\s+1\.\s+Introduction.*)", md, re.S)
    body = body_m.group(1).strip() if body_m else md
    return title, abstract, body


def strip_html_comments(md: str) -> str:
    return re.sub(r"<!--.*?-->", "", md, flags=re.S)


def main() -> int:
    intro = (SEC / "01_introduction.md").read_text(encoding="utf-8")
    title, abstract, intro_body = parse_title_abstract_body(intro)

    avail = (
        "## Code and Data Availability\n\n"
        "Code, training/evaluation scripts, configuration, and the synthetic "
        f"datasets are available at <{REPO_URL}>. Model weights and large training "
        "outputs are not included; the base model is Qwen3.5-0.8B-Base. The "
        "released datasets are model-generated (teacher-distilled) and contain no "
        "personal data.\n"
    )

    yaml_abstract = "\n".join("  " + l for l in abstract.splitlines())
    # Emit bibliography into the YAML metadata so the merged .md is self-describing:
    # a plain `pandoc --citeproc paper_full.md` (or a venue template) resolves the
    # References section without needing the --bibliography flag we also pass below.
    # We reference a BARE filename and copy references.bib into build/ next to the
    # .md (below), so the path resolves whenever the two files travel together —
    # independent of the working directory the compile is run from. (A relative
    # `../references.bib` would only resolve from one specific CWD.)
    front = (
        "---\n"
        f'title: "{title}"\n'
        'author: "Independent Research"\n'
        "abstract: |\n"
        f"{yaml_abstract}\n"
        "bibliography: references.bib\n"
        "link-citations: true\n"
        "---\n"
    )

    parts = [front]
    for name in ORDER:
        md = intro_body if name == "01_introduction" else (
            SEC / f"{name}.md").read_text(encoding="utf-8")
        md = strip_html_comments(md)
        parts.append("\n\n" + md.strip() + "\n")
        if name == "08_appendix_repro":
            parts.append("\n\n" + avail + "\n")

    # references heading so citeproc appends the bibliography under it
    parts.append("\n\n# References\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged = normalize_unicode_math("\n".join(parts))
    # Give each table column a width proportional to its content so wide-first-
    # column tables don't overflow into the next column (see build_paper_merged).
    merged = base.rebalance_pipe_table_widths(merged)
    OUT.write_text(merged, encoding="utf-8")
    print(f"wrote {OUT}  ({len(merged)} chars)")

    # Bundle references.bib next to paper_full.md so the YAML `bibliography:
    # references.bib` resolves regardless of the compile CWD (self-contained build/).
    bib_local = OUT.parent / "references.bib"
    bib_local.write_bytes(BIB.read_bytes())
    print(f"bundled {bib_local}")

    pdf = OUT.with_suffix(".pdf")
    docx = OUT.with_suffix(".docx")
    # Point --bibliography at the build-local copy (absolute) so the build is also
    # CWD-independent; the YAML bare path covers third-party plain-pandoc compiles.
    common = [str(OUT), "--citeproc", f"--bibliography={bib_local}"]
    rc_pdf = subprocess.run(
        ["pandoc", *common, "-o", str(pdf), "--pdf-engine=xelatex",
         # Match the DOCX page geometry (US Letter, 1-in margins, 11pt) so the
         # two exports share the same page size, and shrink any wide table to
         # the text width (see base._pdf_header).
         "-V", "mainfont=Cambria", "-V", "papersize=letter",
         "-V", "geometry:margin=1in", "-V", "fontsize=11pt",
         "-V", "linkcolor=blue",
         "--include-in-header", str(base._pdf_header())],
        cwd=str(ROOT),
    ).returncode
    rc_docx = subprocess.run(
        ["pandoc", *common, "-o", str(docx)], cwd=str(ROOT),
    ).returncode
    print(f"PDF  {pdf}  rc={rc_pdf}")
    print(f"DOCX {docx} rc={rc_docx}")
    return 0 if (rc_pdf == 0 and rc_docx == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
