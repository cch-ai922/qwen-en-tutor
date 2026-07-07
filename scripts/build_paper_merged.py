"""build_paper_merged.py — merge the paper's section markdown into ONE
self-contained markdown file, then (separately) convert to PDF + DOCX.

- Section order matches paper/latex/main.tex (using the FINAL 05_results.md).
- Each ```{=latex} ... ``` table block is converted to a native markdown
  table via pandoc (latex -> markdown), so the tables render in BOTH pdf and
  docx with no manual number transcription.
- Title/author/abstract are lifted from main.tex.
- A "Code and Data Availability" section with the public repo URL is inserted
  after the conclusion.

Output: paper/build/paper_full.md  (then convert with pandoc — see bottom).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEC = ROOT / "paper" / "sections"
MAINTEX = ROOT / "paper" / "latex" / "main.tex"
OUT = ROOT / "paper" / "build" / "paper_full.md"
REPO_URL = "https://github.com/cch-ai922/tutor-train"

ORDER = [
    "01_introduction", "02_related_work", "03_method", "04_experimental_setup",
    "05_results", "06_discussion", "07_conclusion", "08_appendix",
]


def latex_to_md(latex: str) -> str:
    """Convert a LaTeX fragment to pandoc markdown (preserves numbers/bold/caption)."""
    r = subprocess.run(
        ["pandoc", "-f", "latex", "-t", "markdown"],
        input=latex, capture_output=True, text=True, encoding="utf-8",
    )
    if r.returncode != 0 or not r.stdout.strip():
        # fall back to a fenced raw block so nothing is silently lost
        return "```\n" + latex.strip() + "\n```"
    return r.stdout.strip()


def convert_latex_blocks(md: str) -> str:
    return re.sub(r"```\{=latex\}\n(.*?)\n```",
                  lambda m: latex_to_md(m.group(1)), md, flags=re.S)


def clean_inline_latex(md: str) -> str:
    """Neutralise inline LaTeX-isms that don't survive to a portable doc.

    Cross-references (\\ref) become nothing: in the merged single file every
    table/figure sits inline right where it is discussed, so the numeric
    pointer is redundant and would render as "??" (PDF) or drop (DOCX).
    \\texttt{x} -> `x` so monospace shows in both outputs.
    """
    md = re.sub(r"\s*~?\\[cC]?ref\{[^}]*\}", "", md)
    md = re.sub(r"\\texttt\{([^}]*)\}", r"`\1`", md)
    return md


# Literal unicode math symbols (emitted by the latex->markdown table/abstract
# conversion) that the default PDF font lacks. Wrap as LaTeX math so they
# render in BOTH the xelatex PDF and the DOCX (as Word equations). These never
# appear inside existing $...$ in the source, so wrapping is safe.
_UNICODE_MATH = {
    "≤": r"$\le$", "≥": r"$\ge$", "≈": r"$\approx$",
    "±": r"$\pm$", "→": r"$\rightarrow$", "×": r"$\times$",
    "−": "-", "∼": r"$\sim$", "≃": r"$\simeq$",
    "≅": r"$\cong$", "≠": r"$\ne$",
    "⟹": r"$\Longrightarrow$", "⟸": r"$\Longleftarrow$",
    "∈": r"$\in$", "∉": r"$\notin$", "∪": r"$\cup$", "∩": r"$\cap$",
    "α": r"$\alpha$", "𝛼": r"$\alpha$", "β": r"$\beta$",
    "≪": r"$\ll$", "≫": r"$\gg$",
    # stray unicode spaces (thin/nbsp) the table conversion can emit -> normal space
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
}


def normalize_unicode_math(md: str) -> str:
    for u, tex in _UNICODE_MATH.items():
        md = md.replace(u, tex)
    return md


def parse_intro(md: str) -> tuple[str, str, str]:
    """Split 01_introduction.md into (title, abstract, body).

    The file opens with a level-1 title, then `## Abstract`, then a `---`
    rule, then the `# 1. Introduction` body. We lift the title + abstract for
    the document header and return the body from `# 1. Introduction` onward so
    they are not duplicated.
    """
    title_m = re.match(r"#\s+(.+)", md)
    title = title_m.group(1).strip() if title_m else "Untitled"
    abs_m = re.search(r"##\s+Abstract\s*\n+(.*?)\n+---", md, re.S)
    abstract = abs_m.group(1).strip() if abs_m else ""
    body_m = re.search(r"(#\s+1\.\s+Introduction.*)", md, re.S)
    body = body_m.group(1).strip() if body_m else md
    return title, abstract, body


def main() -> int:
    # Title + abstract are the canonical ones in main.tex (01_introduction.md
    # is body-only). Convert the LaTeX abstract to markdown for the header.
    maintex = MAINTEX.read_text(encoding="utf-8")
    tm = re.search(r"\\title\{(.*?)\}", maintex, re.S)
    title = re.sub(r"\s+", " ", tm.group(1).replace("\\\\", " ")).strip() if tm else "Untitled"
    abm = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", maintex, re.S)
    abstract_md = latex_to_md(abm.group(1).strip()) if abm else ""

    avail = (
        "## Code and Data Availability\n\n"
        "Code, training/evaluation scripts, configuration, and the synthetic "
        f"training/evaluation datasets are available at <{REPO_URL}>. "
        "Model weights and large training outputs are not included; the base "
        "model is Qwen3.5-0.8B-Base. The released datasets are model-generated "
        "(teacher-distilled) and contain no personal data.\n"
    )

    # YAML metadata (title/author/abstract). Bibliography is passed on the
    # pandoc CLI so the relative path resolves from the project root.
    yaml_abstract = "\n".join("  " + l for l in abstract_md.splitlines())
    front = (
        "---\n"
        f'title: "{title}"\n'
        'author: "Miles Yung — Independent Research (milesyung2026@gmail.com)"\n'
        "abstract: |\n"
        f"{yaml_abstract}\n"
        "---\n"
    )

    parts = [front]
    for name in ORDER:
        md = (SEC / f"{name}.md").read_text(encoding="utf-8")
        md = convert_latex_blocks(md)
        md = clean_inline_latex(md)
        parts.append("\n\n" + md.strip() + "\n")
        if name == "07_conclusion":
            parts.append("\n\n" + avail + "\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged = normalize_unicode_math("\n".join(parts))
    OUT.write_text(merged, encoding="utf-8")
    print(f"wrote {OUT}  ({len(merged)} chars)")

    # --- convert to PDF (xelatex; Cambria covers the math glyphs) + DOCX ---
    bib = ROOT / "paper" / "references.bib"
    pdf = OUT.with_suffix(".pdf")
    docx = OUT.with_suffix(".docx")
    common = [str(OUT), "--citeproc", f"--bibliography={bib}"]
    rc_pdf = subprocess.run(
        ["pandoc", *common, "-o", str(pdf), "--pdf-engine=xelatex",
         "-V", "mainfont=Cambria", "-V", "geometry:margin=1in",
         "-V", "fontsize=11pt", "-V", "linkcolor=blue"],
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
