"""build_workshop_merged.py — merge the 8-page WORKSHOP paper's section
markdown into ONE self-contained markdown file, then convert to DOCX + PDF.

Mirrors build_paper_merged.py but targets paper_workshop/. Reuses that module's
helpers (latex->md table conversion, table/figure numbering, figure embedding,
unicode-math normalisation) so the two builds stay in lockstep.

Output: paper_workshop/build/paper_workshop_full.{md,docx,pdf}
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import build_paper_merged as base  # sibling module; reuse its helpers

ROOT = Path(__file__).resolve().parent.parent
SEC = ROOT / "paper_workshop" / "sections"
MAINTEX = ROOT / "paper_workshop" / "latex" / "main.tex"
OUT = ROOT / "paper_workshop" / "build" / "paper_workshop_full.md"
FIG_DIR_REL = "paper_workshop/figures"
REPO_URL = "https://github.com/cch-ai922/tutor-train"

ORDER = [
    "01_introduction", "02_related_work", "03_method", "04_experimental_setup",
    "05_results", "06_discussion_conclusion", "07_appendix",
]


def _figures_to_native_md_ws(md: str, tmap: dict, fmap: dict) -> str:
    """Same as base._figures_to_native_md but points image paths at the
    workshop figures dir."""
    def conv(m: "re.Match[str]") -> str:
        block = m.group(0)
        src_m = re.search(r'<img[^>]*src="([^"]+)"', block)
        cap_m = re.search(r"<figcaption>(.*?)</figcaption>", block, re.S)
        if not src_m:
            return block
        fname = src_m.group(1).split("/")[-1]
        path = f"{FIG_DIR_REL}/{fname}"
        caption = cap_m.group(1) if cap_m else ""
        caption = base._resolve_caption_refs(caption, tmap, fmap)
        caption = re.sub(r"</?strong>", "**", caption)
        caption = re.sub(r"</?em>", "*", caption)
        caption = re.sub(r"<[^>]+>", "", caption)
        caption = re.sub(r"\s+", " ", caption).strip()
        caption = re.sub(r"^\*\*Figure\s+\d+\.\s*", "**", caption)
        caption = re.sub(r"^Figure\s+\d+\.\s*", "", caption)
        return f"![{caption}]({path})"
    return re.sub(r"<figure\b.*?</figure>", conv, md, flags=re.S)


def main() -> int:
    maintex = MAINTEX.read_text(encoding="utf-8")
    tm = re.search(r"\\title\{(.*?)\}", maintex, re.S)
    title = re.sub(r"\s+", " ", tm.group(1).replace("\\\\", " ")).strip() if tm else "Untitled"
    abm = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", maintex, re.S)
    abstract_md = base.latex_to_md(abm.group(1).strip()) if abm else ""

    avail = (
        "## Code and Data Availability\n\n"
        "Code, training/evaluation scripts, configuration, and the synthetic "
        f"training/evaluation datasets are available at <{REPO_URL}>. "
        "The released datasets are model-generated (teacher-distilled) and "
        "contain no personal data. A full-length version of this paper reports "
        "the per-capability statistics and the complete confound analysis.\n"
    )

    yaml_abstract = "\n".join("  " + l for l in abstract_md.splitlines())
    front = (
        "---\n"
        f'title: "{title}"\n'
        'author: "Miles Yung — Independent Research (milesyung2026@gmail.com)"\n'
        "abstract: |\n"
        f"{yaml_abstract}\n"
        "---\n"
    )

    raw = {name: (SEC / f"{name}.md").read_text(encoding="utf-8") for name in ORDER}
    tmap = base.build_table_map([raw[name] for name in ORDER])
    fmap = base.build_figure_map([raw[name] for name in ORDER])

    parts = [front]
    for name in ORDER:
        md = base.convert_latex_blocks(raw[name], tmap, fmap)
        md = base.clean_inline_latex(md, tmap, fmap)
        parts.append("\n\n" + md.strip() + "\n")
        if name == "06_discussion_conclusion":
            parts.append("\n\n" + avail + "\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged = base.normalize_unicode_math("\n".join(parts))
    merged = _figures_to_native_md_ws(merged, tmap, fmap)
    merged = base.rebalance_pipe_table_widths(merged)
    merged = re.sub(r"`<!-- -->`(\{=html\})?", " ", merged)
    # Drop the injected "Table N." lead in pipe-table captions (def-list form
    # ``: **Table N.** ...``): pandoc's table writer adds its own "Table N:"
    # number, so ours would double it. Keep the bold so the caption title stays
    # bold.
    merged = re.sub(r"(?m)^(\s*:\s*)\*\*Table\s+\d+\.\s*", r"\1**", merged)
    OUT.write_text(merged, encoding="utf-8")
    print(f"wrote {OUT}  ({len(merged)} chars)")

    bib = ROOT / "paper_workshop" / "references.bib"
    pdf = OUT.with_suffix(".pdf")
    docx = OUT.with_suffix(".docx")
    common = [str(OUT), "--citeproc", f"--bibliography={bib}"]
    rc_pdf = subprocess.run(
        ["pandoc", *common, "-o", str(pdf), "--pdf-engine=xelatex",
         # Match the DOCX page geometry (US Letter, 1-in margins, 11pt).
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
