# Building the paper as LaTeX

This directory contains the canonical LaTeX entry point for the
paper (the version you'd submit to a venue like BEA / ACL).  The
section bodies are auto-generated from `paper/sections/*.md` at
build time, so the markdown remains the source of truth for prose
and tables.

## One-time setup (Windows)

### 1. Install MiKTeX and Pandoc

```powershell
winget install MiKTeX.MiKTeX
winget install JohnMacFarlane.Pandoc
```

After install, **open a fresh PowerShell** so `pdflatex`, `bibtex`,
and `pandoc` end up on `PATH`.  MiKTeX is a lazy installer — the
first build will fetch any missing packages automatically (you'll
see a prompt the first time; click *Install*).  Once warmed, the
build is fully offline.

Sanity check:

```powershell
pdflatex --version
bibtex   --version
pandoc   --version
```

### 2. Drop in the ACL style files

The paper uses the ACL Anthology style (the same one BEA workshops
require).  Download two files from
<https://github.com/acl-org/acl-style-files>:

| File              | Where it goes                       |
|-------------------|-------------------------------------|
| `acl.sty`         | `qwen-en-tutor/paper/latex/acl.sty` |
| `acl_natbib.bst`  | `qwen-en-tutor/paper/latex/acl_natbib.bst` |

These are licensed under their own permissive terms; we don't
vendor them in this repo so you always pick up the latest version
the venue is using.

### 3. Verify the markdown source has pandoc-style citations

The build script converts each `paper/sections/*.md` to LaTeX
via pandoc.  Citations in the markdown must use pandoc's
`[@bibkey]` syntax (which pandoc turns into `\citep{bibkey}`),
e.g.:

```markdown
Self-Instruct [@wang2023selfinstruct] popularised this approach.
Multiple judges share a self-preference bias
[@wang2023pandalm; @saito2023verbosity; @panickssery2024selfpreference].
```

All current sections have already been converted to this format
(see `paper/sections/02_related_work.md`, `04_experimental_setup.md`,
`06_discussion.md`).  When you add a new citation, look up the key
in `paper/references.bib` and use `[@that-key]`.

## Build

From the `qwen-en-tutor/` directory:

```powershell
pwsh ./scripts/build_paper_latex.ps1
```

Optional flags:

| Flag    | What it does                                |
|---------|---------------------------------------------|
| `-Clean`| Wipe `paper/latex/generated/` before build  |
| `-Open` | Open the resulting PDF after a successful build |

The script:

1. Verifies `pandoc`, `pdflatex`, `bibtex` are on `PATH`.
2. Strips `# N. Heading` / `## N.N Subheading` numbering from each
   markdown file (LaTeX auto-numbers via `\section` / `\subsection`).
3. Runs `pandoc` on each section to produce
   `paper/latex/generated/sections/0X_*.tex`.
4. Copies `main.tex`, `references.bib`, `acl.sty`, `acl_natbib.bst`
   into `paper/latex/generated/`.
5. Runs `pdflatex -> bibtex -> pdflatex -> pdflatex` (the classic
   four-pass dance needed to resolve cross-references and the
   bibliography).
6. Copies the final PDF to `paper/build/paper.pdf` (the canonical
   "latest rendered paper" path, used by both the LaTeX path here
   and the Chrome HTML-preview path).
7. Reports both paths: the canonical `paper/build/paper.pdf` and
   the intermediate `paper/latex/generated/main.pdf`.

## What's in this directory

```
paper/latex/
  main.tex          # static shell: title, author, abstract, \input chain
  README.md         # this file
  .gitignore        # ignores generated/, *.aux, *.log, *.bbl, *.blg
  acl.sty           # (you download, not tracked)
  acl_natbib.bst    # (you download, not tracked)
  generated/        # produced by the build script (gitignored)
    main.tex
    references.bib
    acl.sty
    acl_natbib.bst
    sections/
      01_introduction.tex
      02_related_work.tex
      ...
    main.pdf        # the final paper
    main.log        # pdflatex log; first place to look for errors
```

## Troubleshooting

**`pdflatex: command not found`** — MiKTeX installed but PATH not
refreshed.  Close and reopen PowerShell.

**`Package acl not found`** — `acl.sty` not in
`paper/latex/` *or* in MiKTeX's package tree.  Drop it next to
`main.tex` (the script copies it into `generated/` from there).

**`bibtex: cannot open references.aux`** — pdflatex's first pass
errored.  Open `paper/latex/generated/main.log` and search for
`! ` (LaTeX error sigil).  Common causes: an unescaped `_` or `&`
in a markdown section that pandoc passed through literally; or a
missing citation key.

**Citation comes out as `[?]`** — the bib key in the markdown
doesn't match a key in `references.bib`.  Pandoc emits a warning,
and `bibtex` exits 0 but with a `Warning--I didn't find a database
entry for "key"` line in `main.blg`.

**Tables look squished** — pandoc converts markdown pipe-tables to
plain `tabular`, which is narrow in the two-column ACL layout.
Either widen with `\begin{table*}...\end{table*}` (full-width
spans) by post-editing the generated `.tex`, or rewrite the
markdown table using a `<table>` HTML block and let pandoc pass it
through as raw LaTeX.

**MiKTeX hangs on first build asking to install packages** — accept
each prompt; you'll need `acl` dependencies (`titlesec`,
`microtype`, `inconsolata`, `natbib`) plus whatever the build
references that's not in the basic install.  Once warmed, future
builds are silent.

## Why two sources (HTML and LaTeX)?

`paper/build/paper.html` (rendered to PDF via Chrome) is the fast
preview: zero dependencies beyond a browser, takes about 2 seconds
to rebuild, looks roughly like an ACL paper.  Use it during
drafting.

`paper/latex/` is the canonical submission build: matches what the
venue's typesetters expect, produces the polished
microtype/Times-with-real-kerning output that reviewers see, and
the `.tex` source is what an accepted-paper camera-ready packet
ships.  Use it when you're close to submission.

Both pipelines read the same `paper/sections/*.md` files, so the
content stays in sync as long as you edit the markdown rather than
the HTML or generated `.tex`.
