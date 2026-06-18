# =============================================================================
# build_paper_latex.ps1 -- LaTeX build pipeline for paper/latex/main.tex
# =============================================================================
#
# Converts paper/sections/*.md -> .tex via pandoc, copies references.bib,
# then runs pdflatex + bibtex + pdflatex + pdflatex from paper/latex/generated/.
#
# Prerequisites:
#   1. MiKTeX or TeX Live installed and on PATH (pdflatex, bibtex)
#        winget install MiKTeX.MiKTeX
#   2. Pandoc installed and on PATH
#        winget install JohnMacFarlane.Pandoc
#   3. ACL style files in paper/latex/:
#        acl.sty, acl_natbib.bst
#      Download from https://github.com/acl-org/acl-style-files
#
# Usage:
#   From the qwen-en-tutor/ directory:
#       pwsh ./scripts/build_paper_latex.ps1
#   Or with explicit flags:
#       pwsh ./scripts/build_paper_latex.ps1 -Clean        # wipe generated/
#       pwsh ./scripts/build_paper_latex.ps1 -Open         # open PDF after build
#
# Output:
#   paper/latex/generated/main.pdf  (the rendered paper)
#   paper/latex/generated/main.log  (pdflatex log; check for warnings)
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Open
)

$ErrorActionPreference = 'Stop'

# ---- Paths ------------------------------------------------------------------
$repoRoot   = Resolve-Path (Join-Path $PSScriptRoot '..')
$paperDir   = Join-Path $repoRoot 'paper'
$sectionsMd = Join-Path $paperDir 'sections'
$latexDir   = Join-Path $paperDir 'latex'
$genDir     = Join-Path $latexDir 'generated'
$genSecDir  = Join-Path $genDir 'sections'
$bibFile    = Join-Path $paperDir 'references.bib'
$mainTex    = Join-Path $latexDir 'main.tex'

# ---- Tool checks ------------------------------------------------------------
function Test-Tool {
    param([string]$Name, [string]$WingetId)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Error @"
$Name not found on PATH. Install with:
    winget install $WingetId
or see paper/latex/README.md for alternatives.
"@
    }
    return $cmd.Source
}

Write-Host '== Checking tools ==' -ForegroundColor Cyan
$pandoc   = Test-Tool 'pandoc'   'JohnMacFarlane.Pandoc'
$pdflatex = Test-Tool 'pdflatex' 'MiKTeX.MiKTeX'
$bibtex   = Test-Tool 'bibtex'   'MiKTeX.MiKTeX'
Write-Host "  pandoc:   $pandoc"
Write-Host "  pdflatex: $pdflatex"
Write-Host "  bibtex:   $bibtex"

# ---- ACL style file check ---------------------------------------------------
$aclSty = Join-Path $latexDir 'acl.sty'
$aclBst = Join-Path $latexDir 'acl_natbib.bst'
if (-not (Test-Path $aclSty) -or -not (Test-Path $aclBst)) {
    Write-Warning @"
ACL style files missing from $latexDir.
Download from https://github.com/acl-org/acl-style-files and copy:
    acl.sty         -> $aclSty
    acl_natbib.bst  -> $aclBst
Build will continue but may fail if these are not in the LaTeX search path.
"@
}

# ---- Clean / prepare generated/ --------------------------------------------
if ($Clean -and (Test-Path $genDir)) {
    Write-Host '== Cleaning generated/ ==' -ForegroundColor Cyan
    Remove-Item -Recurse -Force $genDir
}
New-Item -ItemType Directory -Force -Path $genSecDir | Out-Null

# ---- Stage main.tex + bib + style into generated/ ---------------------------
Write-Host '== Staging files into generated/ ==' -ForegroundColor Cyan
Copy-Item $mainTex (Join-Path $genDir 'main.tex') -Force
Copy-Item $bibFile (Join-Path $genDir 'references.bib') -Force
if (Test-Path $aclSty) { Copy-Item $aclSty $genDir -Force }
if (Test-Path $aclBst) { Copy-Item $aclBst $genDir -Force }

# ---- Pandoc: md -> tex per section ------------------------------------------
Write-Host '== Converting sections (pandoc md -> tex) ==' -ForegroundColor Cyan
$sections = Get-ChildItem -Path $sectionsMd -Filter '*.md' | Sort-Object Name
foreach ($md in $sections) {
    $stem    = [System.IO.Path]::GetFileNameWithoutExtension($md.Name)
    $tmpMd   = Join-Path $env:TEMP "$stem.md"
    $outTex  = Join-Path $genSecDir "$stem.tex"

    # Strip leading "# N." and "## N.N" numbering from headings (LaTeX
    # auto-numbers via \section / \subsection). Handles both
    # "# 2. Title" (trailing period after the number) and
    # "## 2.1 Title" (no trailing period).
    # IMPORTANT: read+write as UTF-8 no-BOM via .NET API. PowerShell 5.1's
    # default `Get-Content`/`Set-Content -Encoding utf8` round-trips through
    # the system code page, which mangles em-dashes, section signs, etc.
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    $content = [System.IO.File]::ReadAllText($md.FullName, $utf8NoBom)
    $content = [regex]::Replace($content, '(?m)^(#{1,3})\s+\d+(\.\d+)*\.?\s+', '$1 ')
    [System.IO.File]::WriteAllText($tmpMd, $content, $utf8NoBom)

    # --natbib makes pandoc emit \citep{key} from [@key] so the bib
    # entries in references.bib resolve through bibtex+acl_natbib.bst.
    & $pandoc `
        --from='markdown+pipe_tables+raw_tex+tex_math_dollars' `
        --to=latex `
        --natbib `
        --top-level-division=section `
        --output=$outTex `
        $tmpMd
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pandoc failed on $($md.Name) (exit $LASTEXITCODE)"
    }
    Remove-Item $tmpMd -Force
    Write-Host "  $($md.Name) -> $outTex"
}

# ---- LaTeX compile: pdflatex + bibtex + pdflatex + pdflatex -----------------
Write-Host '== LaTeX compile (pdflatex/bibtex/pdflatex x2) ==' -ForegroundColor Cyan
Push-Location $genDir
try {
    function Invoke-LaTeX {
        param([string]$Cmd, [string[]]$LaTeXArgs, [string]$SuccessFile)
        Write-Host "  $Cmd $($LaTeXArgs -join ' ')" -ForegroundColor DarkGray
        # MiKTeX (and pdflatex when it triggers METAFONT for first-time fonts)
        # writes informational chatter and a "you have not checked for MiKTeX
        # updates" line to stderr. With the script-wide $ErrorActionPreference
        # = 'Stop', PowerShell wraps each stderr line in a NativeCommandError
        # exception that aborts the script. We locally drop back to 'Continue'
        # and verify success by checking that the expected output file was
        # produced/updated, not by exit code.
        $ErrorActionPreference = 'Continue'
        $beforeMtime = if ($SuccessFile -and (Test-Path $SuccessFile)) {
            (Get-Item $SuccessFile).LastWriteTimeUtc
        } else { [DateTime]::MinValue }
        # Redirect stderr to $null entirely; the .log file holds everything
        # we'd need for debugging. Suppressing stdout via $null= avoids
        # flooding the terminal.
        $null = & $Cmd @LaTeXArgs 2>$null
        if ($SuccessFile) {
            if (-not (Test-Path $SuccessFile)) {
                throw "$Cmd produced no $SuccessFile. See $genDir\main.log"
            }
            $afterMtime = (Get-Item $SuccessFile).LastWriteTimeUtc
            if ($afterMtime -le $beforeMtime) {
                throw "$Cmd did not update $SuccessFile (still $afterMtime). See $genDir\main.log"
            }
        }
    }

    # -interaction=nonstopmode keeps pdflatex from blocking on missing
    # references during the first pass. We check the .pdf / .bbl file
    # mtime rather than $LASTEXITCODE to detect real failures.
    Invoke-LaTeX 'pdflatex' @('-interaction=nonstopmode', 'main.tex') -SuccessFile 'main.pdf'
    Invoke-LaTeX 'bibtex'   @('main')                                  -SuccessFile 'main.bbl'
    Invoke-LaTeX 'pdflatex' @('-interaction=nonstopmode', 'main.tex') -SuccessFile 'main.pdf'
    Invoke-LaTeX 'pdflatex' @('-interaction=nonstopmode', 'main.tex') -SuccessFile 'main.pdf'
} finally {
    Pop-Location
}

$outPdf = Join-Path $genDir 'main.pdf'
if (-not (Test-Path $outPdf)) {
    Write-Error "Build finished without producing $outPdf -- check $genDir\main.log"
}

Write-Host ''
Write-Host '== Build OK ==' -ForegroundColor Green
Write-Host "PDF: $outPdf"
Write-Host "Log: $(Join-Path $genDir 'main.log')"

if ($Open) {
    Start-Process $outPdf
}
