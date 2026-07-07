# =============================================================================
# build_workshop_latex.ps1 -- LaTeX build for the 8-page WORKSHOP version.
#
# Converts paper_workshop/sections/*.md -> .tex via pandoc, copies
# references.bib + figures + style files, then runs the four-pass LaTeX
# dance. Mirrors scripts/build_paper_latex.ps1 but targets paper_workshop/.
#
# Usage (from qwen-en-tutor/):
#     pwsh ./scripts/build_workshop_latex.ps1 [-Clean] [-Open]
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Open
)

$ErrorActionPreference = 'Stop'

$repoRoot   = Resolve-Path (Join-Path $PSScriptRoot '..')
$paperDir   = Join-Path $repoRoot 'paper_workshop'
$sectionsMd = Join-Path $paperDir 'sections'
$latexDir   = Join-Path $paperDir 'latex'
$genDir     = Join-Path $latexDir 'generated'
$genSecDir  = Join-Path $genDir 'sections'
$bibFile    = Join-Path $paperDir 'references.bib'
$mainTex    = Join-Path $latexDir 'main.tex'

function Test-Tool {
    param([string]$Name, [string]$WingetId)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) { Write-Error "$Name not found on PATH (winget install $WingetId)." }
    return $cmd.Source
}

Write-Host '== Checking tools ==' -ForegroundColor Cyan
$pandoc   = Test-Tool 'pandoc'   'JohnMacFarlane.Pandoc'
$pdflatex = Test-Tool 'pdflatex' 'MiKTeX.MiKTeX'
$bibtex   = Test-Tool 'bibtex'   'MiKTeX.MiKTeX'

$aclSty = Join-Path $latexDir 'acl.sty'
$aclBst = Join-Path $latexDir 'acl_natbib.bst'

if ($Clean -and (Test-Path $genDir)) {
    Remove-Item -Recurse -Force $genDir
}
New-Item -ItemType Directory -Force -Path $genSecDir | Out-Null

Write-Host '== Staging files ==' -ForegroundColor Cyan
Copy-Item $mainTex (Join-Path $genDir 'main.tex') -Force
Copy-Item $bibFile (Join-Path $genDir 'references.bib') -Force
if (Test-Path $aclSty) { Copy-Item $aclSty $genDir -Force }
if (Test-Path $aclBst) { Copy-Item $aclBst $genDir -Force }

$figSrc = Join-Path $paperDir 'figures'
$figDst = Join-Path $genDir 'figures'
if (Test-Path $figSrc) {
    New-Item -ItemType Directory -Force -Path $figDst | Out-Null
    Get-ChildItem -Path $figSrc -Filter '*.png' | ForEach-Object { Copy-Item $_.FullName $figDst -Force }
}

Write-Host '== Converting sections (pandoc) ==' -ForegroundColor Cyan
$sections = Get-ChildItem -Path $sectionsMd -Filter '*.md' | Sort-Object Name
foreach ($md in $sections) {
    $stem   = [System.IO.Path]::GetFileNameWithoutExtension($md.Name)
    $tmpMd  = Join-Path $env:TEMP "ws_$stem.md"
    $outTex = Join-Path $genSecDir "$stem.tex"

    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    $content = [System.IO.File]::ReadAllText($md.FullName, $utf8NoBom)
    $content = [regex]::Replace($content, '(?m)^(#{1,3})\s+\d+(\.\d+)*\.?\s+', '$1 ')
    # Appendix uses "# A. ..." headings -- strip the letter prefix too.
    $content = [regex]::Replace($content, '(?m)^(#)\s+[A-Z]\.\s+', '$1 ')
    [System.IO.File]::WriteAllText($tmpMd, $content, $utf8NoBom)

    $beforeMtime = if (Test-Path $outTex) { (Get-Item $outTex).LastWriteTimeUtc } else { [DateTime]::MinValue }
    & {
        $ErrorActionPreference = 'Continue'
        $null = & $pandoc `
            --from='markdown+pipe_tables+raw_tex+tex_math_dollars+tex_math_single_backslash' `
            --to=latex --natbib --no-highlight --top-level-division=section `
            --output=$outTex $tmpMd 2>$null
    }
    if (-not (Test-Path $outTex) -or (Get-Item $outTex).LastWriteTimeUtc -le $beforeMtime) {
        throw "pandoc produced no fresh $outTex"
    }
    Remove-Item $tmpMd -Force
    Write-Host "  $($md.Name)"
}

Write-Host '== LaTeX compile ==' -ForegroundColor Cyan
Push-Location $genDir
try {
    function Invoke-LaTeX {
        param([string]$Cmd, [string[]]$LaTeXArgs, [string]$SuccessFile)
        $ErrorActionPreference = 'Continue'
        $beforeMtime = if ($SuccessFile -and (Test-Path $SuccessFile)) { (Get-Item $SuccessFile).LastWriteTimeUtc } else { [DateTime]::MinValue }
        $null = & $Cmd @LaTeXArgs 2>$null
        if ($SuccessFile) {
            if (-not (Test-Path $SuccessFile)) { throw "$Cmd produced no $SuccessFile. See $genDir\main.log" }
            if ((Get-Item $SuccessFile).LastWriteTimeUtc -le $beforeMtime) { throw "$Cmd did not update $SuccessFile. See $genDir\main.log" }
        }
    }
    Invoke-LaTeX 'pdflatex' @('-interaction=nonstopmode', 'main.tex') -SuccessFile 'main.pdf'
    Invoke-LaTeX 'bibtex'   @('main')                                 -SuccessFile 'main.bbl'
    Invoke-LaTeX 'pdflatex' @('-interaction=nonstopmode', 'main.tex') -SuccessFile 'main.pdf'
    Invoke-LaTeX 'pdflatex' @('-interaction=nonstopmode', 'main.tex') -SuccessFile 'main.pdf'
} finally {
    Pop-Location
}

$outPdf = Join-Path $genDir 'main.pdf'
$buildDir = Join-Path $paperDir 'build'
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
$canonicalPdf = Join-Path $buildDir 'paper_workshop.pdf'
Copy-Item $outPdf $canonicalPdf -Force

Write-Host ''
Write-Host '== Workshop build OK ==' -ForegroundColor Green
Write-Host "PDF: $canonicalPdf"
Write-Host "Log: $(Join-Path $genDir 'main.log')"
if ($Open) { Start-Process $canonicalPdf }
