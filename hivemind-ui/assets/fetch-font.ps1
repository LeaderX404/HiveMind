# Fetch the Arabic UI font (wave #51). Pinned to a commit so
# the bytes cannot drift; CI (.github/workflows/build.yml)
# runs the same logic before cargo build.
$ErrorActionPreference = "Stop"
$dest = Join-Path $PSScriptRoot "NotoNaskhArabic-Regular.ttf"
if (-not (Test-Path $dest)) {
  Invoke-WebRequest -Uri "https://raw.githubusercontent.com/notofonts/notofonts.github.io/cb61ee5541f7c90cd00720f3eadf898f1ebf9c3c/fonts/NotoNaskhArabic/hinted/ttf/NotoNaskhArabic-Regular.ttf" -OutFile $dest
}
$size = (Get-Item $dest).Length
echo "font bytes: $size"
if ($size -lt 100KB -or $size -gt 5MB) { throw "font size sanity failed: $size" }
