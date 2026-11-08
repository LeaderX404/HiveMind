# Downloads the HiveMind core bundle and verifies it before
# extraction (wave #47). Called by the installer via
# powershell.exe -ExecutionPolicy Bypass -File fetch_core.ps1
# with -Url and -Sha256 and -Dest, so no unverified code or
# archive ever lands in the install directory.
param(
    [Parameter(Mandatory = $true)]
    [string]$Url,
    [Parameter(Mandatory = $true)]
    [string]$Sha256,
    [Parameter(Mandatory = $true)]
    [string]$Dest
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

Write-Output "core: downloading $Url"
$zip = Join-Path $env:TEMP "HiveMind-Core.zip.part"
try {
    Invoke-WebRequest -Uri $Url -OutFile $zip -UseBasicParsing
} catch {
    Write-Output "core: download failed: $_"
    exit 43
}

$actual = (Get-FileHash -Path $zip -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $Sha256.ToLower()) {
    Write-Output "core: sha256 mismatch - expected $Sha256 got $actual"
    Remove-Item $zip -ErrorAction SilentlyContinue
    exit 42
}
Write-Output "core: archive verified, extracting to $Dest"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
try {
    Expand-Archive -Path $zip -DestinationPath $Dest -Force
} catch {
    Write-Output "core: extraction failed: $_"
    exit 44
}
Remove-Item $zip -ErrorAction SilentlyContinue
if (-not (Test-Path (Join-Path $Dest "torch"))) {
    Write-Output "core: archive did not contain torch/"
    exit 45
}
Write-Output "core: installed at $Dest"
exit 0
