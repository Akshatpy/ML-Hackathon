# Laptop 1: start the PathSnap API (and optionally a public HTTPS tunnel).
# Usage (from the project root, in PowerShell):
#   .\scripts\start_api.ps1                 # LAN only: http://<this-ip>:8000
#   .\scripts\start_api.ps1 -Key demo2026   # require an access key
#   .\scripts\start_api.ps1 -Tunnel         # also open a Cloudflare quick tunnel (https URL)
param(
    [string]$Key = "",
    [switch]$Tunnel,
    [int]$Port = 8000
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { Write-Error "venv missing: run 'python -m venv .venv' and 'pip install -r requirements.txt'"; exit 1 }

$env:PATHSNAP_KEY = $Key

$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown" } | Select-Object -First 1).IPAddress
Write-Host "LAN address for the UI: http://${ip}:$Port"

if ($Tunnel) {
    if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
        Write-Error "cloudflared not found. Install once with: winget install --id Cloudflare.cloudflared"
        exit 1
    }
    # the tunnel prints its https://....trycloudflare.com address in this separate window
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cloudflared tunnel --url http://localhost:$Port"
    Write-Host "Tunnel window opened: copy the https://...trycloudflare.com URL into the UI."
}

# request logs appear in this window; keep it visible during the evaluation
& $python -m uvicorn api.main:app --host 0.0.0.0 --port $Port
