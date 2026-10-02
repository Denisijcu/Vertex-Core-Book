# ==========================================================
# Vertex Lab - Reconstruye el venv limpio para GTX 1660 Ti
# Uso (desde F:\vertex-lab):   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
# El venv anterior NO se borra: se renombra a venv_bak_<fecha>.
# ==========================================================
param(
    [string]$Python = "D:\python312\python.exe",
    [string[]]$CudaTags = @("cu128", "cu126", "cu130")
)

Set-Location $PSScriptRoot
Write-Host "== Vertex Lab setup en $PSScriptRoot ==" -ForegroundColor Cyan

if (-not (Test-Path $Python)) { Write-Host "[!] No existe $Python. Pasa -Python <ruta a python 3.12>" -ForegroundColor Red; exit 1 }

if (Test-Path .\venv) {
    $bak = "venv_bak_" + (Get-Date -Format "yyyyMMdd_HHmm")
    try {
        Rename-Item .\venv $bak -ErrorAction Stop
        Write-Host "[i] venv anterior renombrado a $bak (bórralo cuando confirmes que todo va bien)"
    } catch {
        Write-Host "[!] No pude renombrar venv (¿VS Code/terminal/Python lo tiene abierto?). Ciérralos y repite." -ForegroundColor Red
        exit 1
    }
}

& $Python -m venv venv
$py = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
& $py -m pip install --upgrade pip wheel

$ok = $false
foreach ($tag in $CudaTags) {
    Write-Host "[*] Instalando torch ($tag)..." -ForegroundColor Cyan
    & $py -m pip install torch --index-url "https://download.pytorch.org/whl/$tag"
    if ($LASTEXITCODE -eq 0) {
        & $py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
        if ($LASTEXITCODE -eq 0) { $ok = $true; Write-Host "[+] torch con CUDA OK ($tag)" -ForegroundColor Green; break }
        & $py -m pip uninstall -y torch
    }
}
if (-not $ok) { Write-Host "[!] No logré un torch con CUDA funcional." -ForegroundColor Red; exit 1 }

& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "[!] Falló requirements.txt" -ForegroundColor Red; exit 1 }

& $py scripts\00_check_env.py
