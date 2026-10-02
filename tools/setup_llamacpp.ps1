# ==========================================================
# Descarga llama.cpp: codigo fuente (convert_hf_to_gguf.py + gguf-py) y binarios
# precompilados de Windows (llama-quantize.exe). No compila nada.
# llama.cpp publica los binarios en builds "bNNNNN" (marcadas como pre-release),
# asi que se busca la build mas reciente que traiga el zip win-cpu-x64.
# Uso (desde F:\vertex-lab):   powershell -ExecutionPolicy Bypass -File .\tools\setup_llamacpp.ps1
# ==========================================================
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest va mucho mas rapido sin barra
$root  = Split-Path $PSScriptRoot -Parent
$dest  = Join-Path $PSScriptRoot "llama.cpp"
$tmp   = Join-Path $PSScriptRoot "_tmp_llamacpp"
$py    = Join-Path $root "venv\Scripts\python.exe"

$rels = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20" -Headers @{ "User-Agent" = "vertex-lab" }
$rel = $null; $bin = $null
foreach ($r in $rels) {
    $a = $r.assets | Where-Object { $_.name -match "^llama-.*-bin-win-cpu-x64\.zip$" } | Select-Object -First 1
    if ($a) { $rel = $r; $bin = $a; break }
}
if (-not $bin) { throw "No encontre un zip win-cpu-x64 en las ultimas 20 releases de llama.cpp" }
$tag = $rel.tag_name
Write-Host "[*] llama.cpp $tag ($($bin.name))" -ForegroundColor Cyan

if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item -ItemType Directory $tmp | Out-Null

Write-Host "[*] Bajando codigo fuente del tag $tag..."
Invoke-WebRequest "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$tag.zip" -OutFile "$tmp\src.zip"
Write-Host "[*] Bajando binarios..."
Invoke-WebRequest $bin.browser_download_url -OutFile "$tmp\bin.zip"

Write-Host "[*] Descomprimiendo..."
Expand-Archive "$tmp\src.zip" "$tmp\src"
if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }   # solo la copia de llama.cpp de este repo
Move-Item (Get-ChildItem "$tmp\src" -Directory | Select-Object -First 1).FullName $dest
Expand-Archive "$tmp\bin.zip" (Join-Path $dest "bin")
Remove-Item $tmp -Recurse -Force
Set-Content (Join-Path $dest "VERSION_VERTEX.txt") $tag

$q = Get-ChildItem $dest -Recurse -Filter "llama-quantize.exe" | Select-Object -First 1
if ($q) { Write-Host "[+] llama-quantize: $($q.FullName)" -ForegroundColor Green } else { Write-Host "[!] No aparecio llama-quantize.exe" -ForegroundColor Red }

if (Test-Path $py) {
    Write-Host "[*] Probando convert_hf_to_gguf.py con el venv..."
    $env:PYTHONPATH = Join-Path $dest "gguf-py"
    & $py (Join-Path $dest "convert_hf_to_gguf.py") --help *> $null
    if ($LASTEXITCODE -eq 0) { Write-Host "[+] convert_hf_to_gguf.py OK" -ForegroundColor Green }
    else {
        Write-Host "[!] convert_hf_to_gguf.py no arranca. Ejecutalo a mano para ver que modulo falta:" -ForegroundColor Yellow
        Write-Host "    `$env:PYTHONPATH='$dest\gguf-py'; & '$py' '$dest\convert_hf_to_gguf.py' --help"
    }
}
