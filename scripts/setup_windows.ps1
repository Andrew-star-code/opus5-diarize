# Проверка окружения перед первым запуском.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
#
# Ничего не ломает: только смотрит, чего не хватает, и говорит,
# что с этим делать. Правило брандмауэра добавляется, только если
# скрипт запущен от администратора.

$root = Split-Path -Parent $PSScriptRoot
$problems = @()

function Test-Step {
    param([string]$Name, [scriptblock]$Check, [string]$Fix)
    Write-Host -NoNewline ("  {0,-34}" -f $Name)
    $result = $null
    try { $result = & $Check } catch { $result = $null }
    if ($result) {
        Write-Host "OK  $result" -ForegroundColor Green
        return $true
    }
    Write-Host "нет" -ForegroundColor Yellow
    if ($Fix) { $script:problems += "$Name -> $Fix" }
    return $false
}

Write-Host ""
Write-Host "=== Проверка окружения ===" -ForegroundColor Cyan
Write-Host ""

Test-Step "Docker" {
    (docker --version) 2>$null
} "установите Docker Desktop с включённой поддержкой WSL2" | Out-Null

Test-Step "Docker Compose" {
    (docker compose version) 2>$null
} "обновите Docker Desktop" | Out-Null

Test-Step "WSL2" {
    $out = (wsl -l -v) 2>$null
    if ($out) { "установлен" } else { $null }
} "выполните: wsl --install" | Out-Null

$gpuOk = Test-Step "Видеокарта NVIDIA" {
    (nvidia-smi --query-gpu=name --format=csv,noheader) 2>$null
} "нужен драйвер NVIDIA не ниже 570 (для RTX 50xx)"

Test-Step "Драйвер и CUDA в контейнере" {
    $out = (docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi -L) 2>$null
    if ($out) { "проброс работает" } else { $null }
} "установите NVIDIA Container Toolkit и перезапустите Docker Desktop" | Out-Null

# ── .env ────────────────────────────────────────────────────────
$envFile = Join-Path $root ".env"
$envExample = Join-Path $root ".env.example"
Write-Host -NoNewline ("  {0,-34}" -f "Файл .env")
if (Test-Path $envFile) {
    Write-Host "OK" -ForegroundColor Green
} else {
    Copy-Item $envExample $envFile
    Write-Host "создан из .env.example" -ForegroundColor Yellow
    $problems += "Файл .env -> впишите HF_TOKEN, он нужен один раз для скачивания моделей"
}

# ── сертификат ──────────────────────────────────────────────────
Write-Host -NoNewline ("  {0,-34}" -f "TLS-сертификат")
if (Test-Path (Join-Path $root "certs\lan.pem")) {
    Write-Host "OK" -ForegroundColor Green
} else {
    Write-Host "нет" -ForegroundColor Yellow
    $problems += "TLS-сертификат -> выполните scripts\setup_tls.ps1 (без него не будет микрофона)"
}

# ── брандмауэр ──────────────────────────────────────────────────
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

Write-Host -NoNewline ("  {0,-34}" -f "Правило брандмауэра")
$rule = Get-NetFirewallRule -DisplayName "Scribe (HTTP/HTTPS)" -ErrorAction SilentlyContinue
if ($rule) {
    Write-Host "OK" -ForegroundColor Green
} elseif ($isAdmin) {
    New-NetFirewallRule -DisplayName "Scribe (HTTP/HTTPS)" `
        -Direction Inbound -Action Allow -Protocol TCP `
        -LocalPort 80, 443 -Profile Private | Out-Null
    Write-Host "добавлено" -ForegroundColor Green
} else {
    Write-Host "нет" -ForegroundColor Yellow
    $problems += "Правило брандмауэра -> запустите этот скрипт от администратора"
}

# ── адрес в сети ────────────────────────────────────────────────
$ip = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object {
        $_.IPAddress -notlike "127.*" -and
        $_.IPAddress -notlike "169.254.*" -and
        $_.InterfaceAlias -notlike "*vEthernet*" -and
        $_.InterfaceAlias -notlike "*Loopback*"
    } |
    Sort-Object InterfaceMetric |
    Select-Object -First 1 -ExpandProperty IPAddress

Write-Host ""
if ($problems.Count -eq 0) {
    Write-Host "Всё на месте. Запуск:" -ForegroundColor Green
    Write-Host "    docker compose up -d --build"
    if ($ip) {
        Write-Host ""
        Write-Host "Адрес для остальных в сети: https://$ip"
    }
} else {
    Write-Host "Осталось сделать:" -ForegroundColor Yellow
    foreach ($item in $problems) { Write-Host "  - $item" }
}

if (-not $gpuOk) {
    Write-Host ""
    Write-Host "Без видеокарты сервис запустится, но распознавание пойдёт" -ForegroundColor Yellow
    Write-Host "на процессоре и будет медленнее в десятки раз." -ForegroundColor Yellow
}
Write-Host ""
