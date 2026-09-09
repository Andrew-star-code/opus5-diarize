# Выпуск TLS-сертификата для локальной сети.
#
# Зачем это вообще нужно: браузер отдаёт микрофон только на защищённом
# соединении. По адресу http://192.168.x.x живая запись работать не будет
# никогда — это ограничение браузера, а не сервиса. Поэтому поднимаем
# локальный удостоверяющий центр через mkcert и выписываем сертификат
# на имя и адрес этой машины.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_tls.ps1

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$certDir = Join-Path $root "certs"

Write-Host ""
Write-Host "=== Сертификат для локальной сети ===" -ForegroundColor Cyan
Write-Host ""

# ── mkcert ──────────────────────────────────────────────────────
$mkcert = Get-Command mkcert -ErrorAction SilentlyContinue
if (-not $mkcert) {
    Write-Host "mkcert не найден. Установите его одной из команд:" -ForegroundColor Yellow
    Write-Host "    winget install FiloSottile.mkcert"
    Write-Host "    choco install mkcert"
    Write-Host ""
    Write-Host "После установки запустите этот скрипт снова."
    exit 1
}

# ── имя и адрес ─────────────────────────────────────────────────
$lanHost = $env:LAN_HOST
if (-not $lanHost) {
    $envFile = Join-Path $root ".env"
    if (Test-Path $envFile) {
        $line = Select-String -Path $envFile -Pattern "^LAN_HOST=" | Select-Object -First 1
        if ($line) { $lanHost = ($line.Line -split "=", 2)[1].Trim() }
    }
}
if (-not $lanHost) { $lanHost = "scribe.local" }

# Определить единственный "правильный" адрес нельзя: на машине запросто
# живут Docker, WSL, ZeroTier, Radmin и VPN-туннель, и любой из них может
# оказаться первым по метрике интерфейса. Поэтому не угадываем, а вписываем
# в сертификат все частные адреса сразу — какой бы из них ни набрал клиент,
# сертификат подойдёт.
$addresses = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object {
        $_.IPAddress -notlike "127.*" -and
        $_.IPAddress -notlike "169.254.*"      # APIPA: сеть не сконфигурирована
    } |
    Select-Object -ExpandProperty IPAddress -Unique

if (-not $addresses) {
    Write-Host "Не удалось найти ни одного сетевого адреса." -ForegroundColor Red
    exit 1
}

# Для инструкций пользователю нужен именно тот адрес, по которому машину
# видят соседи по локальной сети: физический адаптер со шлюзом, без
# виртуальных и туннельных.
$virtual = "vEthernet|ZeroTier|Radmin|Hyper-V|VirtualBox|VMware|TAP|TUN|tun|Loopback|Bluetooth|Docker"
$primary = Get-NetIPConfiguration |
    Where-Object {
        $_.IPv4Address -and $_.IPv4DefaultGateway -and
        $_.InterfaceAlias -notmatch $virtual
    } |
    Sort-Object { $_.NetIPv4Interface.InterfaceMetric } |
    Select-Object -First 1

$ip = if ($primary) { ($primary.IPv4Address | Select-Object -First 1).IPAddress } else { $addresses[0] }

Write-Host "Имя:            $lanHost"
Write-Host "Адрес в сети:   $ip" -NoNewline
if ($primary) { Write-Host "  ($($primary.InterfaceAlias))" } else { Write-Host "  (определён приблизительно)" }
Write-Host "В сертификате:  $($addresses -join ', ')"
Write-Host ""

# ── локальный удостоверяющий центр ──────────────────────────────
Write-Host "Устанавливаю локальный корневой сертификат..." -ForegroundColor Cyan
& mkcert -install
if (-not $?) {
    Write-Host "mkcert -install не отработал." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $certDir)) {
    New-Item -ItemType Directory -Path $certDir | Out-Null
}

$certFile = Join-Path $certDir "lan.pem"
$keyFile = Join-Path $certDir "lan-key.pem"

Write-Host "Выписываю сертификат..." -ForegroundColor Cyan
$names = @($lanHost, "localhost", "127.0.0.1") + $addresses
& mkcert -cert-file $certFile -key-file $keyFile @names
if (-not $?) {
    Write-Host "Не удалось выписать сертификат." -ForegroundColor Red
    exit 1
}

$caRoot = (& mkcert -CAROOT).Trim()

Write-Host ""
Write-Host "Готово." -ForegroundColor Green
Write-Host "  Сертификат: $certFile"
Write-Host "  Ключ:       $keyFile"
Write-Host ""
Write-Host "=== Что сделать на других компьютерах сети ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Скопируйте туда файл rootCA.pem из папки:"
Write-Host "   $caRoot"
Write-Host ""
Write-Host "2. Установите его как доверенный корневой сертификат:"
Write-Host "   Windows - двойной клик, 'Установить сертификат',"
Write-Host "             'Локальный компьютер', 'Доверенные корневые центры'."
Write-Host "   Android - Настройки, Безопасность, Учётные данные, Установить."
Write-Host "   iOS     - открыть файл, затем Настройки, Профиль, и включить"
Write-Host "             доверие в разделе 'Об этом устройстве'."
Write-Host ""
Write-Host "3. Пропишите имя в файле hosts (нужны права администратора):"
Write-Host "   $ip`t$lanHost"
Write-Host "   Windows: C:\Windows\System32\drivers\etc\hosts"
Write-Host "   macOS и Linux: /etc/hosts"
Write-Host ""
Write-Host "Без корневого сертификата браузер посчитает адрес небезопасным"
Write-Host "и не даст доступ к микрофону."
Write-Host ""
