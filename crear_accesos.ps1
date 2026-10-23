<#
    crear_accesos.ps1

    Genera los accesos directos del escritorio para el proyecto "fc-filtro-inverso".

    Uso:
        powershell -ExecutionPolicy Bypass -File .\crear_accesos.ps1
        powershell -ExecutionPolicy Bypass -File .\crear_accesos.ps1 -Force

    Crea tres accesos:
        Filtro Inverso.lnk              -> interfaz grafica (sin consola)
        Filtro Inverso (Consola).lnk    -> interfaz grafica con consola para depurar
        Calibrar Filtro Inverso.lnk     -> calibracion hardware (log sine sweep)
#>

[CmdletBinding()]
param(
    [string]$ProjectDir = '',
    [string]$PythonExe = '',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

# --- 1. Carpeta del proyecto ----------------------------------------------
if (-not $ProjectDir) {
    $ProjectDir = $PSScriptRoot
}
if (-not $ProjectDir) {
    $ProjectDir = (Get-Location).Path
}
if (-not (Test-Path -LiteralPath $ProjectDir)) {
    throw "La carpeta del proyecto no existe: $ProjectDir"
}
$ProjectDir = (Get-Item -LiteralPath $ProjectDir).FullName.TrimEnd('\')

# --- 2. Interprete de Python con las dependencias -------------------------
function Test-Dependencies {
    param([string]$Exe)
    if (-not $Exe -or -not (Test-Path -LiteralPath $Exe)) { return $false }
    $code = 'import numpy, sounddevice, pyqtgraph, PySide6'
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Exe -c $code 2>$null | Out-Null
    $ok = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prev
    return $ok
}

function Resolve-Python {
    param([string]$Candidate)

    if ($Candidate) {
        if (-not (Test-Path -LiteralPath $Candidate)) {
            throw "El interprete indicado no existe: $Candidate"
        }
        return (Get-Item -LiteralPath $Candidate).FullName
    }

    # El launcher "py" apunta al interprete instalado.
    $pyCmd = Get-Command 'py.exe' -ErrorAction SilentlyContinue
    if (-not $pyCmd) { $pyCmd = Get-Command 'py' -ErrorAction SilentlyContinue }
    if ($pyCmd) {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $out = & $pyCmd.Source -c 'import sys; print(sys.executable)' 2>$null
        $ErrorActionPreference = $prev
        foreach ($line in @($out)) {
            $exe = "$line".Trim()
            if ($exe -and (Test-Path -LiteralPath $exe) -and (Test-Dependencies $exe)) {
                return $exe
            }
        }
    }

    # Cualquier python.exe del PATH que ya tenga las dependencias.
    foreach ($cmd in @(Get-Command 'python.exe' -ErrorAction SilentlyContinue)) {
        if (Test-Dependencies $cmd.Source) { return $cmd.Source }
    }

    throw 'No se encontro un Python con numpy, sounddevice, pyqtgraph y PySide6. Instala: py -m pip install -r requirements.txt'
}

$python = Resolve-Python -Candidate $PythonExe
$pythonw = Join-Path (Split-Path -Parent $python) 'pythonw.exe'
if (-not (Test-Path -LiteralPath $pythonw)) { $pythonw = $python }

$desktop = [Environment]::GetFolderPath('Desktop')
if (-not $desktop) { throw 'No se pudo resolver la carpeta del escritorio.' }

Write-Host "Proyecto   : $ProjectDir"
Write-Host "Python     : $python"
Write-Host "Escritorio : $desktop"
Write-Host ''

# --- 3. Crear accesos ------------------------------------------------------
$shell = New-Object -ComObject WScript.Shell

function New-Shortcut {
    param(
        [string]$Name,
        [string]$TargetPath,
        [string]$Arguments = '',
        [string]$Description = ''
    )

    $linkPath = Join-Path $desktop $Name
    if ((Test-Path -LiteralPath $linkPath) -and -not $Force) {
        Write-Host "[skip] $Name  (ya existe; usa -Force para sobrescribir)"
        return
    }

    $shortcut = $shell.CreateShortcut($linkPath)
    $shortcut.TargetPath = $TargetPath
    $shortcut.Arguments = $Arguments
    $shortcut.WorkingDirectory = $ProjectDir
    $shortcut.Description = $Description
    $shortcut.IconLocation = "$python,0"
    $shortcut.WindowStyle = 1
    $shortcut.Save()

    Write-Host "[ok]   $Name"
}

$mainPy = Join-Path $ProjectDir 'main.py'
$calibPy = Join-Path $ProjectDir 'calibrate_hardware.py'

New-Shortcut -Name 'Filtro Inverso.lnk' `
    -TargetPath $pythonw `
    -Arguments ('"' + $mainPy + '"') `
    -Description 'Ecualizador FFT full-duplex en tiempo real (interfaz grafica)'

New-Shortcut -Name 'Filtro Inverso (Consola).lnk' `
    -TargetPath $python `
    -Arguments ('"' + $mainPy + '"') `
    -Description 'Ecualizador FFT en tiempo real con consola para ver errores'

New-Shortcut -Name 'Calibrar Filtro Inverso.lnk' `
    -TargetPath $python `
    -Arguments ('"' + $calibPy + '"') `
    -Description 'Medir la respuesta del circuito con un barrido logaritmico (log sine sweep)'

Write-Host ''
Write-Host 'Accesos directos listos en el escritorio.'
