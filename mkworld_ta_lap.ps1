$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = $scriptDir

$recorderScript = Join-Path $projectRoot "scripts\ta_lap_recorder.py"

Write-Host "Starting Time Attack Lap Timer..."
Set-Location $projectRoot
uv run python $recorderScript

Write-Host "Done."