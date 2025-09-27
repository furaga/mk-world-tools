$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = $scriptDir

Write-Host "Checking for existing plot_mkworld_rate.py processes..."
$existingProcesses = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" | Where-Object { $_.CommandLine -like '*plot_mkworld_rate.py*' }

if ($existingProcesses) {
    Write-Host "Found existing plot_mkworld_rate.py processes. Stopping them..."
    $existingProcesses | ForEach-Object {
        Write-Host "Stopping process with ID: $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Finished stopping existing processes."
} else {
    Write-Host "No existing plot_mkworld_rate.py processes found."
}

Get-Job | Remove-Job

$csvPath = Join-Path $projectRoot ".cache\mkworld-race.csv"
$plotScript = Join-Path $projectRoot "plot_mkworld_rate.py"
$recorderScript = Join-Path $projectRoot "scripts\auto_recorder.py"

$plotJob = Start-Job -ScriptBlock {
    param($workDir, $plotScript, $csvPath)
    Set-Location $workDir
    uv run python $plotScript --file $csvPath
} -ArgumentList $projectRoot, $plotScript, $csvPath

Write-Host "Starting rate plot display in background (Job ID: $($plotJob.Id))..."

Write-Host "Running auto_recorder"
Set-Location $projectRoot
uv run python $recorderScript --out_csv_path $csvPath --game mkworld-race

Write-Host "`nStopping plot job..."
Stop-Job -Job $plotJob
Remove-Job -Job $plotJob

Write-Host "Done."