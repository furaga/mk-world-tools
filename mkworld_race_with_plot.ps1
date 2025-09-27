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

$csvPath = ".cache/mkworld-race.csv"

$plotJob = Start-Job -ScriptBlock {
    param($workDir, $csvPath)
    Set-Location $workDir
    uv run python plot_mkworld_rate.py --file $csvPath
} -ArgumentList $PWD, $csvPath

Write-Host "Starting rate plot display in background (Job ID: $($plotJob.Id))..."

Write-Host "Running make mkworld-race --debug..."
make mkworld-race --debug

Write-Host "`nStopping plot job..."
Stop-Job -Job $plotJob
Remove-Job -Job $plotJob

Write-Host "Done."