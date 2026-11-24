@echo off
rem Stop Personal Health Assistant server and its related processes.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
"$runner = [System.IO.Path]::GetFullPath('%~dp0run.py');" ^
"$processes = Get-CimInstance Win32_Process;" ^
"$runnerProcess = $processes | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($runner) };" ^
"if (-not $runnerProcess) { Write-Host 'Personal Health Assistant is not running.'; exit 0 };" ^
"$children = $processes | Where-Object { $_.ParentProcessId -eq $runnerProcess.ProcessId };" ^
"foreach ($child in $children) { Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue; Write-Host 'Stopped:' $child.Name $child.ProcessId };" ^
"Stop-Process -Id $runnerProcess.ProcessId -Force -ErrorAction SilentlyContinue;" ^
"Write-Host 'Personal Health Assistant stopped.'"

echo.
pause
