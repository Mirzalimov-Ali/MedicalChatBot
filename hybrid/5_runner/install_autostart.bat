@echo off
rem Automatically start Personal Health Assistant when you log in to Windows.
rem While connected to AC power, prevent the computer from entering sleep or hibernation.
rem No administrator rights required: a shortcut is placed in the current user's Startup folder.
rem To undo these changes, run uninstall_autostart.bat.

powershell -NoProfile -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Startup') + '\Personal Health Assistant.lnk'); $s.TargetPath = (Get-Command pythonw.exe).Source; $s.Arguments = '\"%~dp0run.py\"'; $s.WorkingDirectory = '%~dp0'; $s.Save()"

powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

echo.
echo Done: Personal Health Assistant will start automatically when you log in to Windows.
echo Sleep and hibernation are disabled while connected to AC power.
echo.
echo Note: After restarting the computer, you must log in to Windows for the server to start.