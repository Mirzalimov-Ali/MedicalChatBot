@echo off
rem Undo install_autostart.bat: disable automatic startup and restore sleep settings.

del "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Personal Health Assistant.lnk" 2>nul

powercfg /change standby-timeout-ac 30
powercfg /change hibernate-timeout-ac 0

echo.
echo Personal Health Assistant automatic startup has been disabled.
echo Sleep timeout restored to 30 minutes.
echo Hibernation remains disabled while connected to AC power.