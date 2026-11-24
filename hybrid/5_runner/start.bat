@echo off
rem Start Personal Health Assistant in the background.
rem The server link will be saved to data/backend_link.txt.

rem Start the main runner without opening a console window.
start "" pythonw "%~dp0run.py"

echo.
echo Personal Health Assistant is starting in the background.
echo.
echo Server link: %~dp0data\backend_link.txt
echo Main log:    %~dp0data\run.log
echo.
echo To stop the server, run stop.bat.
echo.
