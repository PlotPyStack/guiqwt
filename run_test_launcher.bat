REM This code was copied from PythonQwt project (license: MIT, copyright Pierre Raybaut)
@echo off
setlocal
set PYTHONPATH=%cd%\..\PythonQwt;%cd%\..\guidata;%cd%
if defined WINPYDIRBASE (
    call %WINPYDIRBASE%\scripts\env.bat
    @echo ==============================================================================
    @echo:
    @echo Using WinPython from %WINPYDIRBASE%
    @echo:
    @echo ==============================================================================
    @echo:
    )
python -m guiqwt.tests.__init__