@echo off
setlocal
if "%~1"=="" (
  echo Usage: run_case.bat QUERY_IMAGE [IDENTITY_LIBRARY] [SOURCE_LIBRARY]
  exit /b 2
)
set "PRODUCT_ROOT=%~dp0"
set "PYTHON_CMD=python"
py -3.14 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.14"
if "%~2"=="" (
  %PYTHON_CMD% -B "%PRODUCT_ROOT%launcher.py" --query "%~1"
) else if "%~3"=="" (
  %PYTHON_CMD% -B "%PRODUCT_ROOT%launcher.py" --query "%~1" --identity-library "%~2"
) else (
  %PYTHON_CMD% -B "%PRODUCT_ROOT%launcher.py" --query "%~1" --identity-library "%~2" --source-library "%~3"
)
exit /b %ERRORLEVEL%
