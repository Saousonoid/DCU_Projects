@echo off

set TF_CPP_MIN_LOG_LEVEL=2
:: Run the scripts in parallel
start "" cmd /c "python server.py test.csv 30 & pause"
timeout /t 5 /nobreak >nul 2>&1
start "" cmd /c "python client.py dia.csv diabetes 0 & pause"
start "" cmd /c "python client.py ckd.csv ckd 1 & pause"

:: Wait for all processes to finish
echo All scripts are running in parallel.
