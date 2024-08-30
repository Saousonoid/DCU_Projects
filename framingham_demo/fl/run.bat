@echo off
call conda activate fedenv
set TF_CPP_MIN_LOG_LEVEL=2

:: Run the scripts in parallel
start "" cmd /c "python server.py test.csv 30 & pause"
start "" cmd /c "python client.py client1.csv prevalentHyp 0  & pause"
start "" cmd /c "python client.py client2.csv TenYearCHD  1 & pause"

:: Wait for all processes to finish
echo All scripts are running in parallel.
