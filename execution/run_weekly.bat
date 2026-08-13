@echo off
REM Weekly paper-trading cycle. Hardened after incident #8:
REM   - waits for DNS/network before starting (task fired pre-connectivity)
REM   - aborts on failure instead of continuing
REM   - propagates a non-zero exit code so Task Scheduler's LastTaskResult
REM     reflects reality (the old version always exited 0 via git push)

setlocal

set REPO=C:\Users\yjv25\Documents\quant-research
set LOG=%REPO%\execution\cycle_log.txt

echo ===== cycle started %date% %time% ===== >> "%LOG%"

REM --- wait up to 5 minutes for name resolution to work ---
set TRIES=0
:waitnet
ping -n 1 api.binance.com >nul 2>&1
if %errorlevel%==0 goto netok
set /a TRIES+=1
if %TRIES% GEQ 10 (
  echo NETWORK UNAVAILABLE after 5 minutes - aborting >> "%LOG%"
  echo ===== cycle FAILED %date% %time% ===== >> "%LOG%"
  exit /b 1
)
timeout /t 30 /nobreak >nul
goto waitnet
:netok
echo network ok after %TRIES% retries >> "%LOG%"

cd /d %REPO%\execution

python run_cycle.py >> "%LOG%" 2>&1
if errorlevel 1 (
  echo run_cycle.py FAILED >> "%LOG%"
  echo ===== cycle FAILED %date% %time% ===== >> "%LOG%"
  exit /b 1
)

python report.py >> "%LOG%" 2>&1
if errorlevel 1 (
  echo report.py FAILED >> "%LOG%"
  echo ===== cycle FAILED %date% %time% ===== >> "%LOG%"
  exit /b 1
)

cd /d %REPO%
git add execution/journal.db docs/live.html
git commit -m "weekly cycle" >> "%LOG%" 2>&1
git push >> "%LOG%" 2>&1
if errorlevel 1 (
  echo git push FAILED - journal is safe locally, will push next run >> "%LOG%"
  echo ===== cycle FINISHED WITH PUSH ERROR %date% %time% ===== >> "%LOG%"
  exit /b 1
)

echo ===== cycle finished OK %date% %time% ===== >> "%LOG%"
exit /b 0
