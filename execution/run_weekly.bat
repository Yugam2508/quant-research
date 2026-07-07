@echo off
REM Weekly paper-trading cycle: rebalance -> report -> push to GitHub.
REM Scheduled via Windows Task Scheduler, Mondays 09:00.
REM API keys are read from user environment variables (set once with setx).

cd /d C:\Users\yjv25\Documents\quant-research\execution

echo ===== cycle started %date% %time% ===== >> cycle_log.txt
python run_cycle.py >> cycle_log.txt 2>&1
python report.py >> cycle_log.txt 2>&1

cd /d C:\Users\yjv25\Documents\quant-research
git add execution/journal.db docs/live.html
git commit -m "weekly cycle" >> execution\cycle_log.txt 2>&1
git push >> execution\cycle_log.txt 2>&1

echo ===== cycle finished %date% %time% ===== >> execution\cycle_log.txt
