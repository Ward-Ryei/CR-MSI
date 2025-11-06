@echo off
setlocal enabledelayedexpansion
echo 当前脚本的绝对路径: %~dp0
cd /d %~dp0
python unit_run.py
pause