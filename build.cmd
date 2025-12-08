@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title Qpass打包脚本 ...
color 0A

echo ========================================================
echo        Qpass打包脚本
echo ========================================================
echo.

REM --- 1. 检测与创建环境 ---
if exist venv goto :ACTIVATE_VENV

echo [1/5] 创建虚拟环境...
python -m venv venv
if %errorlevel% neq 0 (
    echo [错误] 创建失败，请检查 Python 是否安装。
    pause
    exit /b
)

:ACTIVATE_VENV
echo [2/5] 激活环境...
call venv\Scripts\activate.bat

REM --- 2. 安装依赖 (使用 requirements.txt) ---
echo [3/5] 安装/补全依赖...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [警告] 依赖安装出错，请检查网络或 requirements.txt 文件。
    pause
)

REM --- 3. 清理 ---
echo [4/5] 清理旧文件...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.spec del /q *.spec

REM --- 4. 打包 ---
echo [5/5] 正在打包 (开启控制台以便调试)...
set "APP_NAME=Qpass"

pyinstaller --console --onefile --clean --name "%APP_NAME%" ^
    --hidden-import pynput.keyboard._win32 ^
    --hidden-import pynput.mouse._win32 ^
    --hidden-import comtypes ^
    --hidden-import cv2 ^
    --exclude-module matplotlib ^
    --exclude-module pandas ^
    --exclude-module scipy ^
    --exclude-module PyQt5 ^
    --exclude-module wx ^
    --exclude-module email ^
    --exclude-module http ^
    --exclude-module xml ^
    --exclude-module unittest ^
    main.py

if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [失败] 打包出错！
    pause
    exit /b
)

echo.
echo ========================================================
echo [成功] 打包完成！
echo.
echo 程序位置: dist\%APP_NAME%.exe
echo.
echo 请务必将 exe 放在一个新的文件夹中运行，
echo 因为它需要生成 projects 文件夹。
echo ========================================================
pause
start dist