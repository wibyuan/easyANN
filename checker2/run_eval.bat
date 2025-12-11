@echo off
setlocal enabledelayedexpansion

REM === 消融实验评估脚本 ===
REM 用法: run_eval.bat <method> [dataset]
REM 示例: run_eval.bat hnsw2\baseline DEBUG

if "%~1"=="" (
    echo 用法: run_eval.bat ^<method^> [dataset]
    echo   method: 相对于 ablation 目录的路径 ^(如 hnsw2\baseline^)
    echo   dataset: DEBUG, SIFT, GLOVE, ALL ^(默认 ALL^)
    echo.
    echo 示例:
    echo   run_eval.bat hnsw2\baseline DEBUG
    echo   run_eval.bat hnsw2\baseline_with_fixed_beam SIFT
    echo   run_eval.bat onng1_test2_sq16\baseline ALL
    exit /b 1
)

set METHOD=%~1
set METHOD_PATH=..\ablation\%METHOD%
set DATASET=%~2
if "%DATASET%"=="" set DATASET=ALL

echo === 消融实验评估 ===
echo 方法: %METHOD%
echo 路径: %METHOD_PATH%
echo 数据集: %DATASET%

REM 检查源文件是否存在
if not exist "%METHOD_PATH%\MySolution.cpp" (
    echo 错误: 找不到 %METHOD_PATH%\MySolution.cpp
    echo 请确认路径是否正确
    exit /b 1
)

REM 编译
echo.
echo --- 编译中 ---
g++ -O3 -std=c++17 -DCOUNT_DIST -DTEST_GRAPH -I"%METHOD_PATH%" evaluate.cpp "%METHOD_PATH%\MySolution.cpp" -o eval_ablation.exe

if %ERRORLEVEL% neq 0 (
    echo 编译失败！
    exit /b 1
)

echo 编译成功！

REM 运行评估
echo.
echo --- 运行评估 ---
eval_ablation.exe "%METHOD_PATH%" %DATASET%

echo.
echo === 完成 ===
