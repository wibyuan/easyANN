@echo off
REM 一键评测脚本
REM 
REM 用法: .\run_eval.bat [method_name]
REM 示例: .\run_eval.bat brute
REM
REM 
REM 这脚本会干：
REM 1. 检查您是不是给了 "brute" 这种方法名
REM 2. 告诉 g++ 去 ../method/brute 目录里找 .h (include) 和 .cpp (link)
REM 3. 编译
REM 4. 运行 .\evaluate.exe，并把结果文件路径 ../method/brute/evaluation_results.txt 传进去

REM 检查参数
IF "%1"=="" (
    echo 错误: 未指定评测方法名。
    echo.
    echo 用法: .\run_eval.bat [method_name]
    echo 示例: .\run_eval.bat brute
    goto :eof
)

SET METHOD_NAME=%1
SET METHOD_DIR=../method/%METHOD_NAME%
SET INCLUDE_PATH=-I%METHOD_DIR%
SET SRC_FILE=%METHOD_DIR%/MySolution.cpp
SET OUTPUT_FILE=%METHOD_DIR%/evaluation_results.txt

echo --- 正在自动链接: %METHOD_NAME% ---
echo --- Include 路径: %INCLUDE_PATH%
echo --- 源文件: %SRC_FILE%
echo --- 输出路径: %OUTPUT_FILE%
echo.

REM --- 1. 编译 ---
echo 正在编译 (带性能计数)...
REM 加上 -DCOUNT_DIST
g++ %INCLUDE_PATH% evaluate.cpp %SRC_FILE% -o evaluate -O3 -std=c++17 -DCOUNT_DIST -DTEST_GRAPH

IF %ERRORLEVEL% NEQ 0 (
    echo 错误: 编译失败！
    goto :eof
)
echo 编译成功。

REM --- 2. 运行 ---
echo.
echo 正在执行评测...
.\evaluate.exe %OUTPUT_FILE%

echo.
echo --- 评测完毕 ---
echo --- 结果已保存到 %OUTPUT_FILE% ---