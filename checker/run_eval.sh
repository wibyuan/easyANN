#!/bin/bash
# 操，这是一键评测脚本
#
# 用法: ./run_eval.sh [method_name]
# 示例: ./run_eval.sh brute
#
# 这脚本会干：
# 1. 检查您是不是给了 "brute" 这种方法名
# 2. 告诉 g++ 去 ../method/brute 目录里找 .h (include) 和 .cpp (link)
# 3. 编译
# 4. 运行 ./evaluate，并把结果文件路径 ../method/brute/evaluation_results.txt 传进去

# 检查参数
# $1 是第一个参数。"-z" 检查字符串是否为空。
if [ -z "$1" ]; then
    echo "操，你他妈没告诉我评测哪个方法名。"
    echo ""
    echo "用法: ./run_eval.sh [method_name]"
    echo "示例: ./run_eval.sh brute"
    exit 1 # "goto :eof" 在 bash 里就是 "exit"
fi

# "SET VAR=VALUE" 在 bash 里是 "VAR=VALUE"
METHOD_NAME="$1"
METHOD_DIR="../method/$METHOD_NAME"
INCLUDE_PATH="-I$METHOD_DIR"
SRC_FILE="$METHOD_DIR/MySolution.cpp"
OUTPUT_FILE="$METHOD_DIR/evaluation_results.txt"

# 变量在 bash 里用 "$VAR" 引用
echo "--- 正在自动链接: $METHOD_NAME ---"
echo "--- Include 路径: $INCLUDE_PATH"
echo "--- 源文件: $SRC_FILE"
echo "--- 输出路径: $OUTPUT_FILE"
echo ""

# --- 1. 编译 ---
echo "正在编译..."
# 建议给变量加上引号，防止路径里有空格
# "$?" 是上一条命令的退出码 (等价于 %ERRORLEVEL%)
g++ "$INCLUDE_PATH" evaluate.cpp "$SRC_FILE" -o evaluate -O3 -std=c++17

if [ $? -ne 0 ]; then # "-ne 0" 等价于 "NEQ 0"
    echo "操，编译他妈的失败了！"
    exit 1
fi
echo "编译成功。"

# --- 2. 运行 ---
echo ""
echo "正在执行评测..."
# 在 Linux/macOS 上，可执行文件通常没有 .exe 后缀
# 并且必须用 "./" 来执行当前目录下的文件
./evaluate "$OUTPUT_FILE"

echo ""
echo "--- 评测完毕 ---"
echo "--- 结果已保存到 $OUTPUT_FILE ---"