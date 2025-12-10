#!/bin/bash
# 操，这是一键评测脚本 (Linux/macOS 版)

# 检查参数
if [ -z "$1" ]; then
    echo "操，你他妈没告诉我评测哪个方法名。"
    echo ""
    echo "用法: ./run_eval.sh [method_name] [mode]"
    echo "示例 1 (极速模式): ./run_eval.sh hnsw2"
    echo "示例 2 (计数模式): ./run_eval.sh hnsw2 count"
    exit 1
fi

METHOD_NAME="$1"
MODE="$2"

METHOD_DIR="../method/$METHOD_NAME"
INCLUDE_PATH="-I$METHOD_DIR"
SRC_FILE="$METHOD_DIR/MySolution.cpp"
OUTPUT_FILE="$METHOD_DIR/evaluation_results.txt"

# --- 编译器参数配置 ---
# 1. -O3: 开启最高优化
# 2. -std=c++17: 必须的
# 3. -fopenmp: 操，这个必须加，否则多线程 build 无效！
CXX_FLAGS="-O3 -std=c++17 -fopenmp"

# 检查是否开启计数模式
if [ "$MODE" == "count" ]; then
    echo "!!! 警告：已开启性能计数模式 (-DCOUNT_DIST) !!!"
    echo "!!! 这会拖慢运行速度，仅用于调试算法效率 !!!"
    CXX_FLAGS="$CXX_FLAGS -DCOUNT_DIST -DTEST_GRAPH"
fi

echo "--- 正在自动链接: $METHOD_NAME ---"
echo "--- 编译参数: $CXX_FLAGS"
echo "--- 输出路径: $OUTPUT_FILE"
echo ""

# --- 1. 编译 ---
echo "正在编译..."
# 这里的变量加引号是为了防止路径带空格
g++ $INCLUDE_PATH evaluate.cpp "$SRC_FILE" -o evaluate $CXX_FLAGS

if [ $? -ne 0 ]; then
    echo "操，编译他妈的失败了！"
    exit 1
fi
echo "编译成功。"

# --- 2. 运行 ---
echo ""
echo "正在执行评测..."

./evaluate "$OUTPUT_FILE"

echo ""
echo "--- 评测完毕 ---"
echo "--- 结果已保存到 $OUTPUT_FILE ---"