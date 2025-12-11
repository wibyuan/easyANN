#!/bin/bash

# === 消融实验评估脚本 ===
# 用法: ./run_eval.sh <method> [dataset]
# 示例: ./run_eval.sh hnsw2/baseline DEBUG

if [ -z "$1" ]; then
    echo "用法: ./run_eval.sh <method> [dataset]"
    echo "  method: 相对于 ablation 目录的路径 (如 hnsw2/baseline)"
    echo "  dataset: DEBUG, SIFT, GLOVE, ALL (默认 ALL)"
    echo ""
    echo "示例:"
    echo "  ./run_eval.sh hnsw2/baseline DEBUG"
    echo "  ./run_eval.sh hnsw2/baseline_with_fixed_beam SIFT"
    echo "  ./run_eval.sh onng1_test2_sq16/baseline ALL"
    exit 1
fi

METHOD="$1"
METHOD_PATH="../ablation/$METHOD"
DATASET="${2:-ALL}"

echo "=== 消融实验评估 ==="
echo "方法: $METHOD"
echo "路径: $METHOD_PATH"
echo "数据集: $DATASET"

# 检查源文件是否存在
if [ ! -f "$METHOD_PATH/MySolution.cpp" ]; then
    echo "错误: 找不到 $METHOD_PATH/MySolution.cpp"
    echo "请确认路径是否正确"
    exit 1
fi

# 编译
echo ""
echo "--- 编译中 ---"
g++ -O3 -std=c++17 -DCOUNT_DIST -DTEST_GRAPH -I"$METHOD_PATH" evaluate.cpp "$METHOD_PATH/MySolution.cpp" -o eval_ablation

if [ $? -ne 0 ]; then
    echo "编译失败！"
    exit 1
fi

echo "编译成功！"

# 运行评估
echo ""
echo "--- 运行评估 ---"
./eval_ablation "$METHOD_PATH" "$DATASET"

echo ""
echo "=== 完成 ==="
