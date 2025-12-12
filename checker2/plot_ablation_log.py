#!/usr/bin/env python3
"""
消融实验画图脚本 (对数坐标版)
用法: python plot_ablation_log.py <method_group> [dataset]
示例: python plot_ablation_log.py hnsw2 SIFT

特点:
- X 轴使用 -log10(1-recall) 变换，高召回率区域更清晰
- Y 轴使用 log10 刻度
"""

import os
import sys
import csv
import math
import matplotlib.pyplot as plt

ABLATION_BASE = "../ablation"

def read_csv(filepath):
    """读取 CSV 文件，返回字典列表"""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def transform_recall(recall):
    """将 recall 转换为 -log10(1-recall)，高召回率时值更大"""
    if recall >= 1.0:
        recall = 0.9999
    if recall <= 0:
        recall = 0.0001
    return -math.log10(1 - recall)

def plot_qps_recall_log(ax, data_dict, title):
    """绘制 QPS-Recall 曲线 (对数坐标)"""
    for name, rows in data_dict.items():
        if not rows:
            continue
        sorted_rows = sorted(rows, key=lambda x: float(x['recall']))
        recalls_transformed = [transform_recall(float(r['recall'])) for r in sorted_rows]
        qps = [float(r['QPS']) for r in sorted_rows]
        ax.plot(recalls_transformed, qps, 'o-', label=name, markersize=3)

    ax.set_xlabel('-log10(1 - Recall@10)')
    ax.set_ylabel('QPS (log scale)')
    ax.set_yscale('log')
    ax.set_title(f'{title}: QPS vs Recall (log scale)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # 添加更多原始 recall 值的刻度标签
    recall_ticks = [0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995, 0.999]
    tick_positions = [transform_recall(r) for r in recall_ticks]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f'{r*100:.1f}%' if r < 0.99 else f'{r*100:.2f}%' for r in recall_ticks], rotation=45, ha='right', fontsize=8)

def plot_distops_recall_log(ax, data_dict, title):
    """绘制 dist_ops-Recall 曲线 (对数坐标)"""
    for name, rows in data_dict.items():
        if not rows:
            continue
        sorted_rows = sorted(rows, key=lambda x: float(x['recall']))
        recalls_transformed = [transform_recall(float(r['recall'])) for r in sorted_rows]
        dist_ops = [float(r['avg_dist_ops']) for r in sorted_rows]
        ax.plot(recalls_transformed, dist_ops, 'o-', label=name, markersize=3)

    ax.set_xlabel('-log10(1 - Recall@10)')
    ax.set_ylabel('Avg Distance Operations (log scale)')
    ax.set_yscale('log')
    ax.set_title(f'{title}: dist_ops vs Recall (log scale)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    recall_ticks = [0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995, 0.999]
    tick_positions = [transform_recall(r) for r in recall_ticks]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f'{r*100:.1f}%' if r < 0.99 else f'{r*100:.2f}%' for r in recall_ticks], rotation=45, ha='right', fontsize=8)

def load_variant_data(base_dir, dataset):
    """加载所有变体的数据"""
    search_data = {}

    for variant in os.listdir(base_dir):
        variant_path = os.path.join(base_dir, variant)
        if not os.path.isdir(variant_path):
            continue

        search_csv = os.path.join(variant_path, f'search_{dataset}.csv')
        data = read_csv(search_csv)
        if data:
            search_data[variant] = data

    return search_data

def main():
    if len(sys.argv) < 2:
        print("用法: python plot_ablation_log.py <method_group> [dataset]")
        print("  method_group: 算法组 (如 hnsw2, onng1_test2)")
        print("  dataset: DEBUG, SIFT, GLOVE (默认 SIFT)")
        sys.exit(1)

    method_group = sys.argv[1]
    base_dir = os.path.join(ABLATION_BASE, method_group)
    dataset = sys.argv[2] if len(sys.argv) >= 3 else 'SIFT'

    print(f"=== 消融实验画图 (对数坐标) ===")
    print(f"算法组: {method_group}")
    print(f"数据集: {dataset}")

    if not os.path.exists(base_dir):
        print(f"错误: 目录不存在 {base_dir}")
        sys.exit(1)

    search_data = load_variant_data(base_dir, dataset)

    if not search_data:
        print(f"警告: 未找到搜索结果数据 (search_{dataset}.csv)")
        sys.exit(1)

    # 创建图表: 上下两张图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Ablation Study (Log Scale): {method_group} on {dataset}', fontsize=14)

    plot_qps_recall_log(ax1, search_data, dataset)
    plot_distops_recall_log(ax2, search_data, dataset)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = os.path.join(base_dir, f'ablation_{dataset}_log.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {output_file}")

    plt.show()

if __name__ == '__main__':
    main()
