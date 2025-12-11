#!/usr/bin/env python3
"""
消融实验画图脚本
用法: python plot_ablation.py <method_group> [dataset]
示例: python plot_ablation.py hnsw2 SIFT
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 自动拼接 ablation 路径
ABLATION_BASE = "../ablation"

def plot_qps_recall(ax, data_dict, title):
    """绘制 QPS-Recall 曲线"""
    for name, df in data_dict.items():
        # 按 recall 排序
        df_sorted = df.sort_values('recall')
        ax.plot(df_sorted['recall'], df_sorted['QPS'], 'o-', label=name, markersize=3)
    ax.set_xlabel('Recall@10')
    ax.set_ylabel('QPS')
    ax.set_title(f'{title}: QPS vs Recall')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

def plot_distops_recall(ax, data_dict, title):
    """绘制 dist_ops-Recall 曲线"""
    for name, df in data_dict.items():
        df_sorted = df.sort_values('recall')
        ax.plot(df_sorted['recall'], df_sorted['avg_dist_ops'], 'o-', label=name, markersize=3)
    ax.set_xlabel('Recall@10')
    ax.set_ylabel('Avg Distance Operations')
    ax.set_title(f'{title}: dist_ops vs Recall')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

def plot_degree_distribution(ax, data_dict, title):
    """绘制度数分布（已经是直方图格式）"""
    for name, df in data_dict.items():
        ax.bar(df['degree'], df['count'], alpha=0.5, label=name, width=0.8)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.set_title(f'{title}: Degree Distribution')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

def load_variant_data(base_dir, dataset):
    """加载所有变体的数据"""
    search_data = {}
    degree_data = {}
    build_data = {}

    # 遍历所有子目录
    for variant in os.listdir(base_dir):
        variant_path = os.path.join(base_dir, variant)
        if not os.path.isdir(variant_path):
            continue

        # 搜索结果
        search_csv = os.path.join(variant_path, f'search_{dataset}.csv')
        if os.path.exists(search_csv):
            search_data[variant] = pd.read_csv(search_csv)

        # 度数分布
        degree_csv = os.path.join(variant_path, f'degree_{dataset}.csv')
        if os.path.exists(degree_csv):
            degree_data[variant] = pd.read_csv(degree_csv)

        # Build 统计
        build_csv = os.path.join(variant_path, f'build_stats_{dataset}.csv')
        if os.path.exists(build_csv):
            build_data[variant] = pd.read_csv(build_csv)

    return search_data, degree_data, build_data

def plot_build_comparison(ax, build_data, metric, title, ylabel):
    """绘制 build 指标对比柱状图"""
    names = []
    values = []
    for name, df in build_data.items():
        row = df[df['metric'] == metric]
        if not row.empty:
            names.append(name)
            values.append(float(row['value'].iloc[0]))

    if names:
        x = np.arange(len(names))
        ax.bar(x, values, color='steelblue', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

def main():
    if len(sys.argv) < 2:
        print("用法: python plot_ablation.py <method_group> [dataset]")
        print("  method_group: 算法组 (如 hnsw2, onng1_test2_sq16)")
        print("  dataset: DEBUG, SIFT, GLOVE (默认 SIFT)")
        print("")
        print("示例:")
        print("  python plot_ablation.py hnsw2 SIFT")
        print("  python plot_ablation.py onng1_test2_sq16 GLOVE")
        sys.exit(1)

    method_group = sys.argv[1]
    base_dir = os.path.join(ABLATION_BASE, method_group)
    dataset = sys.argv[2] if len(sys.argv) >= 3 else 'SIFT'

    print(f"=== 消融实验画图 ===")
    print(f"算法组: {method_group}")
    print(f"目录: {base_dir}")
    print(f"数据集: {dataset}")

    if not os.path.exists(base_dir):
        print(f"错误: 目录不存在 {base_dir}")
        sys.exit(1)

    # 加载数据
    search_data, degree_data, build_data = load_variant_data(base_dir, dataset)

    if not search_data:
        print(f"警告: 未找到搜索结果数据 (search_{dataset}.csv)")

    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Ablation Study: {method_group} on {dataset}', fontsize=14)

    # 1. QPS-Recall 曲线
    if search_data:
        ax1 = fig.add_subplot(2, 3, 1)
        plot_qps_recall(ax1, search_data, dataset)

    # 2. dist_ops-Recall 曲线
    if search_data:
        ax2 = fig.add_subplot(2, 3, 2)
        plot_distops_recall(ax2, search_data, dataset)

    # 3. 度数分布
    if degree_data:
        ax3 = fig.add_subplot(2, 3, 3)
        plot_degree_distribution(ax3, degree_data, dataset)

    # 4. Build 时间对比
    if build_data:
        ax4 = fig.add_subplot(2, 3, 4)
        plot_build_comparison(ax4, build_data, 'build_time_ms', f'{dataset}: Build Time', 'Time (ms)')

    # 5. Build dist_ops 对比
    if build_data:
        ax5 = fig.add_subplot(2, 3, 5)
        plot_build_comparison(ax5, build_data, 'build_dist_ops', f'{dataset}: Build dist_ops', 'Distance Operations')

    # 6. 导航准确率对比
    if build_data:
        ax6 = fig.add_subplot(2, 3, 6)
        plot_build_comparison(ax6, build_data, 'nav_accuracy', f'{dataset}: Navigation Accuracy', 'Accuracy')

    plt.tight_layout()

    # 保存图片
    output_file = os.path.join(base_dir, f'ablation_{dataset}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {output_file}")

    plt.show()

if __name__ == '__main__':
    main()
