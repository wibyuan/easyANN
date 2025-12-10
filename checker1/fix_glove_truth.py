import os

# --- 配置 ---
FILE_PATH = "../data/glove/groundtruth.txt" # 目标文件路径
TARGET_K = 10 # 目标截取长度

print(f"--- 正在修正 GLOVE Ground Truth: {FILE_PATH} ---")

if not os.path.exists(FILE_PATH):
    print(f"错误: 找不到文件 {FILE_PATH}")
    print("请确保您在 /checker 目录下运行此脚本，且文件路径正确。")
    exit(1)

try:
    # 1. 读取所有行
    new_lines = []
    with open(FILE_PATH, 'r') as f:
        lines = f.readlines()
        print(f"读取了 {len(lines)} 行数据。")
        
        if len(lines) > 0:
            # 检查第一行来看看现在的 K 是多少
            first_line_len = len(lines[0].strip().split())
            print(f"检测到原始每行包含 {first_line_len} 个 ID。")
            if first_line_len < TARGET_K:
                print(f"警告: 原始 K ({first_line_len}) 小于目标 K ({TARGET_K})，无需裁剪或数据不足。")
            elif first_line_len == TARGET_K:
                print("提示: 原始 K 已经是 10 了。")

        # 2. 处理每一行：只保留前 10 个
        for line in lines:
            ids = line.strip().split()
            # 截取前 TARGET_K 个
            kept_ids = ids[:TARGET_K]
            # 重新组合成字符串，保留换行符
            new_line = " ".join(kept_ids) + "\n"
            new_lines.append(new_line)

    # 3. 覆盖写入原文件
    with open(FILE_PATH, 'w') as f:
        f.writelines(new_lines)
        
    print(f"[成功] 已将 {FILE_PATH} 裁剪为 K={TARGET_K} 并覆盖保存。")

except Exception as e:
    print(f"处理过程中出错: {e}")