import os
import shutil

print("--- GLOVE 数据集物理拆分脚本 ---")

# --- GLOVE 配置 (固定) ---
D = 100
BASE_N = 1183514
QUERY_N = 10000
BASE_FLOATS = BASE_N * D
TOTAL_FLOATS = BASE_FLOATS + (QUERY_N * D)

# --- 路径定义 ---
INPUT_FILE = "./data/glove/base.txt"
QUERY_OUT_FILE = "./data/glove/query.txt"
TEMP_BASE_OUT_FILE = "./data/glove/base.txt.tmp"

# 检查输入文件
if not os.path.exists(INPUT_FILE):
    print(f"操，输入文件 {INPUT_FILE} 不存在。")
    exit(1)

print(f"输入文件: {INPUT_FILE}")
print(f"查询输出: {QUERY_OUT_FILE}")
print(f"临时底库: {TEMP_BASE_OUT_FILE}")

try:
    current_count = 0
    with open(INPUT_FILE, 'r') as f_in, \
         open(TEMP_BASE_OUT_FILE, 'w') as f_base_out, \
         open(QUERY_OUT_FILE, 'w') as f_query_out:
        
        for line in f_in:
            numbers = line.split()
            if not numbers:
                continue
            
            for num_str in numbers:
                current_count += 1
                
                if current_count <= BASE_FLOATS:
                    # 写入底库文件
                    f_base_out.write(num_str + " ")
                elif current_count <= TOTAL_FLOATS:
                    # 写入查询库文件
                    f_query_out.write(num_str + " ")
                
                # 每 100 个数（即一个向量）换行
                if current_count % D == 0:
                    if current_count <= BASE_FLOATS:
                        f_base_out.write("\n")
                    else:
                        f_query_out.write("\n")

    print(f"\n处理完成。总共 {current_count:,} 个浮点数。")
    
    if current_count != TOTAL_FLOATS:
        print(f"[失败] 操，文件总数 {current_count} 和预期 {TOTAL_FLOATS} 不符。")
        os.remove(TEMP_BASE_OUT_FILE)
        os.remove(QUERY_OUT_FILE)
    else:
        # 操，最关键的一步：用新的底库覆盖旧的
        shutil.move(TEMP_BASE_OUT_FILE, INPUT_FILE)
        print(f"[成功] 已创建 {QUERY_OUT_FILE}")
        print(f"[成功] 已覆盖 {INPUT_FILE} (现在只包含底库)")

except Exception as e:
    print(f"操，处理过程中发生错误: {e}")
    # 清理垃圾
    if os.path.exists(TEMP_BASE_OUT_FILE):
        os.remove(TEMP_BASE_OUT_FILE)
    if os.path.exists(QUERY_OUT_FILE):
        os.remove(QUERY_OUT_FILE)