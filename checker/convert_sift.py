import numpy as np
import os

# --- 辅助函数 (读取 .fvecs) ---
def fvecs_read(filename, c_contiguous=True):
    try:
        fv = np.fromfile(filename, dtype=np.float32)
        if fv.size == 0: return np.zeros((0, 0))
        dim = fv.view(np.int32)[0]
        #
        # 操，之前的 assert 是坨屎，这是新的
        #
        assert fv.size % (dim + 1) == 0, f"Invalid fvecs file: {filename}. fv.size={fv.size}, dim={dim}"
        
        fv = fv.reshape(-1, 1 + dim)
        if not all(fv.view(np.int32)[:, 0] == dim):
            raise IOError("File not in fvecs format (dims not consistent)")
        fv = fv[:, 1:]
        if c_contiguous: fv = fv.copy()
        return fv
    except Exception as e:
        print(f"操, 读取 {filename} 失败: {e}")
        print("请确保文件存在且格式正确。")
        raise e

# --- 辅助函数 (读取 .ivecs) ---
def ivecs_read(filename, c_contiguous=True):
    try:
        iv = np.fromfile(filename, dtype=np.int32)
        if iv.size == 0: return np.zeros((0, 0))
        dim = iv.view(np.int32)[0]
        #
        # 操，之前的 assert 是坨屎，这是新的
        #
        assert iv.size % (dim + 1) == 0, f"Invalid ivecs file: {filename}. iv.size={iv.size}, dim={dim}"
        
        iv = iv.reshape(-1, 1 + dim)
        if not all(iv.view(np.int32)[:, 0] == dim):
            raise IOError("File not in ivecs format (dims not consistent)")
        iv = iv[:, 1:]
        if c_contiguous: iv = iv.copy()
        return iv
    except Exception as e:
        print(f"操, 读取 {filename} 失败: {e}")
        print("请确保文件存在且格式正确。")
        raise e

# --- 0. 定义路径 ---
#
# 操，我们假设 .fvecs 和 .ivecs 文件
# 就在这个脚本的**同一个**目录 (checker 目录)
#
QUERY_FVECS = 'sift_query.fvecs'
TRUTH_IVECS = 'sift_groundtruth.ivecs'

#
# 输出路径（相对于 checker 目录）
#
OUTPUT_QUERY_TXT = "../data/sift/query.txt"
OUTPUT_TRUTH_TXT = "../data/sift/groundtruth.txt"

# 确保 /data/sift 目录存在
os.makedirs("../data/sift", exist_ok=True)


# --- 1. 转换 Query 文件 ---
print("--- 正在转换 SIFT Query ---")
try:
    data_sift_query = fvecs_read(QUERY_FVECS)
    np.savetxt(OUTPUT_QUERY_TXT, data_sift_query, fmt='%f')
    print(f"[成功] 10,000 个查询向量已保存到: {OUTPUT_QUERY_TXT}")
except Exception as e:
    print(f"[失败] 转换 '{QUERY_FVECS}' 出错。")


# --- 2. 转换并验证 Groundtruth 文件 ---
print("\n--- 正在转换 SIFT Groundtruth ---")
try:
    truth_data_k100 = ivecs_read(TRUTH_IVECS)
    print(f"  已加载 '{TRUTH_IVECS}'。Shape: {truth_data_k100.shape} (预期 10000, 100)")

    # --- 关键：执行您要的“确认” ---
    print("--- 正在确认 K-NN 排序 (Sanity Check) ---")
    nn_1 = truth_data_k100[0, 0]  # 第0个查询的第1个邻居
    nn_10 = truth_data_k100[0, 9] # 第0个查询的第10个邻居
    nn_100 = truth_data_k100[0, 99]# 第0个查询的第100个邻居
    
    print(f"  (检查Query 0) 1st NN ID: {nn_1}")
    print(f"  (检查Query 0) 10th NN ID: {nn_10}")
    print(f"  (检查Query 0) 100th NN ID: {nn_100}")

    if nn_1 != nn_10 and nn_10 != nn_100 and nn_1 != nn_100:
        print("  [确认成功] ID 均不相同。文件 100% 是 K=100 的排序列表。")
    else:
        print("  [确认失败] ID 有重复！操，这个文件是坨屎。")

    # --- S截取 K=10 并导出 ---
    truth_data_k10 = truth_data_k100[:, :10]
    print(f"\n  已截取 K=10。Shape: {truth_data_k10.shape}")
    
    np.savetxt(OUTPUT_TRUTH_TXT, truth_data_k10, fmt='%d') # 保存为整数
    print(f"[成功] K=10 标准答案已保存到: {OUTPUT_TRUTH_TXT}")

except Exception as e:
    print(f"[失败] 转换 '{TRUTH_IVECS}' 出错。")