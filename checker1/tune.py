import optuna
import subprocess
import sys

# 配置
EXE_PATH = "tune.exe"

def objective(trial):
    # 1. 定义搜索空间
    ef_cons = trial.suggest_int("efConstruction", 100, 140)
    m_max0 = trial.suggest_int("Mmax0", 120, 130)
    min_edges = trial.suggest_int("MIN_EDGES", 2, 30)
    gamma = trial.suggest_float("gamma", 0.2, 0.3, step=0.001)

    # 2. 调用 C++ 程序
    # 参数顺序: [ef] [Mmax0] [min_edges] [gamma]
    cmd = [EXE_PATH, str(ef_cons), str(m_max0), str(min_edges), str(gamma)]
    
    print(f"\nTrial {trial.number}: {cmd}")
    
    try:
        # 设置 2000 秒 超时熔断
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2000)
        output = result.stdout
        
        # 3. 解析输出 "[RESULT] recall qps avg_dist_ops build_time"
        line = [l for l in output.split('\n') if l.startswith("[RESULT]")]
        if not line:
            print("Error: No result line found.")
            return float('inf') # 失败
            
        parts = line[0].split()
        recall = float(parts[1])
        qps = float(parts[2])
        avg_dist = float(parts[3])
        build_time = float(parts[4])
        
        print(f"  -> Recall: {recall:.4f}, DistOps: {avg_dist:.0f}, QPS: {qps:.0f}, Build: {build_time:.0f}s")

        # 4. 目标函数 (Minimize Distance Ops)
        # 硬约束: Recall >= 0.99
        if recall < 0.99:
            # 惩罚项: 距离无限大
            # 为了给优化器梯度，可以返回一个巨大的基数 + 差距
            return 1e9 + (0.99 - recall) * 1e9
            
        return avg_dist

    except subprocess.TimeoutExpired:
        print("  -> Timeout (Build > 2000s)!")
        return float('inf') # 超时惩罚
    except Exception as e:
        print(f"  -> Exception: {e}")
        return float('inf')

if __name__ == "__main__":
    # 最小化距离计算次数
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    
    print("\nBest Params:", study.best_params)
    print("Min Dist Ops:", study.best_value)