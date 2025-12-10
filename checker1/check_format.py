import os

def check_file_format(filepath, expected_n, expected_d):
    """
    流式读取 .txt 文件，确认其格式和总数是否符合预期。
    """
    print(f"--- 正在检查: {filepath} ---")
    
    if not os.path.exists(filepath):
        print(f"错误: 文件未找到: {filepath}")
        return

    expected_total = expected_n * expected_d
    actual_count = 0
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # split() 默认会处理所有空白（空格和换行符）
                numbers = line.split()
                if not numbers:
                    continue
                    
                actual_count += len(numbers)
                
                # 检查第一个数是否为浮点数
                if actual_count > 0 and actual_count <= len(numbers):
                    try:
                        float(numbers[0])
                    except ValueError:
                        print(f"错误: 文件中包含非数字内容: '{numbers[0]}'")
                        return

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    print(f"预期维度 (d): {expected_d}")
    print(f"预期向量数 (N): {expected_n:,}")
    print(f"预期总浮点数 (N * d): {expected_total:,}")
    print(f"实际在文件中找到的浮点数: {actual_count:,}")

    if actual_count == expected_total:
        print("\n[成功] 文件格式和总数均符合预期！")
    else:
        print(f"\n[失败] 文件总数与预期不符！")
        print(f"  缺少或多出了 {abs(actual_count - expected_total):,} 个数字。")

if __name__ == "__main__":
    # 根据 PJ.pdf 第 11 页定义
    # SIFT: N=1,000,000, d=128
    # GLOVE: N=1,183,514, d=100
    
    # 检查 SIFT
    sift_n = 1000000
    sift_d = 128
    sift_path = "./data/sift/base.txt"
    check_file_format(sift_path, sift_n, sift_d)
    
    print("\n" + "="*30 + "\n")
    
    # 检查 GLOVE
    glove_n = 1183514
    glove_d = 100
    glove_path = "./data/glove/base.txt"
    check_file_format(glove_path, glove_n, glove_d)