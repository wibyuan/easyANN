// 编译时需要包含您的暴力搜索实现
#include "../method/brute/MySolution.h" 

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip> // std::setprecision

using namespace std;

// --- GLOVE 配置 (固定) ---
const string GLOVE_INPUT_FILE = "../data/glove/base.txt";
const string GLOVE_OUTPUT_FILE = "../data/glove/groundtruth.txt";
const int GLOVE_D = 100;
const long long GLOVE_BASE_N = 1183514;
const long long GLOVE_QUERY_N = 10000;
const long long GLOVE_BASE_FLOATS = GLOVE_BASE_N * GLOVE_D;
const long long GLOVE_TOTAL_FLOATS = GLOVE_BASE_FLOATS + (GLOVE_QUERY_N * GLOVE_D);

// --- SIFT 配置 (自洽版本) ---
const string SIFT_INPUT_FILE = "../data/sift/base.txt";
const string SIFT_OUTPUT_FILE = "../data/sift/groundtruth.txt";
const int SIFT_D = 128;
const long long SIFT_TOTAL_N = 1000000;
const long long SIFT_QUERY_N = 10000; // 我们将使用前 10k 作为查询
const long long SIFT_BASE_N = SIFT_TOTAL_N - SIFT_QUERY_N; // 剩下 990k 作为底库
const long long SIFT_QUERY_FLOATS = SIFT_QUERY_N * SIFT_D;
const long long SIFT_BASE_FLOATS = SIFT_BASE_N * SIFT_D;
const long long SIFT_TOTAL_FLOATS = SIFT_TOTAL_N * SIFT_D;

// --- 项目配置 ---
const int K = 10; // 查找 10 个最近邻

/**
 * 辅助函数：从 .txt 文件加载数据到一维 vector
 */
bool load_data_from_txt(const string& filename, vector<float>& data_vector, long long expected_count) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "错误: 无法打开 " << filename << endl;
        return false;
    }
    cout << "正在读取 " << filename << " (预期 " << expected_count << " 个)..." << endl;
    
    data_vector.reserve(expected_count); // 预分配内存

    float val;
    long long count = 0;
    while (fin >> val) {
        data_vector.push_back(val);
        count++;
    }
    fin.close();

    cout << "读取完成: " << count << " / " << expected_count << " 个浮点数。" << endl;
    if (count != expected_count) {
        cerr << "严重错误: 文件大小与预期不符！" << endl;
        return false;
    }
    return true;
}


/**
 * 流程 B：处理 SIFT (在内存中拆分 base.txt 为 query 和 base)
 */
void process_sift() {
    cout << "\n--- 正在处理 SIFT 数据集 (内存拆分模式) ---" << endl;
    
    // 1. 加载 SIFT 完整数据
    vector<float> all_sift_data;
    if (!load_data_from_txt(SIFT_INPUT_FILE, all_sift_data, SIFT_TOTAL_FLOATS)) return;

    // 2. 构建底库 (后 990k 向量)
    //    我们创建一个新的向量 `base_data`，只包含底库部分
    vector<float> base_data(all_sift_data.begin() + SIFT_QUERY_FLOATS, all_sift_data.end());
    
    Solution brute_force_solution;
    cout << "正在构建暴力搜索索引 (基于 SIFT 后 " << SIFT_BASE_N << " 个向量)..." << endl;
    brute_force_solution.build(SIFT_D, base_data);
    
    // 释放底库拷贝的内存
    base_data.clear();
    base_data.shrink_to_fit();
    cout << "索引构建完成。" << endl;

    // 3. 循环搜索并生成答案 (使用前 10k 向量)
    ofstream fout(SIFT_OUTPUT_FILE);
    if (!fout.is_open()) {
        cerr << "错误: 无法创建答案文件 " << SIFT_OUTPUT_FILE << endl;
        return;
    }

    cout << "正在为 SIFT 前 " << SIFT_QUERY_N << " 个查询生成基准答案..." << endl;
    vector<float> current_query(SIFT_D);
    int results[K]; // 存放的是 [0, 990k-1] 范围内的 ID

    for (long long i = 0; i < SIFT_QUERY_N; ++i) {
        // 从 all_sift_data 的“前半部分”提取查询向量
        for(int j = 0; j < SIFT_D; ++j) {
            current_query[j] = all_sift_data[i * SIFT_D + j];
        }
        
        brute_force_solution.search(current_query, results);

        // 写入答案, **必须加上偏移量**
        for (int k = 0; k < K; ++k) {
            // SIFT_QUERY_N (10000) 就是偏移量
            // 例子: 暴力搜索返回 ID 0 (这是 990k 底库中的第0个)
            // 它在 1M 原始文件中的真实 ID 是 10000
            fout << (results[k] + SIFT_QUERY_N) << (k == K - 1 ? "" : " ");
        }
        fout << "\n";
    }

    fout.close();
    cout << "\n[成功] SIFT 基准答案已生成: " << SIFT_OUTPUT_FILE << endl;
}


/**
 * 流程 A：处理 GLOVE (在内存中拆分 base.txt 为 base 和 query)
 */
void process_glove() {
    cout << "\n--- 正在处理 GLOVE 数据集 (内存拆分模式) ---" << endl;
    
    // 1. 一次性加载所有数据
    vector<float> all_data;
    if (!load_data_from_txt(GLOVE_INPUT_FILE, all_data, GLOVE_TOTAL_FLOATS)) return;

    // 2. 构建索引 (使用底库数据)
    vector<float> base_data(all_data.begin(), all_data.begin() + GLOVE_BASE_FLOATS);
    
    Solution brute_force_solution;
    cout << "正在构建暴力搜索索引 (基于 GLOVE " << GLOVE_BASE_N << " 个向量)..." << endl;
    brute_force_solution.build(GLOVE_D, base_data);
    
    // 释放底库拷贝的内存
    base_data.clear();
    base_data.shrink_to_fit();
    cout << "索引构建完成。" << endl;

    // 3. 循环搜索并生成答案 (使用 all_data 中的查询库数据)
    ofstream fout(GLOVE_OUTPUT_FILE);
    if (!fout.is_open()) {
        cerr << "错误: 无法创建答案文件 " << GLOVE_OUTPUT_FILE << endl;
        return;
    }

    cout << "正在为 GLOVE " << GLOVE_QUERY_N << " 个查询生成基准答案..." << endl;
    vector<float> current_query(GLOVE_D);
    int results[K];

    for (long long i = 0; i < GLOVE_QUERY_N; ++i) {
        long long query_start_index = GLOVE_BASE_FLOATS + (i * GLOVE_D);
        for(int j = 0; j < GLOVE_D; ++j) {
            current_query[j] = all_data[query_start_index + j];
        }
        brute_force_solution.search(current_query, results);
        // GLOVE 的 ID 是从 0 开始的，不需要偏移
        for (int k = 0; k < K; ++k) fout << results[k] << (k == K - 1 ? "" : " ");
        fout << "\n";
    }

    fout.close();
    cout << "\n[成功] GLOVE 基准答案已生成: " << GLOVE_OUTPUT_FILE << endl;
}


// --- 主函数：自动按顺序执行所有任务 ---
int main() {
    
    // 自动按顺序执行
    process_glove();
    
    // process_sift();
    

    cout << "\n===============================" << endl;
    cout << "所有基准答案已生成完毕。" << endl;
    cout << "===============================" << endl;

    return 0;
}
//g++ generate_truth.cpp ../method/brute/MySolution.cpp -o generate_truth -O3 -std=c++17      