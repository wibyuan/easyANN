#include "MySolution.h" 
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono> 
#include <set>    

using namespace std;

// --- DEBUG 配置 (现在指向 .bin 文件) ---
const string DEBUG_BASE_FILE = "../data/debug/base.bin";
const string DEBUG_QUERY_FILE = "../data/debug/query.bin";
const string DEBUG_TRUTH_FILE = "../data/debug/groundtruth.bin";
const int DEBUG_D = 16;
const int DEBUG_BASE_N = 1000;
const int DEBUG_QUERY_N = 100;

// --- SIFT 配置 (现在指向 .bin 文件) ---
const string SIFT_BASE_FILE = "../data/sift/base.bin";
const string SIFT_QUERY_FILE = "../data/sift/query.bin";
const string SIFT_TRUTH_FILE = "../data/sift/groundtruth.bin";
const int SIFT_D = 128;
const long long SIFT_BASE_N = 1000000;
const long long SIFT_QUERY_N = 10000;

// --- GLOVE 配置 (现在指向 .bin 文件) ---
const string GLOVE_BASE_FILE = "../data/glove/base.bin";
const string GLOVE_QUERY_FILE = "../data/glove/query.bin";
const string GLOVE_TRUTH_FILE = "../data/glove/groundtruth.bin";
const int GLOVE_D = 100;
const long long GLOVE_BASE_N = 1183514;
const long long GLOVE_QUERY_N = 10000;

// --- 项目配置 ---
const int K = 10; // 查找 10 个最近邻

/**
 * 模板函数：从二进制文件 (BIN) 快速加载数据
 */
template <typename T>
bool load_from_bin(const string& filename, vector<T>& data_vector, long long expected_count) {
    ifstream fin(filename, ios::binary);
    if (!fin.is_open()) {
        cerr << "  错误: 无法打开 (BIN) " << filename << endl;
        cerr << "  请确保您已运行 'convert_to_binary' 脚本。" << endl;
        return false;
    }
    cout << "  正在读取 (BIN): " << filename << "..." << endl;

    // 调整 vector 大小以容纳所有数据
    data_vector.resize(expected_count);
    
    // 一次性将文件内容读入 vector 内存
    fin.read(reinterpret_cast<char*>(data_vector.data()), expected_count * sizeof(T));
    
    if (fin.gcount() != expected_count * sizeof(T)) {
         cerr << "  严重错误: " << filename << " 文件大小不符，读取字节失败！" << endl;
         fin.close();
         return false;
    }
    fin.close();
    return true;
}


/**
 * 统一的、干活的评测函数
 */
void process_dataset(
    const string& name,
    const string& base_path,
    const string& query_path,
    const string& truth_path,
    int D,
    long long BASE_N,
    long long QUERY_N,
    ofstream& result_file 
) {
    cout << "\n=============================================" << endl;
    cout << "--- 正在评测 " << name << " 数据集 ---" << endl;
    
    // 1. 加载所有数据 (现在使用快速的二进制加载)
    vector<float> base_data;
    vector<float> query_data;
    vector<vector<int>> truth_data_flat; // 基准答案 (先读入一维)
    vector<vector<int>> truth_data_2d(QUERY_N, vector<int>(K)); // 二维格式

    if (!load_from_bin(base_path, base_data, BASE_N * D)) return;
    if (!load_from_bin(query_path, query_data, QUERY_N * D)) return;
    
    // 加载一维的 groundtruth.bin
    vector<int> truth_flat;
    if (!load_from_bin(truth_path, truth_flat, QUERY_N * K)) return;
    // 将一维的 truth 转换为二维
    for (long long i = 0; i < QUERY_N; ++i) {
        for (int k = 0; k < K; ++k) {
            truth_data_2d[i][k] = truth_flat[i * K + k];
        }
    }


    Solution solution; 

    // 2. 评测 Build 时延
    cout << "\n--- 评测 Build ---\n";
    auto start_build = chrono::high_resolution_clock::now();
    solution.build(D, base_data);
    auto end_build = chrono::high_resolution_clock::now();
    
    base_data.clear();
    base_data.shrink_to_fit();
    
    double build_time_ms = chrono::duration<double, std::milli>(end_build - start_build).count();
    cout << "  Build 时延: " << build_time_ms << " ms" << endl;

    // 3. 评测 Search 时延 和 精度
    cout << "\n--- 评测 Search ---\n";
    double total_search_time_ms = 0;
    double total_recall_score = 0; 

    vector<float> current_query(D);
    int my_results[K];

    for (long long i = 0; i < QUERY_N; ++i) {
        for(int j = 0; j < D; ++j) {
            current_query[j] = query_data[i * D + j];
        }

        auto start_search = chrono::high_resolution_clock::now();
        solution.search(current_query, my_results);
        auto end_search = chrono::high_resolution_clock::now();
        
        total_search_time_ms += chrono::duration<double, std::milli>(end_search - start_search).count();

        // 计算召回率 (精度)
        const vector<int>& correct_results = truth_data_2d[i];
        set<int> correct_set(correct_results.begin(), correct_results.end());
        
        int intersection_count = 0;
        for (int k = 0; k < K; ++k) {
            if (correct_set.count(my_results[k])) {
                intersection_count++;
            }
        }
        total_recall_score += (double)intersection_count / K;
    }

    // 4. 输出最终结果
    double T_avg_ms = total_search_time_ms / QUERY_N;
    double delta_accuracy = total_recall_score / QUERY_N;

    cout << "--- " << name << " 评测结果 ---\n";
    result_file << "\n--- " << name << " 评测结果 ---\n";
    
    cout << "  Build 时延: " << build_time_ms << " ms\n";
    result_file << "  Build 时延: " << build_time_ms << " ms\n";

    cout << "  平均 Search 时延 (T_avg): " << T_avg_ms << " ms\n";
    result_file << "  平均 Search 时延 (T_avg): " << T_avg_ms << " ms\n";

    cout << "  检索精度 (delta): " << delta_accuracy << "\n";
    result_file << "  检索精度 (delta): " << delta_accuracy << "\n";
    
    if (delta_accuracy < 0.99) {
        cout << "  [!!! 失败 !!!] 精度未达到 0.99 约束！\n";
        result_file << "  [!!! 失败 !!!] 精度未达到 0.99 约束！\n";
    } else {
        cout << "  [成功] 精度满足 >= 0.99 约束。\n";
        result_file << "  [成功] 精度满足 >= 0.99 约束。\n";
    }
}


int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "错误: 此脚本不应被直接运行！" << endl;
        cerr << "请运行: .\\run_eval.bat [method_name]" << endl;
        return 1;
    }
    
    const string RESULT_FILE_PATH = argv[1];
    ofstream result_file(RESULT_FILE_PATH);
    if (!result_file.is_open()) {
        cerr << "错误: 无法在 " << RESULT_FILE_PATH << " 创建结果文件！" << endl;
        return 1;
    }
    
    cout << "========= 本地评测器 (v1.4 - 二进制模式) =========\n";
    cout << "========= 结果将保存到 " << RESULT_FILE_PATH << " =========" << endl;
    
    result_file << "========= 本地评测器 (v1.4 - 二进制模式) =========\n";
    result_file << "========= 目标: 评测 " << argv[1] << " =========" << endl; 

    // 1. 评测 DEBUG
    process_dataset(
        "DEBUG",
        DEBUG_BASE_FILE, DEBUG_QUERY_FILE, DEBUG_TRUTH_FILE,
        DEBUG_D, DEBUG_BASE_N, DEBUG_QUERY_N,
        result_file
    );
    process_dataset(
        "DEBUG",
        DEBUG_BASE_FILE, DEBUG_QUERY_FILE, DEBUG_TRUTH_FILE,
        DEBUG_D, DEBUG_BASE_N, DEBUG_QUERY_N,
        result_file
    );

    // 2. 评测 SIFT
    process_dataset(
        "SIFT",
        SIFT_BASE_FILE, SIFT_QUERY_FILE, SIFT_TRUTH_FILE,
        SIFT_D, SIFT_BASE_N, SIFT_QUERY_N,
        result_file
    );
    
    // 3. 评测 GLOVE
    process_dataset(
        "GLOVE",
        GLOVE_BASE_FILE, GLOVE_QUERY_FILE, GLOVE_TRUTH_FILE,
        GLOVE_D, GLOVE_BASE_N, GLOVE_QUERY_N,
        result_file
    );

    cout << "\n===============================" << endl;
    cout << "所有评测已执行完毕。" << endl;
    cout << "===============================" << endl;
    
    result_file << "\n===============================\n";
    result_file << "所有评测已执行完毕。\n";
    result_file << "===============================\n";
    result_file.close();

    return 0;
}