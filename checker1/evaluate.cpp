#include "MySolution.h" 
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono> 
#include <set>
#include <atomic> 
#include <filesystem> // C++17，用于获取文件大小

namespace fs = std::filesystem;
using namespace std;

// --- 全局计数器 ---
#ifdef COUNT_DIST
    atomic<unsigned long long> g_dist_calc_count{0};
#endif
#ifdef TEST_GRAPH
    atomic<unsigned long long> g_acc{0}, g_tot{0};
#endif

// --- 路径配置 (只保留路径和维度 D) ---
// DEBUG
const string DEBUG_BASE_FILE = "../data/debug/base.bin";
const string DEBUG_QUERY_FILE = "../data/debug/query.bin";
const string DEBUG_TRUTH_FILE = "../data/debug/groundtruth.bin";
const int DEBUG_D = 16;

// SIFT
const string SIFT_BASE_FILE = "../data/sift/base.bin";
const string SIFT_QUERY_FILE = "../data/sift/query.bin";
const string SIFT_TRUTH_FILE = "../data/sift/groundtruth.bin";
const int SIFT_D = 128;

// GLOVE
const string GLOVE_BASE_FILE = "../data/glove/base.bin";
const string GLOVE_QUERY_FILE = "../data/glove/query.bin";
const string GLOVE_TRUTH_FILE = "../data/glove/groundtruth.bin";
const int GLOVE_D = 100;

// --- 项目配置 ---
const int K = 10; 

/**
 * 模板函数：从二进制文件 (BIN) 动态加载数据
 * 返回加载的元素数量 (行数 N)
 */
template <typename T>
long long load_from_bin_dynamic(const string& filename, vector<T>& data_vector, int dim) {
    ifstream fin(filename, ios::binary | ios::ate); // 打开并定位到末尾
    if (!fin.is_open()) {
        cerr << "  错误: 无法打开 (BIN) " << filename << endl;
        return 0;
    }

    streamsize file_size = fin.tellg(); // 获取文件大小 (字节)
    fin.seekg(0, ios::beg); // 回到开头

    if (file_size % sizeof(T) != 0) {
        cerr << "  严重错误: 文件大小不是数据类型的整数倍！" << endl;
        return 0;
    }

    long long total_elements = file_size / sizeof(T);
    long long N = total_elements / dim;

    if (total_elements % dim != 0) {
        cerr << "  严重错误: 文件总元素数 (" << total_elements << ") 不能被维度 (" << dim << ") 整除！" << endl;
        return 0;
    }

    cout << "  正在读取 " << filename << " (Detected N=" << N << ")..." << endl;

    data_vector.resize(total_elements);
    fin.read(reinterpret_cast<char*>(data_vector.data()), file_size);
    
    if (fin.gcount() != file_size) {
         cerr << "  严重错误: 读取字节数不符！" << endl;
         fin.close();
         return 0;
    }
    fin.close();
    return N;
}

/**
 * 评测函数 (动态版)
 */
void process_dataset(
    const string& name,
    const string& base_path,
    const string& query_path,
    const string& truth_path,
    int D,
    ofstream& result_file 
) {
    cout << "\n=============================================" << endl;
    cout << "--- 正在评测 " << name << " 数据集 ---" << endl;
    
    vector<float> base_data;
    vector<float> query_data;
    vector<int> truth_flat;

    // 1. 动态加载数据
    long long base_N = load_from_bin_dynamic(base_path, base_data, D);
    if (base_N == 0) return;

    long long query_N = load_from_bin_dynamic(query_path, query_data, D);
    if (query_N == 0) return;
    
    // Groundtruth 的维度是 K
    long long truth_N = load_from_bin_dynamic(truth_path, truth_flat, K);
    if (truth_N == 0) return;

    // 2. 验证数据一致性
    if (query_N != truth_N) {
        cerr << "  严重错误: Query数量 (" << query_N << ") 与 Truth数量 (" << truth_N << ") 不匹配！" << endl;
        return;
    }

    // 3. 解析 Truth
    vector<vector<int>> truth_data_2d(query_N, vector<int>(K));
    for (long long i = 0; i < query_N; ++i) {
        for (int k = 0; k < K; ++k) truth_data_2d[i][k] = truth_flat[i * K + k];
    }
    #ifdef TEST_GRAPH
        g_acc = g_tot = 0;
    #endif 

    Solution solution; 

    // --- 4. 评测 Build ---
    cout << "\n--- 评测 Build (N=" << base_N << ") ---\n";
    
    #ifdef COUNT_DIST
        g_dist_calc_count = 0;
    #endif

    auto start_build = chrono::high_resolution_clock::now();
    solution.build(D, base_data);
    auto end_build = chrono::high_resolution_clock::now();
    
    double build_time_ms = chrono::duration<double, std::milli>(end_build - start_build).count();
    
    unsigned long long build_ops = 0;
    #ifdef COUNT_DIST
        build_ops = g_dist_calc_count.load();
    #endif

    base_data.clear();
    base_data.shrink_to_fit();
    
    // --- 5. 评测 Search ---
    cout << "\n--- 评测 Search (Q=" << query_N << ") ---\n";
    double total_search_time_ms = 0;
    double total_recall_score = 0; 

    #ifdef COUNT_DIST
        g_dist_calc_count = 0;
    #endif

    vector<float> current_query(D);
    int my_results[K];

    for (long long i = 0; i < query_N; ++i) {
        for(int j = 0; j < D; ++j) current_query[j] = query_data[i * D + j];

        auto start_search = chrono::high_resolution_clock::now();
        solution.search(current_query, my_results);
        auto end_search = chrono::high_resolution_clock::now();
        
        total_search_time_ms += chrono::duration<double, std::milli>(end_search - start_search).count();

        const vector<int>& correct_results = truth_data_2d[i];
        set<int> correct_set(correct_results.begin(), correct_results.end());
        int intersection_count = 0;
        for (int k = 0; k < K; ++k) {
            if (correct_set.count(my_results[k])) intersection_count++;
        }
        total_recall_score += (double)intersection_count / K;
    }

    unsigned long long search_ops_total = 0;
    double search_ops_avg = 0;
    #ifdef COUNT_DIST
        search_ops_total = g_dist_calc_count.load();
        search_ops_avg = (double)search_ops_total / query_N;
    #endif

    // --- 6. 输出最终结果 ---
    double T_avg_ms = total_search_time_ms / query_N;
    double delta_accuracy = total_recall_score / query_N;

    string report = "\n--- " + name + " 评测结果 ---\n";
    report += "  Build 时延: " + to_string(build_time_ms) + " ms\n";
    
    #ifdef COUNT_DIST
    report += "  Build 距离计算次数: " + to_string(build_ops) + "\n";
    #endif
    #ifdef TEST_GRAPH
    report += "贪心导航准确率：" + to_string(1.0*g_acc.load()/10000)+ "\n";
    #endif
    #ifdef TEST_GRAPH
    report += "贪心导航边数：" + to_string(1.0*g_tot.load()/10000)+ "\n";
    #endif

    report += "  平均 Search 时延 (T_avg): " + to_string(T_avg_ms) + " ms\n";
    
    #ifdef COUNT_DIST
    report += "  平均 Search 距离计算次数: " + to_string(search_ops_avg) + "\n";
    #endif

    report += "  检索精度 (delta): " + to_string(delta_accuracy) + "\n";

    cout << report;
    result_file << report;
    
    if (delta_accuracy < 0.99) {
        string msg = "  [!!! 失败 !!!] 精度未达到 0.99 约束！\n";
        cout << msg; result_file << msg;
    } else {
        string msg = "  [成功] 精度满足 >= 0.99 约束。\n";
        cout << msg; result_file << msg;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "请运行: .\\run_eval.bat [method_name]" << endl;
        return 1;
    }
    const string RESULT_FILE_PATH = argv[1];
    ofstream result_file(RESULT_FILE_PATH);
    
    #ifdef COUNT_DIST
    cout << "!!! 性能计数模式已开启 (COUNT_DIST) !!!" << endl;
    cout << "警告: 原子操作会轻微影响运行时延，仅供算法优化分析。" << endl;
    result_file << "!!! 性能计数模式已开启 (COUNT_DIST) !!!\n";
    #endif

    process_dataset("DEBUG", DEBUG_BASE_FILE, DEBUG_QUERY_FILE, DEBUG_TRUTH_FILE, DEBUG_D, result_file);
    process_dataset("SIFT", SIFT_BASE_FILE, SIFT_QUERY_FILE, SIFT_TRUTH_FILE, SIFT_D, result_file);
    process_dataset("GLOVE", GLOVE_BASE_FILE, GLOVE_QUERY_FILE, GLOVE_TRUTH_FILE, GLOVE_D, result_file);

    result_file.close();
    return 0;
}