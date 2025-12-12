#include "MySolution.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <set>
#include <atomic>
#include <filesystem>
#include <iomanip>
#include <algorithm>

namespace fs = std::filesystem;
using namespace std;

// --- 全局计数器 ---
#ifdef COUNT_DIST
    atomic<unsigned long long> g_dist_calc_count{0};
#endif
#ifdef TEST_GRAPH
    atomic<unsigned long long> g_acc{0}, g_tot{0};
#endif

// --- 路径配置 ---
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

// --- gamma 参数扫描配置 ---
const float GAMMA_START = 0.00f;
const float GAMMA_END = 0.50f;
const float GAMMA_STEP = 0.01f;


/**
 * 从二进制文件动态加载数据
 */
template <typename T>
long long load_from_bin_dynamic(const string& filename, vector<T>& data_vector, int dim) {
    ifstream fin(filename, ios::binary | ios::ate);
    if (!fin.is_open()) {
        cerr << "  错误: 无法打开 (BIN) " << filename << endl;
        return 0;
    }

    streamsize file_size = fin.tellg();
    fin.seekg(0, ios::beg);

    if (file_size % sizeof(T) != 0) {
        cerr << "  严重错误: 文件大小不是数据类型的整数倍！" << endl;
        return 0;
    }

    long long total_elements = file_size / sizeof(T);
    long long N = total_elements / dim;

    if (total_elements % dim != 0) {
        cerr << "  严重错误: 文件总元素数不能被维度整除！" << endl;
        return 0;
    }

    cout << "  正在读取 " << filename << " (N=" << N << ")..." << endl;

    data_vector.resize(total_elements);
    fin.read(reinterpret_cast<char*>(data_vector.data()), file_size);
    fin.close();
    return N;
}

/**
 * 保存度数分布到 CSV（直方图形式）
 */
void save_degree_distribution(const vector<int>& degrees, const string& path) {
    map<int, int> degree_count;
    for (int d : degrees) {
        degree_count[d]++;
    }

    ofstream fout(path);
    fout << "degree,count\n";
    for (const auto& [deg, cnt] : degree_count) {
        fout << deg << "," << cnt << "\n";
    }
    fout.close();
    cout << "  度数分布已保存到: " << path << " (" << degree_count.size() << " 种度数)" << endl;
}

/**
 * 保存 build 统计到文件
 */
void save_build_stats(const string& path, double build_time_ms,
                      unsigned long long build_dist_ops,
                      double nav_accuracy, double nav_steps) {
    ofstream fout(path);
    fout << "metric,value\n";
    fout << "build_time_ms," << fixed << setprecision(2) << build_time_ms << "\n";
    fout << "build_dist_ops," << build_dist_ops << "\n";
    fout << "nav_accuracy," << fixed << setprecision(6) << nav_accuracy << "\n";
    fout << "nav_avg_steps," << fixed << setprecision(2) << nav_steps << "\n";
    fout.close();
    cout << "  Build 统计已保存到: " << path << endl;
}

/**
 * 评测单个数据集 - 消融实验版
 */
void process_dataset_ablation(
    const string& dataset_name,
    const string& base_path,
    const string& query_path,
    const string& truth_path,
    int D,
    const string& output_dir
) {
    cout << "\n=============================================" << endl;
    cout << "--- 消融评测: " << dataset_name << " ---" << endl;

    fs::create_directories(output_dir);

    string graph_cache_path = output_dir + "/graph_" + dataset_name + ".bin";
    string degree_csv_path = output_dir + "/degree_" + dataset_name + ".csv";
    string build_stats_path = output_dir + "/build_stats_" + dataset_name + ".csv";
    string search_csv_path = output_dir + "/search_" + dataset_name + ".csv";

    // 1. 加载数据
    vector<float> base_data;
    vector<float> query_data;
    vector<int> truth_flat;

    long long base_N = load_from_bin_dynamic(base_path, base_data, D);
    if (base_N == 0) return;

    long long query_N = load_from_bin_dynamic(query_path, query_data, D);
    if (query_N == 0) return;

    long long truth_N = load_from_bin_dynamic(truth_path, truth_flat, K);
    if (truth_N == 0) return;

    if (query_N != truth_N) {
        cerr << "  错误: Query数量与Truth数量不匹配！" << endl;
        return;
    }

    vector<vector<int>> truth_data_2d(query_N, vector<int>(K));
    for (long long i = 0; i < query_N; ++i) {
        for (int k = 0; k < K; ++k) {
            truth_data_2d[i][k] = truth_flat[i * K + k];
        }
    }

    Solution solution;

    // 2. 检查图缓存
    bool cache_exists = fs::exists(graph_cache_path);

    if (cache_exists) {
        cout << "\n--- 发现图缓存，直接加载 ---" << endl;
        if (!solution.load_graph(graph_cache_path)) {
            cerr << "  加载缓存失败，重新构建..." << endl;
            cache_exists = false;
        } else {
            cout << "  图缓存加载成功！" << endl;
        }
    }

    if (!cache_exists) {
        // 3. 构建图
        cout << "\n--- 构建图 (N=" << base_N << ") ---" << endl;

        #ifdef COUNT_DIST
            g_dist_calc_count = 0;
        #endif
        #ifdef TEST_GRAPH
            g_acc = 0;
            g_tot = 0;
        #endif

        auto start_build = chrono::high_resolution_clock::now();
        solution.build(D, base_data);
        auto end_build = chrono::high_resolution_clock::now();

        double build_time_ms = chrono::duration<double, milli>(end_build - start_build).count();

        unsigned long long build_dist_ops = 0;
        #ifdef COUNT_DIST
            build_dist_ops = g_dist_calc_count.load();
        #endif

        cout << "  Build 完成: " << fixed << setprecision(2) << build_time_ms << " ms" << endl;
        cout << "  Build dist_ops: " << build_dist_ops << endl;

        // 4. 收集图统计
        vector<int> degrees = solution.get_degree_distribution();
        save_degree_distribution(degrees, degree_csv_path);

        // 5. 获取 KNN_check 结果
        double nav_accuracy = 0.0, nav_steps = 0.0;
        #ifdef TEST_GRAPH
            nav_accuracy = (double)g_acc.load() / 10000.0;
            nav_steps = (double)g_tot.load() / 10000.0;
            cout << "  贪心导航准确率: " << nav_accuracy << endl;
            cout << "  平均导航步数: " << nav_steps << endl;
        #endif

        // 6. 保存 build 统计
        save_build_stats(build_stats_path, build_time_ms, build_dist_ops, nav_accuracy, nav_steps);

        // 7. 保存图缓存
        solution.save_graph(graph_cache_path);
        cout << "  图缓存已保存到: " << graph_cache_path << endl;
    }

    // 8. 搜索评测 - 参数扫描（支持断点续传）
    cout << "\n--- 搜索评测 (参数扫描) ---" << endl;

    vector<float> current_query(D);
    int my_results[K];

    int total_params = (int)((GAMMA_END - GAMMA_START) / GAMMA_STEP) + 1;
    int start_idx = 0;
    
    // 初始化状态变量
    double prev_recall = -1.0; 
    int stable_count = 0; 

    // 检查是否存在已有的搜索结果，支持断点续传并恢复状态
    if (fs::exists(search_csv_path)) {
        ifstream existing_csv(search_csv_path);
        string line;
        vector<double> past_recalls;

        // 跳过 header
        if (getline(existing_csv, line)) {
            while (getline(existing_csv, line)) {
                if (line.empty()) continue;
                
                size_t last_comma = line.find_last_of(',');
                if (last_comma != string::npos) {
                    try {
                        double r = stod(line.substr(last_comma + 1));
                        past_recalls.push_back(r);
                    } catch (...) {}
                }
            }
        }
        existing_csv.close();

        start_idx = past_recalls.size();

        // 核心修正：从历史数据恢复 stable_count
        if (start_idx > 0) {
            prev_recall = past_recalls.back();
            
            // 回溯计算连续稳定次数，直接用 == 判断
            for (int i = start_idx - 2; i >= 0; --i) {
                if (past_recalls[i] == prev_recall) {
                    stable_count++;
                } else {
                    break;
                }
            }
        }

        if (start_idx > 0 && start_idx < total_params && stable_count < 3 && prev_recall < 1) {
            cout << "  发现已有结果，从第 " << start_idx << "/" << total_params << " 个参数继续..." << endl;
            cout << "  [状态恢复] 上次 Recall: " << fixed << setprecision(6) << prev_recall 
                 << ", 已连续稳定次数: " << stable_count << endl;
        } else if (start_idx >= total_params || stable_count >= 3 || prev_recall == 1) {
            cout << "  搜索结果已完成，跳过。" << endl;
            return;
        }
    }

    ofstream search_csv;
    if (start_idx == 0) {
        search_csv.open(search_csv_path);
        search_csv << "gamma,efSearch,QPS,avg_dist_ops,recall\n";
    } else {
        search_csv.open(search_csv_path, ios::app);
    }

    float gamma = GAMMA_START;
    int efSearch = 10;

    for (int idx = start_idx; idx < total_params; ++idx) {
        gamma = GAMMA_START + idx * GAMMA_STEP;
        efSearch = 10 * pow(1.2, idx);

        solution.set_gamma(gamma);
        solution.set_ef_search(efSearch);

        #ifdef COUNT_DIST
            g_dist_calc_count = 0;
        #endif

        double total_recall = 0.0;

        auto start_search = chrono::high_resolution_clock::now();

        for (long long i = 0; i < query_N; ++i) {
            for (int j = 0; j < D; ++j) {
                current_query[j] = query_data[i * D + j];
            }

            solution.search(current_query, my_results);

            const vector<int>& correct_results = truth_data_2d[i];
            set<int> correct_set(correct_results.begin(), correct_results.end());
            int hits = 0;
            for (int k = 0; k < K; ++k) {
                if (correct_set.count(my_results[k])) hits++;
            }
            total_recall += (double)hits / K;
        }

        auto end_search = chrono::high_resolution_clock::now();
        double total_time_sec = chrono::duration<double>(end_search - start_search).count();

        double QPS = query_N / total_time_sec;
        double recall = total_recall / query_N;

        unsigned long long search_dist_ops_total = 0;
        double avg_dist_ops = 0;
        #ifdef COUNT_DIST
            search_dist_ops_total = g_dist_calc_count.load();
            avg_dist_ops = (double)search_dist_ops_total / query_N;
        #endif

        search_csv << fixed << setprecision(2) << gamma << ","
                   << efSearch << ","
                   << fixed << setprecision(1) << QPS << ","
                   << fixed << setprecision(1) << avg_dist_ops << ","
                   << fixed << setprecision(6) << recall << "\n";
        search_csv.flush();

        cout << "\r  进度: " << (idx + 1) << "/" << total_params
             << " gamma=" << fixed << setprecision(2) << gamma
             << " ef=" << efSearch
             << " QPS=" << (int)QPS << " recall=" << fixed << setprecision(4) << recall << "   " << flush;

        // --- 终止条件判断 ---
        
        // 1. 达到 1
        if (recall >= 1) {
            cout << "\n  [终止] Recall 已达到 1，停止后续参数扫描。" << endl;
            break;
        }

        // 2. 连续三次召回率不变，且大于 0.999
        if (recall == prev_recall) {
            stable_count++;
        } else {
            stable_count = 0;
        }

        if (stable_count >= 3 && recall > 0.999) {
            cout << "\n  [终止] Recall 已连续 3 次保持不变 (" << fixed << setprecision(6) << recall << ") 且 > 0.999，停止扫描。" << endl;
            break;
        }

        prev_recall = recall;
        // ------------------
    }

    search_csv.close();
    cout << "\n  搜索结果已保存到: " << search_csv_path << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "用法: " << argv[0] << " <output_dir> [dataset]" << endl;
        cerr << "  output_dir: 输出目录" << endl;
        cerr << "  dataset: DEBUG, SIFT, GLOVE, ALL (默认 ALL)" << endl;
        return 1;
    }

    string output_dir = argv[1];
    string dataset = (argc >= 3) ? argv[2] : "ALL";

    cout << "=== 消融实验评估框架 ===" << endl;
    cout << "输出目录: " << output_dir << endl;
    cout << "数据集: " << dataset << endl;

    #ifdef COUNT_DIST
    cout << "!!! 性能计数模式已开启 (COUNT_DIST) !!!" << endl;
    #endif

    if (dataset == "DEBUG" || dataset == "ALL") {
        process_dataset_ablation("DEBUG", DEBUG_BASE_FILE, DEBUG_QUERY_FILE, DEBUG_TRUTH_FILE, DEBUG_D, output_dir);
    }
    if (dataset == "SIFT" || dataset == "ALL") {
        process_dataset_ablation("SIFT", SIFT_BASE_FILE, SIFT_QUERY_FILE, SIFT_TRUTH_FILE, SIFT_D, output_dir);
    }
    if (dataset == "GLOVE" || dataset == "ALL") {
        process_dataset_ablation("GLOVE", GLOVE_BASE_FILE, GLOVE_QUERY_FILE, GLOVE_TRUTH_FILE, GLOVE_D, output_dir);
    }

    cout << "\n=== 评估完成 ===" << endl;
    return 0;
}