#include "MySolution.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <set>
#include <atomic>
#include <filesystem>

using namespace std;

// ================= [核心] 引入全局参数 =================
#ifdef TUNING_MODE
    extern int K;
    extern int efConstruction;
    extern int Mmax0;
    extern int MIN_EDGES;
    extern float gamma;
#endif

// ================= 配置 =================
const string GLOVE_BASE_FILE = "../data/glove/base.bin";
const string GLOVE_QUERY_FILE = "../data/glove/query.bin";
const string GLOVE_TRUTH_FILE = "../data/glove/groundtruth.bin";
const int GLOVE_D = 100;

// 全局计数器引用 (在 MySolution.cpp 中定义)
#ifdef COUNT_DIST
    std::atomic<unsigned long long> g_dist_calc_count{0};
#endif

// 加载函数
template <typename T>
long long load_bin(const string& filename, vector<T>& data_vector) {
    ifstream fin(filename, ios::binary | ios::ate);
    if (!fin) return 0;
    streamsize size = fin.tellg();
    fin.seekg(0, ios::beg);
    data_vector.resize(size / sizeof(T));
    fin.read((char*)data_vector.data(), size);
    return data_vector.size();
}

int main(int argc, char* argv[]) {
    // 1. 解析参数: [exe] [efConstruction] [Mmax0] [MIN_EDGES] [gamma]
    #ifdef TUNING_MODE
    if (argc >= 5) {
        efConstruction = atoi(argv[1]);
        Mmax0 = atoi(argv[2]);
        MIN_EDGES = atoi(argv[3]);
        gamma = (float)atof(argv[4]);
        // K 默认 10，通常不调
    }
    #endif

    // 2. 加载数据
    vector<float> base, query;
    vector<int> truth;
    if (!load_bin(GLOVE_BASE_FILE, base)) { cerr << "No Base" << endl; return -1; }
    if (!load_bin(GLOVE_QUERY_FILE, query)) { cerr << "No Query" << endl; return -1; }
    if (!load_bin(GLOVE_TRUTH_FILE, truth)) { cerr << "No Truth" << endl; return -1; }

    int N = base.size() / GLOVE_D;
    int Q = query.size() / GLOVE_D;

    Solution sol;

    // 3. Build 阶段 (带超时熔断逻辑在 python 端控制，这里只管跑)
    auto t1 = chrono::high_resolution_clock::now();
    sol.build(GLOVE_D, base);
    auto t2 = chrono::high_resolution_clock::now();
    double build_time = chrono::duration<double>(t2 - t1).count();

    // 4. Search 阶段
    #ifdef COUNT_DIST
        g_dist_calc_count = 0;
    #endif
    
    double total_recall = 0;
    vector<int> res(10); // K=10
    int truth_K = 10; // Groundtruth width

    auto t3 = chrono::high_resolution_clock::now();
    for(int i=0; i<Q; ++i) {
        // 构造查询向量
        vector<float> q_vec(query.begin() + i*GLOVE_D, query.begin() + (i+1)*GLOVE_D);
        sol.search(q_vec, res.data());
        
        // 计算 Recall
        int hits = 0;
        // 假设 Truth 也是紧凑排列的
        const int* gt_ptr = &truth[i * truth_K]; 
        for(int k=0; k<10; ++k) {
            for(int j=0; j<10; ++j) {
                if(res[k] == gt_ptr[j]) { hits++; break; }
            }
        }
        total_recall += (double)hits / 10.0;
    }
    auto t4 = chrono::high_resolution_clock::now();
    double search_time = chrono::duration<double>(t4 - t3).count();

    // 5. 收集指标
    double avg_recall = total_recall / Q;
    double qps = Q / search_time;
    unsigned long long total_dist_ops = 0;
    #ifdef COUNT_DIST
        total_dist_ops = g_dist_calc_count.load();
    #endif
    double avg_dist_ops = (double)total_dist_ops / Q;

    // 6. 输出标准格式供 Python 解析
    // 格式: [RESULT] recall qps avg_dist_ops build_time
    cout << "[RESULT] " << avg_recall << " " << qps << " " << avg_dist_ops << " " << build_time << endl;

    return 0;
}