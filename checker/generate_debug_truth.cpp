#include "../method/brute/MySolution.h" // 包含您的暴力搜索
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

// --- 调试数据配置 ---
const string DEBUG_DIR = "../data/debug";
const string BASE_FILE = DEBUG_DIR + "/base.txt";
const string QUERY_FILE = DEBUG_DIR + "/query.txt";
const string OUTPUT_FILE = DEBUG_DIR + "/groundtruth.txt";
const int D = 16;
const int BASE_N = 1000;
const int QUERY_N = 100;
const int K = 10;
// --------------------

// 辅助函数：加载 TXT 数据
bool load_data_from_txt(const string& filename, vector<float>& data_vector, long long expected_count) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "错误: 无法打开 " << filename << endl; return false;
    }
    cout << "  正在读取 " << filename << "..." << endl;
    data_vector.reserve(expected_count);
    float val;
    long long count = 0;
    while (fin >> val) { data_vector.push_back(val); count++; }
    fin.close();
    if (count != expected_count) {
        cerr << "  错误: " << filename << " 文件大小与预期 " << expected_count << " 不符！" << endl;
        return false;
    }
    return true;
}

int main() {
    cout << "正在为 DEBUG 数据集生成 Ground Truth..." << endl;

    // 1. 加载底库
    vector<float> base_data;
    if (!load_data_from_txt(BASE_FILE, base_data, BASE_N * D)) return 1;

    // 2. 构建暴力索引
    Solution brute_force_solution;
    brute_force_solution.build(D, base_data);
    base_data.clear();
    
    // 3. 加载查询
    vector<float> query_data;
    if (!load_data_from_txt(QUERY_FILE, query_data, QUERY_N * D)) return 1;

    // 4. 生成答案
    ofstream fout(OUTPUT_FILE);
    vector<float> current_query(D);
    int results[K];

    for (long long i = 0; i < QUERY_N; ++i) {
        for(int j = 0; j < D; ++j) {
            current_query[j] = query_data[i * D + j];
        }
        brute_force_solution.search(current_query, results);
        for (int k = 0; k < K; ++k) {
            fout << results[k] << (k == K - 1 ? "" : " ");
        }
        fout << "\n";
    }
    fout.close();

    cout << "DEBUG Ground Truth 已生成: " << OUTPUT_FILE << endl;
    return 0;
}