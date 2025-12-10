#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem> // C++17

namespace fs = std::filesystem;
using namespace std;

// --- DEBUG 配置 ---
const string DEBUG_DIR = "../data/debug";
const int DEBUG_D = 16;
const int DEBUG_BASE_N = 1000;
const int DEBUG_QUERY_N = 100;

// --- SIFT 配置 ---
const string SIFT_DIR = "../data/sift";
const int SIFT_D = 128;
const long long SIFT_BASE_N = 1000000;
const long long SIFT_QUERY_N = 10000;

// --- GLOVE 配置 ---
const string GLOVE_DIR = "../data/glove";
const int GLOVE_D = 100;
const long long GLOVE_BASE_N = 1183514;
const long long GLOVE_QUERY_N = 10000;

// --- 项目配置 ---
const int K = 10; // 查找 10 个最近邻

/**
 * 模板函数：读取文本文件 (TXT)
 */
template <typename T>
bool load_from_txt(const string& filename, vector<T>& data_vector, long long expected_count) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "  错误: 无法打开 (TXT) " << filename << endl;
        return false;
    }
    cout << "  正在读取 (TXT): " << filename << "..." << endl;
    
    data_vector.reserve(expected_count);
    T val;
    long long count = 0;
    while (fin >> val) {
        data_vector.push_back(val);
        count++;
    }
    fin.close();

    if (count != expected_count) {
        cerr << "  严重错误: 文件 " << filename << " 大小 (" << count << ") 与预期 (" << expected_count << ") 不符！" << endl;
        return false;
    }
    return true;
}

/**
 * 模板函数：写入二进制文件 (BIN)
 */
template <typename T>
bool write_to_bin(const string& filename, const vector<T>& data_vector) {
    ofstream fout(filename, ios::binary | ios::out);
    if (!fout.is_open()) {
        cerr << "  错误: 无法创建 (BIN) " << filename << endl;
        return false;
    }
    
    fout.write(reinterpret_cast<const char*>(data_vector.data()), data_vector.size() * sizeof(T));
    fout.close();
    
    cout << "  [成功] 已写入 (BIN): " << filename << endl;
    return true;
}

/**
 * 转换一个数据集（例如 "SIFT"）
 */
bool process_dataset(
    const string& name,
    const string& dir,
    int D,
    long long BASE_N,
    long long QUERY_N
) {
    cout << "\n--- 正在转换 " << name << " 数据集 ---" << endl;
    
    // --- 1. 处理 Base (float) ---
    vector<float> base_data;
    string base_txt = dir + "/base.txt";
    string base_bin = dir + "/base.bin";
    if (!load_from_txt(base_txt, base_data, BASE_N * D)) return false;
    if (!write_to_bin(base_bin, base_data)) return false;

    // --- 2. 处理 Query (float) ---
    vector<float> query_data;
    string query_txt = dir + "/query.txt";
    string query_bin = dir + "/query.bin";
    if (!load_from_txt(query_txt, query_data, QUERY_N * D)) return false;
    if (!write_to_bin(query_bin, query_data)) return false;

    // --- 3. 处理 Groundtruth (int) ---
    vector<int> truth_data;
    string truth_txt = dir + "/groundtruth.txt";
    string truth_bin = dir + "/groundtruth.bin";
    if (!load_from_txt(truth_txt, truth_data, QUERY_N * K)) return false;
    if (!write_to_bin(truth_bin, truth_data)) return false;

    return true;
}


int main() {
    cout << "========= 数据集 TXT -> BIN 转换器 =========\n";
    cout << "此脚本将读取所有 .txt 文件并创建 .bin 文件，\n";
    cout << "这将极大提高后续评测的加载速度。\n";

    // 1. 转换 DEBUG
    process_dataset("DEBUG", DEBUG_DIR, DEBUG_D, DEBUG_BASE_N, DEBUG_QUERY_N);
    
    // 2. 转换 SIFT
    process_dataset("SIFT", SIFT_DIR, SIFT_D, SIFT_BASE_N, SIFT_QUERY_N);
    
    // 3. 转换 GLOVE
    process_dataset("GLOVE", GLOVE_DIR, GLOVE_D, GLOVE_BASE_N, GLOVE_QUERY_N);

    cout << "\n===============================" << endl;
    cout << "所有数据集已成功转换为 .bin 格式。" << endl;
    cout << "===============================" << endl;

    return 0;
}