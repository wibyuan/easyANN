#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <filesystem> // C++17

namespace fs = std::filesystem;
using namespace std;

// --- 配置：您可以随意修改这些参数 ---
const string DEBUG_DIR = "../data/debug"; // 输出目录
const int D = 16;       // 调试用的维度 (小一点)
const int BASE_N = 1000;  // 1000 个底库向量
const int QUERY_N = 100;  // 100 个查询向量
const unsigned int SEED = 2025; // 固定种子
// ---------------------------------

const string BASE_PATH = DEBUG_DIR + "/base.txt";
const string QUERY_PATH = DEBUG_DIR + "/query.txt";

// 使用 mt19937 生成数据并写入文件
void generate_file(const string& filename, int N, int D, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "错误: 无法创建 " << filename << endl;
        return;
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            fout << dist(gen) << (j == D - 1 ? "" : " ");
        }
        fout << "\n";
    }
    fout.close();
    cout << "已生成: " << filename << " (N=" << N << ", D=" << D << ")" << endl;
}

int main() {
    cout << "正在生成调试数据集..." << endl;

    // 1. 创建目录 (如果不存在)
    try {
        fs::create_directory(DEBUG_DIR);
    } catch (const std::exception& e) {
        cerr << "创建目录 " << DEBUG_DIR << " 失败: " << e.what() << endl;
    }

    // 2. 初始化生成器
    std::mt19937 gen(SEED);

    // 3. 生成 base.txt
    generate_file(BASE_PATH, BASE_N, D, gen);

    // 4. 生成 query.txt
    generate_file(QUERY_PATH, QUERY_N, D, gen);

    cout << "调试数据集生成完毕。" << endl;
    return 0;
}