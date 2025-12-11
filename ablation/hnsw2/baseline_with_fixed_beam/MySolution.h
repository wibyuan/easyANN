#ifndef CPP_SOLUTION_H
#define CPP_SOLUTION_H
#include<bits/stdc++.h>
using namespace std;

class Solution {
public:
    // 原有接口
    void build(int d, const vector<float>& base);
    void search(const vector<float>& query, int* res);

    // === 新增接口 ===

    // 搜索参数设置（二选一生效）
    void set_gamma(float gamma_val);    // 自适应搜索（本变体忽略）
    void set_ef_search(int ef_val);     // 固定 beam

    // Build 统计
    vector<int> get_degree_distribution();  // 返回完整度数序列 (每个节点在 layer 0 的邻居数)

    // 图缓存
    void save_graph(const string& path);    // 序列化图到文件
    bool load_graph(const string& path);    // 从文件加载图，返回是否成功
};

#endif