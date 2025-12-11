#ifndef CPP_SOLUTION_H
#define CPP_SOLUTION_H
#include<bits/stdc++.h>
using namespace std;
class Solution {
public:
    void build(int d, const vector<float>& base);
    void search(const vector<float>& query, int* res);

    // === 消融实验接口 ===
    void set_gamma(float gamma_val);
    void set_ef_search(int ef_val);
    vector<int> get_degree_distribution();
    void save_graph(const string& path);
    bool load_graph(const string& path);
};
#endif
