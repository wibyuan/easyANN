#pragma once
#include "hnswlib/hnswlib.h"
#include <vector>
#include <memory>


// // Forward declare hnswlib classes
// namespace hnswlib {
//     template<typename> class HierarchicalNSW;
//     template<typename> class L2Space;
// }

class Solution {
public:
    Solution();
    ~Solution();

    void build(int d, const std::vector<float>& base);
    void search(const std::vector<float>& query, int* res);

private:
    int dim;
    const int K = 10;

    std::unique_ptr<hnswlib::L2Space> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_index;
};