#include "MySolution.h"

// --- 所有的 flatnav 头文件，现在只在这里被包含 ---
#include "flatnav/index/Index.h"
#include "flatnav/distances/SquaredL2Distance.h"
#include "flatnav/util/Datatype.h"

// --- FlatNav 的参数 ---
const int M = 16;
const int efConstruction = 200;
const int efSearch = 500;

// --- 1. 定义那个被隐藏的实现类 ---
class Solution::SolutionImpl {
public:
    // 把之前 Solution 类的所有私有成员和逻辑都搬到这里
    int dim;
    static const int K = 10;

    using DistanceType = flatnav::distances::SquaredL2Distance<flatnav::util::DataType::float32>;
    using IndexType = flatnav::Index<DistanceType, int>;
    
    std::unique_ptr<IndexType> index;

    // --- build 和 search 的真正实现 ---
    void build(int d, const std::vector<float>& base) {
        this->dim = d;
        const int num_vectors = base.size() / d;

        auto distance = DistanceType::create(dim);
        index = std::make_unique<IndexType>(std::move(distance), num_vectors, M);
        
        std::vector<int> labels(num_vectors);
        for (int i = 0; i < num_vectors; ++i) {
            labels[i] = i;
        }

        if (!base.empty()) {
            index->addBatch<float>(
                const_cast<float*>(base.data()),
                labels,
                efConstruction
            );
        }
    }

    void search(const std::vector<float>& query, int* res) {
        if (!index || query.empty()) {
            return;
        }

        auto results = index->search(query.data(), K, efSearch);

        for (int i = 0; i < K && i < results.size(); ++i) {
            res[i] = results[i].second;
        }
    }
};

// --- 2. 实现 Solution 类的接口函数，让它们去调用真正的实现 ---

Solution::Solution() : pimpl(std::make_unique<SolutionImpl>()) {
    // 构造函数，创建 pimpl 实例
}

Solution::~Solution() {
    // 析构函数，必须在这里定义，即使是空的
}

void Solution::build(int d, const std::vector<float>& base) {
    pimpl->build(d, base);
}

void Solution::search(const std::vector<float>& query, int* res) {
    pimpl->search(query, res);
}