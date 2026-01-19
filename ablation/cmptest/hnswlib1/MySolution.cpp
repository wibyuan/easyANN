#include "MySolution.h"

#include <bits/stdc++.h>
using namespace std;

int dim;
const int K = 10;

std::unique_ptr<hnswlib::L2Space> space;
std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_index;
void Solution::build(int d, const std::vector<float>& base) {
    dim = d;
    int num_elements = base.size() / dim;

    const int M = 32;
    const int ef_construction = 256;

    space = std::make_unique<hnswlib::L2Space>(dim);
    hnsw_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), num_elements, M, ef_construction);


    int num_threads = thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        
        vector<thread> workers;
        atomic<int> current_idx(1); 
        for(int t = 0; t < num_threads; ++t) {
            workers.emplace_back([&]() {
                while(true) {
                    int i = current_idx++;
                    if(i >= num_elements) break;
        hnsw_index->addPoint(base.data() + i * dim, i);
                    // build_progress++;
                }
            });
        }

        for(auto& t : workers) {
            if(t.joinable()) t.join();
        }

}
int ef_search = 200;
void Solution::search(const std::vector<float>& query, int* res) {
    if (!hnsw_index) {
        return;
    }

    // Set the search-time ef parameter
    // ef > K is required.
    
    hnsw_index->setEf(ef_search);

    // 1. Perform the KNN search
    // The result is a std::priority_queue, which is a max-heap.
    // The top() element is the FARTHEST of the K neighbors.
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result_queue =
        hnsw_index->searchKnn(query.data(), K);

    // 2. Correctly extract results into the res array (from nearest to farthest)
    int result_size = result_queue.size();
    
    // We need to fill the res array from the end, because the max-heap
    // gives us the farthest elements first.
    for (int i = result_size - 1; i >= 0; --i) {
        res[i] = result_queue.top().second; // .second is the label
        result_queue.pop();
    }

    // If the number of results found is less than K, fill the rest with -1 or a default value.
    // (This part depends on the evaluator's requirement for partial results, but it's good practice)
    for (int i = result_size; i < K; ++i) {
        res[i] = -1; 
    }
}
// === 消融实验接口实现 ===

void Solution::set_gamma(float gamma_val){
}

void Solution::set_ef_search(int ef_val){
    ef_search = ef_val;
}

vector<int> Solution::get_degree_distribution(){
    return vector<int>();
}

void Solution::save_graph(const string& path){
}

bool Solution::load_graph(const string& path){
    return false;
}