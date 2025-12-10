#include "MySolution.h"
// SIMD
#include <immintrin.h>
// CPUID
#if defined(_MSC_VER)
#include <intrin.h> // MSVC
#else
#include <cpuid.h>  // GCC/Clang
#endif
#ifdef COUNT_DIST
    extern atomic<unsigned long long> g_dist_calc_count;
#endif
#ifdef TEST_GRAPH
extern atomic<unsigned long long> g_acc, g_tot;
#endif
namespace {
    class GorderPriorityQueue {
        struct Node {
            int key;      
            int priority; 
        };

        vector<Node> _list;
        unordered_map<int, int> _index_table; 
        void swap(int i, int j) {
            Node tmp = _list[i];
            _list[i] = _list[j];
            _list[j] = tmp;
            _index_table[_list[i].key] = i;
            _index_table[_list[j].key] = j;
        }

    public:
        GorderPriorityQueue(int N) {
            _list.reserve(N);
            _index_table.reserve(N);
            for (int i = 0; i < N; i++) {
                _list.push_back({i, 0});
                _index_table[i] = i;
            }
        }

        void increment(int key) {
            auto it_map = _index_table.find(key);
            if (it_map == _index_table.end()) return; 

            int idx = it_map->second;
            int current_priority = _list[idx].priority;

            auto it = upper_bound(_list.begin(), _list.end(), current_priority, 
                [](int val, const Node& n) { return val < n.priority; });
            
            int new_index = (it - _list.begin()) - 1;

            if (idx != new_index) {
                swap(idx, new_index);
            }
            _list[new_index].priority++;
        }

        void decrement(int key) {
            auto it_map = _index_table.find(key);
            if (it_map == _index_table.end()) return;

            int idx = it_map->second;
            int current_priority = _list[idx].priority;

            auto it = lower_bound(_list.begin(), _list.end(), current_priority, 
                [](const Node& n, int val) { return n.priority < val; });
            
            int new_index = (it - _list.begin());

            if (idx != new_index) {
                swap(idx, new_index);
            }
            _list[new_index].priority--;
        }

        int pop() {
            if (_list.empty()) return -1;
            Node max_node = _list.back();
            _list.pop_back();
            _index_table.erase(max_node.key);
            return max_node.key;
        }
        
        bool empty() const { return _list.empty(); }
    };

    vector<int> compute_gorder_mapping(int N, const function<vector<int>(int)>& get_neighbors, int w = 5) {

        vector<vector<int>> outdegree_table(N);
        vector<vector<int>> indegree_table(N);
        
        for(int i = 0; i < N; ++i) {
            auto neighbors = get_neighbors(i);
            outdegree_table[i] = neighbors;
            for(int neighbor : neighbors) {
                if (neighbor < N) { 
                    indegree_table[neighbor].push_back(i);
                }
            }
        }

        GorderPriorityQueue Q(N);
        vector<int> P(N, 0);

        int seed_node = 0;
        Q.increment(seed_node);
        P[0] = Q.pop();

        for (int i = 1; i < N; i++) {
            int v_e = P[i - 1];

            for (int u : outdegree_table[v_e]) {
                Q.increment(u);
            }
            for (int u : indegree_table[v_e]) {
                Q.increment(u);
                for (int v : outdegree_table[u]) {
                    Q.increment(v);
                }
            }

            if (i > w + 1) {
                int v_b = P[i - w - 1];
                
                for (int u : outdegree_table[v_b]) {
                    Q.decrement(u);
                }
                for (int u : indegree_table[v_b]) {
                    Q.decrement(u);
                    for (int v : outdegree_table[u]) {
                        Q.decrement(v);
                    }
                }
            }
            
            int next_node = Q.pop();
            P[i] = next_node;
        }

        vector<int> mapping(N);
        for(int new_id = 0; new_id < N; ++new_id) {
            mapping[P[new_id]] = new_id;
        }
        return mapping;
    }
    vector<int> reordered_to_original;
    vector<int> compute_rcm_mapping(int N, const function<vector<int>(int)>& get_neighbors) {
        vector<pair<int, int>> sorted_nodes(N); // {node_id, degree}
        vector<int> degrees(N);
        for(int i=0; i<N; ++i) {
            auto neighbors = get_neighbors(i);
            degrees[i] = neighbors.size();
            sorted_nodes[i] = {i, degrees[i]};
        }

        sort(sorted_nodes.begin(), sorted_nodes.end(), 
            [](const pair<int,int>& a, const pair<int,int>& b){
                return a.second < b.second;
            });

        vector<int> P;
        P.reserve(N);
        vector<bool> visited(N, false);

        for(const auto& item : sorted_nodes) {
            int start_node = item.first;
            if(visited[start_node]) continue;

            queue<int> Q;
            Q.push(start_node);
            visited[start_node] = true;
            P.push_back(start_node);

            while(!Q.empty()) {
                int curr = Q.front(); Q.pop();
                
                // 获取邻居并按度数排序
                auto neighbors = get_neighbors(curr);
                sort(neighbors.begin(), neighbors.end(), 
                    [&](int a, int b){ return degrees[a] < degrees[b]; });

                for(int neighbor : neighbors) {
                    if(!visited[neighbor]) {
                        visited[neighbor] = true;
                        P.push_back(neighbor);
                        Q.push(neighbor);
                    }
                }
            }
        }

        reverse(P.begin(), P.end());

        vector<int> mapping(N);
        for(int new_id = 0; new_id < N; ++new_id) {
            mapping[P[new_id]] = new_id;
        }
        return mapping;
    }
    void cpuid(int info[4], int infoType){
        #if defined(_MSC_VER)
            __cpuidex(info, infoType, 0);
        #else
            __cpuid_count(infoType, 0, info[0], info[1], info[2], info[3]);
        #endif
    }
    float distance(const float* a, const float* b, const int dimension) {
        #ifdef COUNT_DIST
        g_dist_calc_count++;
        #endif
        static const struct CPUFeatures {
            bool sse = false, sse3 = false, avx = false, avx512 = false;
            CPUFeatures() {
                int info[4]; cpuid(info, 0); int nIds = info[0];
                if (nIds >= 1) { 
                    cpuid(info, 1); 
                    sse = (info[3] & (1 << 25)) != 0; 
                    sse3 = (info[2] & (1 << 0)) != 0; // SSE3 check
                    avx = (info[2] & (1 << 28)) != 0; 
                }
                if (nIds >= 7) { cpuid(info, 7); avx512 = (info[1] & (1 << 16)) != 0; }
            }
        } features;

        static auto scalar_l2 = [](const float* v1, const float* v2, int d) -> float {
            float res = 0.0f; for (int i = 0; i < d; ++i) { const float diff = v1[i] - v2[i]; res += diff * diff; } return res;
        };

        #if defined(__AVX512F__)
        if (features.avx512) {
            int d16 = dimension & -16;
            __m512 sum = _mm512_setzero_ps();
            for (int i = 0; i < d16; i += 16) {
                __m512 m_v1 = _mm512_load_ps(a + i); __m512 m_v2 = _mm512_load_ps(b + i);
                __m512 diff = _mm512_sub_ps(m_v1, m_v2); sum = _mm512_fmadd_ps(diff, diff, sum);
            }
            float res = _mm512_reduce_add_ps(sum);
            if (dimension > d16) res += scalar_l2(a + d16, b + d16, dimension - d16);
            return res;
        }
        #endif
        #if defined(__AVX__)
        if (features.avx) {
            int d8 = dimension & -8;
            __m256 sum = _mm256_setzero_ps();
            for (int i = 0; i < d8; i += 8) {
                __m256 m_v1 = _mm256_load_ps(a + i); __m256 m_v2 = _mm256_load_ps(b + i);
                __m256 diff = _mm256_sub_ps(m_v1, m_v2); sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
            }
            __m128 sum_high = _mm256_extractf128_ps(sum, 1); __m128 sum_low = _mm256_castps256_ps128(sum);
            __m128 final_sum = _mm_add_ps(sum_high, sum_low); 
            // use SSE3
            #if defined(__SSE3__)
            if (features.sse3) {
                final_sum = _mm_hadd_ps(final_sum, final_sum);
                final_sum = _mm_hadd_ps(final_sum, final_sum);
            } else
            #endif
            { // pure SSE
                final_sum = _mm_add_ps(final_sum, _mm_shuffle_ps(final_sum, final_sum, _MM_SHUFFLE(2, 3, 0, 1)));
                final_sum = _mm_add_ps(final_sum, _mm_shuffle_ps(final_sum, final_sum, _MM_SHUFFLE(1, 0, 3, 2)));
            }
            float res = _mm_cvtss_f32(final_sum);
            if (dimension > d8) res += scalar_l2(a + d8, b + d8, dimension - d8);
            return res;
        }
        #endif
        #if defined(__SSE__)
        if (features.sse) {
            int d4 = dimension & -4;
            __m128 sum = _mm_setzero_ps();
            for (int i = 0; i < d4; i += 4) {
                __m128 m_v1 = _mm_load_ps(a + i); __m128 m_v2 = _mm_load_ps(b + i);
                __m128 diff = _mm_sub_ps(m_v1, m_v2); sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
            }
            // if SSE3 use hadd
            #if defined(__SSE3__)
            if (features.sse3) {
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
            } else 
            #endif
            { // pure SSE
                sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1)));
                sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2)));
            }
            float res = _mm_cvtss_f32(sum);
            if (dimension > d4) res += scalar_l2(a + d4, b + d4, dimension - d4);
            return res;
        }
        #endif
        return scalar_l2(a, b, dimension);
    }
    class VisitedList{
        public:
            unsigned short curV,*mass;
            unsigned int numelements;
            VisitedList(int numelements1){
                curV=-1;
                numelements=numelements1;
                mass=new unsigned short [numelements];
            }
            void reset(){
                if(!++curV)
                    memset(mass,0,sizeof(unsigned short)*numelements),
                    ++curV;
            }
            ~VisitedList(){
                delete[] mass;
            }
    };
    class VisitedListPool{
        stack<VisitedList*>pool;
        mutex poolguard;
        int numelements;
        public:
            VisitedListPool(int initmaxpools,int numelements1){
                numelements=numelements1;
                for(int i=0;i<initmaxpools;++i)
                    pool.push(new VisitedList(numelements));
            }
            VisitedList *getFreeVisitedList(){
                VisitedList *rez;
                unique_lock<mutex>lock(poolguard);
                if(pool.size()>0)
                    rez=pool.top(),pool.pop();
                else rez=new VisitedList(numelements);
                rez->reset();
                return rez;
            }
            void releaseVisitedList(VisitedList *vl){
                unique_lock<mutex>lock(poolguard);
                pool.push(vl);
            }
            ~VisitedListPool(){
                while(!pool.empty()){
                    VisitedList *rez=pool.top();
                    pool.pop(),delete rez;
                }
            }
    };
    char* memory_pool_ = nullptr;
    size_t size_per_element_, offset_links_, offset_data_;
    // memory help
    inline int* get_links_ptr(int id) {
        return (int*)(memory_pool_ + (size_t)id * size_per_element_ + offset_links_);
    }
    inline int get_neighbor_count(int id) {
        return (int)(*get_links_ptr(id));
    }
    inline void set_neighbor_count(int id, int cnt) {
        *get_links_ptr(id) = (int)cnt;
    }
    inline int* get_neighbor_ptr(int id) {
        return (int*)(get_links_ptr(id) + 1);
    }
    inline float* get_data_ptr(int id) {
        return (float*)(memory_pool_ + (size_t)id * size_per_element_ + offset_data_);
    }

    const int K=10, M=16, efConstruction=200, Mmax0=M*2;
    const float gamma=0.3;
    int N, dim, enter_point;
    unique_ptr<VisitedListPool> visited_list_pool_;
    // mutex locks with unique ptr
    vector<unique_ptr<mutex>> node_locks;
    float distance_id(int id, int jd) {
        return distance(get_data_ptr(id), get_data_ptr(jd), dim);
    }
    float distance_qry(int id, const vector<float>& b) {
        return distance(get_data_ptr(id), b.data(), dim);
    }
    void set_memory(int d, int N){
        dim = d;
        if(memory_pool_) {
            #if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
                _aligned_free(memory_pool_);
            #else
                free(memory_pool_);
            #endif
            memory_pool_ = nullptr;
        }

        size_t size_links = sizeof(int) + Mmax0 * sizeof(int);
        offset_links_ = 0;
        offset_data_ = size_links;

        // 64 bytes
        size_t align_requirement = 64; // for AVX-512
        offset_data_ = (offset_data_ + align_requirement - 1) & ~(align_requirement - 1);
        size_per_element_ = offset_data_ + d * sizeof(float);
        size_t cache_line_size = 64; 
        size_per_element_ = (size_per_element_ + cache_line_size - 1) & ~(cache_line_size - 1);
        
        // use _aligned_malloc / posix_memalign
        size_t total_size = (size_t)N * size_per_element_;
        #if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
            // Windows (MSVC or MinGW)
            memory_pool_ = (char*)_aligned_malloc(total_size, align_requirement);
            if (!memory_pool_) throw runtime_error("Failed to allocate aligned memory pool (Windows)");
        #else
            // Linux/macOS
            if (posix_memalign((void**)&memory_pool_, align_requirement, total_size) != 0) {
                throw runtime_error("Failed to allocate aligned memory pool (Linux/macOS)");
            }
        #endif
        if (!memory_pool_) throw runtime_error("Failed to allocate memory pool");
        // locks and neighbors
        node_locks.clear();
        node_locks.reserve(N);
        for(int i=0; i<N; ++i) {
            set_neighbor_count(i, 0);
            node_locks.emplace_back(new mutex());
        }

        // num of threads used to determine VisitedListPool
        int num_threads = thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        visited_list_pool_.reset(new VisitedListPool(num_threads, N));
    }

    void search_layer(int q, int ep, int ef, priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>>& C){
        priority_queue<pair<float,int>> W;
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        unsigned short* mass = vl->mass;
        unsigned short curV = vl->curV;        
        mass[ep] = curV,mass[q]=curV;
        _mm_prefetch((const char*)get_data_ptr(ep), _MM_HINT_T0);
        float xx = distance_id(ep, q);        
        C.push(make_pair(xx, ep));
        W.push(make_pair(xx, ep));       
        while(!C.empty()){
            pair<float,int> c = C.top();
            pair<float,int> f = W.top();
            C.pop();            
            if(c.first > f.first) break;
            unique_lock<mutex> lock_c(*node_locks[c.second]);
            int *c_ptr = get_neighbor_ptr(c.second);
            int c_sz = get_neighbor_count(c.second);
            if (c_sz > 0) {
                int first_neighbor_id = c_ptr[0];
                _mm_prefetch((const char*)(mass + first_neighbor_id), _MM_HINT_T0);
                _mm_prefetch((const char*)get_data_ptr(first_neighbor_id), _MM_HINT_T0);
            }           
            for(int i = 0; i < c_sz; ++i){
                int e = c_ptr[i];
                if (i + 1 < c_sz) {
                    int next_neighbor_id = c_ptr[i+1];
                    _mm_prefetch((const char*)(mass + next_neighbor_id), _MM_HINT_T0);
                    _mm_prefetch((const char*)get_data_ptr(next_neighbor_id), _MM_HINT_T0);
                }

                if(mass[e] != curV){
                    mass[e] = curV;
                    f = W.top();
                    xx = distance_id(e, q);
                    if(W.size() < ef || xx < f.first){
                        C.push(make_pair(xx, e));
                        W.push(make_pair(xx, e));
                        if(W.size() > ef) W.pop();
                    }
                }
            }
        }
        
        while(!W.empty()) C.push(W.top()), W.pop();
        visited_list_pool_->releaseVisitedList(vl);
    }
    void select_neighbors_heuristic(int q, priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>>& C){
        int *q_ptr = get_neighbor_ptr(q);
        int q_sz = 0;
        while(!C.empty() && q_sz < Mmax0){
            pair<float,int> e = C.top(); C.pop();
            bool ok = 1;
            for(int i = 0; i < q_sz; ++i)
                if(distance_id(e.second, q_ptr[i]) < distance_id(e.second, q)){
                    ok = 0; break;
                }
            if(ok) q_ptr[q_sz++] = e.second;
        }
        set_neighbor_count(q, q_sz);
    }
    void insert(int q){
        // use unique_ptr
        unique_lock<mutex> lock_q(*node_locks[q]);
        priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>> W;
        search_layer(q, enter_point, efConstruction, W);
        select_neighbors_heuristic(q, W);
        
        int *q_ptr = get_neighbor_ptr(q), q_sz = get_neighbor_count(q);
        for(int i = 0; i < q_sz; ++i){
            int e = q_ptr[i];
            int *e_ptr = get_neighbor_ptr(e);
            int e_sz = get_neighbor_count(e);            
            // use unique_ptr
            unique_lock<mutex> lock_e(*node_locks[e]);
            if(e_sz < Mmax0) {
                e_ptr[e_sz] = q;
                set_neighbor_count(e, ++e_sz);
            } else {
                for(int j = 0; j < e_sz; ++j)
                    W.push(make_pair(distance_id(e, e_ptr[j]), e_ptr[j]));
                W.push(make_pair(distance_id(e, q), q));
                select_neighbors_heuristic(e, W);
            }
            lock_e.unlock();
        }
        lock_q.unlock();
    }

    atomic<int> build_progress(0);

    string format_duration(long long seconds) {
        if (seconds < 0) return "??:??";
        long long m = seconds / 60;
        long long s = seconds % 60;
        stringstream ss;
        ss << setfill('0') << setw(2) << m << ":" << setfill('0') << setw(2) << s;
        return ss.str();
    }

    void print_progress(int total) {
        auto start_time = chrono::high_resolution_clock::now();
        while (build_progress < total) {
            int current = build_progress.load();
            if (current == 0) {
                cout << "\r0.00% 00:00/??:??" << flush;
                this_thread::sleep_for(chrono::milliseconds(100));
                continue;
            }
            auto now = chrono::high_resolution_clock::now();
            auto elapsed_duration = chrono::duration_cast<chrono::seconds>(now - start_time);
            long long elapsed_seconds = elapsed_duration.count();
            double time_per_item = static_cast<double>(elapsed_seconds) / current;
            long long all_seconds = static_cast<long long>(total * time_per_item);
            
            cout << "\r" << (100 * current / total) << "% " 
                 << format_duration(elapsed_seconds) << "/" 
                 << format_duration(all_seconds) << flush;
            this_thread::sleep_for(chrono::milliseconds(1000));
        }
        auto end_time = chrono::high_resolution_clock::now();
        auto total_duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);
        cout << "\r100.00% " << format_duration(total_duration.count()) << '/' << format_duration(total_duration.count()) << endl;
    }
    void reorder_memory(int N) {
        auto get_neighbors_func = [&](int id) -> vector<int> {
            int* links = get_neighbor_ptr(id);
            int count = get_neighbor_count(id);
            return vector<int>(links, links + count);
        };

        vector<int> old_to_new = compute_gorder_mapping(N, get_neighbors_func, 5);
        
        vector<int> new_to_old(N);
        for(int old_id=0; old_id<N; ++old_id) {
            new_to_old[old_to_new[old_id]] = old_id;
        }

        char* new_pool = nullptr;
        size_t total_size = (size_t)N * size_per_element_;
        size_t align_req = 64;
        
        #if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
            new_pool = (char*)_aligned_malloc(total_size, align_req);
        #else
            posix_memalign((void**)&new_pool, align_req, total_size);
        #endif

        for(int new_id = 0; new_id < N; ++new_id) {
            int old_id = new_to_old[new_id];
            
            char* src_ptr = memory_pool_ + (size_t)old_id * size_per_element_;
            char* dst_ptr = new_pool + (size_t)new_id * size_per_element_;
            
            memcpy(dst_ptr, src_ptr, size_per_element_);

            int* links_ptr = (int*)(dst_ptr + offset_links_); 
            int count = *links_ptr; 
            int* neighbors = links_ptr + 1; 
            for(int j = 0; j < count; ++j) {
                neighbors[j] = old_to_new[neighbors[j]]; 
            }
        }

        #if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
            _aligned_free(memory_pool_);
        #else
            free(memory_pool_);
        #endif
        memory_pool_ = new_pool;

        // 7. 更新入口点 (因为它的 ID 也变了)
        enter_point = old_to_new[enter_point];
        reordered_to_original = new_to_old;
    }
    void KNN_build(int d, const vector<float>& base){
        N = base.size() / d;
        set_memory(d, N);
        for(int i = 0; i < N; ++i)
            memcpy(get_data_ptr(i), base.data() + (size_t)i * d, dim * sizeof(float));
        
        enter_point = 0;
        // build_progress = 1; // 0 for enter_point

        // thread progress_thread(print_progress, N);
        
        int num_threads = thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        
        vector<thread> workers;
        atomic<int> current_idx(1); 
        for(int t = 0; t < num_threads; ++t) {
            workers.emplace_back([&]() {
                while(true) {
                    int i = current_idx++;
                    if(i >= N) break;
                    insert(i);
                    // build_progress++;
                }
            });
        }

        for(auto& t : workers) {
            if(t.joinable()) t.join();
        }
        reorder_memory(N);
        // if (progress_thread.joinable())
        //     progress_thread.join();
    }

    void KNN_search(const vector<float>& q, int *res){
        priority_queue<pair<float,int>> Bk;
        priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>> C;
        
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        unsigned short* mass = vl->mass;
        unsigned short curV = vl->curV;
        
        int ep = enter_point;
        _mm_prefetch((const char*)get_data_ptr(ep), _MM_HINT_T0);
        float xx = distance_qry(ep, q);
        int num_nodes_built = N; 

        if (num_nodes_built > 0) {

            const int num_initializations = 100; 
            int step_size = num_nodes_built / num_initializations;
            if (step_size == 0) step_size = 1;

            float min_dist = numeric_limits<float>::max();
            int best_ep = 0;

            for (int node = 0; node < num_nodes_built; node += step_size) {
                _mm_prefetch((const char*)get_data_ptr(node), _MM_HINT_T0);
                float dist = distance_qry(node, q); 
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_ep = node;
                }
            }
            ep = best_ep; 
            xx = min_dist;
        }
        mass[ep] = curV;
        C.push(make_pair(xx, ep));
        Bk.push(make_pair(xx, ep));
        
        while(!C.empty()){
            auto c = C.top();
            C.pop();
            
            if(Bk.size() >= K && c.first > (1.0f + gamma) * Bk.top().first) break;
            
            int *c_ptr = get_neighbor_ptr(c.second);
            int c_sz = get_neighbor_count(c.second);

            if (c_sz > 0) {
                int first_neighbor_id = c_ptr[0];
                _mm_prefetch((const char*)(mass + first_neighbor_id), _MM_HINT_T0);
                _mm_prefetch((const char*)get_data_ptr(first_neighbor_id), _MM_HINT_T0);
            }

            for(int i = 0; i < c_sz; ++i){
                int e = c_ptr[i];
                if (i + 1 < c_sz) {
                    int next_neighbor_id = c_ptr[i+1];
                    _mm_prefetch((const char*)(mass + next_neighbor_id), _MM_HINT_T0);
                    _mm_prefetch((const char*)get_data_ptr(next_neighbor_id), _MM_HINT_T0);
                }

                if(mass[e] != curV){
                    mass[e] = curV;
                    xx = distance_qry(e, q);
                    
                    if(Bk.size() < K || xx < Bk.top().first){
                        Bk.push(make_pair(xx, e));
                        if(Bk.size() > K) Bk.pop();
                    }
                    
                    if(Bk.size() < K || xx < (1.0f + gamma) * Bk.top().first)
                        C.push(make_pair(xx, e));
                }
            }
        }
        
        int nm = K;
        while(!Bk.empty() && nm > 0) {
            if (!reordered_to_original.empty()) 
                res[--nm] = reordered_to_original[Bk.top().second];
            else 
                res[--nm] = Bk.top().second; 
            
            Bk.pop();
        }
        visited_list_pool_->releaseVisitedList(vl);
    }
}

void Solution::build(int d, const vector<float>& base){
    KNN_build(d, base);
}

void Solution::search(const vector<float>& query, int* res){
    KNN_search(query, res);
}