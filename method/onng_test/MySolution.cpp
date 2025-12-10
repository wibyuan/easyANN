#include "MySolution.h"
// SIMD
#include<immintrin.h>
// CPUID
#if defined(_MSC_VER)
#include<intrin.h> // MSVC
#else
#include<cpuid.h>  // GCC/Clang
#include<x86intrin.h>
#endif
#ifdef COUNT_DIST
extern atomic<unsigned long long> g_dist_calc_count;
#endif
#ifdef TEST_GRAPH
extern atomic<unsigned long long> g_acc, g_tot;
#endif
namespace {
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
    struct CpuFlags {
        bool sse = false;
        bool sse2 = false;
        bool sse3 = false;
        bool ssse3 = false;
        bool sse41 = false;
        bool sse42 = false;
        bool avx = false;
        bool avx2 = false;
        bool fma = false;
        bool avx512f = false;
        bool avx512dq = false;
        bool avx512bw = false;
        bool avx512vl = false;
        
        CpuFlags() { detect(); }

        void detect() {
            int info[4];
            // CPUID 0
            cpuid(info, 0);
            int nIds = info[0];

            // CPUID 1
            if (nIds >= 1) {
                cpuid(info, 1);
                sse    = (info[3] & (1 << 25)) != 0;
                sse2   = (info[3] & (1 << 26)) != 0;
                sse3   = (info[2] & (1 << 0))  != 0;
                ssse3  = (info[2] & (1 << 9))  != 0;
                sse41  = (info[2] & (1 << 19)) != 0;
                sse42  = (info[2] & (1 << 20)) != 0;
                avx    = (info[2] & (1 << 28)) != 0;
                fma    = (info[2] & (1 << 12)) != 0;
            }

            // CPUID 7
            if (nIds >= 7) {
                cpuid(info, 7);
                avx2     = (info[1] & (1 << 5))  != 0;
                avx512f  = (info[1] & (1 << 16)) != 0;
                avx512dq = (info[1] & (1 << 17)) != 0;
                avx512bw = (info[1] & (1 << 30)) != 0;
                avx512vl = (info[1] & (1 << 31)) != 0;
            }
        }

        static void cpuid(int info[4], int infoType) {
            #if defined(_MSC_VER)
                __cpuidex(info, infoType, 0);
            #else
                __cpuid_count(infoType, 0, info[0], info[1], info[2], info[3]);
            #endif
        }
    };
    // ---------------------------------------------------------
    // Distance Kernels 
    // ---------------------------------------------------------
    
    // 1. Scalar 兜底
    float dist_scalar(const float* a, const float* b, int dim) {
        float res = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            res += diff * diff;
        }
        return res;
    }

    // 2. SSE1 
    float dist_sse1(const float* a, const float* b, int dim) {
        #if defined(__SSE__)
        __m128 sum = _mm_setzero_ps();
        for (int i = 0; i < dim; i += 4) {
            __m128 v1 = _mm_load_ps(a + i);
            __m128 v2 = _mm_load_ps(b + i);
            __m128 diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        // Pure SSE Reduction (No SSE3 instructions)
        // [0, 1, 2, 3] -> [2, 3, 2, 3]
        __m128 t1 = _mm_movehl_ps(sum, sum); 
        // [0+2, 1+3, 2+2, 3+3]
        __m128 t2 = _mm_add_ps(sum, t1);     
        // [1+3, 1+3, 1+3, 1+3] (Shuffle index 1 to all)
        __m128 t3 = _mm_shuffle_ps(t2, t2, 0x55); 
        // [0+2+1+3, ...]
        __m128 t4 = _mm_add_ss(t2, t3);      
        return _mm_cvtss_f32(t4);
        #else
        return dist_scalar(a, b, dim);
        #endif
    }

    // 3. SSE3 (HADD)
    float dist_sse3(const float* a, const float* b, int dim) {
        #if defined(__SSE3__)
        __m128 sum = _mm_setzero_ps();
        for (int i = 0; i < dim; i += 4) {
            __m128 v1 = _mm_load_ps(a + i);
            __m128 v2 = _mm_load_ps(b + i);
            __m128 diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        // Optimized Reduction (Hardware HADD)
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
        #else
        // 如果编译器不支持 SSE3 宏，回退到 SSE1
        return dist_sse1(a, b, dim);
        #endif
    }

    // 4. AVX (256-bit)
    // 既然有 AVX，那必定支持 SSE3，所以 Reduction 可以放心用 hadd
    float dist_avx(const float* a, const float* b, int dim) {
        #if defined(__AVX__)
        __m256 sum = _mm256_setzero_ps();
        for (int i = 0; i < dim; i += 8) {
            __m256 v1 = _mm256_load_ps(a + i);
            __m256 v2 = _mm256_load_ps(b + i);
            __m256 diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }
        // Reduction
        __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
        __m128 sum_lo = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(sum_hi, sum_lo);
        
        // 使用 SSE3 hadd
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        return _mm_cvtss_f32(sum128);
        #else
        return dist_sse3(a, b, dim);
        #endif
    }

    // 5. AVX2 + FMA (256-bit) 
    float dist_avx2_fma(const float* a, const float* b, int dim) {
        #if defined(__AVX2__) && defined(__FMA__)
        __m256 sum = _mm256_setzero_ps();
        for (int i = 0; i < dim; i += 8) {
            __m256 v1 = _mm256_load_ps(a + i);
            __m256 v2 = _mm256_load_ps(b + i);
            __m256 diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_fmadd_ps(diff, diff, sum); // FMA: d*d + sum
        }
        __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
        __m128 sum_lo = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(sum_hi, sum_lo);
        
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        return _mm_cvtss_f32(sum128);
        #else
        return dist_avx(a, b, dim);
        #endif
    }

    // 6. AVX-512 (512-bit)
    float dist_avx512(const float* a, const float* b, int dim) {
        #if defined(__AVX512F__)
        __m512 sum = _mm512_setzero_ps();
        for (int i = 0; i < dim; i += 16) {
            __m512 v1 = _mm512_load_ps(a + i);
            __m512 v2 = _mm512_load_ps(b + i);
            __m512 diff = _mm512_sub_ps(v1, v2);
            sum = _mm512_fmadd_ps(diff, diff, sum);
        }
        return _mm512_reduce_add_ps(sum);
        #else
        return dist_avx2_fma(a, b, dim);
        #endif
    }

    // ---------------------------------------------------------
    // 全局配置与分发器
    // ---------------------------------------------------------
    typedef float (*DistFunc)(const float*, const float*, int);
    DistFunc g_dist_func = dist_scalar;
    int g_simd_width = 1;

    void prepare_simd() {
        static bool inited = false;
        if (inited) return;
        
        CpuFlags flags;
        
        // 优先级倒序：最强指令集优先
        
        // 1. AVX-512
        #if defined(__AVX512F__)
        if (flags.avx512f) {
            g_dist_func = dist_avx512;
            g_simd_width = 16;
            inited = true; return;
        }
        #endif

        // 2. AVX2 + FMA
        #if defined(__AVX2__) && defined(__FMA__)
        if (flags.avx2 && flags.fma) {
            g_dist_func = dist_avx2_fma;
            g_simd_width = 8;
            inited = true; return;
        }
        #endif

        // 3. AVX (包含 SSE3)
        #if defined(__AVX__)
        if (flags.avx) {
            g_dist_func = dist_avx;
            g_simd_width = 8;
            inited = true; return;
        }
        #endif

        // 4. SSE3
        #if defined(__SSE3__)
        if (flags.sse3) {
            g_dist_func = dist_sse3;
            g_simd_width = 4;
            inited = true; return;
        }
        #endif

        // 5. SSE1 (Legacy Shuffle)
        #if defined(__SSE__)
        if (flags.sse) {
            g_dist_func = dist_sse1;
            g_simd_width = 4;
            inited = true; return;
        }
        #endif

        // 6. Scalar Fallback
        g_dist_func = dist_scalar;
        g_simd_width = 1;
        inited = true;
    }
    // 包装函数
    float distance(const float* a, const float* b, int dim_padded) {
        #ifdef COUNT_DIST
        g_dist_calc_count++;
        #endif
        return g_dist_func(a, b, dim_padded);
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
    const int K=10, efConstruction=140, Mmax0=128;
    const float epsilon=0.23,gamma=0.22;
    int N, dim, dim_padded, enter_point;
    unique_ptr<VisitedListPool> visited_list_pool_;
    // mutex locks with unique ptr
    vector<unique_ptr<mutex>> node_locks;
    float distance_id(int id, int jd) {
        return distance(get_data_ptr(id), get_data_ptr(jd), dim);
    }
    float distance_qry_padded(int id, const float* padded_query) {
        return distance(get_data_ptr(id), padded_query, dim_padded);
    }
    // ---------------------------------------------------------
    // 内存对齐辅助函数
    // ---------------------------------------------------------
    void* aligned_alloc_wrapper(size_t size, size_t alignment) {
        #if defined(_MSC_VER) || defined(__MINGW32__)
            return _aligned_malloc(size, alignment);
        #else
            void* p = nullptr;
            if (posix_memalign(&p, alignment, size) != 0) return nullptr;
            return p;
        #endif
    }

    void aligned_free_wrapper(void* p) {
        #if defined(_MSC_VER) || defined(__MINGW32__)
            _aligned_free(p);
        #else
            free(p);
        #endif
    }
    void set_memory(int d, int N_val) {
        dim = d;
        // 关键：计算补齐后的维度
        dim_padded = (d + g_simd_width - 1) & ~(g_simd_width - 1);

        if (memory_pool_) {
            aligned_free_wrapper(memory_pool_);
            memory_pool_ = nullptr;
        }

        size_t size_links = sizeof(int) + Mmax0 * sizeof(int);
        offset_links_ = 0;
        offset_data_ = size_links;

        size_t align_requirement = 64; 
        offset_data_ = (offset_data_ + align_requirement - 1) & ~(align_requirement - 1);
        
        // 使用 dim_padded 预留空间
        size_per_element_ = offset_data_ + dim_padded * sizeof(float);
        
        size_t cache_line_size = 64;
        size_per_element_ = (size_per_element_ + cache_line_size - 1) & ~(cache_line_size - 1);

        size_t total_size = (size_t)N_val * size_per_element_;
        
        memory_pool_ = (char*)aligned_alloc_wrapper(total_size, align_requirement);
        if (!memory_pool_) throw runtime_error("Failed to allocate memory pool");

        node_locks.clear();
        node_locks.reserve(N_val);
        for (int i = 0; i < N_val; ++i) {
            set_neighbor_count(i, 0);
            node_locks.emplace_back(new mutex());
        }

        int num_threads = thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        visited_list_pool_.reset(new VisitedListPool(num_threads, N_val));
    }

    inline bool has_edge(int u, int v) {
        int *u_ptr=get_neighbor_ptr(u), u_sz=get_neighbor_count(u);
        for (int i = 0; i < u_sz; ++i) {
            if (u_ptr[i] == v) return true;
        }
        return false;
    }

    struct NodeDist {
        int id;
        float dist;
        bool operator<(const NodeDist& other) const {
            return dist < other.dist;
        }
    };

    // this was translated from NGT GraphReconstructor::adjustPathsEffectively
    // 我忽然觉得标记借鉴的代码库很有必要，到时候 readme.md 漏了就不好
    void shortcut_reduction(int N,double epsilon) {
        const int MIN_EDGES = 5; 
        for (int u = 0; u < N; ++u) {
            int* u_ptr = get_neighbor_ptr(u), u_sz = get_neighbor_count(u);

            if (u_sz <= MIN_EDGES) continue;

            // Snapshot & Sort
            vector<NodeDist> neighbors;
            neighbors.reserve(u_sz);
            for (int i = 0; i < u_sz; ++i) {
                int v = u_ptr[i];
                neighbors.push_back({v, distance_id(u, v)});
            }
            sort(neighbors.begin(), neighbors.end());

            vector<bool> remove_mask(u_sz, false);
            int current_degree = u_sz;

            for (int i = 0; i < u_sz; ++i) {
                int v = neighbors[i].id;
                float d_uv = neighbors[i].dist;

                for (int j = 0; j < u_sz; ++j) {
                    if (i == j) continue; 
                    if (remove_mask[j]) continue; 

                    int w = neighbors[j].id;
                    float d_uw = neighbors[j].dist;

                    if (d_uw >= (1.0f + epsilon)*d_uv) continue;

                    if (!has_edge(w, v)) continue;

                    float d_wv = distance_id(w, v);
                    if (d_wv < d_uv) {
                        remove_mask[i] = true;
                        current_degree--;
                        break; 
                    }
                }

                if (current_degree <= MIN_EDGES) break;
            }

            // Rewrite in-place
            if (current_degree < u_sz) {
                int new_idx = 0;
                for (int i = 0; i < u_sz; ++i) {
                    if (!remove_mask[i]) {
                        u_ptr[new_idx] = neighbors[i].id;
                        new_idx++;
                    }
                }
                set_neighbor_count(u, new_idx);
            }
        }
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

        vector<int> old_to_new = compute_rcm_mapping(N, get_neighbors_func);
        
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

        enter_point = old_to_new[enter_point];
        reordered_to_original = new_to_old;
    }
    #ifdef TEST_GRAPH
    void quick_search(int u,int v){
        while(1){
            int su=u,*u_ptr=get_neighbor_ptr(u),u_sz=get_neighbor_count(u);
            for(int z=0;z<u_sz;++z)
                if(distance_id(u,v)>distance_id(u_ptr[z],v))
                    u=u_ptr[z];
            if(u==su)break;
            g_tot++;
        }
        if(u==v)g_acc++;
    }
    void KNN_check(){
        mt19937 rng{114514};
        uniform_int_distribution dist{0,N-1};
        for(int i=0;i<10000;++i){
            int u=dist(rng),v=dist(rng);
            quick_search(u,v);
        }
    }
    #endif
    void KNN_build(int d, const vector<float>& base){
        prepare_simd();
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
        shortcut_reduction(N,0.1);
        shortcut_reduction(N,0.23);
        #ifdef TEST_GRAPH
        KNN_check();
        #endif
        reorder_memory(N);
        // if (progress_thread.joinable())
        //     progress_thread.join();
    }
    void KNN_search_internal(const float* padded_query, int *res){
        priority_queue<pair<float,int>> Bk;
        priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>> C;
        
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        unsigned short* mass = vl->mass;
        unsigned short curV = vl->curV;
        
        int ep = enter_point;
        _mm_prefetch((const char*)get_data_ptr(ep), _MM_HINT_T0);
        float xx = distance_qry_padded(ep, padded_query);
        int num_nodes_built = N; 

        if (num_nodes_built > 0) {
            const int num_initializations = 100; 
            int step_size = num_nodes_built / num_initializations;
            if (step_size == 0) step_size = 1;
            float min_dist = numeric_limits<float>::max();
            int best_ep = 0;
            for (int node = 0; node < num_nodes_built; node += step_size) {
                _mm_prefetch((const char*)get_data_ptr(node), _MM_HINT_T0);
                float dist = distance_qry_padded(node, padded_query); 
                if (dist < min_dist) { min_dist = dist; best_ep = node; }
            }
            ep = best_ep; xx = min_dist;
        }
        mass[ep] = curV;
        C.push(make_pair(xx, ep));
        Bk.push(make_pair(xx, ep));
        
        while(!C.empty()){
            auto c = C.top(); C.pop();
            if(Bk.size() >= K && c.first > (1.0f + gamma) * Bk.top().first) break;
            
            int *c_ptr = get_neighbor_ptr(c.second);
            int c_sz = get_neighbor_count(c.second);
            if (c_sz > 0) {
                int first = c_ptr[0];
                _mm_prefetch((const char*)(mass + first), _MM_HINT_T0);
                _mm_prefetch((const char*)get_data_ptr(first), _MM_HINT_T0);
            }           
            for(int i = 0; i < c_sz; ++i){
                int e = c_ptr[i];
                if (i + 1 < c_sz) {
                    int next = c_ptr[i+1];
                    _mm_prefetch((const char*)(mass + next), _MM_HINT_T0);
                    _mm_prefetch((const char*)get_data_ptr(next), _MM_HINT_T0);
                }
                if(mass[e] != curV){
                    mass[e] = curV;
                    xx = distance_qry_padded(e, padded_query);
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
            if (!reordered_to_original.empty()) res[--nm] = reordered_to_original[Bk.top().second];
            else res[--nm] = Bk.top().second; 
            Bk.pop();
        }
        visited_list_pool_->releaseVisitedList(vl);
    }
    void KNN_search(const vector<float>& q, int *res){
        static const size_t align = 64;
        void* ptr = aligned_alloc_wrapper(dim_padded * sizeof(float), align);
        if(!ptr) return; // fatal

        float* padded_q = (float*)ptr;
        memcpy(padded_q, q.data(), dim * sizeof(float));
        if (dim_padded > dim) {
            memset(padded_q + dim, 0, (dim_padded - dim) * sizeof(float));
        }
        
        KNN_search_internal(padded_q, res);
        aligned_free_wrapper(ptr);
    }
}

void Solution::build(int d, const vector<float>& base){
    KNN_build(d, base);
}

void Solution::search(const vector<float>& query, int* res){
    KNN_search(query, res);
}