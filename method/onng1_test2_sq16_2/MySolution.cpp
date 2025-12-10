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
    // ---------------------------------------------------------
    // SQ16 全局配置
    // ---------------------------------------------------------
    // 范围 [-15000, 15000]，跨度 30000。
    const float SQ_MAX = 15000.0f;
    const float SQ_MIN = -15000.0f;
    float g_min_val = 0.0f;
    float g_scale = 1.0f;

    inline int16_t quantize_one(float x) {
        float v = (x - g_min_val) * g_scale + SQ_MIN;
        if (v < SQ_MIN) v = SQ_MIN;
        if (v > SQ_MAX) v = SQ_MAX;
        return (int16_t)(v + 0.5f);
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
    struct CpuFlags {
        bool sse = false;
        bool sse2 = false; // Baseline for SQ16
        bool sse3 = false;
        bool avx = false;
        bool avx2 = false;
        bool avx512f = false;
        bool avx512bw = false;
        
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
                avx    = (info[2] & (1 << 28)) != 0;
            }

            // CPUID 7
            if (nIds >= 7) {
                cpuid(info, 7);
                avx2     = (info[1] & (1 << 5))  != 0;
                avx512f  = (info[1] & (1 << 16)) != 0;
                avx512bw = (info[1] & (1 << 30)) != 0;
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
    // Distance Kernels (SQ16 - Int16)
    // ---------------------------------------------------------
    
    // 1. Scalar 兜底 (Int16)
    float dist_sq16_scalar(const int16_t* a, const int16_t* b, int dim) {
        float res = 0.0f;
        for (int i = 0; i < dim; ++i) {
            int diff = (int)a[i] - (int)b[i];
            res += (float)(diff * diff);
        }
        return res;
    }

    // 2. SSE2 (Native Integer Support)
    float dist_sq16_sse2(const int16_t* a, const int16_t* b, int dim) {
        #if defined(__SSE2__)
        __m128 sum_f = _mm_setzero_ps();
        int i = 0;
        for (; i <= dim - 8; i += 8) {
            __m128i v1 = _mm_loadu_si128((const __m128i*)(a + i));
            __m128i v2 = _mm_loadu_si128((const __m128i*)(b + i));
            __m128i diff = _mm_sub_epi16(v1, v2);
            // madd: res[0]=d0*d0+d1*d1, ... (returns 4x int32)
            __m128i sq_sum = _mm_madd_epi16(diff, diff); 
            // 累加到 float 防止 int32 溢出 (对于高维数据)
            sum_f = _mm_add_ps(sum_f, _mm_cvtepi32_ps(sq_sum));
        }
        // Reduction
        __m128 t1 = _mm_movehl_ps(sum_f, sum_f); 
        __m128 t2 = _mm_add_ps(sum_f, t1);     
        __m128 t3 = _mm_shuffle_ps(t2, t2, 1); 
        __m128 t4 = _mm_add_ss(t2, t3);      
        float res = _mm_cvtss_f32(t4);

        // Handle tail
        for (; i < dim; ++i) {
            int d = (int)a[i] - (int)b[i];
            res += (float)(d * d);
        }
        return res;
        #else
        return dist_sq16_scalar(a, b, dim);
        #endif
    }

    // 3. AVX2 (256-bit Int16)
    float dist_sq16_avx2(const int16_t* a, const int16_t* b, int dim) {
        #if defined(__AVX2__)
        __m256 sum_f = _mm256_setzero_ps();
        int i = 0;
        for (; i <= dim - 16; i += 16) {
            __m256i v1 = _mm256_loadu_si256((const __m256i*)(a + i));
            __m256i v2 = _mm256_loadu_si256((const __m256i*)(b + i));
            __m256i diff = _mm256_sub_epi16(v1, v2);
            __m256i sq_sum = _mm256_madd_epi16(diff, diff);
            sum_f = _mm256_add_ps(sum_f, _mm256_cvtepi32_ps(sq_sum));
        }
        __m128 sum_hi = _mm256_extractf128_ps(sum_f, 1);
        __m128 sum_lo = _mm256_castps256_ps128(sum_f);
        __m128 sum128 = _mm_add_ps(sum_hi, sum_lo);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float res = _mm_cvtss_f32(sum128);

        for (; i < dim; ++i) {
            int d = (int)a[i] - (int)b[i];
            res += (float)(d * d);
        }
        return res;
        #else
        return dist_sq16_sse2(a, b, dim);
        #endif
    }

    // 4. AVX-512 (512-bit Int16)
    float dist_sq16_avx512(const int16_t* a, const int16_t* b, int dim) {
        #if defined(__AVX512F__) && defined(__AVX512BW__)
        __m512 sum_f = _mm512_setzero_ps();
        int i = 0;
        for (; i <= dim - 32; i += 32) {
            __m512i v1 = _mm512_loadu_si512(a + i);
            __m512i v2 = _mm512_loadu_si512(b + i);
            __m512i diff = _mm512_sub_epi16(v1, v2);
            __m512i sq_sum = _mm512_madd_epi16(diff, diff);
            sum_f = _mm512_add_ps(sum_f, _mm512_cvtepi32_ps(sq_sum));
        }
        float res = _mm512_reduce_add_ps(sum_f);
        for (; i < dim; ++i) {
            int d = (int)a[i] - (int)b[i];
            res += (float)(d * d);
        }
        return res;
        #else
        return dist_sq16_avx2(a, b, dim);
        #endif
    }

    // ---------------------------------------------------------
    // 全局配置与分发器
    // ---------------------------------------------------------
    typedef float (*DistFunc)(const int16_t*, const int16_t*, int);
    DistFunc g_dist_func = dist_sq16_scalar;
    int g_simd_width = 1;

    void prepare_simd() {
        static bool inited = false;
        if (inited) return;
        
        CpuFlags flags;
        
        // 1. AVX-512 (BW needed for int16 ops)
        #if defined(__AVX512F__) && defined(__AVX512BW__)
        if (flags.avx512f && flags.avx512bw) {
            g_dist_func = dist_sq16_avx512;
            g_simd_width = 32; // 32 * 16bit = 512bit
            inited = true; return;
        }
        #endif

        // 2. AVX2
        #if defined(__AVX2__)
        if (flags.avx2) {
            g_dist_func = dist_sq16_avx2;
            g_simd_width = 16;
            inited = true; return;
        }
        #endif

        // 3. SSE2 (Baseline for SQ16)
        #if defined(__SSE2__)
        if (flags.sse2) {
            g_dist_func = dist_sq16_sse2;
            g_simd_width = 8;
            inited = true; return;
        }
        #endif

        // Fallback
        g_dist_func = dist_sq16_scalar;
        g_simd_width = 1;
        inited = true;
    }
    // 包装函数
    float distance(const int16_t* a, const int16_t* b, int dim_padded) {
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
    // [CHANGE] 返回 int16_t 指针
    inline int16_t* get_data_ptr(int id) {
        return (int16_t*)(memory_pool_ + (size_t)id * size_per_element_ + offset_data_);
    }
    const int K=10, efConstruction=256, Mmax0=256;
    const float epsilon=0.20, gamma=0.19;
    int N, dim, dim_padded, enter_point;
    unique_ptr<VisitedListPool> visited_list_pool_;
    // mutex locks with unique ptr
    vector<unique_ptr<mutex>> node_locks;
    float distance_id(int id, int jd) {
        return distance(get_data_ptr(id), get_data_ptr(jd), dim_padded);
    }
    // [CHANGE] Search-time optimized: padded_query 也是 int16_t
    float distance_qry_padded(int id, const int16_t* padded_query) {
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
        
        // [CHANGE] 使用 dim_padded * sizeof(int16_t)
        size_per_element_ = offset_data_ + dim_padded * sizeof(int16_t);
        
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
    // [CHANGE] Pass explicit q_vec for build time (NULL) vs search time
    void search_layer(int q, int ep, int ef, priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>>& C){
        priority_queue<pair<float,int>> W;
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        unsigned short* mass = vl->mass;
        unsigned short curV = vl->curV;        
        mass[ep] = curV;
        if(q != -1) mass[q]=curV;
        
        _mm_prefetch((const char*)get_data_ptr(ep), _MM_HINT_T0);
        
        float xx;
        xx = distance_id(ep, q);

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
        vector<pair<float,int>> candidates;
        while(!C.empty()) {
            candidates.push_back(C.top());
            C.pop();
        }
        
        for(const auto& item : candidates) {
            if(q_sz >= Mmax0) break;
            int e = item.second;
            bool ok = 1;
            for(int i = 0; i < q_sz; ++i)
                if(distance_id(e, q_ptr[i]) < item.first){
                    ok = 0; break;
                }
            if(ok) q_ptr[q_sz++] = e;
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
        
        // [CHANGE] Calculate SQ params
        float min_v = numeric_limits<float>::max();
        float max_v = numeric_limits<float>::lowest();
        for(float x : base) {
            if(x < min_v) min_v = x;
            if(x > max_v) max_v = x;
        }
        g_min_val = min_v;
        float range = max_v - min_v;
        if(range < 1e-6) range = 1.0f;
        g_scale = (SQ_MAX - SQ_MIN) / range;

        set_memory(d, N);
        
        // [CHANGE] Quantize base to pool
        // Using int16_t pointer
        for(int i = 0; i < N; ++i) {
            const float* src = base.data() + (size_t)i * d;
            int16_t* dst = get_data_ptr(i);
            for(int j=0; j<d; ++j) dst[j] = quantize_one(src[j]);
            // Padding
            for(int j=d; j<dim_padded; ++j) dst[j] = 0;
        }
        
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
        // shortcut_reduction(N,0);
        // shortcut_reduction(N,epsilon);
        #ifdef TEST_GRAPH
        KNN_check();
        #endif
        reorder_memory(N);
        // if (progress_thread.joinable())
        //     progress_thread.join();
    }
    // [CHANGE] Modified to take quantized query buffer
    void KNN_search_internal(const int16_t* padded_query, int *res){
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
        // [CHANGE] Alloc int16 buffer
        void* ptr = aligned_alloc_wrapper(dim_padded * sizeof(int16_t), align);
        if(!ptr) return; // fatal

        int16_t* padded_q = (int16_t*)ptr;
        // [CHANGE] Quantize query
        for(int i=0; i<dim; ++i) padded_q[i] = quantize_one(q[i]);
        if (dim_padded > dim) {
            memset(padded_q + dim, 0, (dim_padded - dim) * sizeof(int16_t));
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