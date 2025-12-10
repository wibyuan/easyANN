#include"MySolution.h"
// SIMD
#include <immintrin.h>
// CPUID
#if defined(_MSC_VER)
#include <intrin.h> //MSVC
#else
#include <cpuid.h>  //GCC/Clang
#endif
#ifdef COUNT_DIST
    extern atomic<unsigned long long> g_dist_calc_count;
#endif
#ifdef TEST_GRAPH
extern atomic<unsigned long long> g_acc, g_tot;
#endif
namespace{
    const int K=10,M=16,efConstruction=200,Mmax0=M*2;
    const float gamma=0.3;
    int dim;
    vector<vector<int>>hnsw;
    int enter_point=-1;
    vector<vector<float>>base_vectors;
    vector<bool>vis;
    vector<int>visited;
    void cpuid(int info[4], int infoType){
        #if defined(_MSC_VER)
            __cpuidex(info, infoType, 0);
        #else
            __cpuid_count(infoType, 0, info[0], info[1], info[2], info[3]);
        #endif
    }
    float distance(const float* a, const float* b, const int dimension=dim) {
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
                __m512 m_v1 = _mm512_loadu_ps(a + i); __m512 m_v2 = _mm512_loadu_ps(b + i);
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
                __m256 m_v1 = _mm256_loadu_ps(a + i); __m256 m_v2 = _mm256_loadu_ps(b + i);
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
                __m128 m_v1 = _mm_loadu_ps(a + i); __m128 m_v2 = _mm_loadu_ps(b + i);
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
                // sum is [d, c, b, a]
                // shuffle to [c, d, a, b], then add -> [d+c, c+d, b+a, a+b]
                sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1)));
                // shuffle to [b+a, a+b, d+c, c+d], then add -> [d+c+b+a, ...]
                sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2)));
            }
            float res = _mm_cvtss_f32(sum);
            if (dimension > d4) res += scalar_l2(a + d4, b + d4, dimension - d4);
            return res;
        }
        #endif
        return scalar_l2(a, b, dimension);
    }
    float distance(const vector<float>&a,const vector<float>&b){
        return distance(a.data(),b.data());
        // float dis=0,x;
        // for(int i=0;i<dim;++i)
        //     x=a[i]-b[i],dis+=x*x;
        // return dis;
    }
    void search_layer(const vector<float>&q,int ep,int ef,priority_queue<pair<float,int>>&W){
        priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>>C;
        for(int id:visited)vis[id]=0;
        visited.clear();
        vis[ep]=1,visited.push_back(ep);
        float xx=distance(base_vectors[ep],q);
        C.push(make_pair(xx,ep)),
        W.push(make_pair(xx,ep));
        while(!C.empty()){
            auto c=C.top();C.pop();
            auto f=W.top();
            if(c.first>f.first)break;
            for(int e:hnsw[c.second])
                if(!vis[e]){
                    vis[e]=1,visited.push_back(e);
                    f=W.top();
                    xx=distance(base_vectors[e],q);
                    if(W.size()<ef||xx<f.first){
                        C.push(make_pair(xx,e));
                        W.push(make_pair(xx,e));
                        if(W.size()>ef)W.pop();
                    }
                }
        }
    }
    void search_layer_for_KNN(const vector<float>&q,int ep,int K,int lc,priority_queue<pair<float,int>>&Bk){
        priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>>C;
        for(int id:visited)vis[id]=0;
        visited.clear();
        vis[ep]=1,visited.push_back(ep);
        float xx=distance(base_vectors[ep],q);
        C.push(make_pair(xx,ep)),
        Bk.push(make_pair(xx,ep));
        while(!C.empty()){
            auto c=C.top();C.pop();
            if(Bk.size()>=K&&c.first>(1.0f+gamma)*Bk.top().first)break;
            for(int e:hnsw[c.second])
                if(!vis[e]){
                    vis[e]=1,visited.push_back(e);
                    xx=distance(base_vectors[e],q);
                    if(Bk.size()<K||xx<Bk.top().first){
                        Bk.push(make_pair(xx,e));
                        if(Bk.size()>K)Bk.pop();
                    }
                    if(Bk.size()<K||xx<(1.0f+gamma)*Bk.top().first)
                        C.push(make_pair(xx,e));
                }
        }
    }
    void select_neighbors_heuristic(int q,int M,vector<int>&w){
        priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>>C;
        for(int e:w)C.push(make_pair(distance(base_vectors[e],base_vectors[q]),e));
        w.clear();
        while(!C.empty()&&w.size()<M){
            auto e=C.top();C.pop();
            bool ok=1;
            for(int x:w)
                if(distance(base_vectors[e.second],base_vectors[x])<distance(base_vectors[e.second],base_vectors[q])){
                    ok=0;break;
                }
            if(ok)w.push_back(e.second);
        }
    }
    void insert(int q){
        int ep=enter_point;
        if(ep==-1){
            enter_point=q;return;
        }
        int L=0;
        priority_queue<pair<float,int>>W;
        search_layer(base_vectors[q],ep,efConstruction,W);
        auto &qConn=hnsw[q];
        qConn.reserve(W.size());
        while(!W.empty())qConn.push_back(W.top().second),W.pop();
        select_neighbors_heuristic(q,M,qConn);
        for(int e:hnsw[q]){
            hnsw[e].push_back(q);
        }
        int mmx=Mmax0;
        for(int e:hnsw[q]){
            auto &eConn=hnsw[e];
            if(eConn.size()>mmx)
                select_neighbors_heuristic(e,mmx,eConn);
        }
        if(!hnsw[q].empty())
            ep=hnsw[q][0];
    }
    void KNN_search(const vector<float>&q,int K,vector<int>&ans){
        priority_queue<pair<float,int>>W;
        int ep=enter_point;
        search_layer_for_KNN(q,ep,K,0,W);
        ans.clear(),ans.reserve(K);
        while(!W.empty())ans.push_back(W.top().second),W.pop();
        reverse(ans.begin(),ans.end());
    }
    atomic<int> build_progress(0);
    string format_duration(long long seconds) {
        if (seconds < 0) {
            return "??:??";
        }
        long long m = seconds / 60;
        long long s = seconds % 60;
        stringstream ss;
        ss<<setfill('0')<<setw(2)<<m<<":"<<setfill('0') << setw(2) << s;
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
            float percentage = 100.0f * current / total;
            
            cout << "\r"<<(100*current/total)<<"% "<<format_duration(elapsed_seconds)<<"/"<<format_duration(all_seconds)<<flush;
            this_thread::sleep_for(chrono::milliseconds(1000));
        }
        auto end_time = chrono::high_resolution_clock::now();
        auto total_duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);
        cout << "\r100.00% "<<format_duration(total_duration.count())<<'/'<<format_duration(total_duration.count())<<endl;
    }
    vector<int>ans;
}
void Solution::build(int d,const vector<float>&base){
    dim=d,enter_point=-1;
	int N=base.size()/d;
	base_vectors.resize(N);
	for(int i=0;i<N;++i)
		base_vectors[i].assign(base.begin()+i*d,base.begin()+(i+1)*d);
    hnsw.assign(N, vector<int>());
    vis.resize(N),visited.clear(),visited.reserve(N);
    for(int i=0;i<N;++i)vis[i]=0;
    // build_progress=0;
    // thread progress_thread(print_progress,N);
    for(int i=0;i<N;++i){
        insert(i);
        // ++build_progress;
    }
    // if (progress_thread.joinable())
    //     progress_thread.join();
}
void Solution::search(const vector<float>&query,int*res){
    KNN_search(query,K,ans);
    for(int i=0;i<K;++i)
        res[i]=ans[i];
}