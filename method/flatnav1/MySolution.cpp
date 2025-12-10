#include"MySolution.h"
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
    float distance(const vector<float>&a,const vector<float>&b){
        #ifdef COUNT_DIST
        g_dist_calc_count++;
        #endif
        float dis=0,x;
        for(int i=0;i<dim;++i)
            x=a[i]-b[i],dis+=x*x;
        return dis;
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
    atomic<int> build_progress(0);
    string format_duration(long long seconds) {
        if (seconds < 0) {
            return "??:??:??";
        }
        long long h = seconds / 3600;
        long long m = (seconds % 3600) / 60;
        long long s = seconds % 60;
        stringstream ss;
        ss << setfill('0') << setw(2) << h <<":"<< setfill('0') << setw(2) << m << ":"<< setfill('0') << setw(2) << s;
        return ss.str();
    }
    void print_progress(int total) {
        auto start_time = chrono::high_resolution_clock::now();
        while (build_progress < total) {
            int current = build_progress.load();
            if (current == 0) {
                cout << "\rElapsed: 00:00:00 | ETA: ??:??:??" << flush;
                this_thread::sleep_for(chrono::milliseconds(100));
                continue;
            }
            auto now = chrono::high_resolution_clock::now();
            auto elapsed_duration = chrono::duration_cast<chrono::seconds>(now - start_time);
            long long elapsed_seconds = elapsed_duration.count();
            double time_per_item = static_cast<double>(elapsed_seconds) / current;
            int remaining_items = total - current;
            long long eta_seconds = static_cast<long long>(remaining_items * time_per_item);
            float percentage = 100.0f * current / total;
            
            cout << "\rElapsed: " << format_duration(elapsed_seconds)<< " | ETA: " << format_duration(eta_seconds)<< flush;
            this_thread::sleep_for(chrono::milliseconds(200));
        }
        auto end_time = chrono::high_resolution_clock::now();
        auto total_duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);
        cout << "\rTotal time: " << format_duration(total_duration.count()) << "                                " << endl;
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