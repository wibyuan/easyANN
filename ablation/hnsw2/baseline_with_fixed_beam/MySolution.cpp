#include"MySolution.h"
#ifdef COUNT_DIST
    extern atomic<unsigned long long> g_dist_calc_count;
#endif
#ifdef TEST_GRAPH
extern atomic<unsigned long long> g_acc, g_tot;
#endif
namespace{
    const int K=10,M=16,efConstruction=200,Mmax=M,Mmax0=2*M;
    const float mL=1.0/log(M);
    int efSearch_=100;  // 固定 beam width，不用 gamma
    mt19937 rng;
    int dim;
    uniform_real_distribution<float>dist{numeric_limits<float>::min(),1};
    int random_level(){
        return static_cast<int>(-log(dist(rng))*mL);
    }
    struct Node{
        int level;
        vector<vector<int>>neighborhood;
    };
    vector<Node>hnsw;
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
    void search_layer(const vector<float>&q,int ep,int ef,int lc,priority_queue<pair<float,int>>&W){
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
            for(int e:hnsw[c.second].neighborhood[lc])
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
    // Fixed beam 搜索：使用固定 efSearch，不用 gamma 自适应
    void search_layer_for_KNN(const vector<float>&q,int ep,int K,int lc,priority_queue<pair<float,int>>&W){
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
            if(c.first>f.first)break;  // 标准终止条件，不用 gamma
            for(int e:hnsw[c.second].neighborhood[lc])
                if(!vis[e]){
                    vis[e]=1,visited.push_back(e);
                    f=W.top();
                    xx=distance(base_vectors[e],q);
                    if(W.size()<efSearch_||xx<f.first){
                        C.push(make_pair(xx,e));
                        W.push(make_pair(xx,e));
                        if(W.size()>efSearch_)W.pop();
                    }
                }
        }
    }

    void select_neighbors_heuristic(int q,int lc,int M,vector<int>&w){
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
        int ep=enter_point,l=random_level();
        hnsw[q].level=l;
        hnsw[q].neighborhood.resize(l+1);
        int L=ep==-1?ep:hnsw[ep].level;
        for(int lc=L;lc>l;--lc){
            priority_queue<pair<float,int>>W;
            search_layer(base_vectors[q],ep,1,lc,W);
            ep=W.top().second;
        }
        for(int lc=min(L,l);lc>=0;--lc){
            priority_queue<pair<float,int>>W;
            search_layer(base_vectors[q],ep,efConstruction,lc,W);
            auto &qConn=hnsw[q].neighborhood[lc];
            qConn.reserve(W.size());
            while(!W.empty())qConn.push_back(W.top().second),W.pop();
            select_neighbors_heuristic(q,lc,M,qConn);
            for(int e:hnsw[q].neighborhood[lc]){
               hnsw[e].neighborhood[lc].push_back(q);
            }
            int mmx=lc==0?Mmax0:Mmax;
            for(int e:hnsw[q].neighborhood[lc]){
                auto &eConn=hnsw[e].neighborhood[lc];
                if(eConn.size()>mmx)
                    select_neighbors_heuristic(e,lc,mmx,eConn);
            }
            if(!hnsw[q].neighborhood[lc].empty())
                ep=hnsw[q].neighborhood[lc][0];
        }
        if(l>L)enter_point=q;
    }
    void KNN_search(const vector<float>&q,int K,vector<int>&ans){
        priority_queue<pair<float,int>>W;
        int ep=enter_point,L=hnsw[ep].level;
        for(int lc=L;lc>=1;--lc){
            search_layer(q,ep,1,lc,W);
            if(W.size()==1)ep=W.top().second,W.pop();
        }
        search_layer_for_KNN(q,ep,K,0,W);
        ans.clear(),ans.reserve(K);
        while(!W.empty())ans.push_back(W.top().second),W.pop();
        reverse(ans.begin(),ans.end());
    }
    vector<int>ans;

    #ifdef TEST_GRAPH
    // 贪心导航：从 entry 向 target 走，返回是否到达
    void quick_search(int entry, int target){
        int N = hnsw.size();
        if(N <= 0 || enter_point < 0) return;

        // 从入口点逐层向下
        int ep = entry;
        int L = hnsw[ep].level;
        for(int lc = L; lc >= 1; --lc){
            bool improved = true;
            while(improved){
                improved = false;
                if(hnsw[ep].neighborhood.size() > lc){
                    for(int e : hnsw[ep].neighborhood[lc]){
                        if(distance(base_vectors[e], base_vectors[target]) <
                           distance(base_vectors[ep], base_vectors[target])){
                            ep = e;
                            improved = true;
                            g_tot++;
                        }
                    }
                }
            }
        }

        // 在 layer 0 贪心搜索
        bool improved = true;
        while(improved){
            improved = false;
            if(!hnsw[ep].neighborhood.empty()){
                for(int e : hnsw[ep].neighborhood[0]){
                    if(distance(base_vectors[e], base_vectors[target]) <
                       distance(base_vectors[ep], base_vectors[target])){
                        ep = e;
                        improved = true;
                        g_tot++;
                    }
                }
            }
        }

        if(ep == target) g_acc++;
    }

    void KNN_check(){
        int N = hnsw.size();
        if(N <= 0) return;
        mt19937 rng_{114514};
        uniform_int_distribution<int> dist_{0, N-1};
        for(int i = 0; i < 10000; ++i){
            int u = dist_(rng_), v = dist_(rng_);
            quick_search(u, v);
        }
    }
    #endif
}

void Solution::build(int d,const vector<float>&base){
    dim=d,enter_point=-1;
    rng=mt19937(114514);
	int N=base.size()/d;
	base_vectors.resize(N);
	for(int i=0;i<N;++i)
		base_vectors[i].assign(base.begin()+i*d,base.begin()+(i+1)*d);
    hnsw.assign(N, Node());
    vis.resize(N),visited.clear(),visited.reserve(N);
    for(int i=0;i<N;++i)vis[i]=0;
    for(int i=0;i<N;++i){
        insert(i);
    }
    #ifdef TEST_GRAPH
    KNN_check();
    #endif
}

void Solution::search(const vector<float>&query,int*res){
    KNN_search(query,K,ans);
    for(int i=0;i<K;++i)
        res[i]=ans[i];
}

// === 新增接口实现 ===

void Solution::set_gamma(float gamma_val){
    // 本变体使用固定 efSearch，忽略 gamma
    (void)gamma_val;
}

void Solution::set_ef_search(int ef_val){
    efSearch_ = ef_val;
}

vector<int> Solution::get_degree_distribution(){
    vector<int> degrees;
    degrees.reserve(hnsw.size());
    for(const auto& node : hnsw){
        // 返回 layer 0 的度数
        if(!node.neighborhood.empty()){
            degrees.push_back(node.neighborhood[0].size());
        } else {
            degrees.push_back(0);
        }
    }
    return degrees;
}

void Solution::save_graph(const string& path){
    ofstream fout(path, ios::binary);
    if(!fout.is_open()){
        cerr << "Error: Cannot open file for writing: " << path << endl;
        return;
    }

    // 写入元数据
    int N = hnsw.size();
    fout.write(reinterpret_cast<const char*>(&N), sizeof(int));
    fout.write(reinterpret_cast<const char*>(&dim), sizeof(int));
    fout.write(reinterpret_cast<const char*>(&enter_point), sizeof(int));

    // 写入每个节点
    for(int i = 0; i < N; ++i){
        const Node& node = hnsw[i];
        fout.write(reinterpret_cast<const char*>(&node.level), sizeof(int));

        // 写入每层的邻居
        int num_layers = node.neighborhood.size();
        fout.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));
        for(const auto& neighbors : node.neighborhood){
            int num_neighbors = neighbors.size();
            fout.write(reinterpret_cast<const char*>(&num_neighbors), sizeof(int));
            if(num_neighbors > 0){
                fout.write(reinterpret_cast<const char*>(neighbors.data()), num_neighbors * sizeof(int));
            }
        }

        // 写入向量数据
        fout.write(reinterpret_cast<const char*>(base_vectors[i].data()), dim * sizeof(float));
    }

    fout.close();
}

bool Solution::load_graph(const string& path){
    ifstream fin(path, ios::binary);
    if(!fin.is_open()){
        return false;
    }

    // 读取元数据
    int N;
    fin.read(reinterpret_cast<char*>(&N), sizeof(int));
    fin.read(reinterpret_cast<char*>(&dim), sizeof(int));
    fin.read(reinterpret_cast<char*>(&enter_point), sizeof(int));

    // 分配空间
    hnsw.resize(N);
    base_vectors.resize(N);
    vis.resize(N);
    visited.clear();
    visited.reserve(N);
    for(int i = 0; i < N; ++i) vis[i] = 0;

    // 读取每个节点
    for(int i = 0; i < N; ++i){
        Node& node = hnsw[i];
        fin.read(reinterpret_cast<char*>(&node.level), sizeof(int));

        int num_layers;
        fin.read(reinterpret_cast<char*>(&num_layers), sizeof(int));
        node.neighborhood.resize(num_layers);

        for(int l = 0; l < num_layers; ++l){
            int num_neighbors;
            fin.read(reinterpret_cast<char*>(&num_neighbors), sizeof(int));
            node.neighborhood[l].resize(num_neighbors);
            if(num_neighbors > 0){
                fin.read(reinterpret_cast<char*>(node.neighborhood[l].data()), num_neighbors * sizeof(int));
            }
        }

        // 读取向量数据
        base_vectors[i].resize(dim);
        fin.read(reinterpret_cast<char*>(base_vectors[i].data()), dim * sizeof(float));
    }

    fin.close();
    return true;
}