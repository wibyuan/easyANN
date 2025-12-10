#include"MySolution.h"
#ifdef COUNT_DIST
    extern atomic<unsigned long long> g_dist_calc_count;
#endif
#ifdef TEST_GRAPH
extern atomic<unsigned long long> g_acc, g_tot;
#endif
namespace{
    const int K=10,M=16,efConstruction=200,Mmax=M,Mmax0=2*M;
    const float mL=1.0/log(M),gamma=0.3;
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
            for(int e:hnsw[c.second].neighborhood[lc])
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
        // cout<<(i?"\r":"")<<100.0*i/N<<'%';
    }
}
void Solution::search(const vector<float>&query,int*res){
    KNN_search(query,K,ans);
    for(int i=0;i<K;++i)
        res[i]=ans[i];
}