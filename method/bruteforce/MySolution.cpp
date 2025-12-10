#include"MySolution.h"
#ifdef COUNT_DIST
    extern atomic<unsigned long long> g_dist_calc_count;
#endif

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
namespace{
	const int K=10;
	int dim;
	vector<vector<float>>base_vectors;
}
void Solution::build(int d,const vector<float>&base){
	dim=d;
	int N=base.size()/d;
	base_vectors.resize(N);
	for(int i=0;i<N;++i)
		base_vectors[i].assign(base.begin()+i*d,base.begin()+(i+1)*d);
}
void Solution::search(const vector<float>&query,int*res){
	vector<pair<float,int>>dist_id;
	for(int i=0;i<(int)base_vectors.size();++i){
		#ifdef COUNT_DIST
        g_dist_calc_count++;
        #endif
		const vector<float>&v=base_vectors[i];
		float dist=0;
		for(int j=0;j<dim;++j){
			float diff=query[j]-v[j];
			dist+=diff*diff;
		}
		dist_id.emplace_back(dist,i);
	}
	int k=min(K,(int)dist_id.size());
	nth_element(dist_id.begin(),dist_id.begin()+k,dist_id.end());
	sort(dist_id.begin(),dist_id.begin()+k);
	for(int i=0;i<k;++i)
		res[i]=dist_id[i].second;
}
