#ifndef CPP_SOLUTION_H
#define CPP_SOLUTION_H
#include<bits/stdc++.h>
using namespace std;
class Solution {
	public:
		void build(int d,const vector<float>&base);
		void search(const vector<float>&query,int*res);
};
#endif
