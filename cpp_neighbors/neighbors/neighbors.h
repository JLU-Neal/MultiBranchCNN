

#include "../../cpp_utils/cloud/cloud.h"
#include "../../cpp_utils/nanoflann/nanoflann.hpp"
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <list>
#include<string>
#include <cstdint>
#include <math.h>
using namespace std;



void ordered_neighbors(vector<PointXYZ>& queries,
                        vector<PointXYZ>& supports,
                        vector<int>& neighbors_indices,
                        float radius);

void batch_ordered_neighbors(vector<PointXYZ>& queries,
                                vector<PointXYZ>& supports,
                                vector<int>& q_batches,
                                vector<int>& s_batches,
                                vector<int>& neighbors_indices,
                                float radius);

void batch_nanoflann_neighbors(vector<PointXYZ>& queries,
                                vector<PointXYZ>& supports,
                                vector<int>& q_batches,
                                vector<int>& s_batches,
                                vector<int>& neighbors_indices,
                                float radius);

void voxel_index_neighbors(vector<PointXYZ>& queries,
	vector<PointXYZ>& supports,
	vector<int>& q_batches,
	vector<int>& s_batches,
	vector<int>& neighbors_indices,
	float radius);

int find_neighbors(const float*query_pt, int & index_i, int & index_j, int & index_k, 
	float & i_float, float & j_float, float & k_float,
	float r2,
	vector<pair<size_t, float>>& inds_dists,
	vector<vector<vector<list<int>>>>& index_array,
	vector<PointXYZ> &pts);


void hash_index_neighbors(vector<PointXYZ>& queries,
	vector<PointXYZ>& supports,
	vector<int>& q_batches,
	vector<int>& s_batches,
	vector<int>& neighbors_indices,
	float radius);

int find_neighbors_hash(const float*query_pt, int & index_i, int & index_j, int & index_k,
	float & i_float, float & j_float, float & k_float,
	float r2,
	vector<pair<size_t, float>>& inds_dists,
	unordered_multimap<string, int>& index_hash,
	vector<PointXYZ> &pts);
/*
int find_neighbors(const float*query_pt, int & index_i, int & index_j, int & index_k,
	float & i_float, float & j_float, float & k_float,
	float r2,
	vector<pair<size_t, float>>& inds_dists,
	vector<vector<vector<unordered_set<int>>>>& index_array,
	vector<PointXYZ> &pts);
*/



struct index
{
	int i;
	int j;
	int k;
	index(int a, int b, int c)
	{
		i = a;
		j = b;
		k = c;
	}
};