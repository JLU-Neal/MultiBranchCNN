
#include "neighbors.h"
using namespace std;
//int alone = 0;
int offset[3] = { -1, 0, 1 };
void brute_neighbors(vector<PointXYZ>& queries, vector<PointXYZ>& supports, vector<int>& neighbors_indices, float radius, int verbose)
{
	cout<<"============================using func: brute_neighbors====================="<<endl;
	// Initialize variables
	// ******************

	// square radius
	float r2 = radius * radius;

	// indices
	int i0 = 0;

	// Counting vector
	int max_count = 0;
	vector<vector<int>> tmp(queries.size());

	// Search neigbors indices
	// ***********************

	for (auto& p0 : queries)
	{
		int i = 0;
		for (auto& p : supports)
		{
			if ((p0 - p).sq_norm() < r2)
			{
				tmp[i0].push_back(i);
				if (tmp[i0].size() > max_count)
					max_count = tmp[i0].size();
			}
			i++;
		}
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	for (auto& inds : tmp)
	{
		for (int j = 0; j < max_count; j++)
		{
			if (j < inds.size())
				neighbors_indices[i0 * max_count + j] = inds[j];
			else
				neighbors_indices[i0 * max_count + j] = -1;
		}
		i0++;
	}

	return;
}

void ordered_neighbors(vector<PointXYZ>& queries,
                        vector<PointXYZ>& supports,
                        vector<int>& neighbors_indices,
                        float radius)
{	
	cout<<"============================using func: ordered_neighbors====================="<<endl;

	// Initialize variables
	// ******************

	// square radius
	float r2 = radius * radius;

	// indices
	int i0 = 0;

	// Counting vector
	int max_count = 0;
	float d2;
	vector<vector<int>> tmp(queries.size());
	vector<vector<float>> dists(queries.size());

	// Search neigbors indices
	// ***********************

	for (auto& p0 : queries)
	{
		int i = 0;
		for (auto& p : supports)
		{
		    d2 = (p0 - p).sq_norm();
			if (d2 < r2)
			{
			    // Find order of the new point
			    auto it = std::upper_bound(dists[i0].begin(), dists[i0].end(), d2);
			    int index = std::distance(dists[i0].begin(), it);

			    // Insert element
                dists[i0].insert(it, d2);
                tmp[i0].insert(tmp[i0].begin() + index, i);

			    // Update max count
				if (tmp[i0].size() > max_count)
					max_count = tmp[i0].size();
			}
			i++;
		}
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	for (auto& inds : tmp)
	{
		for (int j = 0; j < max_count; j++)
		{
			if (j < inds.size())
				neighbors_indices[i0 * max_count + j] = inds[j];
			else
				neighbors_indices[i0 * max_count + j] = -1;
		}
		i0++;
	}

	return;
}

void batch_ordered_neighbors(vector<PointXYZ>& queries,
                                vector<PointXYZ>& supports,
                                vector<int>& q_batches,
                                vector<int>& s_batches,
                                vector<int>& neighbors_indices,
                                float radius)
{
	cout<<"============================using func: batch_ordered_neighbors====================="<<endl;
	// Initialize variables
	// ******************

	// square radius
	float r2 = radius * radius;

	// indices
	int i0 = 0;

	// Counting vector
	int max_count = 0;
	float d2;
	vector<vector<int>> tmp(queries.size());
	vector<vector<float>> dists(queries.size());

	// batch index
	int b = 0;
	int sum_qb = 0;
	int sum_sb = 0;


	// Search neigbors indices
	// ***********************

	for (auto& p0 : queries)
	{
	    // Check if we changed batch
	    if (i0 == sum_qb + q_batches[b])
	    {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;
	    }

	    // Loop only over the supports of current batch
	    vector<PointXYZ>::iterator p_it;
		int i = 0;
        for(p_it = supports.begin() + sum_sb; p_it < supports.begin() + sum_sb + s_batches[b]; p_it++ )
        {
		    d2 = (p0 - *p_it).sq_norm();
			if (d2 < r2)
			{
			    // Find order of the new point
			    auto it = std::upper_bound(dists[i0].begin(), dists[i0].end(), d2);
			    int index = std::distance(dists[i0].begin(), it);

			    // Insert element
                dists[i0].insert(it, d2);
                tmp[i0].insert(tmp[i0].begin() + index, sum_sb + i);

			    // Update max count
				if (tmp[i0].size() > max_count)
					max_count = tmp[i0].size();
			}
			i++;
		}
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	for (auto& inds : tmp)
	{
		for (int j = 0; j < max_count; j++)
		{
			if (j < inds.size())
				neighbors_indices[i0 * max_count + j] = inds[j];
			else
				neighbors_indices[i0 * max_count + j] = supports.size();
		}
		i0++;
	}

	return;
}


void batch_nanoflann_neighbors(vector<PointXYZ>& queries,
                                vector<PointXYZ>& supports,
                                vector<int>& q_batches,
                                vector<int>& s_batches,
                                vector<int>& neighbors_indices,
                                float radius)
{
	cout<<"============================using func: batch_nanoflann_neighbors====================="<<endl;
	// Initialize variables
	// ******************

	// indices
	int i0 = 0;

	// Square radius
	float r2 = radius * radius;

	// Counting vector
	int max_count = 0;
	float d2;
	vector<vector<pair<size_t, float>>> all_inds_dists(queries.size());

	// batch index
	int b = 0;
	int sum_qb = 0;
	int sum_sb = 0;

	// Nanoflann related variables
	// ***************************

	// CLoud variable
	PointCloud current_cloud;

	// Tree parameters
	nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10 /* max leaf */);

	// KDTree type definition
    typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<float, PointCloud > ,
                                                        PointCloud,
                                                        3 > my_kd_tree_t;

    // Pointer to trees
    my_kd_tree_t* index;

    // Build KDTree for the first batch element
    current_cloud.pts = vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);
    index = new my_kd_tree_t(3, current_cloud, tree_params);
    index->buildIndex();


	// Search neigbors indices
	// ***********************

    // Search params
    nanoflann::SearchParams search_params;
    search_params.sorted = true;

	for (auto& p0 : queries)
	{

	    // Check if we changed batch
	    if (i0 == sum_qb + q_batches[b])
	    {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;

	        // Change the points
	        current_cloud.pts.clear();
            current_cloud.pts = vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);

	        // Build KDTree of the current element of the batch
            delete index;
            index = new my_kd_tree_t(3, current_cloud, tree_params);
            index->buildIndex();
	    }

	    // Initial guess of neighbors size
        all_inds_dists[i0].reserve(max_count);

	    // Find neighbors
	    float query_pt[3] = { p0.x, p0.y, p0.z};
		//The output is given as a vector of pairs, of which the first element is 
		//a point index and the second the corresponding distance.
		//return The number of points within the given radius
		size_t nMatches = index->radiusSearch(query_pt, r2, all_inds_dists[i0], search_params);

        // Update max count
        if (nMatches > max_count)
            max_count = nMatches;

        // Increment query idx
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	sum_sb = 0;
	sum_qb = 0;
	b = 0;
	for (auto& inds_dists : all_inds_dists)
	{
	    // Check if we changed batch
	    if (i0 == sum_qb + q_batches[b])
	    {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;
	    }

		for (int j = 0; j < max_count; j++)
		{
			if (j < inds_dists.size())
				neighbors_indices[i0 * max_count + j] = inds_dists[j].first + sum_sb;
			else
				neighbors_indices[i0 * max_count + j] = supports.size();
		}
		i0++;
	}

	delete index;

	return;
}






void voxel_index_neighbors(vector<PointXYZ>& queries,
                                vector<PointXYZ>& supports,
                                vector<int>& q_batches,
                                vector<int>& s_batches,
                                vector<int>& neighbors_indices,
                                float radius)
{
	//cout<<"============================using func: voxel_index_neighbors====================="<<endl;
	// Initialize variables
	// ******************

	// indices
	int i0 = 0;

	// Square radius
	float r2 = radius * radius;
	float r = radius;
	// Counting vector
	int max_count = 0;
	float d2;
	vector<vector<pair<size_t, float>>> all_inds_dists(queries.size());

	// batch index
	int b = 0;
	int sum_qb = 0;
	int sum_sb = 0;

	// Nanoflann related variables
	// ***************************

	// CLoud variable
	PointCloud current_cloud;

	// Tree parameters
	//nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10 /* max leaf */);

	// KDTree type definition
    //typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<float, PointCloud > ,
                                                        //PointCloud,
                                                        //3 > my_kd_tree_t;

    // Pointer to trees
    //my_kd_tree_t* index;

    // Build KDTree for the first batch element
    //current_cloud.pts = vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);
    //index = new my_kd_tree_t(3, current_cloud, tree_params);
    //index->buildIndex();

	//Build Voxel Index for the first batch element
	current_cloud.pts = vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);
	//Calculate the longest distance of the points for later normalization
	float length = max_length(current_cloud.pts);
	length += 2 * r;//avoid boundary issue
	//Calculate the resolution
	int resolution = ceil((length * 2) / r);
	float grid_size = r / (2 * length);
	r2 = grid_size * grid_size;
	//Initialize the array for indexing
	vector<vector<vector<list<int>>>> index_array(resolution, vector<vector<list<int>>>(resolution, vector<list<int>>(resolution)));
	//Normalization, ensuring that all the points fall in a cube range from [0,1] in each dimension.
	//Register each point to its corresponding grid
	for (vector<PointXYZ>::iterator it = current_cloud.pts.begin(); it != current_cloud.pts.end(); it++)
	{
		it->x += length;
		it->y += length;
		it->z += length;
		it->x /= (2*length);
		it->y /= (2*length);
		it->z /= (2*length);
		int i = it->x / grid_size;
		int j = it->y / grid_size;
		int k = it->z / grid_size;
		//Register the points
		/*
		if (i >= 0 && i < resolution && j >= 0 && j < resolution && k >= 0 && k < resolution)
		{
			
		}
		*/
		index_array[i][j][k].push_back(distance(current_cloud.pts.begin(), it));

	}

	

	// Search neigbors indices
	// ***********************

    // Search params
    //nanoflann::SearchParams search_params;
    //search_params.sorted = true;

	for (auto& p0 : queries)
	{

	    // Check if we changed batch
	    if (i0 == sum_qb + q_batches[b])
	    {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;

	        // Change the points
	        current_cloud.pts.clear();
            current_cloud.pts = vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);

	        // Build KDTree of the current element of the batch
            //delete index;
            //index = new my_kd_tree_t(3, current_cloud, tree_params);
            //index->buildIndex();
		

			//Build Voxel Index for the first batch element
			current_cloud.pts = vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);
			//Calculate the longest distance of the points for later normalization
			length = max_length(current_cloud.pts);
			length += 2 * r;
			//Calculate the resolution
			resolution = ceil((length * 2) / r);
			grid_size = r / (2 * length);
			//Initialize the array for indexing
			vector<vector<vector<list<int>>>> temp(resolution, vector<vector<list<int>>>(resolution, vector<list<int>>(resolution)));
			//vector<vector<vector<list<int>>>>().swap(index_array);
			index_array.swap(temp);
			//Normalization, ensuring that all the points fall in a cube range from [0,1] in each dimension.
			//Register each point to its corresponding grid
			for (vector<PointXYZ>::iterator it = current_cloud.pts.begin(); it != current_cloud.pts.end(); it++)
			{
				it->x += length;
				it->y += length;
				it->z += length;
				it->x /= (2*length);
				it->y /= (2*length);
				it->z /= (2*length);
				int i = it->x / grid_size;
				int j = it->y / grid_size;
				int k = it->z / grid_size;
				//Register the points
				/*
								if (i >= 0 && i < resolution && j >= 0 && j < resolution && k >= 0 && k < resolution)
				{
					
				}
				*/

				index_array[i][j][k].push_back(distance(current_cloud.pts.begin(), it));
			}
	    }

	    // Initial guess of neighbors size
        all_inds_dists[i0].reserve(max_count);

	    
		//Normalize the current point
		float query_pt[3] = { (p0.x + length) / (2 * length), (p0.y + length) / (2 * length), (p0.z + length) / (2 * length) };
		//Project the point to relevant grid.
		float i = query_pt[0] / grid_size;
		float j = query_pt[1] / grid_size;
		float k = query_pt[2] / grid_size;
		int query_i = int(i);
		int query_j = int(j);
		int query_k = int(k);
		float i_float = i - query_i;
		float j_float = i - query_j;
		float k_float = k - query_k;
		// Find neighbors
		
		size_t nMatches = find_neighbors(query_pt, query_i, query_j, query_k, 
			i_float, j_float, k_float, 
			r2, all_inds_dists[i0], index_array, current_cloud.pts);
		//size_t nMatches = index->radiusSearch(query_pt, r2, all_inds_dists[i0], search_params);

        //Update max count
        if (nMatches > max_count)
			max_count = nMatches;

        // Increment query idx
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	sum_sb = 0;
	sum_qb = 0;
	b = 0;
	for (auto& inds_dists : all_inds_dists)
	{
	    // Check if we changed batch
	    if (i0 == sum_qb + q_batches[b])
	    {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;
	    }

		for (int j = 0; j < max_count; j++)
		{
			if (j < inds_dists.size())
				neighbors_indices[i0 * max_count + j] = inds_dists[j].first + sum_sb;
			else
				neighbors_indices[i0 * max_count + j] = supports.size();
		}
		i0++;
	}

	//delete index;
	vector<vector<vector<list<int>>>>().swap(index_array);
	return;
}

int find_neighbors(const float*query_pt, int & index_i, int & index_j, int & index_k,
	float & i_float, float & j_float, float & k_float,
	float r2,
	vector<pair<size_t, float>>& inds_dists,
	vector<vector<vector<list<int>>>>& index_array,
	vector<PointXYZ> &pts)
{
	int count = 0;

	int resolution = index_array.size();
	//by computing the float part of the index to determine the searching direction
	/*
	int x_search_direction[2] = { 0, 0 };
	int y_search_direction[2] = { 0, 0 };
	int z_search_direction[2] = { 0, 0 };
	if (i_float>0.5)
	{
		x_search_direction[1]= 1;
	}
	else
	{
		x_search_direction[1] = -1;
	}
	if (j_float>0.5)
	{
		y_search_direction[1] = 1;
	}
	else
	{
		y_search_direction[1] = -1;
	}
	if (k_float>0.5)
	{
		z_search_direction[1] = 1;
	}
	else
	{
		z_search_direction[1] = -1;
	}
	*/
	

	for (int offset_i=0; offset_i < 3; offset_i++)
	{
		for (int offset_j = 0; offset_j < 3; offset_j++)
		{
			for (int offset_k = 0; offset_k < 3; offset_k++)
			{
				int cur_index_i = index_i + offset[offset_i];
				int cur_index_j = index_j + offset[offset_j];
				int cur_index_k = index_k + offset[offset_k];
				list<int> &cur_set = index_array[cur_index_i][cur_index_j][cur_index_k];
				//cout <<"resolution: "<<resolution<< "cur_index_i:" << cur_index_i << " cur_index_j" <<cur_index_j<< " cur_index_k:" << cur_index_k << endl;
				for (list<int>::iterator it = cur_set.begin(); it != cur_set.end(); it++)
				{
					PointXYZ potential_pts = pts[*it];
					float dist = pow(potential_pts.x - query_pt[0], 2) +
						pow(potential_pts.y - query_pt[1], 2) +
						pow(potential_pts.z - query_pt[2], 2);
					//cout << "dist:" << dist << "  r2:" << r2 << endl;
					if (dist < r2)
					{
						inds_dists.push_back(pair<size_t, float>(*it, dist));
						count++;
					}
				
				}

				
				/*
				if (cur_index_i >= 0 && cur_index_i < resolution)
				{
					if (cur_index_j >= 0 && cur_index_j < resolution)
					{
						if (cur_index_k >= 0 && cur_index_k < resolution)
						{
							
							
						}
					}
				}
								else
				{
					cout << "resolution: " << resolution << " i:" << cur_index_i << " j: " << cur_index_j << " k: " << cur_index_k
						<<" length: "<<length<<" r: "<<r<<" resolution: "<<index_array.size()<<"  grid_size: "<<grid_size<<endl;
				}
				
				*/

			}
		}
	}
	/*
		if (count==0)
	{
		alone++;
		cout << "i: " << index_i << " j: " << index_j << " k:  " << index_k << " alone: " << alone << endl;

	}
	*/

	return count;
}



void hash_index_neighbors(vector<PointXYZ>& queries,
	vector<PointXYZ>& supports,
	vector<int>& q_batches,
	vector<int>& s_batches,
	vector<int>& neighbors_indices,
	float radius)
{
	cout<<"============================using func: hash_index_neighbors====================="<<endl;
	// Initialize variables
	// ******************

	// indices
	int i0 = 0;

	// Square radius
	float r2 = radius * radius;
	float r = radius;
	// Counting vector
	int max_count = 0;
	float d2;
	vector<vector<pair<size_t, float>>> all_inds_dists(queries.size());

	// batch index
	int b = 0;
	int sum_qb = 0;
	int sum_sb = 0;

	// Nanoflann related variables
	// ***************************

	// CLoud variable
	PointCloud current_cloud;

	// Tree parameters
	//nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10 /* max leaf */);

	// KDTree type definition
	//typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<float, PointCloud > ,
														//PointCloud,
														//3 > my_kd_tree_t;

	// Pointer to trees
	//my_kd_tree_t* index;

	// Build KDTree for the first batch element
	current_cloud.pts = vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);
	//index = new my_kd_tree_t(3, current_cloud, tree_params);
	//index->buildIndex();

	//Build Voxel Index for the first batch element
	current_cloud.pts = vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);
	//Calculate the longest distance of the points for later normalization
	float length = max_length(current_cloud.pts);
	//Calculate the resolution
	float grid_size = r;
	int resolution = ceil((length * 2) / grid_size);
	//Initialize the array for indexing
	//vector<vector<vector<unordered_set<int>>>> index_array(resolution, vector<vector<unordered_set<int>>>(resolution));
	unordered_multimap<string,int>index_hash;
	index_hash.reserve(5000);
	//Normalization, ensuring that all the points fall in a cube range from [0,1] in each dimension.
	//Register each point to its corresponding grid
	for (vector<PointXYZ>::iterator it = current_cloud.pts.begin(); it != current_cloud.pts.end(); it++)
	{
		it->x += length;
		it->y += length;
		it->z += length;
		it->x /= (2 * length);
		it->y /= (2 * length);
		it->z /= (2 * length);
		int i = it->x / grid_size;
		int j = it->y / grid_size;
		int k = it->z / grid_size;
		//Register the points
		if (i >= 0 && i < resolution && j >= 0 && j < resolution && k >= 0 && k < resolution)
		{
			string cur_index = to_string(i) + "," + to_string(j) + "," + to_string(k);

			//index_array[i][j][k].insert(distance(current_cloud.pts.begin(), it));
			index_hash.insert(pair< string, int>(cur_index, distance(current_cloud.pts.begin(), it)));
		}

	}



	// Search neigbors indices
	// ***********************

	// Search params
	//nanoflann::SearchParams search_params;
	//search_params.sorted = true;

	for (auto& p0 : queries)
	{

		// Check if we changed batch
		if (i0 == sum_qb + q_batches[b])
		{
			sum_qb += q_batches[b];
			sum_sb += s_batches[b];
			b++;

			// Change the points
			current_cloud.pts.clear();
			current_cloud.pts = vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);

			// Build KDTree of the current element of the batch
			//delete index;
			//index = new my_kd_tree_t(3, current_cloud, tree_params);
			//index->buildIndex();


			//Build Voxel Index for the first batch element
			current_cloud.pts = vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);
			//Calculate the longest distance of the points for later normalization
			double length = max_length(current_cloud.pts);
			//Calculate the resolution
			float grid_size = r;
			int resolution = ceil((length * 2) / grid_size);
			//Initialize the array for indexing
			//vector<vector<vector<unordered_set<int>>>> index_array(resolution, vector<vector<unordered_set<int>>>(resolution, vector<unordered_set<int>>(resolution)));
			unordered_multimap<string, int>().swap(index_hash);
			unordered_multimap<string, int>index_hash;
			index_hash.reserve(5000);
			//Normalization, ensuring that all the points fall in a cube range from [0,1] in each dimension.
			//Register each point to its corresponding grid
			for (vector<PointXYZ>::iterator it = current_cloud.pts.begin(); it != current_cloud.pts.end(); it++)
			{
				it->x += length;
				it->y += length;
				it->z += length;
				it->x /= (2 * length);
				it->y /= (2 * length);
				it->z /= (2 * length);
				int i = it->x / grid_size;
				int j = it->y / grid_size;
				int k = it->z / grid_size;
				//Register the points
				if (i >= 0 && i < resolution && j >= 0 && j < resolution && k >= 0 && k < resolution)
				{
					string cur_index = to_string(i) + "," + to_string(j) + "," + to_string(k);
					index_hash.insert(pair<string, int>(cur_index, distance(current_cloud.pts.begin(), it)));
				}
			}
		}

		// Initial guess of neighbors size
		all_inds_dists[i0].reserve(max_count);


		//Normalize the current point
		float query_pt[3] = { (p0.x + length) / (2 * length), (p0.y + length) / (2 * length), (p0.z + length) / (2 * length) };
		//Project the point to relevant grid.
		float i = query_pt[0] / grid_size;
		float j = query_pt[1] / grid_size;
		float k = query_pt[2] / grid_size;
		int query_i = int(i);
		int query_j = int(j);
		int query_k = int(k);
		float i_float = i - query_i;
		float j_float = i - query_j;
		float k_float = k - query_k;
		// Find neighbors
		size_t nMatches = find_neighbors_hash(query_pt, query_i, query_j, query_k,
			i_float, j_float, k_float,
			r2, all_inds_dists[i0], index_hash, current_cloud.pts);
		//size_t nMatches = index->radiusSearch(query_pt, r2, all_inds_dists[i0], search_params);

		//Update max count
		if (nMatches > max_count)
			max_count = nMatches;

		// Increment query idx
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	sum_sb = 0;
	sum_qb = 0;
	b = 0;
	for (auto& inds_dists : all_inds_dists)
	{
		// Check if we changed batch
		if (i0 == sum_qb + q_batches[b])
		{
			sum_qb += q_batches[b];
			sum_sb += s_batches[b];
			b++;
		}

		for (int j = 0; j < max_count; j++)
		{
			if (j < inds_dists.size())
				neighbors_indices[i0 * max_count + j] = inds_dists[j].first + sum_sb;
			else
				neighbors_indices[i0 * max_count + j] = supports.size();
		}
		i0++;
	}

	//delete index;
	unordered_multimap<string, int>().swap(index_hash);
	return;
}	 


int find_neighbors_hash(const float*query_pt, int & index_i, int & index_j, int & index_k,
	float & i_float, float & j_float, float & k_float,
	float r2,
	vector<pair<size_t, float>>& inds_dists,
	unordered_multimap<string, int>& index_hash,
	vector<PointXYZ> &pts)
{
	int count = 0;
	int offset[3] = { -1, 0, 1 };
	//int resolution = index_array.size();
	//by computing the float part of the index to determine the searching direction
	/*
	int x_search_direction[2] = { 0, 0 };
	int y_search_direction[2] = { 0, 0 };
	int z_search_direction[2] = { 0, 0 };
	if (i_float>0.5)
	{
		x_search_direction[1]= 1;
	}
	else
	{
		x_search_direction[1] = -1;
	}
	if (j_float>0.5)
	{
		y_search_direction[1] = 1;
	}
	else
	{
		y_search_direction[1] = -1;
	}
	if (k_float>0.5)
	{
		z_search_direction[1] = 1;
	}
	else
	{
		z_search_direction[1] = -1;
	}
	*/


	for (int offset_i = 0; offset_i < 3; offset_i++)
	{
		for (int offset_j = 0; offset_j < 3; offset_j++)
		{
			for (int offset_k = 0; offset_k < 3; offset_k++)
			{
				int cur_index_i = index_i + offset[offset_i];
				int cur_index_j = index_j + offset[offset_j];
				int cur_index_k = index_k + offset[offset_k];
				string cur_index = to_string(index_i) + "," + to_string(index_j) + "," + to_string(index_k);
				auto itr = index_hash.equal_range(cur_index);
				for (unordered_multimap<string, int>::iterator it = itr.first; it != itr.second; it++)
				{
					PointXYZ potential_pts = pts[it->second];
					float dist = pow(potential_pts.x - query_pt[0], 2) +
						pow(potential_pts.y - query_pt[1], 2) +
						pow(potential_pts.z - query_pt[2], 2);
					if (dist < r2)
					{
						inds_dists.push_back(pair<size_t, float>(it->second, dist));
						count++;
					}
				}


			}
		}
	}


	return count;
}