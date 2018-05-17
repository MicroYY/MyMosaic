#include "kdtree.h"

#include <stdlib.h>
#include <iostream>
#include <algorithm>


namespace kdTree
{
	inline float squared(const float x)
	{
		return x * x;
	}

	inline bool operator<(const kdTreeResult& r1, const kdTreeResult& r2)
	{
		return r1.distance < r2.distance;
	}

	kdTree::kdTree(kdTreeArray& input_data, bool input_rearrange) :
		data(input_data),
		num(input_data.shape()[0]),
		dim(input_data.shape()[1]),
		root(NULL),
		leaf_idx(num),
		rearrange(input_rearrange)
	{
		for (int i = 0; i < num; i++)
			leaf_idx[i] = i;

		BuildTree();

		if (rearrange)
		{
			rearranged_data.resize(boost::extents[num][dim]);
			for (int i = 0; i < num; i++)
				for (int j = 0; j < dim; j++)
					rearranged_data[i][j] = data[leaf_idx[i]][j];
			data_pointer = &rearranged_data;
		}
		else
			data_pointer = &data;
	}

	kdTree::~kdTree()
	{
		delete root;
	}

	void kdTree::BuildTree()
	{
		root = BuildSubTree(0, num - 1, NULL);
	}

	kdTreeNode* kdTree::BuildSubTree(int lower, int upper, kdTreeNode* parent)
	{
		if (upper < lower)
			return NULL;

		kdTreeNode* node = new kdTreeNode(dim);

		if ((upper - lower) <= bucketsize)
		{
			for (int i = 0; i < dim; i++)
			{
				RangeOfCoordinate(i, lower, upper, node->range_vec[i]);
			}
			node->cut_dim = 0;
			node->cut_val = 0.0;
			node->lower = lower;
			node->upper = upper;
			node->left = node->right = NULL;
		}
		else
		{
			int cut_dim = 0;
			float max_range = 0.0;
			int middle_index;

			for (int i = 0; i < dim; i++)
			{
				if ((parent == NULL) || (parent->cut_dim == i))
				{
					RangeOfCoordinate(i, lower, upper, node->range_vec[i]);
				}
				else
				{
					node->range_vec[i] = parent->range_vec[i];
				}
				float range_of_dim = node->range_vec[i].ub - node->range_vec[i].lb;
				if (range_of_dim > max_range)
				{
					max_range = range_of_dim;
					cut_dim = i;
				}
			}
			float sum = 0.0;
			float average;
			{
				for (int i = lower; i <= upper; i++)
				{
					sum += data[leaf_idx[i]][cut_dim];
				}
				average = sum / static_cast<float>(upper - lower + 1);
				middle_index = MiddleIndex(cut_dim, average, lower, upper);
			}
			node->cut_dim = cut_dim;
			node->lower = lower;
			node->upper = upper;

			node->left = BuildSubTree(lower, middle_index, node);
			node->right = BuildSubTree(middle_index + 1, upper, node);
			///子树建立完成
			
			if (node->right == NULL)
			{
				for (int i = 0; i < dim; i++)
				{
					node->range_vec[i] = node->left->range_vec[i];
				}
				node->cut_val = node->left->range_vec[cut_dim].ub;
				node->cut_left_val = node->cut_right_val = node->cut_val;
			}
			else if (node->left == NULL)
			{
				for (int i = 0; i < dim; i++)
				{
					node->range_vec[i] = node->right->range_vec[i];
				}
				node->cut_val = node->right->range_vec[cut_dim].ub;
				node->cut_left_val = node->cut_right_val = node->cut_val;
			}
			else
			{
				node->cut_left_val = node->left->range_vec[cut_dim].ub;
				node->cut_right_val = node->right->range_vec[cut_dim].lb;
				node->cut_val = (node->cut_left_val + node->cut_right_val) / 2.0F;

				for (int i = 0; i < dim; i++)
				{
					node->range_vec[i].ub = std::max(node->left->range_vec[i].ub,
						node->right->range_vec[i].ub);
					node->range_vec[i].lb = std::min(node->left->range_vec[i].lb,
						node->right->range_vec[i].lb);
				}
			}
		}
		return node;
	}

	///计算每一维度的范围
	void kdTree::RangeOfCoordinate(int dimension, int lower, int upper, Range& range)
	{
		float fmin, fmax;
		float tempmin, tempmax;
		int i;

		fmin = data[leaf_idx[lower]][dimension];
		fmax = fmin;

		for (i = lower + 2; i <= upper; i += 2)
		{
			tempmin = data[leaf_idx[i - 1]][dimension];
			tempmax = data[leaf_idx[i]][dimension];
			if (tempmax > tempmax)
				std::swap(tempmin, tempmax);
			if (fmin > tempmin)
				fmin = tempmin;
			if (fmax < tempmax)
				fmax = tempmax;
		}
		if (i == upper + 1)
		{
			float last = data[leaf_idx[upper]][dimension];
			if (fmin > last)
				fmin = last;
			if (fmax < last)
				fmax = last;
		}
		range.lb = fmin;
		range.ub = fmax;
	}

	///计算中间值索引
	int kdTree::MiddleIndex(int dimension, float average, int lower, int upper)
	{
		int lb = lower, ub = upper;
		while (lb < ub)
		{
			if (data[leaf_idx[lb]][dimension] <= average)
				lb++;
			else
			{
				std::swap(leaf_idx[lb], leaf_idx[ub]);
				ub--;
			}
		}
		if (data[leaf_idx[lb]][dimension] <= average)
			return lb;
		else
			return lb - 1;
	}


	void kdTree::NNearestAroundPoint(std::vector<float>& qv, int nn, kdTreeResultVector& result)
	{
		SearchRecord sr(qv, *this, result);
		std::vector<float> vdiff(dim, 0.0);

		result.clear();

		sr.center_idx = -1;
		sr.correl_time = 0;
		sr.nn = nn;

		root->search(sr);

		if (sort_result)
			sort(result.begin(), result.end());
	}


	void kdTree::NNearestAroundTreeNode(int idx, int correl_time, int nn,
		kdTreeResultVector & result)
	{
		std::vector<float> query_vec(dim);
		result.clear();
		for (int i = 0; i < dim; i++)
			query_vec[i] = data[idx][i];

		SearchRecord sr(query_vec, *this, result);
		sr.center_idx = idx;
		sr.correl_time = correl_time;
		sr.nn = nn;
		root->search(sr);

		if (sort_result)
			sort(result.begin(), result.end());
	}


	kdTreeNode::kdTreeNode(int dim) :range_vec(dim)
	{
		left = right = NULL;
	}

	kdTreeNode::~kdTreeNode()
	{
		if (left != NULL)
			delete left;
		if (right != NULL)
			delete right;
	}

	void kdTreeNode::search(SearchRecord & sr)
	{
		if ((left == NULL) && (right == NULL))
		{
			if (sr.nn == 0)
				ProcessTerminalNodeFixedBall(sr);
			else
				ProcessTerminalNode(sr);
		}
		else
		{
			kdTreeNode *n_close, *n_far;

			float extra;
			float query_val = sr.query_vector[cut_dim];
			if (query_val < cut_val)
			{
				n_close = left;
				n_far = right;
				extra = cut_right_val - query_val;
			}
			else
			{
				n_close = right;
				n_far = left;
				extra = query_val - cut_left_val;
			}
			if (n_close != NULL)
				n_close->search(sr);
			if ((n_far != NULL) && (squared(extra) < sr.ball_size))
			{
				if (n_far->IsWithinRange(sr))
					n_far->search(sr);

			}
		}
	}

	///某维度上与bound的距离
	inline float DistanceFromBound(float x, float min, float max)
	{
		if (x > max)
			return x - max;
		else if (x < min)
			return min - x;
		else
			return 0.0;
	}

	///判断是否在界内
	bool kdTreeNode::IsWithinRange(SearchRecord & sr)
	{
		int dim = sr.dim;
		float distance = 0.0;
		float ball_size = sr.ball_size;
		for (int i = 0; i < dim; i++)
		{
			distance += squared(DistanceFromBound(sr.query_vector[i], range_vec[i].lb, range_vec[i].ub));
			if (distance > ball_size)
				return false;
		}
		return true;
	}

	///？？？
	void kdTreeNode::ProcessTerminalNode(SearchRecord & sr)
	{
		int center_idx = sr.center_idx;
		int correl_time = sr.correl_time;
		unsigned int nn = sr.nn;
		int dim = sr.dim;
		float ball_size = sr.ball_size;

		bool rearrange = sr.ball_size;
		const kdTreeArray& data = *sr.data;

		for (int i = lower; i <= upper; i++)
		{
			int idx;
			float distance;
			bool early_exit;

			if (rearrange)
			{
				early_exit = false;
				distance = 0.0;
				for (int k = 0; k < dim; k++)
				{
					distance += squared(data[i][k] - sr.query_vector[k]);
					if (distance > ball_size)
					{
						early_exit = true;
						break;
					}
				}
				if (early_exit)
					continue;
				idx = sr.index[i];
			}
			else
			{
				idx = sr.index[i];
				early_exit = false;
				distance = 0.0;
				for (int k = 0; k < dim; k++)
				{
					distance += squared(data[idx][k] - sr.query_vector[k]);
					if (distance > ball_size)
					{
						early_exit = true;
						break;
					}
				}
				if (early_exit)
					continue;
			}
			if (center_idx > 0)
				if (abs(idx - center_idx) < correl_time)
					continue;
			if (sr.result.size() < nn)
			{
				kdTreeResult r;
				r.idx = idx;
				r.distance = distance;
				sr.result.PushElementAndHeapify(r);

				if (sr.result.size() == nn)
					ball_size = sr.result.MaxValue();

			}
			else
			{
				kdTreeResult r;
				r.idx = idx;
				r.distance = distance;
				ball_size = sr.result.NewMaxPriority(r);
			}
		}
		sr.ball_size = ball_size;
	}

	///？？？
	void kdTreeNode::ProcessTerminalNodeFixedBall(SearchRecord & sr)
	{
		int center_idx = sr.center_idx;
		int correl_time = sr.correl_time;
		int dim = sr.dim;
		float ball_size = sr.ball_size;

		bool rearrange = sr.rearrange;
		const kdTreeArray& data = *sr.data;

		for (int i = lower; i <= upper; i++)
		{
			int idx = sr.index[i];
			float distance;
			bool early_exit;

			if (rearrange)
			{
				early_exit = false;
				distance = 0.0;
				for (int k = 0; k < dim; k++)
				{
					distance += squared(data[i][k] - sr.query_vector[k]);
					if (distance > ball_size)
					{
						early_exit = true;
						break;
					}
				}
				if (early_exit)
					continue;
				idx = sr.index[i];
			}
			else
			{
				idx = sr.index[i];
				early_exit = false;
				distance = 0.0;
				for (int k = 0; k < dim; k++)
				{
					distance += squared(data[idx][k] - sr.query_vector[k]);
					if (distance > ball_size)
					{
						early_exit = true;
						break;
					}
				}
				if (early_exit)
					continue;
			}
			if (center_idx > 0)
			{
				if (abs(idx - center_idx) < correl_time)
					continue;
			}

			kdTreeResult r;
			r.idx = idx;
			r.distance = distance;
			sr.result.push_back(r);
		}
	}

	static const float infinity = 1.0e38F;

	SearchRecord::SearchRecord(std::vector<float>& input_query_vector,
		kdTree & input_tree, kdTreeResultVector & input_result) :
		query_vector(input_query_vector),
		dim(input_tree.dim),
		nn(0),
		ball_size(infinity),
		result(input_result),
		data(input_tree.data_pointer),
		index(input_tree.leaf_idx)
	{
	}

	void kdTreeResultVector::PushElementAndHeapify(kdTreeResult &r)
	{
		push_back(r);
		push_heap(begin(), end());
	}

	float kdTreeResultVector::MaxValue()
	{
		return (*begin()).distance;
	}

	float kdTreeResultVector::NewMaxPriority(kdTreeResult &r)
	{
		/*pop_heap(begin(), end());
		pop_back();
		push_back(r);
		push_heap(begin(), end());*/
		return (*this)[0].distance;
	}
}
