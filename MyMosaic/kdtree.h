#pragma once

#include <vector>
#include <algorithm>

#include <boost/multi_array.hpp>
#include <boost/array.hpp>


namespace kdTree
{
	typedef boost::multi_array<float, 2> kdTreeArray;

	class kdTree;
	class kdTreeNode;
	class SearchRecord;

	struct Range
	{
		float lb, ub;
	};

	struct kdTreeResult
	{
	public:
		float distance;
		int idx;
	};

	class kdTreeResultVector :public std::vector<kdTreeResult>
	{
	public:
		void PushElementAndHeapify(kdTreeResult&);
		float MaxValue();
		float NewMaxPriority(kdTreeResult&);
	};

	class SearchRecord
	{
	public:
		SearchRecord(std::vector<float>& input_query_vector, kdTree& input_tre,
			kdTreeResultVector& result);

	private:
		friend class kdTree;
		friend class kdTreeNode;

		std::vector<float>& query_vector;
		int dim;
		bool rearrange;
		unsigned int nn;
		float ball_size;
		int center_idx, correl_time;

		kdTreeResultVector& result;
		const kdTreeArray* data;

		//存放节点lower upper
		const std::vector<int>& index;
	};

	class kdTree
	{
	public:
		kdTree(kdTreeArray& inputData, bool input_rearrange = true);
		~kdTree();

		friend class kdTreeNode;
		friend class SearchRecord;

		void NNearestAroundPoint(std::vector<float>& qv, int nn, kdTreeResultVector& result);

		void NNearestAroundTreeNode(int idx, int correltime, int nn, kdTreeResultVector& result);

	private:
		static const int bucketsize = 1;


		//引用
		const kdTreeArray& data;
		kdTreeArray rearranged_data;
		const kdTreeArray* data_pointer;

		const int num;
		const int dim;
		kdTreeNode* root;
		std::vector<int> leaf_idx;
		bool rearrange;

		bool sort_result;

		void BuildTree();
		kdTreeNode* BuildSubTree(int lower, int upper, kdTreeNode* parent);
		void RangeOfCoordinate(int dimension, int lower, int upper, Range& range);
		int MiddleIndex(int dimension, float average, int lower, int upper);
	};

	class kdTreeNode
	{
	public:
		kdTreeNode(int dim);
		~kdTreeNode();

	private:
		friend class kdTree;

		int cut_dim;
		float cut_val, cut_left_val, cut_right_val;
		int lower, upper;

		std::vector<Range> range_vec;

		kdTreeNode* left;
		kdTreeNode* right;

		void search(SearchRecord& sr);

		bool IsWithinRange(SearchRecord& sr);

		void ProcessTerminalNode(SearchRecord& sr);
		void ProcessTerminalNodeFixedBall(SearchRecord& sr);
	};







}