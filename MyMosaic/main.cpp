#include <string>
#include <Windows.h>
#include <algorithm>
#include <time.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <boost/multi_array.hpp>
#include <boost/array.hpp>

#include "kdtree.h"


typedef boost::multi_array<float, 2> MyPoint;


std::string
sanitize_path(std::string const& path)
{
	if (path.empty())
		return "";

	std::string result = path;

	/* Replace backslashes with slashes. */
	std::replace(result.begin(), result.end(), '\\', '/');

	/* Remove double slashes. */
	for (std::size_t i = 0; i < result.size() - 1; )
	{
		if (result[i] == '/' && result[i + 1] == '/')
			result.erase(i, 1);
		else
			i += 1;
	}

	/* Remove trailing slash if result != "/". */
	if (result.size() > 1 && result[result.size() - 1] == '/')
		result.erase(result.end() - 1);

	return result;
}


int main(int argc, char** argv)
{
	clock_t start = clock();

	std::string input_path = sanitize_path(argv[1]);
	//std::string dir = input_path + "*.*";

	WIN32_FIND_DATA data;
	HANDLE hf = FindFirstFile((input_path + "/*").c_str(), &data);

	MyPoint my_data;

	std::vector<cv::Mat> img_list;
	int num_images = 0;
	do
	{
		if (!std::strcmp(data.cFileName, "."))
			continue;
		if (!std::strcmp(data.cFileName, ".."))
			continue;

		std::string img_path = input_path + "/" + data.cFileName;

		cv::Mat img = cv::imread(img_path);
		cv::Size size(20, 20);
		cv::resize(img, img, size);
		//cv::imshow("", img);
		//cv::imwrite(std::to_string(num_images) + ".jpg", img);
		//cv::waitKey(0);
		img_list.push_back(img);
		my_data.resize(boost::extents[num_images + 1][3]);

		int blue_sum = 0;
		int green_sum = 0;
		int red_sum = 0;

		for (int i = 0; i < img.rows; i++)
		{
			uchar* data = img.ptr<uchar>(i);
			for (int j = 0; j < img.cols; j++)
			{
				blue_sum += data[j * 3];
				green_sum += data[j * 3 + 1];
				red_sum += data[j * 3 + 2];
			}
		}
		my_data[num_images][0] = static_cast<float>(blue_sum) / (img.rows * img.cols);
		my_data[num_images][1] = static_cast<float>(green_sum) / (img.rows * img.cols);
		my_data[num_images][2] = static_cast<float>(red_sum) / (img.rows * img.cols);


		num_images++;
		std::cout << num_images << std::endl;

	} while (FindNextFile(hf, &data) != 0);
	/*cv::imshow("", img_list[66]);
	cv::waitKey();*/

	kdTree::kdTree* my_tree = new kdTree::kdTree(my_data);
	//kdTree::kdTreeResultVector result;
	//my_tree->NNearestAroundTreeNode(55, 0, 5, result);

	cv::Mat target = cv::imread(argv[2]);
	std::vector<int> index_list;
	//cv::imshow("", target);
	for (int i = 0; i < target.rows / 20; i++)
	{
		for (int j = 0; j < target.cols / 20; j++)
		{
			cv::Mat roi = target(cv::Rect(j * 20, i * 20, 20, 20));
			int blue_sum = 0;
			int green_sum = 0;
			int red_sum = 0;
			for (int m = 0; m < 20; m++)
			{
				uchar* data = roi.ptr<uchar>(m);
				for (int n = 0; n < 20; n++)
				{
					blue_sum += data[n * 3];
					green_sum += data[n * 3 + 1];
					red_sum += data[n * 3 + 2];
				}
			}
			float blue_aver = blue_sum / (20.0f * 20.0f);
			float green_aver = green_sum / (20.0f * 20.0f);
			float red_aver = red_sum / (20.0f * 20.0f);
			std::vector<float> qv = { blue_aver,green_aver,red_aver };
			kdTree::kdTreeResultVector result;
			my_tree->NNearestAroundPoint(qv, 1, result);
			index_list.push_back(result[0].idx);
		}
	}
	cv::Mat result(target.rows, target.cols, CV_8UC3);
	for (int i = 0; i < index_list.size(); i++)
	{
		//72  54
		//    x = i % 72 
		//    y = i / 72
		//std::cout << i << std::endl;
		int x = i % (target.cols / 20);
		int y = i / (target.cols / 20);
		cv::Rect rect(x * 20, y * 20, 20, 20);
		img_list[index_list[i]].copyTo(result(rect));
	}

	cv::imshow("", result);
	cv::imwrite("result.jpg", result);
	std::cout << (double)(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
	cv::waitKey();

}
