#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void MatInit1()
{
	// 3.1.2 构造法
	Mat r1;
	Mat r2(2, 2, CV_8UC3);
	Mat r3(Size(3, 2), CV_8UC3);
	Mat r4(4, 4, CV_8UC2, Scalar(1, 3));
	Mat r5(Size(3, 5), CV_8UC3, Scalar(4, 5, 6));
	Mat r6(r5);

	int sz[2] = { 3, 3 };
	Mat r7(2, sz, CV_8UC1, cv::Scalar::all(1));

	cout << r1 << endl << r2 << endl << r3 << endl \
		<< r4 << endl << r5 << endl << r6 << endl;

	cout << r7 << endl;

	// 3.1.3 直接赋值法、
	Mat r8 = (Mat_<double>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
	cout << r8 << endl;

	// 3.1.4 数组法
	int a[2][3] = { 1, 2, 3, 4, 5, 6 };
	Mat r9(2, 3, CV_32S, a);

	// 3.1.5 create函数法
	Mat M3;
	M3.create(4, 4, CV_8UC1);
	cout << M3 << endl;
}

void MatInit2()
{
	// 1、定义全0矩阵
	Mat mz1 = Mat::zeros(2, 1, CV_8UC1);
	Mat mz2 = Mat::zeros(cv::Size(2, 3), CV_8UC1);

	int sz1[2] = {3, 2};
	Mat mz3 = Mat::zeros(2, sz1, CV_8UC1);
	//cout << mz1 << endl;
	//cout << mz2 << endl;
	//cout << mz3 << endl;
	
	// 2、定义第一通道全1矩阵
	Mat mz4 = Mat::ones(2, 1, CV_8UC1);
	Mat mz5 = Mat::ones(cv::Size(2, 3), CV_8UC1);

	int sz2[2] = { 3, 2 };
	Mat mz6 = Mat::ones(2, sz2, CV_8UC1);
	//cout << mz4 << endl;
	//cout << mz5 << endl;
	//cout << mz6 << endl;

	// 3、定义对角线为1的矩阵
	Mat mz7 = Mat::eye(3, 3, CV_8UC1);
	Mat mz8 = Mat::eye(Size(3, 3), CV_8UC1);
	cout << mz7 << endl;
	cout << mz8 << endl;
}

void GetRowsColsDims()
{
	// 得到矩阵的行数、列数、维数
	Mat r(Size(3, 3), CV_8UC3);
	cout << "row: " << r.rows << ", col: " << r.cols << endl;
	cout << "dim: " << r.dims << endl;
}

void PointerOfData()
{
	// 矩阵的数据指针
	Mat r1 = (Mat_<double>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
	cout << r1 << endl;

	Mat r2;
	if (r2.data == NULL) cout << "r2.data == NULL" << endl;
	cout << r2 << endl;
}

void CreateANewMatrixHeader()
{
	// 创建新的矩阵头
	Mat r1 = (Mat_<double>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
	Mat r2;
	for (int i = 0; i < r1.rows; ++i)
	{
		r2 = r1.row(i);
		if (r2.data == r1.data)
			cout << "第"  << i << "行指针相同" << endl;

		cout << r2 << endl;
	}

	Mat r3;
	for (int j = 0; j < r1.cols; ++j)
	{
		r3 = r1.col(j);
		if (r3.data == r1.data)
			cout << "第" << j << "列指针相同" << endl;

		cout << r3 << endl;
	}
}

void GetNumOfChannels()
{
	// 得到矩阵通道数
	Mat r1 = (Mat_<double>(2, 3) << 1, 2, 3, 4, 5, 6);
	Mat r2(576, 5 * 768, CV_8UC3);
	Mat r3;
	cout << "r1 channel number: " << r1.channels() << endl;
	cout << "r2 channel number: " << r2.channels() << endl;
	cout << "r3 channel number: " << r3.channels() << endl;
}

void MatCopy() // TODO
{
	// 1、深复制--申请新的空间，完全独立
	Mat A = Mat::ones(4, 5, CV_32F);
	Mat B = A.clone();
	if (B.data != A.data)
		cout << "B-A deep copy" << endl;

	Mat C;
	A.copyTo(C);
	if (C.data != A.data)
		cout << "C-A deep copy" << endl;

	// 2、浅复制--拷贝信息头
	Mat src1 = imread("./image.jpg");
	Mat src2 = src1;
	Mat src3(src2);
	imshow("image1", src2);
	imshow("image2", src3);
	waitKey(0);
}

int main()
{
	//MatInit1();
	//MatInit2();
	//GetRowsColsDims();
	//PointerOfData();
	//CreateANewMatrixHeader();
	//GetNumOfChannels();
	MatCopy();

	return 0;
}