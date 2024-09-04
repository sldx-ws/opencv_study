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

void mask_test()
{
	Mat image, mask;
	Rect r1(10, 10, 60, 100);
	Mat img2;
	image = imread("./image.jpg");
	mask = Mat::zeros(image.size(), CV_8UC1);
	mask(r1).setTo(255);
	image.copyTo(img2, mask);
	imshow("原图像", image);
	imshow("复制后的目标图像", img2);
	imshow("mask", mask);
	waitKey();
}

void isEmpty()
{
	Mat m;
	if (m.empty() == true)
		cout << "NULL" << endl;
}

void ergodic_for()  //  TODO
{
	// 1、指针数组方式
	//Mat mymat = Mat::ones(cv::Size(3, 2), CV_8UC3);
	//Mat mymat = (Mat_<double>(3, 2) << 1, 2, 3, 4, 5, 6);
	
	int a[2][3] = { 1, 2, 3, 4, 5, 6 };
	Mat mymat(2, 3, CV_8U, a);

	uchar* pdata = (uchar*)mymat.data;

	for (int i = 0; i < mymat.rows; ++i)
	{
		for (int j = 0; j < mymat.cols * mymat.channels(); ++j)
		{
			cout << (int)pdata[j] << " ";
		}

		cout << endl;
	}
}

void ergodic_ptr()  //  TODO
{
	// 2、.ptr方式
	// mat.ptr<type>(row)[col]
	
	// 单通道
	Mat image1 = Mat(400, 600, CV_8UC1);
	uchar* data00 = image1.ptr<uchar>(0); // 指向第0行(第0个元素)
	uchar* data10 = image1.ptr<uchar>(1); // 指向第1行(第0个元素)
	
	// 多通道
	Mat image2 = Mat(400, 600, CV_8UC3);
	Vec3b* data000 = image2.ptr<Vec3b>(0);
	Vec3b* data100 = image2.ptr<Vec3b>(1);


	Mat mymat = Mat::ones(Size(3, 2), CV_8UC3);
	for (int i = 0; i < mymat.rows; ++i)
	{
		uchar* pdata = mymat.ptr<uchar>(i);
		for (int j = 0; j < mymat.cols * mymat.channels(); ++j)
		{
			cout << (int)pdata[j] << " ";
		}

		cout << endl;
	}
}

void ergodic_at(Mat& image, int div = 64)
{
	// 3、.at方式
	// 实现colorReduce
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
		{
			image.at<Vec3b>(i, j)[0] = image.at<Vec3b>(i, j)[0] / div * div + div / 2;
			image.at<Vec3b>(i, j)[1] = image.at<Vec3b>(i, j)[1] / div * div + div / 2;
			image.at<Vec3b>(i, j)[2] = image.at<Vec3b>(i, j)[2] / div * div + div / 2;
		}
	}
}

void ergodic_Continuous()
{
	// 4、内存连续法
	Mat mymat = Mat::ones(Size(3, 2), CV_8UC3);
	int nr = mymat.rows;
	int nc = mymat.cols;
	if (mymat.isContinuous())
	{
		nr = 1; // 如果连续，则当成一行
		nc *= mymat.rows; /// 当成一行后的总列数
	}

	for (int i = 0; i < nr; ++i)
	{
		uchar* pdata = mymat.ptr<uchar>(i);
		for (int j = 0; j < nc; ++j)
		{
			cout << (int)pdata[j] << " ";
		}

		cout << endl;
	}
}

void ergodic_Iterator(const Mat& image, Mat& outImage, int div = 64)
{
	// 3、迭代器遍历法
	// 实现colorReduce
	outImage.create(image.size(), image.type());

	MatConstIterator_<Vec3b> it_in = image.begin<Vec3b>();
	MatConstIterator_<Vec3b> itend_in = image.end<Vec3b>();
	MatIterator_<Vec3b> it_out = outImage.begin<Vec3b>();
	MatIterator_<Vec3b> itend_out = outImage.end<Vec3b>();
	while (it_in != itend_in)
	{
		(*it_out)[0] = (*it_in)[0] / div * div + div / 2;
		(*it_out)[1] = (*it_in)[1] / div * div + div / 2;
		(*it_out)[2] = (*it_in)[2] / div * div + div / 2;
		++it_in;
		++it_out;
	}
}

void test_rectangle()
{
	Rect rect(2, 3, 10, 20);
	cout << rect << endl;
	cout << rect.area() << endl;
	cout << rect.size() << endl;
	cout << rect.tl() << " " << rect.br() << endl;
	//cout << rect.width() << " " << rect.height() << endl;
	cout << rect.contains(cv::Point(1, 2)) << endl;
}

int main()
{
	//MatInit1();
	//MatInit2();
	//GetRowsColsDims();
	//PointerOfData();
	//CreateANewMatrixHeader();
	//GetNumOfChannels();
	//MatCopy();
	//isEmpty();
	
	//ergodic_for();
	//ergodic_ptr();

	//Mat A = imread("./image.jpg", 1);
	//namedWindow("原图", WINDOW_FREERATIO);
	//imshow("原图", A);
	//ergodic_at(A);
	//namedWindow("效果图", WINDOW_FREERATIO);
	//imshow("效果图", A);
	//waitKey(0);

	//ergodic_Continuous();

	//Mat A = imread("./image.jpg", 1);
	//Mat B;
	//namedWindow("原图", WINDOW_FREERATIO);
	//imshow("原图", A);
	//ergodic_Iterator(A, B);
	//namedWindow("效果图", WINDOW_FREERATIO);
	//imshow("效果图", B);
	//imwrite("afterimage.jpg", B);
	//waitKey(0);

	//test_rectangle();
	mask_test();

	return 0;
}