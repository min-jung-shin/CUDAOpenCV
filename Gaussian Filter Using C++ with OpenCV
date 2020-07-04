#include <iostream>
#include "opencv2\core.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\highgui.hpp"
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

int main()
{
	Mat image = imread("C:/Users/hse03/Desktop/증명사진.jpg", 1);
	int col_mask5 = 5;
	int row_mask5 = 5;
	int col_mask7 = 7;
	int row_mask7 = 7;

	float *fliter_1d = new float[col_mask5 * row_mask5]();	// 필터 크기 (kernel size) 가 5일때
	float **fliter_2d = new float*[col_mask7]();	// 필터 크기 (kernel size) 가 7일때
	for (int i = 0; i < col_mask7; i++)
		fliter_2d[i] = new float[row_mask7];

	float m5 = 0;
	float m7 = 0;
	for (int i=-((col_mask5* row_mask5)-1 )/2; i< col_mask5 * row_mask5- ((col_mask5 * row_mask5) - 1) / 2 ; i++)	// filter_1d
	{
		fliter_1d[i + ((col_mask5 * row_mask5) - 1) / 2] = 3 * exp(-(pow(i, 2)) / 2 * 0.25);
		m5 += 3 * exp(-(pow(i, 2)) / 2 * 0.25);
		cout << fliter_1d[i + ((col_mask5 * row_mask5) - 1) / 2] << endl;
	}

	for (int i = -((col_mask7-1)/2); i < ((col_mask7 - 1) / 2)+1; i++)	// filter_2d
	{
		for (int j = -((row_mask7 - 1) / 2); j < ((row_mask7 - 1) / 2)+1; j++)
		{
			fliter_2d[i + 3][j + 3] = 3 * exp(-(pow(i, 2) + pow(j, 2)) / 2 * 0.25);
			m7 += 3 * exp(-(pow(i, 2) + pow(j, 2)) / 2 * 0.25);
			cout << fliter_2d[i + 3][j + 3]<<endl;
		}
	}

	vector<Mat> img_split;
	split(image, img_split);

	Mat pad_img5;
	Mat pad_img7;

	copyMakeBorder(image, pad_img5, (col_mask5 -1)/2, (col_mask5 - 1) / 2, (col_mask5 - 1) / 2, (col_mask5 - 1) / 2, BORDER_CONSTANT, Scalar(0));
	copyMakeBorder(image, pad_img7, (col_mask7 - 1) / 2, (col_mask7 - 1) / 2, (col_mask7 - 1) / 2, (col_mask7 - 1) / 2, BORDER_CONSTANT, Scalar(0));	// 패딩
	
	vector<Mat> pad_split5;
	split(pad_img5, pad_split5);
	vector<Mat> pad_split7;
	split(pad_img7, pad_split7);


	
	float* img_buf5 = new float[pad_img5.size().height * pad_img5.size().width * 3]();
	for (int c = 0; c < 3; c++)		// BGR 채널의 영상을 1D buffer에 저장하는 과정
	{
		for (int y = 0; y < pad_img5.size().height; y++)
		{
			for (int x = 0; x < pad_img5.size().width; x++)
			{
				img_buf5[(y*pad_img5.size().width+ x)*3+c] = pad_img5.at<Vec3b>(y, x)[c];
			}
		}
	}


	Mat output5_b(image.size().height, image.size().width, CV_8UC1, Scalar(0));
	Mat output5_g(image.size().height, image.size().width, CV_8UC1, Scalar(0));
	Mat output5_r(image.size().height, image.size().width, CV_8UC1, Scalar(0));
	Mat BGR_img5(image.size().height, image.size().width, CV_8UC3,Scalar(0,0,0));
	for (int c = 0; c < 3; c++)	// 1D buffer에 저장된 영상에 대해 컨볼루션 과정 (필터는 filter_1d 혹은 filter_2d를 사용)
	{
		for (int y = 0; y < image.size().height; y++)
		{
			for (int x = 0; x < image.size().width; x++)
			{
				float sum5 = 0, sum5_1 = 0, sum5_2 = 0, sum5_3 = 0;
				for (int i=0; i<col_mask5; i++)
				{
					for (int j=0; j <row_mask5; j++)
					{
						sum5_1 += pad_split5[0].at<uchar>(y + i, x + j) * fliter_1d[i * col_mask5 + j] / m5;
						sum5_2 += pad_split5[1].at<uchar>(y + i, x + j) * fliter_1d[i * col_mask5 + j] / m5;
						sum5_3 += pad_split5[2].at<uchar>(y + i, x + j) * fliter_1d[i * col_mask5 + j] / m5;
						sum5 += img_buf5[((y+i)*pad_img5.size().width+x+j)*3+c] * fliter_1d[i*col_mask5 +j] / m5;
					}
				}
				
				output5_b.at<uchar>(y, x)= sum5_1;
				output5_g.at<uchar>(y, x) = sum5_2;
				output5_r.at<uchar>(y, x)= sum5_3;
				BGR_img5.at<Vec3b>(y, x)[c] = sum5;
			}
		}
	}

	Mat BGR_img7(image.size().height, image.size().width, CV_8UC3, Scalar(0, 0, 0));
	Mat output7_b(image.size().height, image.size().width, CV_8UC1, Scalar(0));
	Mat output7_g(image.size().height, image.size().width, CV_8UC1, Scalar(0));
	Mat output7_r(image.size().height, image.size().width, CV_8UC1, Scalar(0));

	for (int y = 0; y < image.size().height; y++)	// 2D buffer에 저장된 영상에 대해 컨볼루션 과정 (필터는 filter_1d 혹은 filter_2d를 사용)
	{
		for (int x = 0; x < image.size().width; x++)
		{
			float sum7_1 = 0, sum7_2 = 0, sum7_3 = 0;
			for (int i =0; i < col_mask5; i++)
			{
				for (int j = 0; j < row_mask5; j++)
				{
					sum7_1 += pad_split7[0].at<uchar>(y + i, x + j) * fliter_2d[i][j]/m7;
					sum7_2 += pad_split7[1].at<uchar>(y + i, x + j) * fliter_2d[i][j] /m7;
					sum7_3 += pad_split7[2].at<uchar>(y + i, x + j) * fliter_2d[i][j] /m7;

				}
			}
			output7_b.at<uchar>(y, x) = sum7_1;
			output7_g.at<uchar>(y, x) = sum7_2;
			output7_r.at<uchar>(y, x) = sum7_3;

			BGR_img7.at<Vec3b>(y, x)[0] = sum7_1;
			BGR_img7.at<Vec3b>(y, x)[1] = sum7_2;
			BGR_img7.at<Vec3b>(y, x)[2] = sum7_3;


		}
	}


	// "insert code for imshow"
	// 필터사이즈 5 x 5에 대한 B,G,R 각 채널의 컨볼루션 결과와 BGR 영상의 컨볼루션 결과영상 (총 4개의 이미지가 출력되어야한다)
	// 필터사이즈 7 x 7에 대한 B,G,R 각 채널의 컨볼루션 결과와 BGR 영상의 컨볼루션 결과영상 (총 4개의 이미지가 출력되어야한다)

	imshow("output5_b", output5_b);
	imshow("output5_g", output5_g);
	imshow("output5_r", output5_r);
	imshow("BGR_img5", BGR_img5);

	imshow("output7_b", output7_b);
	imshow("output7_g", output7_g);
	imshow("output7_r", output7_r);
	imshow("BGR_img7", BGR_img7);
	waitKey(0);


	// 동적할당 해체 --> "insert code"

	delete[] fliter_1d;
	delete[] img_buf5;
	for (int i = 0; i < col_mask7; i++)
	{
		delete[] fliter_2d[i];
	}
	delete[] fliter_2d;


	return 0;
}
