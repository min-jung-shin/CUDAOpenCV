
#include <iostream>
#include <vector>
#include "opencv2\imgproc.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\core.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat image = imread("C:/Users/hse03/Desktop/증명사진.jpg", 1);

	Mat gray_img(image.size().height, image.size().width, CV_8UC1, Scalar(0));
	Mat bin_img_256(image.size().height, image.size().width, CV_8UC1, Scalar(0));
	Mat bin_img_128(image.size().height, image.size().width, CV_8UC1, Scalar(0));

	for (int y = 0; y < image.size().height; y++)		// binary img_a_256
	{
		for (int x = 0; x < image.size().width; x++)
		{
			gray_img.at<uchar>(y, x) = (image.at<Vec3b>(y, x)[0] + image.at<Vec3b>(y, x)[1] + image.at<Vec3b>(y, x)[2]) / 3;

			if (gray_img.at<uchar>(y, x) >= 128)
				bin_img_256.at<uchar>(y, x) = 255;
			else
				bin_img_256.at<uchar>(y, x) = 0;
		}
	}

	for (int y = 0; y < image.size().height; y++)		// binary img_a_128
	{
		for (int x = 0; x < image.size().width; x++)
		{
			gray_img.at<uchar>(y, x) = (image.at<Vec3b>(y, x)[0] + image.at<Vec3b>(y, x)[1] + image.at<Vec3b>(y, x)[2]) / 3;

			if (gray_img.at<uchar>(y, x) >= 128)
				bin_img_128.at<uchar>(y, x) = 128;
			else
				bin_img_128.at<uchar>(y, x) = 0;
		}
	}

	int mopology_mask1[7][7] = { { 0,  0,  0,  0,  0,  0,  0},
								 {255,255,255,255,255,255,255},
								 { 0,  0,  0, 255, 0,  0,  0 },
								 { 0,  0,  0, 255, 0,  0,  0 },
								 { 0,  0,  0, 255, 0,  0,  0 },
								 { 0,  0,  0, 255, 0,  0,  0 },
								 { 0,  0,  0, 255, 0,  0,  0}};

	int mopology_mask2[7][7] = { { 0,  0,  0,  0,  0,  0,  0 },
								 { 0,  0,  0,  0,  0,  0,  0 },
								 { 0,  0,  0, 255, 0,  0,  0 },
								 { 0,  0, 255,255,255, 0,  0 },
								 { 0, 255,255,255,255,255, 0 },
								 {255,255,255,255,255,255,255},
								 { 0,  0,  0,  0,  0,  0,  0}};

		// 침식
	Mat Erosion_a_256(image.size().height, image.size().width, CV_8UC1, Scalar(0));	// a, 256 침식 결과 영상
	Mat Erosion_a_128(image.size().height, image.size().width, CV_8UC1, Scalar(0));	// a, 128 침식 결과 영상
	Mat Erosion_b_256(image.size().height, image.size().width, CV_8UC1, Scalar(0));	// b, 256 침식 결과 영상
	Mat Erosion_b_128(image.size().height, image.size().width, CV_8UC1, Scalar(0));	// b, 128 침식 결과 영상

	Mat Eros_pad_256;
	Mat Eros_pad_128;

	copyMakeBorder(bin_img_256, Eros_pad_256, 3, 3, 3, 3, BORDER_CONSTANT, Scalar(0));
	copyMakeBorder(bin_img_128, Eros_pad_128, 3, 3, 3, 3, BORDER_CONSTANT, Scalar(0));

	for (int y = 0; y<image.size().height; y++)
	{
		for (int x = 0;  x < image.size().width; x++)
		{
			int c = 0;
			for (int i = 0; i < 7; i++)
			{
				for (int j = 0; j <7; j++)
				{
					int value = Eros_pad_256.at<uchar>(y + i, x + j);
					if (value == 255 && mopology_mask1[i][j] == 255)
						c++;
				}
			}
			if (c == 12)
				Erosion_a_256.at<uchar>(y, x) = 255;
			else
				Erosion_a_256.at<uchar>(y, x) = 0;
		}
	}

	for (int y = 0; y < image.size().height; y++)
	{
		for (int x = 0; x < image.size().width; x++)
		{
			int c = 0;
			for (int i = 0; i < 7; i++)
			{
				for (int j = 0; j < 7; j++)
				{
					int value = Eros_pad_128.at<uchar>(y + i, x + j);
					if (value == 128 && mopology_mask1[i][j] == 255)
						c++;
				}
			}
			if (c == 12)
				Erosion_a_128.at<uchar>(y, x) = 255;
			else
				Erosion_a_128.at<uchar>(y, x) = 0;
		}
	}
	for (int y = 0; y < image.size().height; y++)
	{
		for (int x = 0; x < image.size().width; x++)
		{
			int c = 0;
			for (int i = 0; i < 7; i++)
			{
				for (int j = 0; j < 7; j++)
				{
					int value = Eros_pad_256.at<uchar>(y + i, x + j);
					if (value == 255 && mopology_mask2[i][j] == 255)
						c++;
				}
			}
			if (c == 16)
				Erosion_b_256.at<uchar>(y, x) = 255;
			else
				Erosion_b_256.at<uchar>(y, x) = 0;
		}
	}

	for (int y = 0; y < image.size().height; y++)
	{
		for (int x = 0; x < image.size().width; x++)
		{
			int c = 0;
			for (int i = 0; i < 7; i++)
			{
				for (int j = 0; j < 7; j++)
				{
					int value = Eros_pad_128.at<uchar>(y + i, x + j);
					if (value == 128 && mopology_mask2[i][j] == 255)
						c++;
				}
			}
			if (c == 16)
				Erosion_b_128.at<uchar>(y, x) = 255;
			else
				Erosion_b_128.at<uchar>(y, x) = 0;
		}
	}


	//팽창
	Mat Dilate_a_256(image.size().height, image.size().width, CV_8UC1, Scalar(0));
	Mat Dilate_a_128(image.size().height, image.size().width, CV_8UC1, Scalar(0));
	Mat Dilate_b_256(image.size().height, image.size().width, CV_8UC1, Scalar(0));
	Mat Dilate_b_128(image.size().height, image.size().width, CV_8UC1, Scalar(0));	// 팽창 결과 영상
	Mat Dil_pad_256;
	Mat Dil_pad_128;
	copyMakeBorder(bin_img_256, Dil_pad_256, 3, 3, 3, 3, BORDER_CONSTANT, Scalar(0));
	copyMakeBorder(bin_img_128, Dil_pad_128, 3, 3, 3, 3, BORDER_CONSTANT, Scalar(0));


	int* Dil_pad_1d_256 = new int[Dil_pad_256.size().height* Dil_pad_256.size().width]();
	for (int y = 0; y < Dil_pad_256.size().height; y++)
	{
		for (int x = 0; x < Dil_pad_256.size().width; x++)
		{
			Dil_pad_1d_256[y * Dil_pad_256.size().width+ x] = Dil_pad_256.at<uchar>(y,x);		// 2차원 이미지를 1차원 메모리에 저장
		}
	}

	int* Dil_pad_1d_128 = new int[Dil_pad_128.size().height * Dil_pad_128.size().width]();
	for (int y = 0; y < Dil_pad_128.size().height; y++)
	{
		for (int x = 0; x < Dil_pad_128.size().width; x++)
		{
			Dil_pad_1d_128[y * Dil_pad_128.size().width + x] = Dil_pad_128.at<uchar>(y, x);		// 2차원 이미지를 1차원 메모리에 저장
		}
	}

	for (int y = 0; y < image.size().height; y++)
	{
		for (int x = 0; x < image.size().width; x++)
		{
			int t = 0;
			for (int i = 0; i <7 ; i++)
			{
				for (int j = 0; j < 7; j++)
				{
					int value2 = Dil_pad_1d_256[(y+i) * Dil_pad_256.size().width + x+j];
					if (value2 == 255 && mopology_mask1)
						t++;
				}
			}
			if (t>=1)
				Dilate_a_256.at<uchar>(y,x)=255;
			else if (t=0)
				Dilate_a_256.at<uchar>(y, x) = 0;
		}
	}

	for (int y = 0; y < image.size().height; y++)
	{
		for (int x = 0; x < image.size().width; x++)
		{
			int t = 0;
			for (int i = 0; i < 7; i++)
			{
				for (int j = 0; j < 7; j++)
				{
					int value2 = Dil_pad_1d_128[(y + i) * Dil_pad_128.size().width + x + j];
					if (value2 == 128 && mopology_mask1)
						t++;
				}
			}
			if (t >= 1)
				Dilate_a_128.at<uchar>(y, x) = 255;
			else if (t = 0)
				Dilate_a_128.at<uchar>(y, x) = 0;
		}
	}

	for (int y = 0; y < image.size().height; y++)
	{
		for (int x = 0; x < image.size().width; x++)
		{
			int t = 0;
			for (int i = 0; i < 7; i++)
			{
				for (int j = 0; j < 7; j++)
				{
					int value2 = Dil_pad_1d_256[(y + i) * Dil_pad_256.size().width + x + j];
					if (value2 == 255 && mopology_mask2)
						t++;
				}
			}
			if (t >= 1)
				Dilate_b_256.at<uchar>(y, x) = 255;
			else if (t = 0)
				Dilate_b_256.at<uchar>(y, x) = 0;
		}
	}

	for (int y = 0; y < image.size().height; y++)
	{
		for (int x = 0; x < image.size().width; x++)
		{
			int t = 0;
			for (int i = 0; i < 7; i++)
			{
				for (int j = 0; j < 7; j++)
				{
					int value2 = Dil_pad_1d_128[(y + i) * Dil_pad_128.size().width + x + j];
					if (value2 == 128 && mopology_mask2)
						t++;
				}
			}
			if (t >= 1)
				Dilate_b_128.at<uchar>(y, x) = 255;
			else if (t = 0)
				Dilate_b_128.at<uchar>(y, x) = 0;
		}
	}

	imshow("그레이 이미지1", gray_img);
	imshow("바이너리 이미지1", bin_img_256);
	imshow("침식 영상a_256", Erosion_a_256);
	imshow("팽창 영상a_256", Dilate_a_256);
	imshow("원본1", image);

	imshow("그레이 이미지2", gray_img);
	imshow("바이너리 이미지2", bin_img_128);
	imshow("침식 영상a_128", Erosion_a_128);
	imshow("팽창 영상a_128", Dilate_a_128);
	imshow("원본2", image);

	imshow("그레이 이미지3", gray_img);
	imshow("바이너리 이미지3", bin_img_256);
	imshow("침식 영상b_256", Erosion_b_256);
	imshow("팽창 영상b_256", Dilate_b_256);
	imshow("원본3", image);

	imshow("그레이 이미지4", gray_img);
	imshow("바이너리 이미지4", bin_img_128);
	imshow("침식 영상b_128", Erosion_b_128);
	imshow("팽창 영상b_128", Dilate_b_128);
	imshow("원본4", image);


	waitKey(0);
	return 0;
}
