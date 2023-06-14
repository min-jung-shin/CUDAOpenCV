#include "opencv2\core\core.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\highgui.hpp"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("C:/Users/hse03/Desktop/증명사진.jpg", 1);
	vector<Mat> img_split;
	split(img, img_split);
	int mask = 15;

	float** filter_2d = new float *[mask]();	// 15 x 15 의 size를 갖는 필터정의 (2차원으로 만들것)
	for (int i = 0; i < mask; i++)
		filter_2d[i] = new float[mask];
	//"Insert code"// 이중포인터를 이용해 동적할당을 했으니 한번 더 초기화를 해주어야함

	float m = 0; // 가우시안 성분의 합.
	for (int i=-(mask-1)/2; i <(mask-1)/2 ; i++)
	{
		for (int j=-(mask-1)/2; j<(mask-1)/2; j++ )
		{
			filter_2d[i+7][j+7]= 3 * exp(-(pow(i, 2) + pow(j, 2)) / 2 * 0.25);
			m += 3 * exp(-(pow(i, 2) + pow(j, 2)) / 2 * 0.25);
			cout << filter_2d[i + 7][j + 7] << endl; // 2차원으로 정의한 필터에 대한 가우시안 성분
		}
	}


	Mat pad_img;	// padded image
	copyMakeBorder(img, pad_img, 7, 7, 7, 7, BORDER_CONSTANT, Scalar(0, 0, 0));
	vector<Mat> pad_split;
	split(pad_img, pad_split);

Mat gaus_output(img.size().height,img.size().width, CV_8UC3, Scalar(0, 0, 0));// 출력될 필터링 이미지를 사전에 정의. 
	for (int c = 0; c < 3; c++)
	{
		for (int y = 0; y < img.size().height; y++)
		{
			for (int x = 0; x < img.size().width; x++)
			{
				float sop1 = 0;
				float sop2 = 0;
				float sop3 = 0;
				for (int i = 0; i < mask; i++)
				{
					for (int j = 0; j < mask; j++)
					{
						sop1 += pad_split[0].at<uchar>(y + i, x + j) * filter_2d[i][j] / m;// 힌트: pad_img를 이용
						sop2 += pad_split[1].at<uchar>(y + i, x + j) * filter_2d[i][j] / m;
						sop3 += pad_split[2].at<uchar>(y + i, x + j) * filter_2d[i][j] / m;
						
					}
				}
				gaus_output.at<Vec3b>(y, x)[0] = sop1;
				gaus_output.at<Vec3b>(y, x)[1] = sop2;
				gaus_output.at<Vec3b>(y, x)[2] = sop3;
			}
		}
	}

	vector<Mat> gaus_output_split;
	split(gaus_output, gaus_output_split);


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//가우시안 된 영상을 가지고 히스토그램을 그려보자 
	//8 채널 영상에 대한 히스토그램의 x좌표는 무조건 0~255 까지의 범위를 가진다. 

	int range = 256;
	int* ori_b_count = new int[range]();
	int* ori_g_count = new int[range]();
	int* ori_r_count = new int[range]();

	int* gau_b_count = new int[range]();
	int* gau_g_count = new int[range]();
	int* gau_r_count = new int[range]();

	int* pixel_bin = new int[range]();      // 각 픽셀의 값을 인텍스 순서대로 하기위한 도구
	for (int i = 0; i < range; i++)
		pixel_bin[i] = i;


	for (int y = 0; y < img.size().height; y++)		// 원본 이미지 빈도수를 구하는 과정
	{
		for (int x = 0; x < img.size().width; x++)
		{
			for (int i = 0; i < range; i++)
			{
				if (pixel_bin[i] == img.at<Vec3b>(y, x)[0])
					ori_b_count[i] += 1;
				if (pixel_bin[i] == img.at<Vec3b>(y, x)[1])
					ori_g_count[i] += 1;
				if (pixel_bin[i] == img.at<Vec3b>(y, x)[2])
					ori_r_count[i] += 1;
			}
		}
	}


	for (int y = 0; y < gaus_output.size().height; y++)		// 가우시안 이미지 빈도수를 구하는 과정
	{
		for (int x = 0; x < gaus_output.size().width; x++)
		{
			for (int i = 0; i < range; i++)
			{
				if (pixel_bin[i] == gaus_output.at<Vec3b>(y, x)[0])
					gau_b_count[i] += 1;      // 가우시안B 채널의 빈도수를 최종적으로 계산함
				if (pixel_bin[i] == gaus_output.at<Vec3b>(y, x)[1])
					gau_g_count[i] += 1;      // 가우시안G 채널의 빈도수를 최종적으로 계산함
				if (pixel_bin[i] == gaus_output.at<Vec3b>(y, x)[2])
					gau_r_count[i] += 1;      // 가우시안R 채널의 빈도수를 최종적으로 계산함
			}
		}
	}
int* ori_max_b_value = max_element(ori_b_count, ori_b_count + 256);
	Mat ori_b_plain(*ori_max_b_value, 256, CV_8UC3, Scalar(0,0,0));
	int* ori_max_g_value = max_element(ori_g_count, ori_g_count + 256);
	Mat ori_g_plain(*ori_max_g_value, 256, CV_8UC3, Scalar(0, 0, 0));
	int* ori_max_r_value = max_element(ori_r_count, ori_r_count + 256);
	Mat ori_r_plain(*ori_max_r_value, 256, CV_8UC3, Scalar(0, 0, 0));

	int* gau_max_b_value = max_element(gau_b_count, gau_b_count + 256);
	Mat gau_b_plain(*gau_max_b_value, 256, CV_8UC3, Scalar(0, 0, 0));
	int* gau_max_g_value = max_element(gau_g_count, gau_g_count + 256);
	Mat gau_g_plain(*gau_max_g_value, 256, CV_8UC3, Scalar(0, 0, 0));
	int* gau_max_r_value = max_element(gau_r_count, gau_r_count + 256);
	Mat gau_r_plain(*gau_max_r_value, 256, CV_8UC3, Scalar(0, 0, 0));


	Mat ori_b_gray_plain(*ori_max_b_value, 256, CV_8UC1, Scalar(0));
	Mat ori_g_gray_plain(*ori_max_g_value, 256, CV_8UC1, Scalar(0));
	Mat ori_r_gray_plain(*ori_max_r_value, 256, CV_8UC1, Scalar(0));

	Mat gau_b_gray_plain(*gau_max_b_value, 256, CV_8UC1, Scalar(0));
	Mat gau_g_gray_plain(*gau_max_g_value, 256, CV_8UC1, Scalar(0));
	Mat gau_r_gray_plain(*gau_max_r_value, 256, CV_8UC1, Scalar(0));; 
	// ( 현재 모든 히스토그램 (원본 영상, 가우시안 영상)은 컬러가 들어가있지만 이 부분은 그레이 영역으로 만드는 정의 코드를 쓰세요. 이 부분에서 정의된 코드는 총 6개입니다)
	// 힌트: Mat example(*max_value, 256, "이 부분을 유심히 보세요", "이 부분을 유심히 보세요"); 
	// ex) B 채널의 히스토그램을 그리면 빈도수 영역이 파랗게 나옵니다. 이 color 영역을 흰색(255) 으로 바꾸라는 뜻입니다.

		for(int i = 0; i < range; i++)
		{
			int c = 0;
			for (int j = 0; j < *ori_max_b_value; j++)// 원본 B 영상 히스토그램
			{
				if (ori_b_count[i] >= c)
				{
					ori_b_plain.at<Vec3b>(*ori_max_b_value - 1 - j, i)[0] = 255;		// 빈도수의 색이 Blue
					ori_b_gray_plain.at<uchar>(*ori_max_b_value - 1 - j, i) = 255;	// 빈도수의 색이 white
				}
				c++;
			}
		}

	for (int i = 0; i < range; i++)
	{
		int c = 0;
		for (int j = 0; j < *ori_max_g_value; j++)// 원본 G 영상 히스토그램
		{
			if (ori_g_count[i] >= c)
			{
				ori_g_plain.at<Vec3b>(*ori_max_g_value - 1 - j, i)[1] = 255;		// 빈도수의 색이 green
				ori_g_gray_plain.at<uchar>(*ori_max_g_value - 1 - j, i) = 255;	// 빈도수의 색이 white
			}

			c++;
		}
	}

	for(int i = 0; i < range; i++)
	{
		int c = 0;
		for (int j = 0; j < *ori_max_r_value; j++)// 원본 R 영상 히스토그램
		{
			if (ori_r_count[i] >= c)
			{
				ori_r_plain.at<Vec3b>(*ori_max_r_value - 1 - j, i)[2] = 255;		// 빈도수의 색이 red
				ori_r_gray_plain.at<uchar>(*ori_max_r_value - 1 - j, i) = 255;	// 빈도수의 색이 white
			}
			c++;
		}
	}
	for (int i = 0; i < range; i++)
	{
		int c = 0;
		for (int j = 0; j < *gau_max_b_value; j++)// 필터링된 B 영상 히스토그램
		{
			if (gau_b_count[i] >= c)
			{
				gau_b_plain.at<Vec3b>(*gau_max_b_value - 1 - j, i)[0] = 255;		// 빈도수의 색이 blue
				gau_b_gray_plain.at<uchar>(*gau_max_b_value - 1 - j, i) = 255;	// 빈도수의 색이 white
			}
			c++;
		}
	}
	for (int i = 0; i < range; i++)
	{
		int c = 0;
		for (int j = 0; j < *gau_max_g_value; j++)// 필터링된 G 영상 히스토그램
		{
			if (gau_g_count[i] >= c)
			{
				gau_g_plain.at<Vec3b>(*gau_max_g_value - 1 - j, i)[1] = 255;		// 빈도수의 색이 green
				gau_g_gray_plain.at<uchar>(*gau_max_g_value - 1 - j, i) = 255;	// 빈도수의 색이 white
			}
			c++;
		}
	}
	for (int i = 0; i < range; i++)
	{
		int c = 0;
		for (int j = 0; j < *gau_max_r_value; j++)// 필터링된 R 영상 히스토그램
		{
			if (gau_r_count[i] >= c)
			{
				gau_r_plain.at<Vec3b>(*gau_max_r_value - 1 - j, i)[2] = 255;		// 빈도수의 색이 red
				gau_r_gray_plain.at<uchar>(*gau_max_r_value - 1 - j, i) = 255;	// 빈도수의 색이 white
			}
			c++;
		}
	}
int* b_sum = new int[range]();
	int* g_sum = new int[range]();
	int* r_sum = new int[range]();
	int* b_gau_sum = new int[range]();
	int* g_gau_sum = new int[range]();
	int* r_gau_sum = new int[range]();

	int ori_sum = 0;
	int ori_sum2 = 0; 
	int ori_sum3 = 0;
	int gau_sum = 0;
	int gau_sum2 = 0;
	int gau_sum3 = 0;
	for (int i = 0; i < range; i++)
	{
		ori_sum += ori_b_count[i];
		ori_sum2 += ori_g_count[i];
		ori_sum3 += ori_r_count[i];

		gau_sum += gau_b_count[i];
		gau_sum2 += gau_g_count[i];
		gau_sum3 += gau_r_count[i];

		b_sum[i] = round((255 * ori_sum) / (img.size().height * img.size().width));
		g_sum[i] = round((255 * ori_sum2) / (img.size().height * img.size().width));
		r_sum[i] = round((255 * ori_sum3) / (img.size().height * img.size().width));

		b_gau_sum[i] = round((255 * gau_sum) / (gaus_output.size().height * gaus_output.size().width));
		g_gau_sum[i] = round((255 * gau_sum2) / (gaus_output.size().height * gaus_output.size().width));
		r_gau_sum[i] = round((255 * gau_sum3) / (gaus_output.size().height * gaus_output.size().width));


	}

	Mat eq_b_img(img.size().height, img.size().width, CV_8UC1, Scalar(0));
	Mat eq_g_img(img.size().height, img.size().width, CV_8UC1, Scalar(0));
	Mat eq_r_img(img.size().height, img.size().width, CV_8UC1, Scalar(0));

	Mat eq_b_gau_img(gaus_output.size().height, gaus_output.size().width, CV_8UC1, Scalar(0));
	Mat eq_g_gau_img(gaus_output.size().height, gaus_output.size().width, CV_8UC1, Scalar(0));
	Mat eq_r_gau_img(gaus_output.size().height, gaus_output.size().width, CV_8UC1, Scalar(0));

	int* b_sum_plain = new int[range]();
	int* g_sum_plain = new int[range]();
	int* r_sum_plain = new int[range]();
	int* b_sum_gau_plain = new int[range]();
	int* g_sum_gau_plain = new int[range]();
	int* r_sum_gau_plain = new int[range]();


	for (int y = 0; y < img.size().height; y++)
	{
		for (int x = 0; x < img.size().width; x++)
		{
			eq_b_img.at<uchar>(y, x) = b_sum[img_split[0].at<uchar>(y, x)];
			b_sum_plain[eq_b_img.at<uchar>(y,x)] +=1;

			eq_g_img.at<uchar>(y, x) = g_sum[img_split[1].at<uchar>(y, x)];
			g_sum_plain[eq_g_img.at<uchar>(y, x)] += 1;

			eq_r_img.at<uchar>(y, x) = r_sum[img_split[2].at<uchar>(y, x)];
			r_sum_plain[eq_r_img.at<uchar>(y, x)] += 1;
		}
	}

	for (int y = 0; y < gaus_output.size().height; y++)
	{
		for (int x = 0; x < gaus_output.size().width; x++)
		{
			eq_b_gau_img.at<uchar>(y, x) = b_gau_sum[gaus_output_split[0].at<uchar>(y, x)];
			b_sum_gau_plain[eq_b_gau_img.at<uchar>(y, x)] += 1;

			eq_g_gau_img.at<uchar>(y, x) = g_gau_sum[gaus_output_split[1].at<uchar>(y, x)];
			g_sum_gau_plain[eq_g_gau_img.at<uchar>(y, x)] += 1;


			eq_r_gau_img.at<uchar>(y, x) = r_gau_sum[gaus_output_split[2].at<uchar>(y, x)];
			r_sum_gau_plain[eq_r_gau_img.at<uchar>(y, x)] += 1;
		}
	}



	int* eq_max_b = max_element(b_sum_plain, b_sum_plain + 256);
	int* eq_max_g = max_element(g_sum_plain, g_sum_plain + 256);
	int* eq_max_r = max_element(r_sum_plain, r_sum_plain + 256);
	int* eq_max_gau_b = max_element(b_sum_gau_plain, b_sum_gau_plain + 256);
	int* eq_max_gau_g = max_element(g_sum_gau_plain, g_sum_gau_plain + 256);
	int* eq_max_gau_r = max_element(r_sum_gau_plain, r_sum_gau_plain + 256);

	Mat eq_b_plain(*eq_max_b, range, CV_8UC3, Scalar(0, 0, 0));
	Mat eq_g_plain(*eq_max_g, range, CV_8UC3, Scalar(0, 0, 0));
	Mat eq_r_plain(*eq_max_r, range, CV_8UC3, Scalar(0, 0, 0));

	Mat eq_b_gray_plain(*eq_max_b, range, CV_8UC1, Scalar(0));
	Mat eq_g_gray_plain(*eq_max_g, range, CV_8UC1, Scalar(0));
	Mat eq_r_gray_plain(*eq_max_r, range, CV_8UC1, Scalar(0));

	Mat eq_b_gau_plain(*eq_max_gau_b, range, CV_8UC3, Scalar(0, 0, 0));
	Mat eq_g_gau_plain(*eq_max_gau_g, range, CV_8UC3, Scalar(0, 0, 0));
	Mat eq_r_gau_plain(*eq_max_gau_r, range, CV_8UC3, Scalar(0, 0, 0));

	Mat eq_b_gau_gray_plain(*eq_max_gau_b, range, CV_8UC1, Scalar(0));
	Mat eq_g_gau_gray_plain(*eq_max_gau_g, range, CV_8UC1, Scalar(0));
	Mat eq_r_gau_gray_plain(*eq_max_gau_r, range, CV_8UC1, Scalar(0));



	for (int i = 0; i < range; i++)
	{
		int c = 0;
		for (int j = 0; j < *eq_max_b; j++)
		{
			if (b_sum_plain[i] >= c)
			{
				eq_b_plain.at<Vec3b>(*eq_max_b - 1 - j, i)[0] = 255;
				eq_b_gray_plain.at<uchar>(*eq_max_b - 1 - j, i) = 255;
			}
			c++;
		}
	}
	for (int i = 0; i < range; i++)
	{
		int c = 0;
		for (int j = 0; j < *eq_max_g; j++) {
			if (g_sum_plain[i] >= c)
			{
				eq_g_plain.at<Vec3b>(*eq_max_g - 1 - j, i)[1] = 255;
				eq_g_gray_plain.at<uchar>(*eq_max_g - 1 - j, i) = 255;

			}
			c++;
		}
		
	}
	for (int i = 0; i < range; i++)
	{
		int c = 0;
		for (int j = 0; j < *eq_max_r; j++) {
			if (r_sum_plain[i] >= c)
			{
				eq_r_plain.at<Vec3b>(*eq_max_r - 1 - j, i)[2] = 255;
				eq_r_gray_plain.at<uchar>(*eq_max_r - 1 - j, i) = 255;
			}
			c++;
		}
		
	}

	for (int i = 0; i < range; i++)
	{
		int c = 0;
		for (int j = 0; j < *eq_max_gau_b; j++) {
			if (b_sum_gau_plain[i] >= c)
			{
				eq_b_gau_plain.at<Vec3b>(*eq_max_gau_b - 1 - j, i)[0] = 255;
				eq_b_gau_gray_plain.at<uchar>(*eq_max_gau_b - 1 - j, i) = 255;
			}
			c++;
		}
	}

	for (int i = 0; i < range; i++)
	{
		int c = 0;
		for (int j = 0; j < *eq_max_gau_g; j++) {
			if (g_sum_gau_plain[i] >= c)
			{
				eq_g_gau_plain.at<Vec3b>(*eq_max_gau_g - 1 - j, i)[1] = 255;
				eq_g_gau_gray_plain.at<uchar>(*eq_max_gau_g - 1 - j, i) = 255;
			}
			c++;
		}
		
	}

	for (int i = 0; i < range; i++)
	{
		int c = 0;
		for (int j = 0; j < *eq_max_gau_r; j++) {
			if (r_sum_gau_plain[i] >= c)
			{
				eq_r_gau_plain.at<Vec3b>(*eq_max_gau_r - 1 - j, i)[2] = 255;
				eq_r_gau_gray_plain.at<uchar>(*eq_max_gau_r - 1 - j, i) = 255;
			}
			c++;
		}
	}

resize(ori_b_plain, ori_b_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(ori_g_plain, ori_g_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(ori_r_plain, ori_r_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(gau_b_plain, gau_b_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(gau_g_plain, gau_g_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(gau_r_plain, gau_r_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.

	resize(ori_b_gray_plain, ori_b_gray_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(ori_g_gray_plain, ori_g_gray_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(ori_r_gray_plain, ori_r_gray_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(gau_b_gray_plain, gau_b_gray_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(gau_g_gray_plain, gau_g_gray_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(gau_r_gray_plain, gau_r_gray_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.

	resize(eq_b_plain, eq_b_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(eq_g_plain, eq_g_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(eq_r_plain, eq_r_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(eq_b_gau_plain, eq_b_gau_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(eq_g_gau_plain, eq_g_gau_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(eq_r_gau_plain, eq_r_gau_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.

	resize(eq_b_gray_plain, eq_b_gray_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(eq_g_gray_plain, eq_g_gray_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(eq_r_gray_plain, eq_r_gray_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(eq_b_gau_gray_plain, eq_b_gau_gray_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(eq_g_gau_gray_plain, eq_g_gau_gray_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.
	resize(eq_r_gau_gray_plain, eq_r_gau_gray_plain, Size(256, 128));		// 만들어진 히스토그램에 대해 Size(256, 128)로 resize 하세요.



	imshow("원본", img);
	imshow("Gaussian 영상", gaus_output);

	imshow("B 히스토그램(컬러)", ori_b_plain);
	imshow("G 히스토그램(컬러)", ori_g_plain);
	imshow("R 히스토그램(컬러)", ori_r_plain);

	imshow("B 히스토그램(gray)", ori_b_gray_plain);
	imshow("G 히스토그램(gray)", ori_g_gray_plain);
	imshow("R 히스토그램(gray)", ori_r_gray_plain);


	imshow("가우시안B 히스토그램(컬러)", gau_b_plain);
	imshow("가우시안G 히스토그램(컬러)", gau_g_plain);
	imshow("가우시안R 히스토그램(컬러)", gau_r_plain);

	imshow("가우스B 히스토그램(gray)", gau_b_gray_plain);
	imshow("가우시안G 히스토그램(gray)", gau_g_gray_plain);
	imshow("가우시안R 히스토그램(gray)", gau_r_gray_plain);

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//  평활화 한 이미지에 대한 출력(평활화를 안하신 분은 이 부분을 지우세요.)

	imshow("B 히스토그램 평활화(컬러)", eq_b_plain);
	imshow("G 히스토그램 평활화(컬러)", eq_g_plain);
	imshow("R 히스토그램 평활화(컬러)", eq_r_plain);

	imshow("B 히스토그램 평활화(gray)", eq_b_gray_plain);
	imshow("G 히스토그램 평활화(gray)", eq_g_gray_plain);
	imshow("R 히스토그램 평활화(gray)", eq_r_gray_plain);

	imshow("가우스B 히스토그램 평활화(컬러)", eq_b_gau_plain);
	imshow("가우시안G 히스토그램 평활화(컬러)", eq_g_gau_plain);
	imshow("가우시안R 히스토그램 평활화(컬러)", eq_r_gau_plain);

	imshow("가우스B 히스토그램 평활화(gray)", eq_b_gau_gray_plain);
	imshow("가우시안G 히스토그램 평활화(gray)", eq_g_gau_gray_plain);
	imshow("가우시안R 히스토그램 평활화(gray)", eq_r_gau_gray_plain);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	waitKey(0);

	delete[] filter_2d;
	delete[] pixel_bin;

	delete[] ori_b_count;
	delete[] ori_g_count;
	delete[] ori_r_count;

	delete[] gau_b_count;
	delete[] gau_g_count;
	delete[] gau_r_count;

	delete[] b_sum;
	delete[] g_sum;
	delete[] r_sum;

	delete[] b_gau_sum;
	delete[] g_gau_sum;
	delete[] r_gau_sum;

	delete[] b_sum_plain;
	delete[] g_sum_plain;
	delete[] r_sum_plain;

	delete[] b_sum_gau_plain;
	delete[] g_sum_gau_plain;
	delete[] r_sum_gau_plain;

		// 동적할당을 한 만큼 해체하는 코드를 써주세요.


		return 0;

}
