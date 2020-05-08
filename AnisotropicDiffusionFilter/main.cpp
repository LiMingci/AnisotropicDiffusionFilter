#include <iostream>
#include <string>

#include "CcADF.h"
#include "nldiffusion_functions.h"

/*
int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		std::cout << "please input image" << std::endl;
		return -1;
	}

	cv::Mat imgOrg = cv::imread(argv[1], 0); //convert image to the single channel grayscale image
	//cv::imwrite("../data/org.jpg", imgOrg);
	if (imgOrg.data == nullptr)
	{
		std::cout << "input image is null" << std::endl;
		return -1;
	}
	cv::Mat imgBackup;
	imgOrg.copyTo(imgBackup);

	cv::Mat img32;
	imgOrg.convertTo(img32, CV_32F, 1.0 / 255.0, 0); //convert image to float
	int imgWidth = img32.cols;
	int imgHeight = img32.rows;
	cv::Size imgSize(imgWidth, imgHeight);

	float k = compute_k_percentile(img32, 0.7, 1.0, 300, 0, 0); //calc k

	cv::Mat imgSmooth, imgLx, imgLy;
	imgLx.create(imgSize, CV_32F);
	imgLy.create(imgSize, CV_32F);

	gaussian_2D_convolution(img32, imgSmooth, 0, 0, 1.6);
	image_derivatives_scharr(imgSmooth, imgLx, 1, 0);
	image_derivatives_scharr(imgSmooth, imgLy, 0, 1);

	cv::Mat imgLflow;
	imgLflow.create(imgSize, CV_32F);
	pm_g2(imgLx, imgLy, imgLflow, k);

	float sigma0 = 16.0f;
	//float ttime = 0.0f;
	float ttime = 0.5f * sigma0 * sigma0;
	int naux = 0;
	std::vector<float> tau;
	naux = fed_tau_by_process_time(ttime, 1, 0.25, true, tau);

	cv::Mat imgLstep;
	imgLstep.create(imgSize, CV_32F);
	for (int j = 0; j < naux; j++)
	{
		nld_step_scalar(img32, imgLflow, imgLstep, tau[j]);
	}

	cv::Mat gaussFilted;
	gaussian_2D_convolution(imgBackup, gaussFilted, 0, 0, sigma0);
	cv::imwrite("../data/gauss_filted_t16.jpg", gaussFilted);

	cv::Mat imgFilted;
	img32.convertTo(imgFilted, CV_8U, 255.0, 0.0);
	cv::imwrite("../data/nld_filted_t16.jpg", imgFilted);

	return 0;
}

*/

int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		std::cout << "please input image" << std::endl;
		return -1;
	}

	cv::Mat imgOrg = cv::imread(argv[1]);
											 
	if (imgOrg.data == nullptr)
	{
		std::cout << "input image is null" << std::endl;
		return -1;
	}
	cv::Mat imgBackup;
	imgOrg.copyTo(imgBackup);


	float sigma0 = 32.0f;
	CcADFConfig config;
	config._ttime = 0.5f * sigma0 * sigma0;
	CcADF adf(config);

	cv::Mat nldFilted;
	bool isSucess = adf.AnisotropicDiffusionFilter(imgOrg, nldFilted);
	if (isSucess)
	{
		cv::imwrite("../data/starry_nld_filted_tc32.jpg", nldFilted);
	}

	cv::Mat gaussFilted;
	gaussian_2D_convolution(imgBackup, gaussFilted, 0, 0, sigma0);
	cv::imwrite("../data/starry_gauss_filted_tc32.jpg", gaussFilted);

	
	return 0;
}