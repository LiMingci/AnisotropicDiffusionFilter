#include <iostream>
#include <string>

#include "CcADF.h"
#include "nldiffusion_functions.h"

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


	float sigma0 = 8.0f;
	CcADFConfig config;
	config._ttime = 0.5f * sigma0 * sigma0;
	CcADF adf(config);

	cv::Mat nldFilted;
	bool isSucess = adf.AnisotropicDiffusionFilter(imgOrg, nldFilted);
	if (isSucess)
	{
		cv::imwrite("../data/starry_nld_filted_tc8.jpg", nldFilted);
	}

	cv::Mat gaussFilted;
	gaussian_2D_convolution(imgBackup, gaussFilted, 0, 0, sigma0);
	cv::imwrite("../data/starry_gauss_filted_tc8.jpg", gaussFilted);

	
	return 0;
}