
#include "fed.h"
#include "nldiffusion_functions.h"
#include "CcADF.h"


CcADF::CcADF(const CcADFConfig& config)
{
	_config = config;
}


CcADF::~CcADF()
{
}

bool CcADF::AnisotropicDiffusionFilter(const cv::Mat& srcMat, cv::Mat& dstMat)
{
	int imgChannels = srcMat.channels();

	bool result = false;
	if (imgChannels == 1)
	{
		result = AnisotropicDiffusionFilterSingle(srcMat, dstMat);
	}
	else if(imgChannels >1)
	{
		result = AnisotropicDiffusionFilterMutil(srcMat, dstMat);
	}

	return result;
}

bool CcADF::AnisotropicDiffusionFilterSingle(const cv::Mat& srcMat, cv::Mat& dstMat)
{
	cv::Mat img32;
	srcMat.convertTo(img32, CV_32F, 1.0/255.0, 0);
	int imgWidth = img32.cols;
	int imgHeight = img32.rows;
	cv::Size imgSize(imgWidth, imgHeight);

	float k = compute_k_percentile(img32, 0.7, 1.0, 300, 0, 0);

	cv::Mat imgSmooth, imgLx, imgLy;
	imgLx.create(imgSize, CV_32F);
	imgLy.create(imgSize, CV_32F);

	gaussian_2D_convolution(img32, imgSmooth, 0, 0, _config._soffset);
	image_derivatives_scharr(imgSmooth, imgLx, 1, 0);
	image_derivatives_scharr(imgSmooth, imgLy, 0, 1);

	cv::Mat imgLflow;
	imgLflow.create(imgSize, CV_32F);
	switch (_config._diffusivity)
	{
	case PM_G1:
		pm_g1(imgLx, imgLy, imgLflow, k);
		break;
	case PM_G2:
		pm_g2(imgLx, imgLy, imgLflow, k);
		break;
	case WEICKERT:
		weickert_diffusivity(imgLx, imgLy, imgLflow, k);
		break;
	case CHARBONNIER:
		charbonnier_diffusivity(imgLx, imgLy, imgLflow, k);
		break;
	default:
		std::cout << "Diffusivity: " << _config._diffusivity << " is not supported" << std::endl;
	}

	float ttime = _config._ttime;
	int naux = 0;
	std::vector<float> tau;
	naux = fed_tau_by_process_time(ttime, 1, 0.25, true, tau);

	cv::Mat imgLstep;
	imgLstep.create(imgSize, CV_32F);
	for (int j = 0; j < naux; j++)
	{
		nld_step_scalar(img32, imgLflow, imgLstep, tau[j]);
	}

	img32.convertTo(dstMat, CV_8U, 255.0, 0.0);

	if (dstMat.data == nullptr)
	{
		return false;
	}

	return true;
}


bool CcADF::AnisotropicDiffusionFilterMutil(const cv::Mat& srcMat, cv::Mat& dstMat)
{
	std::vector<cv::Mat> channels;
	cv::split(srcMat, channels);

	bool result = true;
	std::vector<cv::Mat> filtedMats(3);
	for (std::size_t i = 0; i < channels.size(); i++)
	{
		cv::Mat singleMat = channels.at(i);
		bool subResult = AnisotropicDiffusionFilterSingle(singleMat, filtedMats[i]);
		result = result && subResult;
	}
	if (!result)
	{
		return result;
	}

	cv::merge(filtedMats, dstMat);

	return result;
}
