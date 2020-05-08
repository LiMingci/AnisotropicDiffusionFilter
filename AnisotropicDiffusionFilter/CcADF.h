#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

enum DIFFUSIVITY_TYPE
{
	PM_G1 = 0,
	PM_G2 = 1,
	WEICKERT = 2,
	CHARBONNIER = 3
};

struct CcADFConfig
{
	float _ttime;
	DIFFUSIVITY_TYPE _diffusivity;   ///< Diffusivity type
	float _tmax;
	float _soffset;

	CcADFConfig(
		float ttime = 2.0f,
		DIFFUSIVITY_TYPE diffusivity = PM_G2,
		float tmax = 0.25,
		float soffset = 1.6f) 
		: _ttime(ttime), _diffusivity(diffusivity), _tmax(tmax), _soffset(soffset)
	{
		//
	}

};

class CcADF
{
public:
	CcADF(const CcADFConfig& config = CcADFConfig());
	~CcADF();

	bool AnisotropicDiffusionFilter(const cv::Mat& srcMat, cv::Mat& dstMat);

private:
	bool AnisotropicDiffusionFilterSingle(const cv::Mat& srcMat, cv::Mat& dstMat);

	bool AnisotropicDiffusionFilterMutil(const cv::Mat& srcMat, cv::Mat& dstMat);

private:
	CcADFConfig _config;

};

