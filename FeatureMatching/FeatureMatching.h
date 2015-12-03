#ifndef FEATUREMATCHING_H
#define FEATUREMATCHING_H

#include <opencv2\opencv.hpp>

class FeatureMatching
{
public:
	cv::Mat src_image1; // 画像1のファイル名
	cv::Mat src_image2; // 画像2のファイル名
	cv::Mat result; //マッチング結果

	std::string featureDetectorName; // detectorType
	std::string descriptorExtractorName; // descriptorExtractorType
	std::string descriptorMatcherName; // descriptorMatcherType
	bool crossCheck; // マッチング結果をクロスチェックするかどうか

	//コンストラクタ
	FeatureMatching(const char *image1Name, const char *image2Name, 
		const char *_featureDetectorName, const char *_descriptorExtractorName, const char *_descriptorMatcherName, bool _crossCheck)
	{
		src_image1 = cv::imread(image1Name);
		src_image2 = cv::imread(image2Name);
		featureDetectorName = _featureDetectorName;
		descriptorExtractorName = _descriptorExtractorName;
		descriptorMatcherName = _descriptorMatcherName;
		crossCheck = _crossCheck;
	};

	~FeatureMatching(){};

	//特徴点マッチング実行
	void apply()
	{
		if(featureDetectorName == "SIFT" || featureDetectorName == "SURF" 
			|| descriptorExtractorName == "SIFT" || descriptorExtractorName == "SURF")
		{
			// SIFTまたはSURFを使う場合はこれを呼び出す．
			cv::initModule_nonfree();
		}

		// 特徴点抽出
		cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(featureDetectorName);
		std::vector<cv::KeyPoint> keypoint1, keypoint2;
		detector->detect(src_image1, keypoint1);
		detector->detect(src_image2, keypoint2);

		// 特徴記述
		cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create(descriptorExtractorName);
		cv::Mat descriptor1, descriptor2;
		extractor->compute(src_image1, keypoint1, descriptor1);
		extractor->compute(src_image2, keypoint2, descriptor2);

		// マッチング
		cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(descriptorMatcherName);
		std::vector<cv::DMatch> dmatch;
		if (crossCheck)
		{
			// クロスチェックする場合
			std::vector<cv::DMatch> match12, match21;
			matcher->match(descriptor1, descriptor2, match12);
			matcher->match(descriptor2, descriptor1, match21);
			for (size_t i = 0; i < match12.size(); i++)
			{
				cv::DMatch forward = match12[i];
				cv::DMatch backward = match21[forward.trainIdx];
				if (backward.trainIdx == forward.queryIdx)
					dmatch.push_back(forward);
			}
		}
		else
		{
			// クロスチェックしない場合
			matcher->match(descriptor1, descriptor2, dmatch);
		}

		//最小距離
		double min_dist = DBL_MAX;
		for(int j = 0; j < (int)dmatch.size(); j++)
		{
			double dist = dmatch[j].distance;
			if(dist < min_dist) min_dist = (dist < 1.0) ? 1.0 : dist;
		}

		//良いペアのみ残す
		double cutoff = 5.0 * min_dist;
		std::set<int> existing_trainIdx;
		std::vector<cv::DMatch> matches_good;
		for(int j = 0; j < (int)dmatch.size(); j++)
		{
			if(dmatch[j].trainIdx <= 0) dmatch[j].trainIdx = dmatch[j].imgIdx;
			if(dmatch[j].distance > 0.0 && dmatch[j].distance < cutoff){
				if(existing_trainIdx.find(dmatch[j].trainIdx) == existing_trainIdx.end() && dmatch[j].trainIdx >= 0 && dmatch[j].trainIdx < (int)keypoint2.size()) {
					matches_good.push_back(dmatch[j]);
                    existing_trainIdx.insert(dmatch[j].trainIdx);
				}
			}
		}

		// マッチング結果の表示
		cv::drawMatches(src_image1, keypoint1, src_image2, keypoint2, matches_good, result);
		cv::imshow("matching", result);
	}

	void saveResult(const char *resultImageName)
	{
		cv::imwrite(resultImageName, result);
	}
};

#endif