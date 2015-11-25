#ifndef SFM_H
#define SFM_H

#include "WebCamera.h"
#include <opencv2\opencv.hpp>
#include <opencv2\nonfree\nonfree.hpp> // SIFT�܂���SURF���g���ꍇ�͕K�v

class SfM
{
public:
	//�J����
	WebCamera camera;
	//�v���W�F�N�^
	WebCamera projector;

	//��b�s��
	cv::Mat F;
	//��{�s��
	cv::Mat E;

	//�g�p����摜
	cv::Mat src_camImage; // �摜1�̃t�@�C����
	cv::Mat src_projImage; // �摜2�̃t�@�C����
	cv::Mat result; //���ʕ`��p

	//�����_���璊�o�����Ή��_
	std::vector<cv::Point2f>cam_pts, proj_pts;
	
	//�R���X�g���N�^
	SfM(const char *camImageName, const char *projImageName, WebCamera cam, WebCamera proj)
	{
		camera = cam;
		projector = proj;

		std::cout << "cam_K:\n" << camera.cam_K << std::endl;
		std::cout << "cam_dist:\n" << camera.cam_dist << std::endl;
		std::cout << "proj_K:\n" << projector.cam_K << std::endl;
		std::cout << "proj_dist:\n" << projector.cam_dist << std::endl;

		//�c�ݏ������ēǂݍ���(1���ځF�J�����@2����:�v���W�F�N�^)
		//cv::undistort(cv::imread(camImageName), src_camImage, camera.cam_K, camera.cam_dist);
		//cv::undistort(cv::imread(projImageName), src_projImage, projector.cam_K, projector.cam_dist);

		src_camImage = cv::imread(camImageName);
		src_projImage = cv::imread(projImageName);
	};
	~SfM(){};

	void featureMatching(	const char *featureDetectorName, const char *descriptorExtractorName, const char *descriptorMatcherName, bool crossCheck)
	{
		if(featureDetectorName == "SIFT" || featureDetectorName == "SURF" 
			|| descriptorExtractorName == "SIFT" || descriptorExtractorName == "SURF")
		{
			// SIFT�܂���SURF���g���ꍇ�͂�����Ăяo���D
			cv::initModule_nonfree();
		}

		// �����_���o
		cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(featureDetectorName);
		std::vector<cv::KeyPoint> keypoint1, keypoint2;//1->camera 2->projector
		detector->detect(src_camImage, keypoint1);
		detector->detect(src_projImage, keypoint2);

		// �����L�q
		cv::Ptr<cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create(descriptorExtractorName);
		cv::Mat descriptor1, descriptor2;
		extractor->compute(src_camImage, keypoint1, descriptor1);
		extractor->compute(src_projImage, keypoint2, descriptor2);

		// �}�b�`���O
		cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(descriptorMatcherName);
		std::vector<cv::DMatch> dmatch;
		if (crossCheck)
		{
			// �N���X�`�F�b�N����ꍇ
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
			// �N���X�`�F�b�N���Ȃ��ꍇ
			matcher->match(descriptor1, descriptor2, dmatch);
		}

		//�ŏ�����
		double min_dist = DBL_MAX;
		for(int j = 0; j < (int)dmatch.size(); j++)
		{
			double dist = dmatch[j].distance;
			if(dist < min_dist) min_dist = (dist < 1.0) ? 1.0 : dist;
		}

		//�ǂ��y�A�̂ݎc��
		double cutoff = 5.0 * min_dist;
		std::set<int> existing_trainIdx;
		std::vector<cv::DMatch> matches_good;
		for(int j = 0; j < (int)dmatch.size(); j++)
		{
			if(dmatch[j].trainIdx <= 0) dmatch[j].trainIdx = dmatch[j].imgIdx;
			if(dmatch[j].distance > 0.0 && dmatch[j].distance < cutoff){
			//x���W�Ō��ߑł��������l(�}�X�N�̑���)
//			if(dmatch[j].distance > 0.0 && dmatch[j].distance < cutoff && keypoint1[dmatch[j].queryIdx].pt.x > 240 && keypoint2[dmatch[j].trainIdx].pt.x > 240){
				if(existing_trainIdx.find(dmatch[j].trainIdx) == existing_trainIdx.end() && dmatch[j].trainIdx >= 0 && dmatch[j].trainIdx < (int)keypoint2.size()) {
					matches_good.push_back(dmatch[j]);
                    existing_trainIdx.insert(dmatch[j].trainIdx);
				}
			}
		}

        // �Ή��_�̓o�^(5�y�A�ȏ�͕K�v)
        if (matches_good.size() > 10) {
            for (int j = 0; j < (int)matches_good.size(); j++) {
                cam_pts.push_back(keypoint1[matches_good[j].queryIdx].pt);
                proj_pts.push_back(keypoint2[matches_good[j].trainIdx].pt);
            }
		}

		// �}�b�`���O���ʂ̕\��
		cv::drawMatches(src_camImage, keypoint1, src_projImage, keypoint2, matches_good, result);
		//cv::drawMatches(src_image1, keypoint1, src_image2, keypoint2, dmatch, result);
		//cv::Mat resize;
		//result.copyTo(resize);
		//cv::resize(result, resize, resize.size(), 0.5, 0.5);
		cv::imshow("good matching", result);
		cv::waitKey(0);
	}

	void saveResult(const char *resultImageName)
	{
		cv::imwrite(resultImageName, result);
	}

	cv::Mat findEssientialMat(){
		//�Ή��_�𐳋K��(fx=fy=1, cx=cy=0�Ƃ���)


		//��b�s��̎Z�o
		//findfundamentalMat( pt1, pt2, F�s����v�Z�����@, �_����G�s�|�[�����܂ł̍ő勗��, F�̐M���x)
		if(cam_pts.size() == 7)
			F = cv::findFundamentalMat(cam_pts, proj_pts,cv::FM_7POINT, 3.0, 0.99);
		else if(cam_pts.size() == 8)
			F = cv::findFundamentalMat(cam_pts, proj_pts,cv::FM_8POINT, 3.0, 0.99);
		else
			F = cv::findFundamentalMat(cam_pts, proj_pts,cv::RANSAC, 3.0, 0.99);

		//��{�s��̎Z�o
		E = camera.cam_K.t() * F * projector.cam_K;

		return E;
	}


	void findProCamPose(const cv::Mat& E, const cv::Mat& R, const cv::Mat& t)
	{
            cv::Mat R1, R2;
            cv::Mat t_;
			//[R1,t] [R1, -t] [R2, t], [R2, -t]�̉\��������
			decomposeEssentialMat(E, R1, R2, t_);
			std::cout << "\nR1:\n" << R1 << std::endl;
			std::cout << "R2:\n" << R2 << std::endl;
			std::cout << "t:\n" << t_ << std::endl;
	}

	//��{�s�񂩂�R1,R2,t�ɕ���(cv3.0.0�����p)
	void decomposeEssentialMat(const cv::Mat& _E, const cv::Mat& _R1, const cv::Mat& _R2, const cv::Mat& _t )
{
    cv::Mat E = _E.reshape(1, 3);
    CV_Assert(E.cols == 3 && E.rows == 3);

    cv::Mat D, U, Vt;
	cv::SVD::compute(E, D, U, Vt);

    if (determinant(U) < 0) U *= -1.;
    if (determinant(Vt) < 0) Vt *= -1.;

    cv::Mat W = (cv::Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
    W.convertTo(W, E.type());

    cv::Mat R1, R2, t;
    R1 = U * W * Vt;
    R2 = U * W.t() * Vt;
    t = U.col(2) * 1.0;

    R1.copyTo(_R1);
    R2.copyTo(_R2);
    t.copyTo(_t);
}
};
	
#endif