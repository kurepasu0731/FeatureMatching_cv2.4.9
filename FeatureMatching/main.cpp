
#include "Header.h"
#include "FeatureMatching.h"
#include "WebCamera.h"
#include "SfM.h"

//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

//カメラ
WebCamera mainCamera(1920, 1080, "webCamera0");
//プロジェクタ
WebCamera mainProjector(1280, 800, "projector0");
//CalibデータのR,t
cv::Mat calib_R = cv::Mat::eye(3,3,CV_64F);
cv::Mat calib_t;


//**Loading datas**//
cv::Mat K1 = cv::Mat::eye(3,3,CV_64F);
cv::Mat K2 = cv::Mat::eye(3,3,CV_64F);
cv::Mat R2 = cv::Mat::eye(3,3,CV_64F);
cv::Mat t2;

std::vector<cv::Point3d> worldPoints; //対応点の3次元座標
std::vector<cv::Point2d> imagePoints1; //カメラ1画像への射影点
std::vector<cv::Point2d> imagePoints2; //カメラ2画像への射影点

void loadFile(const std::string& filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	cv::FileNode node(fs.fs, NULL);

	read(node["worldPoints"], worldPoints);
	read(node["imagePoints1"], imagePoints1);
	read(node["imagePoints2"], imagePoints2);

	read(node["K1"], K1);
	read(node["K2"], K2);
	read(node["R2"], R2);
	read(node["t2"], t2);

	std::cout << "file loaded." << std::endl;
}

void loadProCamCalibFile(const std::string& filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	cv::FileNode node(fs.fs, NULL);

	//カメラパラメータ読み込み
	read(node["cam_K"], mainCamera.cam_K);
	read(node["cam_dist"], mainCamera.cam_dist);
	//プロジェクタパラメータ読み込み
	read(node["proj_K"], mainProjector.cam_K);
	read(node["proj_dist"], mainProjector.cam_dist);

	read(node["R"], calib_R);
	read(node["T"], calib_t);

	std::cout << "ProCamCalib data file loaded." << std::endl;
}


int main()
{
	//操作説明
		printf("0：特徴点マッチング\n");
		printf("1 : カメラキャリブレーション\n");
		printf("2 : カメラ・プロジェクタのキャリブレーション結果読み込み\n");
		printf("3: カメラ・プロジェクタの相対位置推定\n");
		printf("4: GroundTruth\n");
		printf("5: コーナー検出\n");
		printf("c : 撮影\n"); 


	int frame = 0;

	// キー入力受付用の無限ループ
	while(true){
		printf("====================\n");
		printf("数字を入力してください....\n");
		int command;

		//カメラメインループ
		while(true)
		{
			// 何かのキーが入力されたらループを抜ける
			command = cv::waitKey(33);
			if ( command > 0 ){
				//cキーで撮影
				if(command == 'c')
					mainCamera.capture();
				//m1キーで3sに1回100枚連続撮影
				else if(command == 'm')
				{
					while(mainCamera.capture_num < 100)
					{
						Sleep(3000);
						mainCamera.idle();
						mainCamera.capture();
					}
				}
				else break;
			}
			mainCamera.idle();
		}

		// 条件分岐
		switch (command){

		case '0':
			{
				FeatureMatching featureMatching("./Image/chesspattern/cap0.jpg", "./Image/chesspattern/projector.jpg", "ORB", "ORB", "BruteForce-L1", true);
				featureMatching.apply();
				featureMatching.saveResult("./Image/result/result_10.jpg");
				break;
			}
		case '1':
			{
				mainCamera.initCalibration(10, 7, 24.0);
				mainCamera.cameraCalibration();
			}
			break;
		case '2' :
			{
				loadProCamCalibFile("./calibration.xml");
				//mainCamera.loadCalibParam("WebCamera.xml");
				//printf("カメラキャリブレーションデータ読み込み\n");
				//mainProjector.loadCalibParam("WebCamera.xml");
				//printf("プロジェクタキャリブレーションデータ読み込み\n");
			}
			break;
		case '3':
			{
				//SfM (1枚目：カメラ　2枚目:プロジェクタ)
				SfM sfm("./Image/1210/cap3.jpg", "./Image/1210/projector.jpg", mainCamera, mainProjector);
				//�@特徴点マッチングで対応点取得
				sfm.featureMatching("ORB", "ORB", "BruteForce-L1", true);
				sfm.saveResult("./Image/result/result_1210.jpg");
				//�A基本行列の算出
				cv::Mat E1 = sfm.findEssentialMat(); //cv::calibrationMatrixValues
				cv::Mat E2 = sfm.findEssentialMat2();//内部行列の逆行列を掛ける

				cv::Mat R1 = cv::Mat::eye(3,3,CV_64F);
				cv::Mat t1 = cv::Mat::zeros(3,1,CV_64F);
				cv::Mat R2 = cv::Mat::eye(3,3,CV_64F);
				cv::Mat t2 = cv::Mat::zeros(3,1,CV_64F);

				//�BR,tの算出
				sfm.recoverPose(E1, R1, t1);
				sfm.recoverPose(E2, R2, t2);
				//�B基本行列の分解
				//sfm.findProCamPose(E, R, t);
				std::cout << "\nR1:\n" << R1 << std::endl;
				std::cout << "t1:\n" << t1 << std::endl;
				std::cout << "\nR2:\n" << R2 << std::endl;
				std::cout << "t2:\n" << t2 << std::endl;

				// 3Dビューア
				pcl::visualization::PCLVisualizer viewer("3D Viewer");
				viewer.setBackgroundColor(0, 0, 0);
				viewer.addCoordinateSystem(2.0);
				viewer.initCameraParameters();
				Eigen::Affine3f view1, view2, view3;
				Eigen::Matrix4f _t1, _t2, _t3;
				//E1の結果
				_t1 << (float)R1.at<double>(0,0) , (float)R1.at<double>(0,1) , (float)R1.at<double>(0,2) , (float)t1.at<double>(0,0), 
						  (float)R1.at<double>(1,0) , (float)R1.at<double>(1,1) , (float)R1.at<double>(1,2) , (float)t1.at<double>(1,0), 
						  (float)R1.at<double>(2,0) , (float)R1.at<double>(2,1) , (float)R1.at<double>(2,2) , (float)t1.at<double>(2,0), 
						  0.0f, 0.0f ,0.0f, 1.0f;
				std::cout << "_t1:\n"<< _t1 <<std::endl;
				view1 = _t1;
				//viewer.addCoordinateSystem(0.2, view1);
				//E2の結果
				_t2 << (float)R2.at<double>(0,0) , (float)R2.at<double>(0,1) , (float)R2.at<double>(0,2) , (float)t2.at<double>(0,0), 
						  (float)R2.at<double>(1,0) , (float)R2.at<double>(1,1) , (float)R2.at<double>(1,2) , (float)t2.at<double>(1,0), 
						  (float)R2.at<double>(2,0) , (float)R2.at<double>(2,1) , (float)R2.at<double>(2,2) , (float)t2.at<double>(2,0), 
						  0.0f, 0.0f ,0.0f, 1.0f;
				std::cout << "_t2:\n"<< _t2 <<std::endl;
				view2 = _t2;
				viewer.addCoordinateSystem(0.5, view2);
				//calibファイルの結果(正解)
				_t3 << (float)calib_R.at<double>(0,0) , (float)calib_R.at<double>(0,1) , (float)calib_R.at<double>(0,2) , (float)(calib_t.at<double>(0,0) / 500), 
					(float)calib_R.at<double>(1,0) , (float)calib_R.at<double>(1,1) , (float)calib_R.at<double>(1,2) , (float)(calib_t.at<double>(1,0) / 500), 
					(float)calib_R.at<double>(2,0) , (float)calib_R.at<double>(2,1) , (float)calib_R.at<double>(2,2) , (float)(calib_t.at<double>(2,0) / 500), 
						  0.0f, 0.0f ,0.0f, 1.0f;
				std::cout << "_t3:\n"<< _t3 <<std::endl;
				view3 = _t3;
				viewer.addCoordinateSystem(1.0, view3);

			}
			break;
			case '4':
			{
				//data loading
				loadFile("../groundtruth_1222032.xml");

				//SfM
				SfM sfm("./Image/capture/cap38.jpg", "./Image/capture/cap40.jpg", mainCamera, mainProjector);

				sfm.cam_pts = imagePoints1;
				sfm.proj_pts = imagePoints2;
				sfm.camera.cam_K = K1;
				sfm.projector.cam_K = K2;

				//�A基本行列の算出
				cv::Mat E1 = sfm.findEssentialMat(); //cv::calibrationMatrixValues
				cv::Mat E2 = sfm.findEssentialMat2();//内部行列の逆行列を掛ける

				cv::Mat R1 = cv::Mat::eye(3,3,CV_64F);
				cv::Mat t1 = cv::Mat::zeros(3,1,CV_64F);
				cv::Mat R2_cal = cv::Mat::eye(3,3,CV_64F);
				cv::Mat t2_cal = cv::Mat::zeros(3,1,CV_64F);

				//�BR,tの算出
				sfm.recoverPose(E1, R1, t1);
				sfm.recoverPose(E2, R2_cal, t2_cal);
				//�B基本行列の分解
				//sfm.findProCamPose(E, R, t);
				std::cout << "\nE1 result:\nR1:\n" << R1 << std::endl;
				std::cout << "t1:\n" << t1 << std::endl;
				std::cout << "\nE2 result:\nR2:\n" << R2_cal << std::endl;
				std::cout << "t2:\n" << t2_cal << std::endl;

				std::cout <<"\nground truth\n" << "R2:\n" << R2 <<std:: endl;
				std::cout << "t2:\n" << t2 << std::endl;

				// 3Dビューア
				pcl::visualization::PCLVisualizer viewer("3D Viewer");
				viewer.setBackgroundColor(0, 0, 0);
				viewer.addCoordinateSystem(2.0);
				viewer.initCameraParameters();
				Eigen::Affine3f view1, view2, view3;
				Eigen::Matrix4f _t1, _t2cal, _t2;
				//E1の結果
				_t1 << (float)R1.at<double>(0,0) , (float)R1.at<double>(0,1) , (float)R1.at<double>(0,2) , (float)t1.at<double>(0,0), 
						  (float)R1.at<double>(1,0) , (float)R1.at<double>(1,1) , (float)R1.at<double>(1,2) , (float)t1.at<double>(1,0), 
						  (float)R1.at<double>(2,0) , (float)R1.at<double>(2,1) , (float)R1.at<double>(2,2) , (float)t1.at<double>(2,0), 
						  0.0f, 0.0f ,0.0f, 1.0f;
				std::cout << "_t1:\n"<< _t1 <<std::endl;
				view1 = _t1;
				viewer.addCoordinateSystem(0.2, view1);
				//E2の結果
				_t2cal << (float)R2_cal.at<double>(0,0) , (float)R2_cal.at<double>(0,1) , (float)R2_cal.at<double>(0,2) , (float)t2_cal.at<double>(0,0), 
						  (float)R2_cal.at<double>(1,0) , (float)R2_cal.at<double>(1,1) , (float)R2_cal.at<double>(1,2) , (float)t2_cal.at<double>(1,0), 
						  (float)R2_cal.at<double>(2,0) , (float)R2_cal.at<double>(2,1) , (float)R2_cal.at<double>(2,2) , (float)t2_cal.at<double>(2,0), 
						  0.0f, 0.0f ,0.0f, 1.0f;
				std::cout << "_t2:\n"<< _t2cal <<std::endl;
				view2 = _t2cal;
				viewer.addCoordinateSystem(0.5, view2);
				//ground truth
				_t2 << (float)R2.at<double>(0,0) , (float)R2.at<double>(0,1) , (float)R2.at<double>(0,2) , (float)t2.at<double>(0,0), 
						  (float)R2.at<double>(1,0) , (float)R2.at<double>(1,1) , (float)R2.at<double>(1,2) , (float)t2.at<double>(1,0), 
						  (float)R2.at<double>(2,0) , (float)R2.at<double>(2,1) , (float)R2.at<double>(2,2) , (float)t2.at<double>(2,0), 
						  0.0f, 0.0f ,0.0f, 1.0f;
				std::cout << "_t2:\n"<< _t2 <<std::endl;
				view3 = _t2;
				viewer.addCoordinateSystem(1.0, view3);
			}
			break;
		case '5':
			{
				FeatureMatching featureMatching("./Image/chesspattern/cap4.jpg", "./Image/chesspattern/chess_pattern_1280_800.jpg", "ORB", "ORB", "BruteForce-L1", true);
				
				//プロジェクタ画像のコーナー検出
				cv::imshow("Projector image", featureMatching.findCorners(featureMatching.src_image2));
				cv::namedWindow("Camera image");


				//カメラメインループ
				while(true)
				{
					// 何かのキーが入力されたらループを抜ける
					command = cv::waitKey(33);
					if ( command > 0 ){
						//cキーで撮影
						if(command == 'c')
							mainCamera.capture();
						else break;
					}
					cv::imshow("Camera image", featureMatching.findCorners(mainCamera.getFrame()));
				}

				break;
			}
		default:
			exit(0);
			break;
		}
	}

	return 0;
}