
#include "Header.h"
#include "FeatureMatching.h"
#include "WebCamera.h"
#include "SfM.h"

int main()
{
	//操作説明
		printf("0：特徴点マッチング\n");
		printf("1 : カメラキャリブレーション\n");
		printf("2 : カメラ・プロジェクタのキャリブレーション結果読み込み\n");
		printf("3: カメラ・プロジェクタの相対位置推定\n");
		printf("c : 撮影\n"); 

	//カメラ
	WebCamera mainCamera(640, 480, "webCamera0");
	//プロジェクタ
	WebCamera mainProjector(1280, 800, "projector0");

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
				FeatureMatching featureMatching("./Image/dora_3d/dora02.jpg", "./Image/dora_3d/dora_proj_01.jpg", "SIFT", "SIFT", "BruteForce-L1", true);
				featureMatching.apply();
				featureMatching.saveResult("./Image/result/result_6.jpg");
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
				mainCamera.loadCalibParam("Camera.xml");
				printf("カメラキャリブレーションデータ読み込み\n");
				mainProjector.loadCalibParam("Projector.xml");
				printf("プロジェクタキャリブレーションデータ読み込み\n");
			}
			break;
		case '3':
			{
				//SfM
				SfM sfm("./Image/dora_3d/dora03.jpg", "./Image/dora_3d/dora_proj_01.jpg", mainCamera, mainProjector);
				//①特徴点マッチングで対応点取得
				sfm.featureMatching("SIFT", "SIFT", "BruteForce-L1", true);
				sfm.saveResult("./Image/result/result_8.jpg");
				//②基本行列の算出
				cv::Mat E = sfm.findEssientialMat();
				std::cout << "\nEssentiamMat:\n" << E << std::endl;

				cv::Mat R = cv::Mat::eye(3,3,CV_64F);
				cv::Mat t = cv::Mat::zeros(3,1,CV_64F);
				//③基本行列の分解
				sfm.findProCamPose(E, R, t);
				std::cout << "\nR:\n" << R << std::endl;
				std::cout << "t:\n" << t << std::endl;

			}
			break;
		default:
			exit(0);
			break;
		}
	}

	return 0;
}