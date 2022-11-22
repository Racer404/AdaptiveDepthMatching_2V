#include<opencv2/core.hpp>
#include<opencv2/opencv.hpp> 
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<stdio.h>
#include"stereoMatching.h"

using namespace cv;

Mat Left;
Mat Right;
Mat depthMap;

int main() {

	String inputName = "Sculpture";
	int detailLevel = 4;

	printf("%s\n",inputName.c_str());

	Left = imread("StereoImageDataset/" + inputName + "/left.png");
	Right = imread("StereoImageDataset/" + inputName + "/right.png");

	vector<vector<int>> LeftGrayScale;
	vector<vector<int>> RightGrayScale;

	int totalDis = 0;
	depthMap = Mat(Left.rows, Left.cols, CV_8UC3, Scalar(0, 0, 0));

	for (int width = 0; width < Left.cols; width++) {
		LeftGrayScale.push_back(vector<int>());
		RightGrayScale.push_back(vector<int>());
		for (int height = 0; height < Left.rows; height++)
		{
			int leftGrayScale = Left.at<Vec3b>(height, width)[2] * 0.299 + Left.at<Vec3b>(height, width)[1] * 0.387 + Left.at<Vec3b>(height, width)[0] * 0.114;
			int rightGrayScale = Right.at<Vec3b>(height, width)[2] * 0.299 + Right.at<Vec3b>(height, width)[1] * 0.387 + Right.at<Vec3b>(height, width)[0] * 0.114;
			LeftGrayScale[width].push_back(leftGrayScale);
			RightGrayScale[width].push_back(rightGrayScale);
		}
	}

	std::ostringstream oss;

	stereoMatching sM;
	sM.inputGrayScale(LeftGrayScale, RightGrayScale);
	sM.setPatchSize(10);
	sM.matchMethod(1);
	sM.setGraySumMethod(2);

	stereoMatching outputsM;
	outputsM.inputGrayScale(LeftGrayScale, RightGrayScale);
	outputsM.setPatchSize(2);
	outputsM.matchMethod(3);
	outputsM.setGraySumMethod(2);
	outputsM.setWeightOfDis(1.5);

	for (int height = 0; height < Left.rows; height++) {

		vector<Vec2i> disparityArray;
		for (int width = 0; width < Left.cols; width++) {
			int Z = width - sM.match(width, height)[0];
			bool ifDisExists = false;
			for (int i = 0; i < disparityArray.size(); i++) {
				if (Z == disparityArray[i][0]) {
					disparityArray[i][1]++;
					ifDisExists = true;
					break;
				}
			}
			if (!ifDisExists) {
				disparityArray.push_back(Vec2i(Z, 0));
			}
		}


		vector<int> majorDis;
		for (int i = 0; i < detailLevel; i++) {
			int largest = 0;
			int largestIndex = 0;
			for (int j = 0; j < disparityArray.size(); j++) {
				if (disparityArray[j][1] > largest) {
					largest = disparityArray[j][1];
					largestIndex = j;
				}
			}
			majorDis.push_back(disparityArray[largestIndex][0]);
			disparityArray.erase(disparityArray.begin() + largestIndex);
		}

		outputsM.setMajorDis(majorDis);

		for (int width = 0; width < Left.cols; width++) {
			int Z = width - outputsM.match(width, height)[0];
			depthMap.at<Vec3b>(height, width)[0] = Z;
			depthMap.at<Vec3b>(height, width)[1] = Z;
			depthMap.at<Vec3b>(height, width)[2] = Z;
			totalDis = totalDis + Z;
		}
	}

	oss << "Result/" << inputName << "/Mode=ADM" << " pS1=" << sM.patchSize << " pS2=" << outputsM.patchSize << " wt=" << outputsM.weightOfDis << " dtL=" << detailLevel << " gSM1=" << sM.graySumMethod << " gSM2=" << outputsM.graySumMethod << ".png";
	


	//RENDERING
	float averageDis = (float)totalDis / (float)(Left.cols * Left.rows);

	float a = (float)128 / (float)(averageDis);

	for (int width = 0; width < Left.cols; width++) {
		for (int height = 0; height < Left.rows; height++) {
			int Z = depthMap.at<Vec3b>(height, width)[0] * a;
			if (Z > 255) {
				Z = 255;
			}
			depthMap.at<Vec3b>(height, width)[0] = Z;
			depthMap.at<Vec3b>(height, width)[1] = Z;
			depthMap.at<Vec3b>(height, width)[2] = Z;
		}
	}

	applyColorMap(depthMap, depthMap, COLORMAP_PLASMA);

	medianBlur(depthMap, depthMap, 5);

	imshow("result", depthMap);

	std::string filePath = oss.str();

	imwrite(filePath, depthMap);



	waitKey();
}