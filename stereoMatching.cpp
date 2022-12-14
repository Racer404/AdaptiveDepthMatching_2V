#include"stereoMatching.h"

void stereoMatching::setPatchSize(int inputPatchSize) {
	patchSize = inputPatchSize;
}

void stereoMatching::setGraySumMethod(int inputgraySumMethod) {
	graySumMethod = inputgraySumMethod;
}

void stereoMatching::setWeightOfDis(float inputWeightOfDis) {
	weightOfDis = inputWeightOfDis;
}

void stereoMatching::setBiasOfMinDis(float inputBiasOfMinDis) {
	biasOfMinDis = inputBiasOfMinDis;
}

void stereoMatching::setMajorDis(vector<int> inputMajorDis) {
	majorDis = inputMajorDis;
}

void stereoMatching::inputGrayScale(vector<vector<int>> inputLeft, vector<vector<int>> inputRight) {
	left = inputLeft;
	right = inputRight;
}

void stereoMatching::matchMethod(int methodIndex) {
	matchingMethod = methodIndex;
}

//Vec2f stereoMatching::getSlopeAt(int patchSize, int j) {
//	return Vec2f(1,2);
//}

vector<int> stereoMatching::match(int xPixel, int yPixel) {
	if (matchingMethod == 1) {
		float lowestDifference = 9999999;
		int indexOfLowestDifference = 0;

		if (graySumMethod == 2) {
			vector<int> sample;
			for (int i = 0; i < patchSize; i++) {
				if ((xPixel + i) < left.size()) {
					sample.push_back(left[xPixel + i][yPixel]);
				}
			}
			for (int i = 0; i < xPixel; i++) {
				vector<int> target;
				for (int j = 0; j < patchSize; j++) {
					if ((i + j) < xPixel) {
						target.push_back(right[i + j][yPixel]);
					}
				}

				if (target.size() <= sample.size()) {
					int cost = 0;
					for (int k = 0; k < target.size(); k++) {
						cost = cost + pow((sample[k] - target[k]), 2);
					}

					float difference = cost + abs(weightOfDis * abs(xPixel - i) - biasOfMinDis);

					if (difference < lowestDifference) {
						lowestDifference = difference;
						indexOfLowestDifference = i;
					}
				}
			}
		}
		else if (graySumMethod == 1) {
			int sample = 0;
			for (int i = 0; i < patchSize; i++) {
				if ((xPixel + i) < left.size()) {
					sample = sample + left[xPixel + i][yPixel];
				}
			}

			for (int i = 0; i < xPixel; i++) {
				int target = 0;
				for (int j = 0; j < patchSize; j++) {
					if ((i + j) < xPixel) {
						target = target + right[i + j][yPixel];
					}
				}

				int difference = abs(target - sample)+ abs(weightOfDis * abs(xPixel - i) - biasOfMinDis);

				if (difference < lowestDifference) {
					lowestDifference = difference;
					indexOfLowestDifference = i;
				}
			}
		}
		else if (graySumMethod == 3) {
			vector<int> sample;
			for (int i = 0; i < patchSize; i++) {
				if ((xPixel + i) < left.size()) {
					sample.push_back(left[xPixel + i][yPixel]);
				}
			}
			for (int i = 0; i < xPixel; i++) {
				vector<int> target;
				for (int j = 0; j < patchSize; j++) {
					if ((i + j) < xPixel) {
						target.push_back(right[i + j][yPixel]);
					}
				}

				if (target.size() <= sample.size()) {
					int cost = 0;
					for (int k = 0; k < target.size(); k++) {
						cost = cost + abs(sample[k] - target[k]);
					}

					float difference = cost + abs(weightOfDis * abs(xPixel - i) - biasOfMinDis);

					if (difference < lowestDifference) {
						lowestDifference = difference;
						indexOfLowestDifference = i;
					}
				}
			}
		}
		
		vector<int> resultVector;
		resultVector.push_back(indexOfLowestDifference);
		resultVector.push_back(lowestDifference);
		return resultVector;
	}


	else if (matchingMethod == 3) {

		float lowestDifference = 9999999;
		int indexOfLowestDifference = 0;

		if (graySumMethod == 2) {
			vector<int> sample;
			for (int i = 0; i < patchSize; i++) {
				if ((xPixel + i) < left.size()) {
					sample.push_back(left[xPixel + i][yPixel]);
				}
			}

			for (int i = 0; i < xPixel; i++) {
				vector<int> target;
				for (int j = 0; j < patchSize; j++) {
					if ((i + j) < xPixel) {
						target.push_back(right[i + j][yPixel]);
					}
				}

				float denominator = 0;

				for (int majorNum = 0; majorNum < majorDis.size(); majorNum++) {
					denominator = denominator + (1 / (abs((xPixel - i) - majorDis[majorNum]) + 0.0001));
				}

				if (target.size() <= sample.size()) {
					int cost = 0;
					for (int k = 0; k < target.size(); k++) {
						cost = cost + pow((sample[k] - target[k]), 2);
					}

					float difference = cost + weightOfDis / denominator;

					if (difference < lowestDifference) {
						lowestDifference = difference;
						indexOfLowestDifference = i;
					}
				}
			}
		}
		else if (graySumMethod == 1) {
			int sample = 0;
			for (int i = 0; i < patchSize; i++) {
				if ((xPixel + i) < left.size()) {
					sample = sample + left[xPixel + i][yPixel];
				}
			}

			for (int i = 0; i < xPixel; i++) {
				int target = 0;
				for (int j = 0; j < patchSize; j++) {
					if ((i + j) < xPixel) {
						target = target + right[i + j][yPixel];
					}
				}

				float denominator = 0;

				for (int majorNum = 0; majorNum < majorDis.size(); majorNum++) {
					denominator = denominator + (1 / (abs((xPixel - i) - majorDis[majorNum]) + 0.0001));
				}

				int difference = abs(target - sample) + weightOfDis / denominator;

				if (difference < lowestDifference) {
					lowestDifference = difference;
					indexOfLowestDifference = i;
				}
			}
		}
		
		else if (graySumMethod == 3) {
			vector<int> sample;
			for (int i = 0; i < patchSize; i++) {
				if ((xPixel + i) < left.size()) {
					sample.push_back(left[xPixel + i][yPixel]);
				}
			}

			for (int i = 0; i < xPixel; i++) {
				vector<int> target;
				for (int j = 0; j < patchSize; j++) {
					if ((i + j) < xPixel) {
						target.push_back(right[i + j][yPixel]);
					}
				}

				float denominator = 0;

				for (int majorNum = 0; majorNum < majorDis.size(); majorNum++) {
					denominator = denominator + (1 / (abs((xPixel - i) - majorDis[majorNum]) + 0.0001));
				}

				if (target.size() <= sample.size()) {
					int cost = 0;
					for (int k = 0; k < target.size(); k++) {
						cost = cost + abs(sample[k] - target[k]);
					}

					float difference = cost + weightOfDis / denominator;

					if (difference < lowestDifference) {
						lowestDifference = difference;
						indexOfLowestDifference = i;
					}
				}
			}
		}
		vector<int> resultVector;
		resultVector.push_back(indexOfLowestDifference);
		resultVector.push_back(lowestDifference);
		return resultVector;
	}
}