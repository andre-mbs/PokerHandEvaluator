/*
 *	Computação Visual - OpenCV
 *	Trabalho 2 - Poker Hand Evaluator	
 *		
 *	References:
 * 		- https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector
 * 		- https://github.com/HenryRLee/PokerHandEvaluator
 *
 *	André Santos (84816) - andrembs@ua.pt
 */

#include <iostream>
#include <math.h>

// Basic OpenCV functionalities
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "phevaluator/phevaluator.h"

using namespace std;
using namespace cv;
using namespace phevaluator;

// To store an image
Mat IMAGE2;
Mat IMAGE3;

Mat ranks_inv;
Mat suits_inv;
Mat diffsRank;
Mat diffsSuit;

class QueryCard{
    public:
        vector<Point> contour;
        vector<Point2f> cornerPoints;
		Point center;
		int width;
		int height;
		Mat warp;
		Mat rank_img;
		Mat suit_img;

};

class TrainRank{
    public:
		Mat img;
		String name;
		String shortName;
};

class TrainSuit{
    public:
		Mat img;
		String name;
		String shortName;
};

vector<TrainRank> loadRanks(String path){
	vector<TrainRank> trainRanks;
	vector<String> ranks{"Ace","Two","Three","Four","Five","Six","Seven", "Eight","Nine","Ten","Jack","Queen","King"};
	vector<String> ranksShort{"A","2","3","4","5","6","7", "8","9","T","J","Q","K"};

	for(int i=0; i<13; i++){
		TrainRank tr;
		tr.name = ranks.at(i);
		tr.shortName = ranksShort.at(i);
		String filename = path+ranks.at(i)+".jpg";
		tr.img = imread(filename, IMREAD_GRAYSCALE);
		trainRanks.push_back(tr);
	}

	return trainRanks;
}

vector<TrainSuit> loadSuits(String path){
	vector<TrainSuit> trainSuits;
	vector<String> suits{"Diamonds", "Clubs", "Hearts", "Spades"};
	vector<String> suitsShort{"d", "c", "h", "s"};

	for(int i=0; i<suits.size(); i++){
		TrainSuit ts;
		ts.name = suits.at(i);
		ts.shortName = suitsShort.at(i);
		String filename = path+suits.at(i)+".jpg";
		ts.img = imread(filename, IMREAD_GRAYSCALE);
		trainSuits.push_back(ts);
	}

	return trainSuits;
}

/* 	Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
*/
Mat flattener(Mat image, vector<Point2f> cornerPoints, Point origin, int width, int height){
	// Sort corner point clockwise starting top left
	sort(cornerPoints.begin(), cornerPoints.end(), [origin](Point first, Point second)
		{
			return atan2(first.y - origin.y, first.x - origin.x) < atan2(second.y - origin.y, second.x - origin.x);
		});

	int maxWidth = 200;
	int maxHeight = 300;

	// Create destination array, calculate perspective transform matrix, and warp card image
	vector<Point2f> dst;
	dst.push_back(Point2f(0,0));
	dst.push_back(Point2f(maxWidth-1,0));
	dst.push_back(Point2f(maxWidth-1,maxHeight-1));
	dst.push_back(Point2f(0,maxHeight-1));

	Mat warp;
	Mat m = getPerspectiveTransform(cornerPoints, dst);
	warpPerspective(image, warp, m, Size(maxWidth, maxHeight));

	// Convert to greysacle
	cvtColor(warp, warp, COLOR_BGR2GRAY);

	return warp;
}

QueryCard preprocess_card(vector<Point> contour, Mat image){
    QueryCard qCard;
    qCard.contour = contour;

    // Find perimeter of card and use it to approximate corner points
    vector<Point2f> approx;
    double perimeter = arcLength(contour, true);
    approxPolyDP(contour, approx, 0.02*perimeter, true);
	qCard.cornerPoints = approx;

    // Find width and height of card's bounding rectangle
	Rect boundRect = boundingRect(approx);
	qCard.width = boundRect.width;
	qCard.height = boundRect.height;

	/* Mat cropped = image(boundRect);
	imshow("Cropped Card", cropped); */

	// Find center point of card by taking x and y average of the four corners
	Point sum = Point(0, 0); 
	for(int i=0; i<approx.size(); i++){
		sum.x += approx.at(i).x;
		sum.y += approx.at(i).y;
	}
	sum.x /= approx.size();
	sum.y /= approx.size();
	qCard.center = sum;

	// Warp card into 200x300 flattened image using perspective transform
	qCard.warp = flattener(image, approx, qCard.center, qCard.width, qCard.height);
	imshow("Cropped Card", qCard.warp);

	// Grab corner of warped card image and do a 4x zoom
	Mat corner = qCard.warp(Rect(5, 5, 32, 84));
	resize(corner, corner, Size(), 4, 4);

	// Sample known white pixel intensity to determine good threshold level
	Mat queryThresh;
	int threshLevel = corner.at<uchar>(31, 2) - 30;
	if(threshLevel <= 0) threshLevel = 1; 
	threshold(corner, queryThresh, threshLevel, 255, THRESH_BINARY);

	// Split in to top and bottom half (top shows rank, bottom shows suit)
	Mat qRank = queryThresh(Rect(5, 10, 90, 160));
	Mat qSuit = queryThresh(Rect(1, 165, 80, 130));

	// Get negative of rank and suit images
	Mat qRankInv = 255 - qRank;
	Mat qSuitInv = 255 - qSuit;

	// Find rank contours
    vector<vector<Point> > rankContours;
    vector<Vec4i> hierarchyR;
    findContours( qRankInv, rankContours, hierarchyR, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );

	// Sort rank contours to get the largest
	sort(rankContours.begin(), rankContours.end(), [](vector<Point> c1, vector<Point> c2)
	{
		return contourArea(c1, false) > contourArea(c2, false);
	});

	// Find bounding rectangle for largest contour, use it to resize query rank
    // image to match dimensions of the train rank image
	Rect rankBoundRect = boundingRect(rankContours.at(0));
	Mat qRankSized;
	resize(qRankInv(rankBoundRect), qRankSized, Size(70, 125));
	qCard.rank_img = qRankSized;

	// Find suit contours
    vector<vector<Point> > suitContours;
    vector<Vec4i> hierarchyS;
    findContours( qSuitInv, suitContours, hierarchyS, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );

	// Sort suit contours to get the largest
	sort(suitContours.begin(), suitContours.end(), [](vector<Point> c1, vector<Point> c2)
	{
		return contourArea(c1, false) > contourArea(c2, false);
	});

	// Find bounding rectangle for largest contour, use it to resize query suit
    // image to match dimensions of the train suit image
	Rect suitBoundRect = boundingRect(suitContours.at(0));
	Mat qSuitSized;
	resize(qSuitInv(suitBoundRect), qSuitSized, Size(70, 100));
	qCard.suit_img = qSuitSized;

	hconcat(ranks_inv, qCard.rank_img, ranks_inv);
	hconcat(suits_inv, qCard.suit_img, suits_inv);

	// DEBUG
    for(int i=0; i<approx.size(); i++){
		circle(image, approx.at(i), 3, CV_RGB(0, 255, 0), 2);
    }
	circle(image, qCard.center, 3, CV_RGB(0, 0, 255), 2);
	
	return qCard;
}

/*
	Finds best rank and suit matches for the query card. Differences
    the query card rank and suit images with the train rank and suit images.
    The best match is the rank or suit image that has the least difference.
	Returns a vector of strings with value (rank), suit and short name.
	ex: {"Seven", "Clubs", "7c"}
 */
vector<String> getMatchCard(QueryCard qCard, vector<TrainRank> trainRanks, vector<TrainSuit> trainSuits){
	int bestRankMatchDiff = 10000;
    int bestSuitMatchDiff = 10000;
    String bestRankMatchName = "Unknown";
    String bestRankMatchShortName = "Unknown";
    String bestSuitMatchName = "Unknown";
    String bestSuitMatchShortName = "Unknown";

	Mat minRankDiff;
	Mat minSuitDiff;

	for(int i=0; i<trainRanks.size(); i++){
		TrainRank tRank = trainRanks.at(i);
		Mat diffRankImg;
		absdiff(qCard.rank_img, tRank.img, diffRankImg);
		int rankDiff = sum(diffRankImg).val[0] / 255;

		if(rankDiff < bestRankMatchDiff){
			bestRankMatchDiff = rankDiff;
			bestRankMatchName = tRank.name;
			bestRankMatchShortName = tRank.shortName;
			minRankDiff = diffRankImg;
		}
	}

	for(int i=0; i<trainSuits.size(); i++){
		TrainSuit tSuit = trainSuits.at(i);
		Mat diffSuitImg;
		absdiff(qCard.suit_img, tSuit.img, diffSuitImg);
		int suitDiff = sum(diffSuitImg).val[0] / 255;

		if(suitDiff < bestSuitMatchDiff){
			bestSuitMatchDiff = suitDiff;
			bestSuitMatchName = tSuit.name;
			bestSuitMatchShortName = tSuit.shortName;
			minSuitDiff = diffSuitImg;
		}
	}

	//cout << minSuitDiff.size().height << endl;

	hconcat(diffsRank, minRankDiff, diffsRank);
	hconcat(diffsSuit, minSuitDiff, diffsSuit);

	vector<String> result;
	result.push_back(bestRankMatchName);
	result.push_back(bestSuitMatchName);
	result.push_back(String(bestRankMatchShortName + bestSuitMatchShortName));
	return result;
}

void mouseEvent(int event, int x, int y, int flags, void* userData){
	if  ( event == EVENT_LBUTTONDOWN ){
		Mat* img = (Mat*)userData;
		Mat i = *img;

        cout << "Value: " << to_string(i.at<char>(Point(x, y))) << endl;
    }
}

int main( int argc, char* argv[] )
{
	if(argc != 2){
		cerr << "Usage: " << argv[0] << " <poker_hand_image.jpg>" << std::endl;
		return 0;
	}

	vector<TrainRank> trainRanks = loadRanks("Card_Imgs/");
	vector<TrainSuit> trainSuits = loadSuits("Card_Imgs/");

	suits_inv = Mat(trainSuits.at(0).img.size().height, 1, trainSuits.at(0).img.type(), Scalar(0, 0, 0));
	ranks_inv = Mat(trainRanks.at(0).img.size().height, 1, trainRanks.at(0).img.type(), Scalar(0, 0, 0));
	diffsSuit = Mat(100, 1, trainSuits.at(0).img.type(), Scalar(0, 0, 0));
	diffsRank = Mat(125, 1, trainRanks.at(0).img.type(), Scalar(0, 0, 0));

	// Create a window to display the image	
	namedWindow( "Card Detection", cv::WINDOW_AUTOSIZE );
	namedWindow( "Grayscale", cv::WINDOW_AUTOSIZE );
	namedWindow( "Threshold: 100", cv::WINDOW_AUTOSIZE );
	namedWindow( "Cropped Card", cv::WINDOW_AUTOSIZE );
	namedWindow( "Diff Rank", cv::WINDOW_AUTOSIZE );
	namedWindow( "Diff Suit", cv::WINDOW_AUTOSIZE );
	namedWindow( "Rank", cv::WINDOW_AUTOSIZE );
	namedWindow( "Suit", cv::WINDOW_AUTOSIZE );

	String fileName = "test_images/" + String(argv[1]);
	Mat frame = imread( fileName, IMREAD_UNCHANGED );
	if( frame.empty() )
	{
		// NOT SUCCESSFUL : the data attribute is empty
		cout << "Capture/Openning error: frame or image not found" << endl;
		return 1;
	}

	resize(frame, frame, Size(), 0.3, 0.3);
	if(frame.size().height > frame.size().width)
		rotate(frame, frame, ROTATE_90_CLOCKWISE);
	
	//imshow( "Imagem", frame );

	// Grayscale image
	cvtColor(frame, IMAGE2, COLOR_BGR2GRAY);
	imshow("Grayscale", IMAGE2);

	IMAGE2 += 20;

	// Threshold image and remove unwanted objects
	threshold(IMAGE2, IMAGE2, 100, 255, THRESH_BINARY_INV);
	morphologyEx(IMAGE2, IMAGE2, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(11, 11)));

	imshow("Threshold: 100", IMAGE2);

	// Find contours   
	vector<vector<Point>> contours;
	vector<vector<Point>> cardsContours;
	vector<Vec4i> hierarchy;
	findContours( IMAGE2, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );

	// Draw contours
	Scalar color = Scalar( 255, 0, 0 );
	if(contours.size() > 0){
		for( int i = 0; i< contours.size(); i++ )
		{
			if(contourArea(contours[i], false) > 70000 && contourArea(contours[i], false) < 150000){
				drawContours( frame, contours, i, color, 2, 8 );
				cardsContours.push_back(contours[i]);
			}
		}
	}else{
		cout << "Detection error: 0 countours found" << endl;
	}
	
	vector<String> hand;
	String detectedCards = "";

	// Pre-process the image and get the rank and suit of the cards
	try
	{
		if(cardsContours.size() > 0){
			for(int i=0; i<cardsContours.size(); i++){
				QueryCard card = preprocess_card(cardsContours[i], frame);
				vector<String> a = getMatchCard(card, trainRanks, trainSuits);
				hand.push_back(a.at(2));
				detectedCards += a.at(2) + " ";
			}
		}else{
			cout << "Detection error: 0 cards found" << endl;
		}
	}
	catch(...)
	{
		cout << "Error" << '\n';
	}

	cout << "Detected cards: " << detectedCards << endl;
	
	imshow("Rank", ranks_inv);
	imshow("Suit", suits_inv);
	imshow("Diff Rank", diffsRank);
	imshow("Diff Suit", diffsSuit);

	if(hand.size() == 7){
		Rank rank1 = EvaluateCards(hand.at(0).c_str(), hand.at(1).c_str(), hand.at(2).c_str(),
								   hand.at(3).c_str(), hand.at(4).c_str(), hand.at(5).c_str(), hand.at(6).c_str());
		cout << "Hand evaluation: " << rank1.describeCategory() + ", " + rank1.describeRank() << endl;
	}else
	{
		cout << "Evaluation error: hand.size() != 7 -> Make sure you have 7 cards" << endl;
	}

	imshow( "Card Detection", frame );

	// Wait for esc key
	while (1)
	{
		int key = waitKey(0);
		switch (key){
		case 27:
			return 0;
		}
	}
}