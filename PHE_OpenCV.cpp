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
Mat IMAGE1;
Mat IMAGE2;
Mat IMAGE3;

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

};

class TrainSuit{
    public:
		Mat img;
		String name;

};

vector<TrainRank> loadRanks(String path){
	vector<TrainRank> trainRanks;
	vector<String> ranks{"Ace","Two","Three","Four","Five","Six","Seven", "Eight","Nine","Ten","Jack","Queen","King"};

	for(int i=0; i<13; i++){
		TrainRank tr;
		tr.name = ranks.at(i);
		String filename = path+ranks.at(i)+".jpg";
		tr.img = imread(filename, IMREAD_GRAYSCALE);
		trainRanks.push_back(tr);
	}

	return trainRanks;
}

vector<TrainSuit> loadSuits(String path){
	vector<TrainSuit> trainSuits;
	vector<String> suits{"Diamonds", "Clubs", "Hearts", "Spades"};

	for(int i=0; i<suits.size(); i++){
		TrainSuit ts;
		ts.name = suits.at(i);
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

	// Grab corner of warped card image and do a 4x zoom
	Mat corner = qCard.warp(Rect(5, 5, 32, 84));
	resize(corner, corner, Size(), 4, 4);

	// Sample known white pixel intensity to determine good threshold level
	Mat queryThresh;
	int threshLevel = corner.at<uchar>(31, 2) - 30;
	if(threshLevel <= 0) threshLevel = 1; 
	threshold(corner, queryThresh, threshLevel, 255, THRESH_BINARY);

	// Split in to top and bottom half (top shows rank, bottom shows suit)
	Mat qRank = queryThresh(Rect(10, 10, 100, 160));
	Mat qSuit = queryThresh(Rect(10, 155, 96, 120));

	imshow("Imagem1", qRank);

	// Get negative of rank and suit images
	Mat qRankInv = 255 - qRank;
	Mat qSuitInv = 255 - qSuit;

	imshow("Imagem2", qRankInv);


	/* Mat kernel = getStructuringElement(MORPH_RECT, Size(1, 1));
    for(int i=0; i<1; i++){
        dilate(qRankInv, qRankInv, kernel);
    } */

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
	Returns a string with value (rank) and suit. ex: "Ace Clubs"
 */
String getMatchCard(QueryCard qCard, vector<TrainRank> trainRanks, vector<TrainSuit> trainSuits){
	int bestRankMatchDiff = 10000;
    int bestSuitMatchDiff = 10000;
    String bestRankMatchName = "Unknown";
    String bestSuitMatchName = "Unknown";

	Mat MinDiff;

	for(int i=0; i<trainRanks.size(); i++){
		TrainRank tRank = trainRanks.at(i);
		Mat diffRankImg;
		absdiff(qCard.rank_img, tRank.img, diffRankImg);
		int rankDiff = sum(diffRankImg).val[0] / 255;

		if(rankDiff < bestRankMatchDiff){
			bestRankMatchDiff = rankDiff;
			bestRankMatchName = tRank.name;
			//MinDiff = diffRankImg;
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
			MinDiff = diffSuitImg;
		}
	}
	
	imshow("Imagem0", MinDiff);
	//cout << String(bestRankMatchName + " | " + bestSuitMatchName); 
	return bestRankMatchName + " " + bestSuitMatchName;
}

void mouseEvent(int event, int x, int y, int flags, void* userData){
	if  ( event == EVENT_LBUTTONDOWN ){
		Mat* img = (Mat*)userData;
		Mat i = *img;

        //cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        cout << "Value: " << to_string(i.at<char>(Point(x, y))) << endl;
    }
}

int main( void )
{
	vector<TrainRank> trainRanks = loadRanks("Card_Imgs/");
	vector<TrainSuit> trainSuits = loadSuits("Card_Imgs/");

	// Create a window to display the image	
	namedWindow( "Imagem", cv::WINDOW_AUTOSIZE );
	namedWindow( "Imagem0", cv::WINDOW_AUTOSIZE );
	namedWindow( "Imagem1", cv::WINDOW_AUTOSIZE );
	namedWindow( "Imagem2", cv::WINDOW_AUTOSIZE );
	namedWindow( "Imagem3", cv::WINDOW_AUTOSIZE );
/*	namedWindow( "Imagem4", cv::WINDOW_AUTOSIZE );
	namedWindow( "Imagem5", cv::WINDOW_AUTOSIZE );
	namedWindow( "Imagem6", cv::WINDOW_AUTOSIZE );
*/

	IMAGE1 = imread( "test_images/poker_hand5.jpg", IMREAD_UNCHANGED );

	resize(IMAGE1, IMAGE1, Size(), 0.4, 0.4);
	//rotate(IMAGE1, IMAGE1, ROTATE_90_CLOCKWISE);

	if( IMAGE1.empty() )
	{
		// NOT SUCCESSFUL : the data attribute is empty
		cout << "Image file could not be open !!" << endl;

		return -1;
	}

	cvtColor(IMAGE1, IMAGE2, COLOR_BGR2GRAY);

	//imshow("Imagem3", IMAGE2);

	//GaussianBlur( IMAGE2, IMAGE2, Size( 25, 15 ), 0, 0 );
	//medianBlur( IMAGE2, IMAGE2, 6);

	// REFACTOR: threshold value shouldn't be static...
	threshold(IMAGE2, IMAGE2, 120, 255, THRESH_BINARY_INV);
	
	imshow("Imagem3", IMAGE2);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	for(int i=0; i<1; i++){
		erode(IMAGE2, IMAGE2, kernel);
	}

	/// Find contours   
	vector<vector<Point> > contours;
	vector<vector<Point> > cardsContours;
	vector<Vec4i> hierarchy;
	findContours( IMAGE2, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );
	/// Draw contours
	Scalar color = Scalar( 255, 0, 0 );
	
	if(contours.size() > 0){
		for( int i = 0; i< contours.size(); i++ )
		{
			if(contourArea(contours[i], false) > 70000 && contourArea(contours[i], false) < 150000){
				drawContours( IMAGE1, contours, i, color, 2, 8 );
				cardsContours.push_back(contours[i]);
			}
		}
	}else{
		cout << "No countours found";
	}

	if(cardsContours.size() > 0){
		for(int i=0; i<cardsContours.size(); i++){
			QueryCard card = preprocess_card(cardsContours[i], IMAGE1);
			//imshow( "Imagem"+to_string(i), img);

			String a = getMatchCard(card, trainRanks, trainSuits);
			cout << a + "\n";
		}
	}else{
		cout << "No cards found";
	}
	

	imshow( "Imagem", IMAGE1 );

	Rank rank1 = EvaluateCards("9c", "4c", "4s", "9d", "4h", "Qc", "6c");
	Rank rank2 = EvaluateCards("9c", "4c", "4s", "9d", "4h", "2c", "9c");

	assert(rank1 < rank2);

	// Wait for a pressed key

    waitKey( 0 );

	// Destroy the window --- Actually not needed in such a simple program

	destroyWindow( "Display window" );

	return 0;
}