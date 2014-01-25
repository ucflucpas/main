//============================================================================
// Name        : detectShapes.cpp
// Author      : Lucas Pasqualin
// Version     : v1
// Copyright   : Your copyright notice
// Description : Find shapes in video stream.
//============================================================================
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

////////////////////////////////////////////////////////////////////////////////////////////
//		detectShapes																	  //
////////////////////////////////////////////////////////////////////////////////////////////
//		Detects shapes.																	  //
////////////////////////////////////////////////////////////////////////////////////////////


// not mine! helper function to set label in the middle of the contours
void setLabel(cv::Mat& im, const std::string label, std::vector<cv::Point>& contour)
{
        int fontface = cv::FONT_HERSHEY_SIMPLEX;
        double scale = 0.4;
        int thickness = 1;
        int baseline = 0;

        cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
        cv::Rect r = cv::boundingRect(contour);

        cv::Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
        cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), CV_RGB(255,255,255), CV_FILLED);
        cv::putText(im, label, pt, fontface, scale, CV_RGB(0,0,0), thickness, 8);
}

//detects shapes in an image

Mat detectShapesImg(Mat src){
	Mat tmp,tmp2;
	Mat dst;
	// get image ready
	cvtColor(src,tmp, CV_BGR2GRAY);
	// create binary map of edges using src
	Canny(tmp,tmp2, 150, 250, 3);
	namedWindow("tmp");
	imshow("tmp",tmp2);
	// get ready to detect contours
	vector< vector< Point> > contours;
	findContours(tmp2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); // what are last two paramaters
	vector<Point> approx;		// a place to store approximation curve..
	dst = src.clone();
	for (int i = 0; i < contours.size(); i++){
		approxPolyDP(
				Mat(contours[i]),
				approx,
				arcLength(Mat(contours[i]), true) * 0.02,
				true
				);
		// approx now holds the vertices of the shapes.
		//skips small objects
		if (fabs(contourArea(contours[i])) < 100 || !isContourConvex(approx)){
			continue;
		}
		// recognizes shapes
		if (approx.size() == 3){
			setLabel(dst, "TRI", contours[i]);
		}
		if (approx.size() == 4){
			setLabel(dst, "SQR/RCT", contours[i]);
		}
		if (approx.size() == 5){
			setLabel(dst, "PENT", contours[i]);
		}
		if (approx.size() == 6){
			setLabel(dst, "HEX", contours[i]);
		}
		if (approx.size() >= 7){
			setLabel(dst, "CIR", contours[i]);
		}
	}
	return(dst);
}

// detects shapes in videos. dependent of detectShapesImg and setLabel. input which camera to use.

void detectShapesVid(int cam){
	namedWindow("output", CV_WINDOW_AUTOSIZE);

	VideoCapture cap;									// The video capture
	Mat frame;											// each frame is stored here on each run

	cap.open(cam);

	while (true){										//refreshes each frame
		if (cap.isOpened()){
			Mat frame;
			Mat tmpFrame;
			cap.read(frame);
			tmpFrame =detectShapesImg(frame);
			imshow("output", tmpFrame);
			if (waitKey(10) >= 0){						// escapes when a key is pressed.
				break;
			}
		}
		if (!cap.isOpened()){
			cout << "Video Capture failed." << endl;
			break;
		}
	}//end loop
}

int main(){

	detectShapesVid(0);
}
