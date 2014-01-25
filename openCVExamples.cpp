//============================================================================
// Name        : opencvTest.cpp
// Author      : Lucas
// Version     :
// Copyright   : Your copyright notice
// Description : Test space for opencv
//============================================================================

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;

//////////////////////////////////////////////////////////////////////////////////////
//		Salt																	    //
//////////////////////////////////////////////////////////////////////////////////////
//	This image accesses the image and turns random pixels white 'freq' number of 	//
// times.																			//
//////////////////////////////////////////////////////////////////////////////////////

void salt(Mat &image, int freq){

	for (int index = 0; index < freq; index++){

		int i=rand()%image.cols;
		int j=rand()%image.rows;

		if (image.channels()==1){ 						//gray level image
			image.at<uchar>(j,i) = 255;
		}

		if (image.channels()==3){						//color image
			image.at<Vec3b>(j,i)[0] = 255;
			image.at<Vec3b>(j,i)[1] = 255;
			image.at<Vec3b>(j,i)[2] = 255;
		}
	}

}

//////////////////////////////////////////////////////////////////////////////////////
//		reduceColor																	//
//////////////////////////////////////////////////////////////////////////////////////
//	Reduces the color of an image. . The format is (image, the reduction factor).	//					//
//																					//
//////////////////////////////////////////////////////////////////////////////////////

// This code uses iterators to loop over all pixels.

void reduceColorV3(Mat &image, int div=64){

	if (image.channels() == 3){		//color image
		MatIterator_<Vec3b> start = image.begin<Vec3b>();			//iterator at begin position
		MatIterator_<Vec3b> end = image.end<Vec3b>();				//iterator at end pos.

		// loop over all pixels
		while (start != end){			//why do these need the * ?
			// pixel processing
				(*start)[0] = (((*start)[0]/div)*div)+(div/2);
				(*start)[1] = (((*start)[1]/div)*div)+(div/2);
				(*start)[2] = (((*start)[2]/div)*div)+(div/2);
				// move to next pixel
				start++;
		}
	}

	if (image.channels() == 1){		//grayscale image
			MatIterator_<uchar> start = image.begin<uchar>();			//iterator at begin position
			MatIterator_<uchar> end = image.end<uchar>();				//iterator at end pos.

			// loop over all pixels
			while (start != end){			//why do these need the * ?
				// pixel processing
					(*start) = (((*start)/div)*div)+(div/2);
					// move to next pixel
					start++;
			}
		}
}

//This code runs twice as fast as my code.

void reduceColorV2(Mat &image, int div=64) {
	int nl= image.rows; // number of lines
	// total number of elements per line
	int nc= image.cols * image.channels();
	for (int j=0; j<nl; j++) {
		// get the address of row j
		uchar* data= image.ptr<uchar>(j);
		for (int i=0; i<nc; i++) {
			// process each pixel ---------------------
			data[i]=data[i]/div*div + div/2;

			// end of pixel processing ----------------
		} // end of line
	}
}
// This is my code

void reduceColor(Mat &image, int div){
	int nr = image.rows;						// number of rows in the picture.
	int nc = image.cols;						// number of collumns in the picture.

	if (image.channels() == 3){							// image is in color
		// goes through each row
		for (int indexR = 0; indexR < nr; indexR++){			// goes through each row

			for (int indexC = 0; indexC < nc; indexC++){		// goes through each column in that row
				int dataB = image.at<Vec3b>(indexR,indexC)[0];
				int dataG = image.at<Vec3b>(indexR,indexC)[1];
				int dataR = image.at<Vec3b>(indexR,indexC)[2];

				dataB = ((dataB/div)*div)+(div/2);
				dataG = ((dataG/div)*div)+(div/2);
				dataR = ((dataR/div)*div)+(div/2);
				// this feels like an extra step that
				// could be solved with a reference...

				image.at<Vec3b>(indexR,indexC)[0]=dataB;
				image.at<Vec3b>(indexR,indexC)[1]=dataG;
				image.at<Vec3b>(indexR,indexC)[2]=dataR;

			}
		}
	}

	if (image.channels() == 1){			// image is grayscale
		for (int indexR = 0; indexR < nr; indexR++){			//go through each row
			for (int indexC = 0; indexC < nc; indexC++){		//go through each column in that row
					int data = image.at<uchar>(indexR,indexC);

					data = ((data/div)*div)+(div/2);

					// this feels like an extra step that
					// could be solved with a reference...
					image.at<Vec3b>(indexR,indexC)[0]=data;
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////
//	sharenImg (input image, output image);											  //
////////////////////////////////////////////////////////////////////////////////////////
// Sharpens the image. Fix with iterators....																  //
////////////////////////////////////////////////////////////////////////////////////////

void sharpenImg(Mat image, Mat &imageOut){
	imageOut = image.clone();
	int nr = imageOut.rows;
	int nc = imageOut.cols;

	for (int indexR = 1; indexR < nr-1; indexR++){
		// go through each row except for 1st and last
		for (int indexC = 1; indexC < nc-1; indexC++){
		// go through each col except for 1st and last
			int leftB = image.at<Vec3b>(indexR,indexC-1)[0];
			int leftG = image.at<Vec3b>(indexR,indexC-1)[1];
			int leftR = image.at<Vec3b>(indexR,indexC-1)[2];

			int curB = image.at<Vec3b>(indexR,indexC)[0];
			int curG = image.at<Vec3b>(indexR,indexC)[1];
			int curR = image.at<Vec3b>(indexR,indexC)[2];

			int rightB = image.at<Vec3b>(indexR,indexC+1)[0];
			int rightG = image.at<Vec3b>(indexR,indexC+1)[1];
			int rightR = image.at<Vec3b>(indexR,indexC+1)[2];

			int downB = image.at<Vec3b>(indexR-1,indexC)[0];
			int downG = image.at<Vec3b>(indexR-1,indexC)[1];
			int downR = image.at<Vec3b>(indexR-1,indexC)[2];

			int upB = image.at<Vec3b>(indexR+1,indexC)[0];
			int upG = image.at<Vec3b>(indexR+1,indexC)[1];
			int upR = image.at<Vec3b>(indexR+1,indexC)[2];

			imageOut.at<Vec3b>(indexR,indexC)[0] = saturate_cast<uchar>((curB*5)-leftB-rightB-upB-downB);
			imageOut.at<Vec3b>(indexR,indexC)[1] = saturate_cast<uchar>((curG*5)-leftG-rightG-upG-downG);
			imageOut.at<Vec3b>(indexR,indexC)[2] = saturate_cast<uchar>((curR*5)-leftR-rightR-upR-downR);

		}
	}

}

// accomplishes the same thing but uses a kernel instead. Code is much nicer.
// also, about 3 times as fast.
void sharpenImgV2(Mat image, Mat &imageOut){
	Mat kernel(3,3,CV_32F,Scalar(0));
	kernel.at<float>(1,1)= 5;
	kernel.at<float>(0,1)=-1;
	kernel.at<float>(1,0)=-1;
	kernel.at<float>(1,2)=-1;
	kernel.at<float>(2,1)=-1;

	filter2D(image,imageOut,image.depth(),kernel);
}

///////////////////////////////////////////////////////////////////////////////////////////
//	ColorDetect ( the input image, target color, and tolerance of that color			 //
///////////////////////////////////////////////////////////////////////////////////////////
//	This function returns a binary map of the image with the selected image.			 //
//The distance var is caculated by summing the absolute difference of the RGB values.    //
//																						 //
///////////////////////////////////////////////////////////////////////////////////////////

Mat colorDetect(Mat &image, Vec3b target, int tolerance){
	// initialize the output
	//----------------------------------------------------------------------------------
	Mat output;													// the return image of the method
	output.create(image.rows, image.cols, CV_8U);		// init the output (especially for cases where function called more than once);
	// initialize iterators
	MatIterator_<Vec3b> imCur = image.begin<Vec3b>();			// the o.g image iterator, points to the current pixel. Init at the begin position
	MatIterator_<Vec3b> imEnd = image.end<Vec3b>();				// the o.g image iterator, set at the end position

	MatIterator_<uchar> outCur = output.begin<uchar>();		//output image iterator, point to current pixel. Init at begin position


	//	Loop through all pixels
	//------------------------------------------------------------------------------------
	while (imCur != imEnd){
		// get distance
		// -------------------------------------------------------------------------------
		int colorDistance =
				abs((*imCur)[0]- target[0])+
				abs((*imCur)[1]- target[1])+
				abs((*imCur)[2] -target[2]);
		// create the binary map
		//---------------------------------------------------------------------------------
		if (colorDistance > tolerance){
			(*outCur) = 0;
		} else {
			(*outCur) = 255 ;
		}

		// go to next pixel
		//---------------------------------------------------------------------------------
		imCur++;												//next pixel on the img
		outCur++;												//next pixel on output
	}

	return(output);

}

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

Mat detectShapes(Mat src){
	Mat tmp,tmp2;
	Mat dst;
	// get image ready
	cvtColor(src,tmp, CV_BGR2GRAY);
	// create binary map of edges using src
	Canny(tmp,tmp2, 200, 250, 3);
	namedWindow("tmp");
	imshow("tmp",tmp2);
	waitKey(0);
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



//--------------------------------Below Functions Are Dumb ------------------------------//




////////////////////////////////////////////////////////////////////////////////////////////
//		getHistogram*																	  //
////////////////////////////////////////////////////////////////////////////////////////////
// Computes the histogram of a color image. locMixMax does not work for color images	  //											  //
////////////////////////////////////////////////////////////////////////////////////////////

// getSuitable(image) creates a bw image that is half the size of the original

Mat getSuitable(Mat image){
	Mat tmp,output;
	pyrDown (image,tmp, Size(image.cols/2,image.rows/2));
	cvtColor(image,output, CV_BGR2GRAY);
	return (output);
}

Mat getHistogram(Mat image){
	//Computes Histogram
	//------------------------------------------------------------------------------------

	int imageChannels[3];
	imageChannels[0] = 0;
	imageChannels[1] = 1;
	imageChannels[2] = 2;

	int histSize[3];
	histSize[0] = histSize[1] = histSize[2] = 256;

	float specRanges[2];
	specRanges[0] = 0.0;
	specRanges[1] = 255.0;

	const float* ranges[3];
	ranges[0] = specRanges;
	ranges[1] = specRanges;
	ranges[2] = specRanges;

	MatND hist;
	Mat output;

	calcHist(&image,
			1, 				// histogram of 1 image only
			imageChannels, // the channel used
			Mat(),		// no mask is used
			hist,			// the resulting histogram
			3,				// it is a 3D histogram
			histSize,		// number of bins
			ranges 			// pixel value range
			);

	// draws histogram
	//-----------------------------------------------------------------------------------

	// not really following this code 100%

	// Get min and max bin values
	double maxVal = 0.0;
	double minVal = 0.0;
	/// this peice of shit function throws an exception when i call it.
	cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);


	// Image on which to display histogram
	Mat histImg(histSize[0], histSize[0],CV_8U,cv::Scalar(255));

	// set highest point at 90% of nbins
	int hpt = static_cast<int>(0.9*histSize[0]);

	// Draw a vertical line for each bin
	for( int h = 0; h < histSize[0]; h++ ) {

		float binVal = hist.at<float>(h);
		int intensity = static_cast<int>(binVal*hpt/maxVal);

		// This function draws a line between 2 points
		line(histImg,Point(h,histSize[0]),
				Point(h,histSize[0]-intensity),
				Scalar::all(0));
	}

	return (output);
}



int main() {

	double duration = static_cast<double>(getTickCount()); //the start time

	//place code here


	duration = static_cast<double>(getTickCount())-duration;
	duration = ( duration/getTickFrequency() )* 1000;			//the runtime in secnods
	cout << duration;




	return(0);

}

