#include <sstream>
#include <string>
#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cv.h>

using namespace cv;

// initial default min and max HSV filter values
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;

// program defaults
const int FRAME_WIDTH = 400;
const int FRAME_HEIGHT = 300;
const int MAX_NUM_OBJECTS=50;
const int MIN_OBJECT_AREA = 20*20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH / 1.5;

// gui window titles
const string windowName = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";

void on_trackbar( int, void* ) {
    // gets called whenever a trackbar position is changed
}

string intToString(int number) {
	std::stringstream ss;
	ss << number;
	return ss.str();
}

void createTrackbars() {
	//create window for trackbars
    namedWindow(trackbarWindowName,0);

	//create memory to store trackbar name on window
	char TrackbarName[50];
	sprintf(TrackbarName, "H_MIN", H_MIN);
	sprintf(TrackbarName, "H_MAX", H_MAX);
	sprintf(TrackbarName, "S_MIN", S_MIN);
	sprintf(TrackbarName, "S_MAX", S_MAX);
	sprintf(TrackbarName, "V_MIN", V_MIN);
	sprintf(TrackbarName, "V_MAX", V_MAX);

	// create trackbars and insert them into window
	// 3 parameters are:
	// - address of variable that is changing when trackbar is moved
	// - max value of trackbar can move (eg. H_HIGH),
	// - function that is called whenever the trackbar is moved
	createTrackbar("H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar);
	createTrackbar("H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar);
	createTrackbar("S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar);
	createTrackbar("S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar);
	createTrackbar("V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar);
	createTrackbar("V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar);
}

void drawObject(int x, int y,Mat &frame){
	// draw crossairs on tracked objects
	circle(frame,Point(x, y),20,Scalar(0,255,0),2);
	line(frame,Point(x, y - 5), Point(x, y - 25), Scalar(0, 255, 0), 2);
	line(frame,Point(x, y + 5), Point(x, y + 25), Scalar(0, 255, 0), 2);
	line(frame,Point(x - 5, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	line(frame,Point(x + 5, y), Point(x + 25, y), Scalar(0, 255, 0), 2);

	putText(
		frame,
		intToString(x) + "," + intToString(y),
		Point(x, y + 30),
		1,
		1,
		Scalar(0, 255, 0),
        2
	);
}

void morphOps(Mat &thresh){
	//create structuring element that will be used to "dilate" and "erode" image.
	//the element chosen here is a 3px by 3px rectangle
    Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));

    //dilate with larger element so make sure object is nicely visible
    Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

	erode(thresh, thresh,erodeElement);
	erode(thresh, thresh,erodeElement);

	dilate(thresh, thresh,dilateElement);
	dilate(thresh, thresh,dilateElement);
}

void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed) {
	Mat temp;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	double refArea = 0;
	bool objectFound = false;

	threshold.copyTo(temp);

	// these two vectors needed for output of findContours
	// find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	// use moments method to find our filtered object
	if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();

		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if (numObjects<MAX_NUM_OBJECTS) {
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {
				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				// if the area is less than 20 px by 20px then it is probably just noise
				// if the area is the same as the 3/2 of the image size, probably just a bad filter
				// we only want the object with the largest area so we safe a reference area each
				// iteration and compare it to the area in the next iteration.
				if (area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea) {
					x = moment.m10 / area;
					y = moment.m01 / area;
					objectFound = true;
				} else {
					objectFound = false;
				}
			}
			//let user know you found an object
			if (objectFound == true) {
				putText(cameraFeed,
					"Tracking Object",
					Point(0, 50),
					2,
					1,
					Scalar(0, 255, 0),
					2
				);
				drawObject(x, y, cameraFeed);
			}
		} else {
			putText(cameraFeed,
                "TOO MUCH NOISE! ADJUST FILTER",
                Point(0,50),
                1,
                2,
                Scalar(0,0,255),
                2
			);
		}
	}
}

int main(int argc, char* argv[])
{
	int x = 0;
	int y = 0;
	bool trackObjects = true;
	bool useMorphOps = false;
	Mat cameraFeed;
	Mat HSV;
	Mat threshold;
	VideoCapture capture;

	//create slider bars for HSV filtering and open video capture
	createTrackbars();
	capture.open(0);

	//set height and width of capture frame
	capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

	while(1){
		capture.read(cameraFeed);
		cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);  // convert from BGR to HSV

		// filter HSV image between values and store filtered image to
		// threshold matrix
		inRange(HSV,Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX),threshold);

		// perform morphological operations on thresholded image to eliminate noise
		// and emphasize the filtered object(s)
		if(useMorphOps)
            		morphOps(threshold);

		// pass in thresholded frame to our object tracking function
		// this function will return the x and y coordinates of the
		// filtered object
		if(trackObjects)
			trackFilteredObject(x, y, threshold, cameraFeed);

		// show frames
		imshow(windowName2, threshold);
		imshow(windowName, cameraFeed);
		imshow(windowName1, HSV);

		// delay 30ms so that screen can refresh.
		// image will not appear without this waitKey() command
		waitKey(30);
	}

	return 0;
}
