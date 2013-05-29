#include <sstream>
#include <string>
#include <iostream>

#include <opencv/highgui.h>
#include <opencv/cv.h>

#define FRAME_WIDTH 400
#define FRAME_HEIGHT 300

using namespace cv;
using namespace std;

int detectNumberOfCameras()
{
    int num = 0;
    int finished = 0;

    while(finished == false) {
        CvCapture* capture = cvCreateCameraCapture(num);

        if (!capture) {
            finished = 1;
        } else {
            num++;
            cvReleaseCapture(&capture);
        }
    }

    return num;
}

int main(int argc, char* argv[])
{
    int num_cams = detectNumberOfCameras();
    Mat camera_feed_1;
    Mat camera_feed_2;
	VideoCapture capture_1;
	VideoCapture capture_2;

    cout << "Number of cameras: " << num_cams << endl;
    if (num_cams != 2) {
        cout << "For stereo vision to work you need 2* cameras!" << endl;
        exit(-1);
    }

    // open cameras
    capture_1.open(0);
    capture_2.open(1);

	//set height and width of capture frame
	capture_1.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	capture_2.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

    while(1) {
        // read video streams
		capture_1.read(camera_feed_1);
		capture_2.read(camera_feed_2);

        // display camera feeds
		imshow("Camera 1", camera_feed_1);
		imshow("Camera 2", camera_feed_2);

		// delay 30ms so that screen can refresh.
		waitKey(30);  // IMPORTANT!! IMAGE WILL NOT DISPLAY WITHOUT IT!
    }

    return 0;
}
