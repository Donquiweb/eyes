#include <iostream>
#include <string>

#include <opencv/highgui.h>
#include <opencv/cv.h>

#include <dbg/dbg.h>

#define FRAME_WIDTH 400
#define FRAME_HEIGHT 300
#define CAM_1 "Camera 1"
#define CAM_2 "Camera 2"
#define DISPARITY_MAP "Disparity Map"
#define DISPARITY_CONFIGURATOR "Disparity Configurator"

using namespace cv;
using namespace std;


struct calibration
{
    CvMat *intrinsic_matrix;
    CvMat *distortion_coeffs;
};

struct undistort_image
{
    IplImage *map_x;
    IplImage *map_y;
    IplImage *original_image;
    IplImage *clone_image;
};

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

    // check result
    cout << "Number of cameras: " << num << endl;
    if (num != 2) {
        cout << "Failed to detect 2 cameras for stereo vision!" << endl;
        exit(-1);
    }

    return num;
}

void on_trackbar(int, void*) {
    // gets called whenever a trackbar position is changed
}

// void createDisparityConfigurator()
// {
// 	//create window for trackbars
//     cv::namedWindow(DISPARITY_CONFIGURATOR, CV_WINDOW_AUTOSIZE);
//
// 	//create memory to store trackbar name on window
// 	char TrackbarName[50];
//     sprintf(TrackbarName, "SAD Window Size", 23);
//     sprintf(TrackbarName, "Number of Disparities", 96);
//     sprintf(TrackbarName, "Pre Filter Size", 25);
//     sprintf(TrackbarName, "Pre Filter Cap", 63);
//     sprintf(TrackbarName, "Min Disparity", 0);
//     sprintf(TrackbarName, "Texture Threshold", 20);
//     sprintf(TrackbarName, "Uniqueness Ratio", 10);
//     sprintf(TrackbarName, "Speckle WindowSize", 25);
//     sprintf(TrackbarName, "Speckle Range", 8);
//     sprintf(TrackbarName, "Disp 12 MaxDiff", 1);
//
// 	// create trackbars and insert them into window
// 	// 3 parameters are:
// 	// - address of variable that is changing when trackbar is moved
// 	// - max value of trackbar can move
// 	// - function that is called whenever the trackbar is moved
// 	createTrackbar("SAD Window Size", DISPARITY_CONFIGURATOR, &H_MIN, H_MAX, on_trackbar);
// 	createTrackbar("Number of Disparities", DISPARITY_CONFIGURATOR, &S_MIN, S_MAX, on_trackbar);
// 	createTrackbar("Pre Filter Size", DISPARITY_CONFIGURATOR, &V_MIN, V_MAX, on_trackbar);
// 	createTrackbar("Pre Filter Cap", DISPARITY_CONFIGURATOR, &V_MIN, V_MAX, on_trackbar);
// 	createTrackbar("Min Disparity", DISPARITY_CONFIGURATOR, &V_MIN, V_MAX, on_trackbar);
// 	createTrackbar("Texture Threshold", DISPARITY_CONFIGURATOR, &V_MIN, V_MAX, on_trackbar);
// 	createTrackbar("Uniqueness Ratio", DISPARITY_CONFIGURATOR, &V_MIN, V_MAX, on_trackbar);
// 	createTrackbar("Speckle Window Size", DISPARITY_CONFIGURATOR, &V_MIN, V_MAX, on_trackbar);
// 	createTrackbar("Speckle Range", DISPARITY_CONFIGURATOR, &V_MIN, V_MAX, on_trackbar);
// 	createTrackbar("Disp 12 MaxDiff", DISPARITY_CONFIGURATOR, &V_MIN, V_MAX, on_trackbar);
//
// }

string getImgType(int imgTypeInt)
{
    int img_types = 35;
    // 7 base types, with five channel options each (none or C1, ..., C4)

    int enum_ints[] = {
        CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
        CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
        CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
        CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
        CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
        CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
        CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4
    };

    string enum_strings[] = {
        "CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",
        "CV_8S",  "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",
        "CV_16U", "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
        "CV_16S", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
        "CV_32S", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
        "CV_32F", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
        "CV_64F", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"
    };

    for(int i = 0; i < img_types; i++) {
        if (imgTypeInt == enum_ints[i])
            return enum_strings[i];
    }

    return "unknown image type";
}

cv::StereoBM initDisparityCalculator() {
    cv::StereoBM bm;

    bm.state->SADWindowSize = 23;
    bm.state->numberOfDisparities = 96;
    bm.state->preFilterSize = 25;
    bm.state->preFilterCap = 63;
    bm.state->minDisparity = 0;
    bm.state->textureThreshold = 20;
    bm.state->uniquenessRatio = 10;
    bm.state->speckleWindowSize = 25;
    bm.state->speckleRange = 8;
    bm.state->disp12MaxDiff = 1;

    return bm;
}

Mat calculateDisparity(
    cv::StereoBM bm,
    Mat left,
    Mat right)
{
    Size size = left.size();
    Mat disparity_map = Mat(size, CV_16SC1);
    Mat left_converted;
    Mat right_converted;

    // cout << "Left image type: " << getImgType(left.type()) << endl;
    // cout << "Left converted image type: " << getImgType(left_converted.type()) << endl;

    // calculate disparity map
    bm(left, right, disparity_map);

    return disparity_map;
}

int main(int argc, char* argv[])
{
    int num_cams = detectNumberOfCameras();
    VideoCapture camera_1(0);
    VideoCapture camera_2(1);
    Mat feed_1;
    Mat feed_2;
    Mat gray_feed_1;
    Mat gray_feed_2;
    Size size = feed_1.size();
    Mat disparity_map = Mat(size, CV_16SC1);
    cv::StereoBM bm = initDisparityCalculator();

    // check camera feeds
    if (!camera_1.isOpened() && !camera_2.isOpened()) {
        cout << "Failed to open video feeds!" << endl;
        return -1;
    }

    // std::string intrinsic_filename = "intrinsic.xml";
    // std:string disparity_filename = "disparity.xml";
	// FileStorage intrinsic_file(intrinsic_filename, CV_STORAGE_READ);
	// FileStorage disparity_file(disparity_filename, CV_STORAGE_READ);

    // create gui windows
    cv::namedWindow(CAM_1, CV_WINDOW_AUTOSIZE);
    cv::namedWindow(CAM_2, CV_WINDOW_AUTOSIZE);
    cv::namedWindow(DISPARITY_MAP, CV_WINDOW_AUTOSIZE);

    while(1) {
        // read video streams
        camera_1 >> feed_1;
        camera_2 >> feed_2;
        cvtColor(feed_1, gray_feed_1, CV_BGR2GRAY);
        cvtColor(feed_2, gray_feed_2, CV_BGR2GRAY);

        // display camera feeds
        cv::imshow(CAM_1, feed_1);
        cv::imshow(CAM_2, feed_2);

        // calculate and display disparity map
        disparity_map = calculateDisparity(bm, gray_feed_1, gray_feed_2);
        cv::imshow(DISPARITY_MAP, disparity_map);

		// delay 30ms so that screen can refresh.
		waitKey(30);  // IMPORTANT!! IMAGE WILL NOT DISPLAY WITHOUT IT!
    }

    return 0;
}
