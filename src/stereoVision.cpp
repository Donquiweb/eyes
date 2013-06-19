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
#define DISPARITY_CONFIG "Disparity Configurator"

struct DisparityEventData
{
    int *type;
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
    std::cout << "Number of cameras: " << num << std::endl;
    if (num != 2) {
        std::cout << "Failed to detect 2 cameras for stereo vision!" << std::endl;
        exit(-1);
    }

    return num;
}

void sadWindowSizeEvent(int pos, void *sad_winsize) {
    if (pos > 5 && pos < 255 && pos % 2 != 0)
        *(int *)sad_winsize = pos;
}

void numberOfDisparityEvent(int pos, void *num_of_dispar) {
    if (pos > 0 && pos % 16 == 0)
        *(int *)num_of_dispar = pos;
}

void textureThresholdEvent(int pos, void *texture_threshold) {
    if (pos > 0 && pos < 100)
        *(int *)texture_threshold = pos;
}

void preFilterCap(int pos, void *pre_filter_cap) {
    if (pos > 0 && pos < 63)
        *(int *)pre_filter_cap = pos;
}

void initDisparityConfigurator(cv::StereoBM &bm)
{
	//create window for trackbars
    cv::namedWindow(DISPARITY_CONFIG, CV_WINDOW_AUTOSIZE);

	// create trackbars and insert them into window
    cv::createTrackbar(
        "SAD Window Size",
        DISPARITY_CONFIG,
        NULL,
        255,
        sadWindowSizeEvent,
        (void *) &bm.state->SADWindowSize
    );
    cv::createTrackbar(
        "Number of Disparities",
        DISPARITY_CONFIG,
        NULL,
        200,
        numberOfDisparityEvent,
        (void *) &bm.state->numberOfDisparities
    );

    cv::createTrackbar(
        "Pre-Filter Size",
        DISPARITY_CONFIG,
        NULL,
        100,
        sadWindowSizeEvent,
        (void *) &bm.state->preFilterSize
    );

    cv::createTrackbar(
        "Pre-Filter Cap",
        DISPARITY_CONFIG,
        NULL,
        63,
        textureThresholdEvent,
        (void *) &bm.state->preFilterCap
    );

    cv::createTrackbar(
        "Min Disparity",
        DISPARITY_CONFIG,
        NULL,
        100,
        textureThresholdEvent,
        (void *) &bm.state->minDisparity
    );

    cv::createTrackbar(
        "Texture Threshold",
        DISPARITY_CONFIG,
        NULL,
        200,
        textureThresholdEvent,
        (void *) &bm.state->textureThreshold
    );

    cv::createTrackbar(
        "Uniqueness Ratio",
        DISPARITY_CONFIG,
        NULL,
        100,
        textureThresholdEvent,
        (void *) &bm.state->uniquenessRatio
    );

    cv::createTrackbar(
        "Speckle Window Size",
        DISPARITY_CONFIG,
        NULL,
        100,
        textureThresholdEvent,
        (void *) &bm.state->speckleWindowSize
    );

    cv::createTrackbar(
        "Speckle Range",
        DISPARITY_CONFIG,
        NULL,
        100,
        textureThresholdEvent,
        (void *) &bm.state->speckleRange
    );

    cv::createTrackbar(
        "Disparity Max Diff",
        DISPARITY_CONFIG,
        NULL,
        100,
        textureThresholdEvent,
        (void *) &bm.state->disp12MaxDiff
    );

    // set trackbar positions
    cv::setTrackbarPos(
        "SAD Window Size",
        DISPARITY_CONFIG,
        bm.state->SADWindowSize
    );
    cv::setTrackbarPos(
            "Number of Disparities", DISPARITY_CONFIG, bm.state->numberOfDisparities);
    cv::setTrackbarPos(
        "Pre-Filter Size",
        DISPARITY_CONFIG,
        bm.state->preFilterSize
    );
    cv::setTrackbarPos(
        "Pre-Filter Cap",
        DISPARITY_CONFIG,
        bm.state->preFilterCap
    );
    cv::setTrackbarPos(
        "Min Disparity",
        DISPARITY_CONFIG,
        bm.state->minDisparity
    );
    cv::setTrackbarPos(
        "Texture Threshold",
        DISPARITY_CONFIG,
        bm.state->textureThreshold
    );
    cv::setTrackbarPos(
        "Uniqueness Ratio",
        DISPARITY_CONFIG,
        bm.state->uniquenessRatio
    );
    cv::setTrackbarPos(
        "Speckle Window Size",
        DISPARITY_CONFIG,
        bm.state->speckleWindowSize
    );
    cv::setTrackbarPos(
        "Speckle Range",
        DISPARITY_CONFIG,
        bm.state->speckleRange
    );
    cv::setTrackbarPos(
        "Disparity Max Diff",
        DISPARITY_CONFIG,
        bm.state->disp12MaxDiff
    );
}

cv::string getImgType(int imgTypeInt)
{
    int img_types = 35; // 7 base types, with 5 channel options each
    int enum_ints[] = {
        CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
        CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
        CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
        CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
        CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
        CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
        CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4
    };

    cv::string enum_strings[] = {
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
    cv::StereoBM bm(CV_STEREO_BM_BASIC);

    bm.state->SADWindowSize = 5;
    bm.state->numberOfDisparities = 96;
    bm.state->preFilterSize = 25;
    bm.state->preFilterCap = 63;
    bm.state->minDisparity = 0;
    bm.state->textureThreshold = 20;
    bm.state->uniquenessRatio = 10;
    bm.state->speckleWindowSize = 25;
    bm.state->speckleRange = 32;
    bm.state->disp12MaxDiff = 1;

    return bm;
}

cv::Mat calculateDisparity(
    cv::StereoBM bm,
    cv::Mat left,
    cv::Mat right)
{
    cv::Size size = left.size();
    cv::Mat disparity_map = cv::Mat(size, CV_16SC1);
    cv::Mat left_converted;
    cv::Mat right_converted;

    // calculate disparity map
    bm(left, right, disparity_map);

    return disparity_map;
}

int main(int argc, char* argv[])
{
    int num_cams = detectNumberOfCameras();
    cv::VideoCapture camera_1(0);
    cv::VideoCapture camera_2(1);
    cv::Mat feed_1;
    cv::Mat feed_2;
    cv::Mat gray_feed_1;
    cv::Mat gray_feed_2;
    cv::Size size = feed_1.size();
    cv::Mat disparity_map = cv::Mat(size, CV_16SC1);
    cv::StereoBM bm = initDisparityCalculator();

    // check camera feeds
    if (!camera_1.isOpened() && !camera_2.isOpened()) {
        std::cout << "Failed to open video feeds!" << std::endl;
        return -1;
    }

    // create gui windows
    cv::namedWindow(CAM_1, CV_WINDOW_AUTOSIZE);
    cv::namedWindow(CAM_2, CV_WINDOW_AUTOSIZE);
    cv::namedWindow(DISPARITY_MAP, CV_WINDOW_AUTOSIZE);
    initDisparityConfigurator(bm);

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
        cv::waitKey(30);  // IMPORTANT!! IMAGE WILL NOT DISPLAY WITHOUT IT!
    }

    return 0;
}
