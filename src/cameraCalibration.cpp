#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <opencv/highgui.h>
#include <opencv/cv.h>

#include <dbg/dbg.h>

#define GUI_WIDTH 400
#define GUI_HEIGHT 400
#define CALIBRATION_WINDOW "Calibration Window"
#define LIVE_FEED_WINDOW "Live Feed Window"
#define CALIBRATED_IMAGE "Calibrated Image"
#define UNCALIBRATED_IMAGE "Un-calibrated Image"

using namespace cv;
using namespace std;


struct chessboard_details
{
    int skip_frames;            // wait frames per chessboard view
    int boards_to_capture;      // max number of chessboard to capture
    int width;                  // number of inner corners in the x-axis
    int height;                 // number of inner corners in the y-axis
    int target_corner_count;    // number of corners to capture
    int corner_count;           // current corner count
    int boards_captured;        // number of captured chessboards

    CvSize board_size;          // board size
    CvPoint2D32f *corners;      // corner co-ordinates

    CvMat *img_pts;             // image points
    CvMat *obj_pts;             // object points
    CvMat *pt_cnts;             // point counts
};

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

void init_chessboard(
    struct chessboard_details **cb,
    int boards_to_capture,
    int width,
    int height)
{
    // cb vars
    (*cb)->skip_frames = 20;
    (*cb)->boards_to_capture = boards_to_capture;
    (*cb)->width = width;
    (*cb)->height = height;
    (*cb)->target_corner_count = (*cb)->width * (*cb)->height;
    (*cb)->corner_count = 0;
    (*cb)->boards_captured = 0;
    (*cb)->board_size = cvSize((*cb)->width, (*cb)->height);
    (*cb)->corners = new CvPoint2D32f[(*cb)->target_corner_count];

    // matrix vars
    (*cb)->img_pts = cvCreateMat(
        (*cb)->boards_to_capture * (*cb)->target_corner_count,
        2,
        CV_32FC1
    );
    (*cb)->obj_pts = cvCreateMat(
        (*cb)->boards_to_capture * (*cb)->target_corner_count,
        3,
        CV_32FC1
    );
    (*cb)->pt_cnts = cvCreateMat(
        (*cb)->boards_to_capture,
        1,
        CV_32SC1
    );

}

void analyzeChessboardImage(
        IplImage *image,
        IplImage *gray_image,
        struct chessboard_details **cb)
{
    int step = 0;
    int j = 0;

    // find chess board corners
    int found = cvFindChessboardCorners(
        image,
        (*cb)->board_size,
        (*cb)->corners,
        &(*cb)->corner_count,
        CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS
    );

    // obtain subpixel accuracy on the corners
    cvCvtColor(image, gray_image, CV_BGR2GRAY);
    cvFindCornerSubPix(
        gray_image,
        (*cb)->corners,
        (*cb)->corner_count,
        cvSize(11, 11),
        cvSize(-1, -1),
        cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1)
    );

    // draw the chessboard corners on a GUI window
    cvDrawChessboardCorners(
        image,
        (*cb)->board_size,
        (*cb)->corners,
        (*cb)->corner_count,
        found
    );
    cvShowImage(CALIBRATION_WINDOW, image);

    // if found a good match of the chessboard corners store the data
    if ((*cb)->corner_count == (*cb)->target_corner_count) {
        log_info("Found chessboard corners!");
        step = (*cb)->boards_captured * (*cb)->target_corner_count;

        for (int i = step; j < (*cb)->target_corner_count; i++) {
            CV_MAT_ELEM(*(*cb)->img_pts, float, i , 0) = (*cb)->corners[j].x;
            CV_MAT_ELEM(*(*cb)->img_pts, float, i , 1) = (*cb)->corners[j].y;

            CV_MAT_ELEM(*(*cb)->obj_pts, float, i , 0) = j / (*cb)->width;
            CV_MAT_ELEM(*(*cb)->obj_pts, float, i , 1) = j % (*cb)->width;
            CV_MAT_ELEM(*(*cb)->obj_pts, float, i , 2) = 0.0f;

            j++;
        }

        CV_MAT_ELEM(
            *(*cb)->pt_cnts,
            int,
            (*cb)->boards_captured,
            0
        ) = (*cb)->target_corner_count;

        // increment boards captured
        (*cb)->boards_captured++;
        log_info("Boards captured: %d", (*cb)->boards_captured);
    }
}

int listenForUserEvent()
{
	    int keyboard_event = cvWaitKey(15);

	    switch (keyboard_event) {
        case 'p':  // 'p' key pressed
            log_info("User pressed the 'pause' key!");
            keyboard_event = 0;  // reset keyboard_event
            while (keyboard_event != 'p' && keyboard_event != 27) {
                keyboard_event = cvWaitKey(250);
            }
            break;
        case 27:  // ESC key pressed
            log_info("User pressed the 'ESC' key!");
            log_info("Exiting Calibration ...");
            return 1;
            break;
	    }

	    return 0;
}

int obtainChessboardImages(
        CvCapture *capture,
        IplImage *image,
        IplImage *gray_image,
        struct chessboard_details *chessboard)
{
	int frame = 0;
	int event = 0;

    cvNamedWindow(LIVE_FEED_WINDOW, CV_WINDOW_AUTOSIZE);
    cvNamedWindow(CALIBRATION_WINDOW, CV_WINDOW_AUTOSIZE);

	while (chessboard->boards_captured < chessboard->boards_to_capture) {
        image = cvQueryFrame(capture);

        // skip every skip_frames frames to allow user to move chessboard
	    if (frame++ % chessboard->skip_frames == 0) {
            analyzeChessboardImage(
                image,
                gray_image,
                &chessboard
            );
        }

        // handle user events
        event = listenForUserEvent();
        if (event == 1) {  // quit?
            return 1;
        }

        // display live feed
        cvShowImage(LIVE_FEED_WINDOW, image);
	}

	cvDestroyWindow(LIVE_FEED_WINDOW);
	cvDestroyWindow(CALIBRATION_WINDOW);

	return 0;
}

struct calibration *analyzeFoundChessboardMatrices(
        struct chessboard_details **cb, IplImage *image)
{
    int i = 0;
    int b_captured = (*cb)->boards_captured;
    int tgt_corners = (*cb)->target_corner_count;

    CvMat *obj_pts = cvCreateMat(b_captured * tgt_corners, 3, CV_32FC1);
    CvMat *img_pts = cvCreateMat(b_captured * tgt_corners, 2, CV_32FC1);
    CvMat *pt_cnts = cvCreateMat(b_captured, 1, CV_32SC1);

    struct calibration *results = new calibration();
    results->intrinsic_matrix = cvCreateMat(3, 3, CV_32FC1);
    results->distortion_coeffs = cvCreateMat(5, 1, CV_32FC1);


    // transfer data to correct size matrices
    for (i = 0; i < (b_captured * tgt_corners); i++) {
        // image points
        CV_MAT_ELEM(*img_pts, float, i, 0) =
            CV_MAT_ELEM(*(*cb)->img_pts, float, i, 0);
        CV_MAT_ELEM(*img_pts, float, i, 1) =
            CV_MAT_ELEM(*(*cb)->img_pts, float, i, 1);

        // object points
        CV_MAT_ELEM(*obj_pts, float, i, 0) =
            CV_MAT_ELEM(*(*cb)->obj_pts, float, i, 0);
        CV_MAT_ELEM(*obj_pts, float, i, 1) =
            CV_MAT_ELEM(*(*cb)->obj_pts, float, i, 1);
        CV_MAT_ELEM(*obj_pts, float, i, 2) =
            CV_MAT_ELEM(*(*cb)->obj_pts, float, i, 2);
    }

    for (i = 0; i < b_captured; i++) {
        CV_MAT_ELEM(*pt_cnts, float, i, 0) =
            CV_MAT_ELEM(*(*cb)->pt_cnts, float, i, 0);
    }

    cvReleaseMat(&(*cb)->img_pts);
    cvReleaseMat(&(*cb)->obj_pts);
    cvReleaseMat(&(*cb)->pt_cnts);

    CV_MAT_ELEM(*(results->intrinsic_matrix), float, 0, 0) = 1.0f;
    CV_MAT_ELEM(*(results->intrinsic_matrix), float, 1, 1) = 1.0f;

    // calibrate camera
    cvCalibrateCamera2(
        obj_pts,
        img_pts,
        pt_cnts,
        cvGetSize(image),
        results->intrinsic_matrix,
        results->distortion_coeffs,
        NULL,
        NULL,
        0
    );

    return results;
}

void saveCalibrationResults(struct calibration *results)
{
    cvSave("instrinsics.xml", results->intrinsic_matrix);
    cvSave("distortion.xml", results->distortion_coeffs);
}

void undistortImage(
        struct undistort_image **undistort,
        struct calibration *settings,
        IplImage *image)
{
    (*undistort)->map_x = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
    (*undistort)->map_y = cvCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
    cvInitUndistortMap(
        settings->intrinsic_matrix,
        settings->distortion_coeffs,
        (*undistort)->map_x,
        (*undistort)->map_y
    );
}

int displayCalibrationEffects(
        CvCapture *capture,
        IplImage *image,
        struct undistort_image *undistort)
{
    int event = 0;

    cvNamedWindow(UNCALIBRATED_IMAGE, CV_WINDOW_AUTOSIZE);
    cvNamedWindow(CALIBRATED_IMAGE, CV_WINDOW_AUTOSIZE);

    // display calibrated and uncalibrated image
    while(image) {
        // image before calibration
        IplImage *t = cvCloneImage(image);
        cvShowImage(UNCALIBRATED_IMAGE, image);

        // image after calibration
        cvRemap(t, image, undistort->map_x, undistort->map_y);
        cvReleaseImage(&t);
        cvShowImage(CALIBRATED_IMAGE, image);

        // handle user events
        event = listenForUserEvent();
        if (event == 1) {  // quit?
            return 1;
        }

        // get next frame image
        image = cvQueryFrame(capture);
    }

	cvDestroyWindow(UNCALIBRATED_IMAGE);
	cvDestroyWindow(CALIBRATED_IMAGE);

    return 0;
}

int main(int argc, char* argv[])
{
    // general vars
	int x = 0;
	int y = 0;
	int event = 0;
	struct chessboard_details *chessboard = new chessboard_details();
	struct calibration *results;
	struct undistort_image *undistort = new undistort_image();

    // camera and image vars
	CvCapture *capture;
	Mat camera_feed;
    IplImage *image;
    IplImage *gray_image;



    // START PROGRAM
    log_info("Starting Camera Calibration!");
    log_info("Opening camera stream ...");

    // init video camera
    capture = cvCreateCameraCapture(0);

    // init images
    image = cvQueryFrame(capture);
    gray_image = cvCreateImage(cvGetSize(image), 8, 1);

    // init chessboard
	init_chessboard(
        &chessboard,
        10,  // boards_to_capture
        9,  // width
        6  // height
    );

    // obtain chessboard images
    log_info("Obtain chessboard images ...");
    event = obtainChessboardImages(
        capture,
        image,
        gray_image,
        chessboard
    );
    if (event == 1) return 0;

	// analyze images for calibration and save results
    log_info("Analyze chessboard images for calibration settings ...");
    results = analyzeFoundChessboardMatrices(&chessboard, image);
    saveCalibrationResults(results);

    // load calibration settings
    log_info("Load calibration settings ...");
    undistortImage(&undistort, results, image);

    // display calibration effects
    log_info("Display calibration effects...");
    event = displayCalibrationEffects(capture, image, undistort);
    if (event == 1) return 0;

	return 0;
}
