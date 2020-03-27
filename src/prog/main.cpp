// Manual compiling
// g++ -o main main.cpp -I/usr/local/include/opencv4 
//      -L/usr/local/lib `pkg-config --CXXFLAGS --libs opencv`
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
//#include <stdio.h>
//#include <tensorflow/c/c_api.h>

#include <iostream>

using namespace std;
int main(int argc, char** argv) {
    //printf("Hello from TensorFlow C library version %s\n", TF_Version());

    cv::VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if (!cap.open(0)) return 0;
    for (;;) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;  // end of video stream
        cv::namedWindow( "hej");
        cv::imshow("hej", frame);
        //cv::imshow("this is you, smile! :)", frame);
        if (cv::waitKey(10) == 27) break;  // stop capturing by pressing ESC
    }
    // the camera will be closed automatically upon exit
    // cap.close();
    return 0; 
}
