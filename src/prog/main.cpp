// Manual compiling
// g++ -o main main.cpp -I/usr/local/include/opencv4
//      -L/usr/local/lib `pkg-config --CXXFLAGS --libs opencv`
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/opencv.hpp"

#include "TfLite.h"
#include "bmp.h"
#include "utils.h"

#include <iostream>

using namespace std;

void runImageClassification(const cv::Mat &frame)
{
    static bool initialized = false;
    static TfLite tfLite;

    if (!initialized) {
        tfLite.loadModel("res/mobilenet_v2_1.0_224_quant.tflite");
        tfLite.setInputBmpExport(false);
    }

    tfLite.runInference(frame);
}

void showWebCam()
{
    cv::VideoCapture cap;
    if (!cap.open(0 /* Default camera */))
        return;

    unsigned frameCount = 0;
    for (;;) {
        cv::Mat frame, RGBframe;
        cap >> frame;
        if (frame.empty())
            break;
        cv::cvtColor(frame, RGBframe, CV_BGR2RGB);

        if (frameCount++ % 10 == 0)
            runImageClassification(RGBframe);

        // printFrameInfo(frame);
        // paintRow(frame, 0, 0);
        cv::namedWindow("Webcam");
        cv::imshow("Webcam", frame);
        if (cv::waitKey(10) == 27 /* ESC key */)
            break;
    }
}

int main(int argc, char **argv)
{
    showWebCam();

    return 0;
}
