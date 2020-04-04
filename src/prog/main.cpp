// Manual compiling
// g++ -o main main.cpp -I/usr/local/include/opencv4
//      -L/usr/local/lib `pkg-config --CXXFLAGS --libs opencv`
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include "TfLite.h"
#include "bmp.h"
#include "utils.h"

#include <iostream>

using namespace std;

void printFrameInfo(cv::Mat &frame)
{
    cout << "frame:\n";
    cout << "cols = " << frame.cols << "\n";
    cout << "rows = " << frame.rows << "\n";
    cout << "size = " << frame.size << "\n";
    printf("data = %p\n", (void *)frame.data);
    cout << "dims = " << frame.dims << "\n";
    cout << "type = " << frame.type() << "\n"; // 16=CV_8UC3
    cout << "depth = " << frame.depth() << "\n";
    cout << "channels = " << frame.channels() << "\n";

    uint8_t *data = frame.data;
    for (int i = 0; i < 10; ++i) {
        uint8_t r = *data;
        uint8_t g = *(data + 1);
        uint8_t b = *(data + 2);
        cout << "pixel_" << i << " = (" << (int)r << ", " << (int)g << ", "
             << (int)b << ")\n";
        data += 3;
    }
}

void paintRow(cv::Mat &frame, int row, int color)
{
    uint8_t *data = frame.data;
    size_t width = frame.cols;
    size_t height = frame.rows;

    for (int i = 0; i < width; ++i)
        for (int j = 0; j < height; ++j) {
            *data = 255;
            *(data + 1) = 0;
            *(data + 3) = 0;
            data += 3;
        }
}

void showWebCam()
{
    cv::VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if (!cap.open(0))
        return;

    TfLite tfLite;
    tfLite.loadModel("res/mobilenet_v2_1.0_224_quant.tflite");
    for (;;) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        // tfLite.runInference(frame);
        // printFrameInfo(frame);
        // paintRow(frame, 0, 0);
        cv::namedWindow("Webcam");
        cv::imshow("Webcam", frame);
        if (cv::waitKey(10) == 27)
            break;
    }
}

cv::Mat getFrame()
{
    cv::VideoCapture cap;
    cv::Mat frame;
    // if (!cap.open(0))
    //  return frame;
    cap >> frame;

    return frame;
}

int main(int argc, char **argv)
{
    showWebCam();

    return 0;
}
