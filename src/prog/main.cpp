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
        initialized = true;
    }

    tfLite.runInference(frame);
}

void addBox(cv::Mat &frame, float top, float left, float bottom, float right)
{
    cout << "Adding box (" << top << ", " << left << ", " << bottom << ", "
         << right << ")\n";
    uint8_t *data = frame.data;
    size_t width = frame.cols;
    size_t height = frame.rows;
    top *= height;
    bottom *= height;
    left *= width;
    right *= width;

    size_t pixel = (width * (size_t)top + (size_t)left) * 3;

    for (size_t i = 0; i < right - left; ++i) {
        if (pixel > width * height * 3)
            break;
        data[pixel] = 255;
        data[pixel + 1] = 0;
        data[pixel + 2] = 0;
        pixel += 3;
    }

    pixel = (width * (size_t)top + (size_t)left) * 3;

    for (size_t i = 0; i < right - left; ++i) {
        if (pixel > width * height * 3)
            break;
        data[pixel] = 255;
        data[pixel + 1] = 0;
        data[pixel + 2] = 0;
        pixel += width * 3;
    }

    pixel = (width * (size_t)top + (size_t)right) * 3;

    for (size_t i = 0; i < right - left; ++i) {
        if (pixel > width * height * 3)
            break;
        data[pixel] = 255;
        data[pixel + 1] = 0;
        data[pixel + 2] = 0;
        pixel += width * 3;
    }

    pixel = (width * (size_t)top + (size_t)left) * 3;

    for (size_t i = 0; i < right - left; ++i) {
        if (pixel > width * height * 3)
            break;
        data[pixel] = 255;
        data[pixel + 1] = 0;
        data[pixel + 2] = 0;
        pixel += 3;
    }
}

void runObjectDetection(cv::Mat &frame)
{
    static bool initialized = false;
    static TfLite tfLite;

    if (!initialized) {
        tfLite.loadModel("res/detect.tflite");
        tfLite.printInputOutputInfo();
        tfLite.setInputBmpExport(false);
        initialized = true;
    }

    tfLite.runInference(frame);
    auto outputs = tfLite.getOutputs();
    if (outputs[1]->type != kTfLiteFloat32) {
        errExit("cannot handle output type " + to_string(outputs[1]->type) +
                " yet");
        exit(-1);
    }

    auto boxPtr = reinterpret_cast<const float *>(outputs[0]->data.data);
    for (size_t i = 0; i < 40; ++i)
        cout << "boxx: " << *(boxPtr + i) << endl;

    addBox(frame, boxPtr[0], boxPtr[1], boxPtr[2], boxPtr[3]);
    auto labelPtr = reinterpret_cast<const float *>(outputs[1]->data.data);
    auto scorePtr = reinterpret_cast<const float *>(outputs[2]->data.data);

    for (size_t i = 0; i < 10; ++i) {
        cout << "Label " << i << ": " << labelPtr[i] << endl;
        cout << "Score " << i << ": " << scorePtr[i] << endl;
    }
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

        // if (frameCount++ % 10 == 0)
        //    runImageClassification(RGBframe);
        runObjectDetection(RGBframe);
        cv::cvtColor(RGBframe, frame, CV_RGB2BGR);

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
