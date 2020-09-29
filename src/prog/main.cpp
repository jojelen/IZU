// Manual compiling
// g++ -o main main.cpp -I/usr/local/include/opencv4
//      -L/usr/local/lib `pkg-config --CXXFLAGS --libs opencv`
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/opencv.hpp"

#include "TfLite.h"
#include "bmp.h"
#include "utils.h"

#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

void runImageClassification(const cv::Mat &frame)
{
    TIMER
    static bool initialized = false;
    static TfLite tfLite;

    if (!initialized) {
        tfLite.loadModel("res/mobilenet_v2_1.0_224_quant.tflite");
        tfLite.setInputBmpExport(false);
        initialized = true;
    }

    tfLite.runInference(frame);
}

vector<TfLiteTensor *> runObjectDetection(cv::Mat &frame)
{
    TIMER

    static bool initialized = false;
    static TfLite tfLite;

    if (!initialized) {
        tfLite.loadModel("res/detect.tflite");
        tfLite.printInputOutputInfo();
        tfLite.setInputBmpExport(false);
        initialized = true;
    }

    tfLite.runInference(frame);

    return tfLite.getOutputs();
}

vector<string> createClassMap(const string &classesFile)
{
    TIMER

    ifstream file(classesFile);
    string line;
    vector<string> classesMap;
    while (getline(file, line)) {
        classesMap.push_back(line);
    }

    return classesMap;
}

void postProcessing(cv::Mat &frame, vector<TfLiteTensor *> &output)
{
    TIMER

    // Here, we assume the output to be as in COCO SSD MobileNet v1: Four
    // tensors that specify.
    // 0. Locations: Multidimensional array of [10][4] floating point values
    // between 0 and 1, the inner arrays representing bounding boxes in the form
    // [top, left, bottom, right].
    // 1. Classes: Array of 10 integers (output as floating point values) each
    // indicating the index of a class label from the labels file.
    // 2. Scores Array of 10 floating point values between 0 and 1 representing
    // probability that a class was detected.
    // 3. Number and detections Array of length 1 containing a floating point
    // value expressing the total number of detection results.

    if (output[1]->type != kTfLiteFloat32) {
        errExit("cannot handle output type " + to_string(output[1]->type) +
                " yet");
        exit(-1);
    }

    auto boxPtr = reinterpret_cast<const float *>(output[0]->data.data);
    auto classPtr = reinterpret_cast<const float *>(output[1]->data.data);
    auto scorePtr = reinterpret_cast<const float *>(output[2]->data.data);
    static vector<string> classMap =
        createClassMap("res/coco-labels-paper.txt");

    const static float MIN_SCORE = 0.5;
    const cv::Scalar BOX_COLOR(0, 255, 0);
    const cv::Scalar TEXT_COLOR(255, 255, 0);
    const int FONT = cv::FONT_HERSHEY_SIMPLEX;
    const double FONT_SCALE = 0.5;

    // Add boxes around detections with a score above MIN_SCORE.
    size_t width = frame.cols;
    size_t height = frame.rows;
    for (size_t i = 0; i < 10; ++i) {
        if (*(scorePtr + i) > MIN_SCORE) {
            cv::Point topLeft(boxPtr[4 * i + 1] * width,
                              boxPtr[4 * i] * height);
            cv::Point bottomRight(boxPtr[4 * i + 3] * width,
                                  boxPtr[4 * i + 2] * height);
            cv::rectangle(frame, bottomRight, topLeft, BOX_COLOR);
            int classId = static_cast<int>(classPtr[i]);
            if (classId >= classMap.size()) {
                cout << "[ERROR]: class id not found!\n";
                continue;
            }
            cv::putText(frame, classMap[classId], topLeft, FONT, FONT_SCALE,
                        TEXT_COLOR);
        }
    }
}

void showWebCam()
{
    cv::VideoCapture cap;
    if (!cap.open(0 /* Default camera */))
        return;

    cv::namedWindow("Webcam");
    cv::Mat frame, RGBframe, resized;
    vector<TfLiteTensor *> output;
    size_t frame_nr = 0;
    for (;;) {
#ifdef TIME
        Timer timer("1 frame");
#endif

        cap >> frame;
        if (frame.empty())
            break;

        if (frame_nr % 2 == 0) {
#ifdef TIME
        {
            Timer timer("pre-processing");
#endif
            cv::cvtColor(frame, RGBframe, CV_BGR2RGB);

            cv::resize(RGBframe, resized, cv::Size(300, 300), 0, 0,
                       cv::INTER_CUBIC);
#ifdef TIME
        }
#endif
            output = runObjectDetection(resized);
        }

        postProcessing(frame, output);

        cv::imshow("Webcam", frame);

        if (cv::waitKey(10) == 27 /* ESC key */)
            break;

        ++frame_nr;
    }
}

int main(int argc, char **argv)
{
    TIMER

    showWebCam();

    return 0;
}
