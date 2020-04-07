#pragma once

#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

class TfLite {
  public:
    TfLite();
    ~TfLite();

    void loadModel(const char *modelFile);
    void runInference(const char *inputFile);
    void runInference(const cv::Mat &frame);

    void printOps() const;

  private:
    // Loads a BMP image into the loaded models input tensor.
    void loadBmpImage(const char *bmpFile);
    void loadFrame(const cv::Mat &frame);
    void printInputOutputInfo() const;
    void printInterpreterInfo() const;
    void printTopResults() const;
    std::unique_ptr<tflite::FlatBufferModel> mModel;
    std::unique_ptr<tflite::Interpreter> mInterpreter;
};