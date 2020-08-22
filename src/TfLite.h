#pragma once

#include "opencv2/opencv.hpp"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

class TfLite {
  public:
    TfLite();
    ~TfLite();

    void loadModel(const char *modelFile);
    void runInference(const char *inputFile);
    void runInference(const cv::Mat &frame);
    std::vector<TfLiteTensor *> getOutputs() const;

    void printOps() const;
    void printInputOutputInfo() const;
    void setInputBmpExport(bool value) { mWriteInputBmp = value; }

  private:
    // Loads a BMP image into the loaded models input tensor.
    void loadBmpImage(const char *bmpFile);
    void loadFrame(const cv::Mat &frame);
    void printInterpreterInfo() const;
    void printTopResults() const;
    std::unique_ptr<tflite::FlatBufferModel> mModel;
    std::unique_ptr<tflite::Interpreter> mInterpreter;
    std::unique_ptr<TfLiteDelegate, std::function<void(TfLiteDelegate *)>>
        mDelegate{nullptr,
                  [](TfLiteDelegate *d) { TfLiteGpuDelegateDelete(d); }};
    bool mWriteInputBmp = false;
};
