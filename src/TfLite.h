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
  void printInterpreterInfo() const;
  std::unique_ptr<tflite::FlatBufferModel> mModel;
  std::unique_ptr<tflite::Interpreter> mInterpreter;
};