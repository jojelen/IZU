#pragma once

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"

#include "opencv2/opencv.hpp"

#include <queue>
#include <string>
#include <string_view>
#include <vector>

void errExit(const std::string_view &msg);

// Resizes image data by using the "resize" builtin operator in tflite.
template <class T>
void resize(T *out, uint8_t *in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels)
{
    std::unique_ptr<tflite::Interpreter> interpreter(new tflite::Interpreter);

    // Add tensors. Two inputs: input and new_sizes and one output.
    interpreter->AddTensors(3, nullptr);
    interpreter->SetInputs({0, 1});
    interpreter->SetOutputs({2});

    // Set parameters of tensors.
    TfLiteQuantizationParams quant;
    interpreter->SetTensorParametersReadWrite(
        0, kTfLiteFloat32, "input",
        {1, image_height, image_width, image_channels}, quant);
    interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                              quant);
    interpreter->SetTensorParametersReadWrite(
        2, kTfLiteFloat32, "output",
        {1, wanted_height, wanted_width, wanted_channels}, quant);

    // Add the custom op that does the resizing.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    const TfLiteRegistration *resize_op = resolver.FindOp(
        tflite::BuiltinOperator::BuiltinOperator_RESIZE_BILINEAR, 1);
    // params is freed in AddNodeWithParameters().
    auto *params = reinterpret_cast<TfLiteResizeBilinearParams *>(
        malloc(sizeof(TfLiteResizeBilinearParams)));
    params->align_corners = false;
    params->half_pixel_centers = false;
    interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params,
                                       resize_op, nullptr);

    interpreter->AllocateTensors();

    // Fill first input tensor with image data.
    auto input = interpreter->typed_tensor<float>(0);
    int number_of_pixels = image_height * image_width * image_channels;
    for (int i = 0; i < number_of_pixels; i++) {
        input[i] = in[i];
    }

    // Fill second input tensor with the wanted image size.
    interpreter->typed_tensor<int>(1)[0] = wanted_height;
    interpreter->typed_tensor<int>(1)[1] = wanted_width;

    interpreter->Invoke();

    // Fill out with the output data.
    auto output = interpreter->typed_tensor<float>(2);
    bool floating = std::is_same_v<T, float>;
    static const float input_mean = 0.;
    static const float input_std = 1.;
    auto output_number_of_pixels =
        wanted_height * wanted_width * wanted_channels;
    for (int i = 0; i < output_number_of_pixels; i++) {
        if (floating)
            out[i] = (output[i] - input_mean) / input_std;
        else
            out[i] = (uint8_t)output[i];
    }
}

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
template <class T>
void get_top_n(T *prediction, int prediction_size, size_t num_results,
               float threshold, std::vector<std::pair<float, int>> *top_results,
               bool input_floating)
{
    // Will contain top N results in ascending order.
    std::priority_queue<std::pair<float, int>,
                        std::vector<std::pair<float, int>>,
                        std::greater<std::pair<float, int>>>
        top_result_pq;

    const long count = prediction_size; // NOLINT(runtime/int)
    for (int i = 0; i < count; ++i) {
        float value;
        if (input_floating)
            value = prediction[i];
        else
            value = prediction[i] / 255.0;
        // Only add it if it beats the threshold and has a chance at being in
        // the top N.
        if (value < threshold) {
            continue;
        }

        top_result_pq.push(std::pair<float, int>(value, i));

        // If at capacity, kick the smallest value out.
        if (top_result_pq.size() > num_results) {
            top_result_pq.pop();
        }
    }

    // Copy to output vector and reverse into descending order.
    while (!top_result_pq.empty()) {
        top_results->push_back(top_result_pq.top());
        top_result_pq.pop();
    }
    std::reverse(top_results->begin(), top_results->end());
}

void ReadLabelsFile(const std::string &file_name,
                    std::vector<std::string> *result,
                    size_t *found_label_count);

// Utility functions for OpenCV frames.
void printFrameInfo(cv::Mat &frame);
void paintRow(cv::Mat &frame, int row, int color);