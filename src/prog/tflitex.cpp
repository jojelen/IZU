#include "tensorflow/lite/builtin_op_data.h" //for resize
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h" // For fancy printing of inte
#include "tensorflow/lite/string_util.h"          // for resize
#include <fstream>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <string_view>

using namespace std;

void errExit(const string_view &msg) {
  cerr << "[ERROR]: " << msg << endl;
  exit(-1);
}

template <class T>
void resize(T *out, uint8_t *in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels) {
  int number_of_pixels = image_height * image_width * image_channels;
  std::unique_ptr<tflite::Interpreter> interpreter(new tflite::Interpreter);

  int base_index = 0;

  // two inputs: input and new_sizes
  interpreter->AddTensors(2, &base_index);
  // one output
  interpreter->AddTensors(1, &base_index);
  // set input and output tensors
  interpreter->SetInputs({0, 1});
  interpreter->SetOutputs({2});

  // set parameters of tensors
  TfLiteQuantizationParams quant;
  interpreter->SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "input",
      {1, image_height, image_width, image_channels}, quant);
  interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                            quant);
  interpreter->SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "output",
      {1, wanted_height, wanted_width, wanted_channels}, quant);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  const TfLiteRegistration *resize_op = resolver.FindOp(
      tflite::BuiltinOperator::BuiltinOperator_RESIZE_BILINEAR, 1);
  auto *params = reinterpret_cast<TfLiteResizeBilinearParams *>(
      malloc(sizeof(TfLiteResizeBilinearParams)));
  params->align_corners = false;
  params->half_pixel_centers = false;
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,
                                     nullptr);

  interpreter->AllocateTensors();

  // fill input image
  // in[] are integers, cannot do memcpy() directly
  auto input = interpreter->typed_tensor<float>(0);
  for (int i = 0; i < number_of_pixels; i++) {
    input[i] = in[i];
  }

  // fill new_sizes
  interpreter->typed_tensor<int>(1)[0] = wanted_height;
  interpreter->typed_tensor<int>(1)[1] = wanted_width;

  interpreter->Invoke();

  auto output = interpreter->typed_tensor<float>(2);
  auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;

  bool floating = false;
  float input_mean = 0.;
  float input_std = 1.;

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
               bool input_floating) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
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

std::vector<uint8_t> decode_bmp(const uint8_t *input, int row_size, int width,
                                int height, int channels, bool top_down) {
  std::vector<uint8_t> output(height * width * channels);
  for (int i = 0; i < height; i++) {
    int src_pos;
    int dst_pos;

    for (int j = 0; j < width; j++) {
      if (!top_down) {
        src_pos = ((height - 1 - i) * row_size) + j * channels;
      } else {
        src_pos = i * row_size + j * channels;
      }

      dst_pos = (i * width + j) * channels;

      switch (channels) {
      case 1:
        output[dst_pos] = input[src_pos];
        break;
      case 3:
        // BGR -> RGB
        output[dst_pos] = input[src_pos + 2];
        output[dst_pos + 1] = input[src_pos + 1];
        output[dst_pos + 2] = input[src_pos];
        break;
      case 4:
        // BGRA -> RGBA
        output[dst_pos] = input[src_pos + 2];
        output[dst_pos + 1] = input[src_pos + 1];
        output[dst_pos + 2] = input[src_pos];
        output[dst_pos + 3] = input[src_pos + 3];
        break;
      default:
        errExit("Unexpected number of channels: " + to_string(channels));
        break;
      }
    }
  }
  return output;
}

vector<uint8_t> read_bmp(const std::string &input_bmp_name, int *width,
                         int *height, int *channels) {
  int begin, end;

  ifstream file(input_bmp_name, std::ios::in | std::ios::binary);
  if (!file) {
    errExit("input file " + input_bmp_name + " not found\n");
  }

  begin = file.tellg();
  file.seekg(0, std::ios::end);
  end = file.tellg();
  size_t len = end - begin;

  cout << "len: " << len << "\n";

  // Fill img_bytes with file data
  std::vector<uint8_t> img_bytes(len);
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char *>(img_bytes.data()), len);

  // Extract header info
  const int32_t header_size =
      *(reinterpret_cast<const int32_t *>(img_bytes.data() + 10));
  *width = *(reinterpret_cast<const int32_t *>(img_bytes.data() + 18));
  *height = *(reinterpret_cast<const int32_t *>(img_bytes.data() + 22));
  const int32_t bpp =
      *(reinterpret_cast<const int32_t *>(img_bytes.data() + 28));
  *channels = bpp / 8;

  cout << "width, height, channels: " << *width << ", " << *height << ", "
       << *channels << "\n";

  // there may be padding bytes when the width is not a multiple of 4 bytes
  // 8 * channels == bits per pixel
  const int row_size = (8 * *channels * *width + 31) / 32 * 4;

  // if height is negative, data layout is top down
  // otherwise, it's bottom up
  bool top_down = (*height < 0);

  // Decode image, allocating tensor once the image size is known
  const uint8_t *bmp_pixels = &img_bytes[header_size];
  return decode_bmp(bmp_pixels, row_size, *width, abs(*height), *channels,
                    top_down);
}

TfLiteStatus ReadLabelsFile(const string &file_name,
                            std::vector<string> *result,
                            size_t *found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    errExit("Labels file " + file_name + " not found\n");
    return kTfLiteError;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}

void printOps(const char *file) {
  printf("Loading model\n");
  unique_ptr<tflite::FlatBufferModel> modelPtr =
      tflite::FlatBufferModel::BuildFromFile(file);

  printf("Getting opCodes\n");
  auto opCodes = modelPtr.get()->GetModel()->operator_codes();
  if (opCodes) {
    printf("Found %lu opCodes!\n", (*opCodes).size());

    for (const auto &opcode : *opCodes) {
      cout << "Builtin operator nr: " << opcode->builtin_code() << endl;
      if (opcode->custom_code())
        cout << "Custom operator: " << opcode->custom_code()->str()
             << std::endl;
    }
  } else {
    cout << "Couldn't get operator_codes()\n";
  }
}
// void *operator new(size_t size) {
//  cout << "Allocating " << size << " bytes" << endl;
//  void *p = malloc(size);
//  return p;
//}
//
// void operator delete(void *p) {
//  cout << "Deleting stuff " << endl;
//  free(p);
//}

void runInference(const char *modelFile, const char *inputFile) {
  unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(modelFile);
  if (!model)
    errExit("Couldn't build model from " + string(modelFile));

  tflite::ops::builtin::BuiltinOpResolver resolver;
  unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter)
    errExit("Couldn't build interpreter.");

  // Printing info
  cout << "Interpreter info:\n";
  cout << "tensors size: " << interpreter->tensors_size() << "\n";
  cout << "nodes size: " << interpreter->nodes_size() << "\n";
  cout << "inputs: " << interpreter->inputs().size() << "\n";
  cout << "input(0) name: " << interpreter->GetInputName(0) << "\n";

  const vector<int> inputs = interpreter->inputs();
  const vector<int> outputs = interpreter->outputs();

  if (interpreter->AllocateTensors() != kTfLiteOk)
    errExit("Failed allocating tensors.");

  PrintInterpreterState(interpreter.get());

  int input = inputs[0];
  cout << "input = " << input << "\n";

  TfLiteIntArray *dims = interpreter->tensor(input)->dims;
  cout << "input.dims.size = " << dims->size << "\n";
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];
  cout << "wanted_height = " << wanted_height << "\n";
  cout << "wanted_width = " << wanted_width << "\n";
  cout << "wanted_channels = " << wanted_channels << "\n";

  // Loading bmp image into buffer
  int image_width = 224;
  int image_height = 224;
  int image_channels = 3;
  auto in = read_bmp(inputFile, &image_width, &image_height, &image_channels);

  switch (interpreter->tensor(input)->type) {
  case kTfLiteFloat32:
    cout << "The input should be in float32 format\n";
    resize<float>(interpreter->typed_tensor<float>(input), in.data(),
                  image_height, image_width, image_channels, wanted_height,
                  wanted_width, wanted_channels);
    break;
  case kTfLiteUInt8:
    cout << "The input should be in uint8 format\n";
    resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in.data(),
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels);
    break;
  default:
    cout << "cannot handle input type " << interpreter->tensor(input)->type
         << " yet";
    exit(-1);
  }

  // Running inference
  if (interpreter->Invoke() != kTfLiteOk)
    errExit("Failed to invoke tflite.");

  std::vector<std::pair<float, int>> top_results;

  int output = interpreter->outputs()[0];
  TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = output_dims->data[output_dims->size - 1];
  int results = 10;
  float threshold = 0.001;
  switch (interpreter->tensor(output)->type) {
  case kTfLiteFloat32:
    get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                     results, threshold, &top_results, true);
    break;
  case kTfLiteUInt8:
    get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                       output_size, results, threshold, &top_results, false);
    break;
  default:
    errExit("cannot handle output type " +
            to_string(interpreter->tensor(output)->type) + " yet");
    exit(-1);
  }

  std::vector<string> labels;
  size_t label_count;

  if (ReadLabelsFile(
          "tfliteModels/imageClass/labels_mobilenet_quant_v1_224.txt", &labels,
          &label_count) != kTfLiteOk)
    errExit("Couldn't read labels file.");

  // Print top results
  for (const auto &result : top_results) {
    const float confidence = result.first;
    const int index = result.second;
    cout << confidence << ": " << index << " " << labels[index] << "\n";
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "usage: <tflite model>\n");
    return 1;
  }
  const char *modelFile = argv[1];
  const char *inputFile = argv[2];

  printOps(modelFile);

  runInference(modelFile, inputFile);

  return 0;
}