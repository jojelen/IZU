#include "utils.h"

#include <fstream>
#include <iostream>

using namespace std;

void errExit(const string_view &msg) {
  cerr << "[ERROR]: " << msg << endl;
  exit(-1);
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

vector<uint8_t> read_frame(const cv::Mat &frame) {
  const uint8_t *bmp_pixels = frame.data;
  int row_size = frame.rows;
  int height = frame.cols;

  if (!frame.isContinuous())
    errExit("Data is continous!\n");

  // Decode image, allocating tensor once the image size is known
  return decode_bmp(bmp_pixels, row_size, row_size, abs(height), 3, true);
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

void ReadLabelsFile(const string &file_name, std::vector<string> *result,
                    size_t *found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    errExit("Labels file " + file_name + " not found\n");
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
}