#include "utils.h"
#include "bmp.h"

#include <fstream>
#include <iostream>

using namespace std;

void errExit(const string_view &msg)
{
    cerr << "[ERROR]: " << msg << endl;
    exit(-1);
}

vector<uint8_t> read_frame(const cv::Mat &frame)
{
    const uint8_t *bmp_pixels = frame.data;
    int row_size = frame.rows;
    int height = frame.cols;

    if (!frame.isContinuous())
        errExit("Data is continous!\n");

    // Decode image, allocating tensor once the image size is known
    return decode_bmp(bmp_pixels, row_size, row_size, abs(height), 3, true);
}

void ReadLabelsFile(const string &file_name, std::vector<string> *result,
                    size_t *found_label_count)
{
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