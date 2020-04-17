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