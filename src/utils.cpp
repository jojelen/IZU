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

Timer::Timer() : mMessage("Timer:")
{
    mStart = std::chrono::steady_clock::now();
}

Timer::Timer(std::string &&msg) : mMessage(msg)
{
    mStart = std::chrono::steady_clock::now();
}

Timer::~Timer()
{
  printDuration();
}

void Timer::printDuration() const
{
    auto mEnd = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = mEnd - mStart;
    auto ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    if (ns < 1e6) {
        cout << mMessage << ": " << ns << " ns\n";
    }
    else if (ns >= 1e6 && ns <= 1e9) {
        auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                .count();
        cout << mMessage << ": " << ms << " ms\n";
    }
    else {
        size_t sec = static_cast<size_t>(
            std::chrono::duration_cast<std::chrono::seconds>(duration).count());

        size_t min = sec / 60;
        size_t hours = min / 60;
        min %= 60;
        sec = sec - 60 * 60 * hours - 60 * min;
        cout << mMessage << ": " << hours << " h " << min << " m " << sec
             << " s\n";
    }
}
