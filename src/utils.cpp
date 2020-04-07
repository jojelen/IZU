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