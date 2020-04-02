#pragma once

#include <stdint.h>
#include <vector>

void writeBmp(size_t width, size_t height, size_t channels,
              const std::vector<uint8_t> &data, const char *fileName);