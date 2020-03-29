#pragma once

#include <stdint.h>
#include <vector>

void writeBmp(size_t width, size_t height, const std::vector<uint8_t> &data);

void testBmpClass();