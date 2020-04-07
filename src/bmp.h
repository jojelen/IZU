#pragma once

#include <stdint.h>
#include <vector>

void writeBmp(size_t width, size_t height, size_t channels,
              const std::vector<uint8_t> &data, const char *fileName);

void writeBmp(size_t width, size_t height, size_t channels, uint8_t *data,
              const char *fileName);

// Decodes BMP data to RGB data.
//
// This means getting rid of padding and converting to RGB (or RGBA for 4
// channels).
//
// Ex:
// BGRBGR00
// BGRBGR00 -> RGBRGBRGBRGB
std::vector<uint8_t> decodeBmpData(const uint8_t *input, int width, int height,
                                   int channels);

// Returns a RGB(A) data vector containing the BMP file data.
std::vector<uint8_t> readBmp(const char *fileName, int *width, int *height,
                             int *channels);