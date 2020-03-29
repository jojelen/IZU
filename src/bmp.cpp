#include "bmp.h"
#include "utils.h"

#include <fstream>
#include <iostream>

using namespace std;

struct BMPHeader {
  uint16_t file_type = 0x4D42; // File type always BM which is 0x4D42.
  uint32_t file_size = 0;      // Size of the file (in bytes).
  uint16_t reserved1 = 0;      // Reserved, always 0.
  uint16_t reserved2 = 0;      // Reserved, always 0.
  uint32_t offset_data = 0;    // Start of pixel data.
};

struct DIBHeader {
  uint32_t size = 40; // Byte size of DIBHeader + ColorHeader.
  int32_t width = 0;  // Width in pixels.
  int32_t height = 0; // Height in pixels.
  // (if positive, bottom-up, with origin in lower left corner)
  // (if negative, top-down, with origin in upper left corner)

  uint16_t planes = 1;
  uint16_t bit_count = 32; // No. of bits per pixel.
  uint32_t compression = 3;
  uint32_t size_image = 0; // 0 - for uncompressed images.
  int32_t x_pixels_per_meter = 0;
  int32_t y_pixels_per_meter = 0;
  uint32_t colors_used = 0; // No. color indexes in the color table. Use 0 for
                            // the max number of colors allowed by bit_count.
  uint32_t colors_important = 0; // No. of colors used for displaying the
                                 // bitmap. If 0 all colors are required.
};

struct ColorHeader {
  uint32_t red_mask = 0x00ff0000;         // Bit mask for the red channel.
  uint32_t green_mask = 0x0000ff00;       // Bit mask for the green channel.
  uint32_t blue_mask = 0x000000ff;        // Bit mask for the blue channel.
  uint32_t alpha_mask = 0xff000000;       // Bit mask for the alpha channel.
  uint32_t color_space_type = 0x73524742; // Default "sRGB" (0x73524742).
  uint32_t unused[16]{0};                 // Unused data for sRGB color space.
};

class BMP {
public:
  BMP();
  BMP(const char *file);
  ~BMP();

  void read(const char *file);
  void write(const char *file);
  void printInfo() const;

private:
  void checkColorHeaderFormat(ColorHeader &mColorHeader);
  void writeHeadersAndData(ofstream &of);
  uint32_t makeStrideAligned(uint32_t align_stride);

  void readHeaders(ifstream &inp);
  void readData(ifstream &inp);
  void writeHeaders(ofstream &of);
  void writeData(ofstream &of);

  constexpr static size_t HEADER_SIZE = 14;
  constexpr static size_t DIB_HEADER_SIZE = 40;
  constexpr static size_t COLOR_HEADER_SIZE = 84;

  BMPHeader mFileHeader;
  DIBHeader mDibHeader;
  ColorHeader mColorHeader;
  vector<uint8_t> mData;
  uint32_t mRowStride = 0;
};

BMP::BMP() {}
BMP::BMP(const char *file) { read(file); }
BMP::~BMP() {}

void BMP::readHeaders(ifstream &inp) {
  inp.read((char *)&mFileHeader, HEADER_SIZE);
  if (mFileHeader.file_type != 0x4D42)
    errExit("File type error when reading BMP.");

  inp.read((char *)&mDibHeader, DIB_HEADER_SIZE);

  // Color header
  if (mDibHeader.bit_count == 32) {
    if (mDibHeader.size >= (DIB_HEADER_SIZE + COLOR_HEADER_SIZE)) {
      inp.read((char *)&mColorHeader, COLOR_HEADER_SIZE);
      checkColorHeaderFormat(mColorHeader);
    } else {
      std::cerr << "Warning! The file does not seem to contain bit mask "
                   "information\n";
      errExit("Unrecognized file format.");
    }
  }
}

void BMP::read(const char *file) {
  std::ifstream inp{file, ios::in | ios::binary};
  if (!inp)
    errExit("Unable to open " + string(file));

  readHeaders(inp);

  // Setting inp to point to data beginning
  inp.seekg(HEADER_SIZE + mDibHeader.size, ios::beg);

  // Overwrite size info, since we may throw away some info.
  mDibHeader.size = DIB_HEADER_SIZE + COLOR_HEADER_SIZE;
  mFileHeader.offset_data = HEADER_SIZE + DIB_HEADER_SIZE + COLOR_HEADER_SIZE;
  mFileHeader.file_size = mFileHeader.offset_data;

  readData(inp);
}

void BMP::readData(ifstream &inp) {
  if (mDibHeader.height < 0) {
    errExit("The program can treat only BMP images with the "
            "origin in the bottom left corner!");
  }

  size_t nrPixels = mDibHeader.width * mDibHeader.height;
  mData.resize(nrPixels * 4 /* bytes per pixel */);

  switch (mDibHeader.bit_count) {
  case 32: {
    inp.read((char *)mData.data(), mData.size());
    break;
  }
  case 24: {
    // Transform data to 32bit BGRA format.
    size_t padding = (((mDibHeader.width * 3) % 4) == 0)
                         ? 0
                         : 4 - ((mDibHeader.width * 3) % 4);
    vector<uint8_t> temp(padding);

    uint8_t *dataPtr = mData.data(); // Points to each pixel in mData.
    for (int y = 0; y < mDibHeader.height; ++y) {
      for (int x = 0; x < mDibHeader.width; ++x) {
        // BGR -> BGRA
        inp.read(reinterpret_cast<char *>(dataPtr), 3);
        *(dataPtr + 3) = 255;
        dataPtr += 4;
      }
      inp.read(reinterpret_cast<char *>(temp.data()), temp.size());
    }
    mDibHeader.bit_count = 32;
    mDibHeader.compression = 3;
    break;
  }
  default: { errExit("Invalid bit count when reading BMP file."); }
  }
  mDibHeader.size_image = mData.size();
  mFileHeader.file_size += mData.size();
}

void BMP::write(const char *file) {
  ofstream of{file, std::ios_base::binary};
  if (!of)
    errExit("Can't open the output image file.");

  printInfo();
  of.write((const char *)&mFileHeader, HEADER_SIZE);
  of.write((const char *)&mDibHeader, DIB_HEADER_SIZE);
  of.write((const char *)&mColorHeader, COLOR_HEADER_SIZE);
  of.write((const char *)mData.data(), mData.size());
}

void BMP::checkColorHeaderFormat(ColorHeader &mColorHeader) {
  ColorHeader expected_color_header;
  if (expected_color_header.red_mask != mColorHeader.red_mask ||
      expected_color_header.blue_mask != mColorHeader.blue_mask ||
      expected_color_header.green_mask != mColorHeader.green_mask ||
      expected_color_header.alpha_mask != mColorHeader.alpha_mask) {
    errExit(
        "Unexpected color mask format! The program expects the pixel data to "
        "be in the BGRA format");
  }
  if (expected_color_header.color_space_type != mColorHeader.color_space_type) {
    errExit("Unexpected color space type! The program expects sRGB values");
  }
}

void BMP::printInfo() const {
  cout << "BMP file header (" << HEADER_SIZE << " bytes):"
       << "\nfile type = " << hex << mFileHeader.file_type << dec
       << "\nfile size = " << mFileHeader.file_size
       << "\nreserved1 = " << mFileHeader.reserved1
       << "\nreserved2 = " << mFileHeader.reserved2
       << "\noffset = " << mFileHeader.offset_data << "\nBMP info header:("
       << DIB_HEADER_SIZE << " bytes):"
       << "\nsize = " << mDibHeader.size << "\nwidth = " << mDibHeader.width
       << "\nheight = " << mDibHeader.height
       << "\nplanes = " << mDibHeader.planes
       << "\nbit count = " << mDibHeader.bit_count
       << "\ncompression = " << mDibHeader.compression
       << "\nsize image = " << mDibHeader.size_image
       << "\nx ppm = " << mDibHeader.x_pixels_per_meter
       << "\ny ppm = " << mDibHeader.y_pixels_per_meter
       << "\ncolors used = " << mDibHeader.colors_used
       << "\ncolors important = " << mDibHeader.colors_important << "\n";
  if (mDibHeader.bit_count == 32) {
    cout << "Color header (" << COLOR_HEADER_SIZE << " bytes):"
         << "\nred mask = " << hex << mColorHeader.red_mask
         << "\ngreen mask = " << mColorHeader.green_mask
         << "\nblue mask = " << mColorHeader.blue_mask
         << "\nalpha mask = " << mColorHeader.alpha_mask << dec
         << "\ncolor space type = " << mColorHeader.color_space_type << "\n";
  }
}

void testBmpClass() {
  BMP image("res/Shapes.bmp");
  image.write("Shapes_copy.bmp");
  BMP image2("res/midsommer.bmp");
  image2.write("midsommer_copy.bmp");
  BMP image3("square.bmp");
  image3.write("square_copy.bmp");
}