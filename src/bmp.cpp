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
    uint16_t bpp = 32; // No. of bytes per pixel.
    uint32_t compression = 3;
    uint32_t size_image = 0; // 0 - for uncompressed images.
    int32_t x_pixels_per_meter = 0;
    int32_t y_pixels_per_meter = 0;
    uint32_t colors_used = 0; // No. color indexes in the color table. Use 0 for
                              // the max number of colors allowed by bpp.
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

// Class for managing import/export bmp pictures
class BMP {
  public:
    BMP();
    BMP(const char *file);
    ~BMP();

    void read(const char *file);
    void write(const char *file) const;
    void printInfo() const;
    void addData(size_t width, size_t height, const vector<uint8_t> &data);
    void addData(size_t width, size_t height, uint8_t *data, size_t len);

    int getWidth() const { return static_cast<int>(mDibHeader.width); }
    int getHeight() const { return static_cast<int>(mDibHeader.height); }
    int getChannels() const;
    vector<uint8_t> getData() const { return mData; };

  private:
    void checkColorHeaderFormat(ColorHeader &mColorHeader);
    void writeHeadersAndData(ofstream &of);
    uint32_t makeStrideAligned(uint32_t align_stride);

    void overwriteHeaders();
    void readHeaders(ifstream &inp);
    void readData(ifstream &inp);
    void writeHeaders(ofstream &of);
    void writeData(ofstream &of);

    // Get black BMP image when exporting 3 channel images (something missing
    // in header?). Temp solution is to always convert to 4 channels before
    // writing.
    void convertTo4Channels();

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

int BMP::getChannels() const { return static_cast<int>(mDibHeader.bpp) / 8; }

void BMP::readHeaders(ifstream &inp)
{
    inp.read(reinterpret_cast<char *>(&mFileHeader), HEADER_SIZE);
    if (mFileHeader.file_type != 0x4D42)
        errExit("File type error when reading BMP.");

    inp.read(reinterpret_cast<char *>(&mDibHeader), DIB_HEADER_SIZE);

    // Color header
    if (mDibHeader.bpp == 32) {
        if (mDibHeader.size >= (DIB_HEADER_SIZE + COLOR_HEADER_SIZE)) {
            inp.read((char *)&mColorHeader, COLOR_HEADER_SIZE);
            checkColorHeaderFormat(mColorHeader);
        }
        else {
            std::cerr << "Warning! The file does not seem to contain bit mask "
                         "information\n";
            errExit("Unrecognized file format.");
        }
    }
}

void BMP::read(const char *file)
{
    std::ifstream inp{file, ios::in | ios::binary};
    if (!inp)
        errExit("Unable to open " + string(file));

    readHeaders(inp);
    readData(inp);
}

void BMP::overwriteHeaders()
{
    mDibHeader.size = DIB_HEADER_SIZE + COLOR_HEADER_SIZE;
    mFileHeader.offset_data = HEADER_SIZE + DIB_HEADER_SIZE + COLOR_HEADER_SIZE;
    mFileHeader.file_size = mFileHeader.offset_data;
}

void BMP::addData(size_t width, size_t height, const vector<uint8_t> &data)
{
    overwriteHeaders();
    mFileHeader.file_size = mFileHeader.offset_data + data.size();
    mDibHeader.width = static_cast<int32_t>(width);
    mDibHeader.height = static_cast<int32_t>(height);
    mDibHeader.size_image = static_cast<uint32_t>(data.size());
    mData = data;
}

void BMP::addData(size_t width, size_t height, uint8_t *data, size_t len)
{
    overwriteHeaders();
    mFileHeader.file_size = mFileHeader.offset_data + len;
    mDibHeader.width = static_cast<int32_t>(width);
    mDibHeader.height = static_cast<int32_t>(height);
    mDibHeader.size_image = static_cast<uint32_t>(len);
    mData.clear();
    mData.insert(mData.begin(), data, data + len);
}

void BMP::readData(ifstream &inp)
{
    if (mDibHeader.height < 0) {
        errExit("The program can treat only BMP images with the "
                "origin in the bottom left corner!");
    }

    // Calc size of data in bytes.
    inp.seekg(0, ios::end);
    int end = inp.tellg();
    size_t len = end - HEADER_SIZE - mDibHeader.size;

    // Setting inp to point to data beginning and read into mData.
    inp.seekg(HEADER_SIZE + mDibHeader.size, ios::beg);
    mData.resize(len);
    inp.read(reinterpret_cast<char *>(mData.data()), mData.size());

    // Update headers.
    overwriteHeaders();
    mDibHeader.size_image = mData.size();
    mFileHeader.file_size += mData.size();
}

void BMP::convertTo4Channels()
{
    if (mDibHeader.bpp == 24) // Transform data to 32bit BGRA format.
    {
        vector<uint8_t> newData;
        size_t pixels = mDibHeader.width * mDibHeader.height * 3;
        for (size_t i = 0; i < pixels; i += 3) {
            newData.push_back(mData[i]);
            newData.push_back(mData[i + 1]);
            newData.push_back(mData[i + 2]);
            newData.push_back(255);
        }
        addData(mDibHeader.width, mDibHeader.height, newData);
    }
}

void BMP::write(const char *file) const
{
    ofstream of{file, std::ios_base::binary};
    if (!of)
        errExit("Can't open the output image file.");

    // printInfo(); //DEBUG
    of.write((const char *)&mFileHeader, HEADER_SIZE);
    of.write((const char *)&mDibHeader, DIB_HEADER_SIZE);
    of.write((const char *)&mColorHeader, COLOR_HEADER_SIZE);
    of.write((const char *)mData.data(), mData.size());
}

void BMP::checkColorHeaderFormat(ColorHeader &mColorHeader)
{
    ColorHeader expected_color_header;
    if (expected_color_header.red_mask != mColorHeader.red_mask ||
        expected_color_header.blue_mask != mColorHeader.blue_mask ||
        expected_color_header.green_mask != mColorHeader.green_mask ||
        expected_color_header.alpha_mask != mColorHeader.alpha_mask) {
        errExit("Unexpected color mask format! The program expects the pixel "
                "data to "
                "be in the BGRA format");
    }
    if (expected_color_header.color_space_type !=
        mColorHeader.color_space_type) {
        errExit("Unexpected color space type! The program expects sRGB values");
    }
}

void BMP::printInfo() const
{
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
         << "\nbit count = " << mDibHeader.bpp
         << "\ncompression = " << mDibHeader.compression
         << "\nsize image = " << mDibHeader.size_image
         << "\nx ppm = " << mDibHeader.x_pixels_per_meter
         << "\ny ppm = " << mDibHeader.y_pixels_per_meter
         << "\ncolors used = " << mDibHeader.colors_used
         << "\ncolors important = " << mDibHeader.colors_important << "\n";
    if (mDibHeader.bpp == 32) {
        cout << "Color header (" << COLOR_HEADER_SIZE << " bytes):"
             << "\nred mask = " << hex << mColorHeader.red_mask
             << "\ngreen mask = " << mColorHeader.green_mask
             << "\nblue mask = " << mColorHeader.blue_mask
             << "\nalpha mask = " << mColorHeader.alpha_mask << dec
             << "\ncolor space type = " << mColorHeader.color_space_type
             << "\n";
    }
}

void writeBmp(size_t width, size_t height, size_t channels,
              const std::vector<uint8_t> &data, const char *fileName)
{
    TIMER

    BMP image;
    if (channels == 4) {
        image.addData(width, height, data);
    }
    else if (channels == 3) {
        vector<uint8_t> newData;
        size_t pixels = width * height * channels;
        for (size_t i = 0; i < pixels; i += 3) {
            newData.push_back(data[i + 2]);
            newData.push_back(data[i + 1]);
            newData.push_back(data[i]);
            newData.push_back(255);
        }
        image.addData(width, height, newData);
    }
    else {
        errExit("Invalid channels argument for creating bmp.");
    }

    image.write(fileName);
}

void writeBmp(size_t width, size_t height, size_t channels, uint8_t *data,
              const char *fileName)
{
    TIMER

    BMP image;
    size_t pixels = width * height * channels;
    if (channels == 4) {
        image.addData(width, height, data, pixels);
    }
    else if (channels == 3) {
        vector<uint8_t> newData;
        for (size_t i = 0; i < pixels; i += 3) {
            newData.push_back(data[i + 2]);
            newData.push_back(data[i + 1]);
            newData.push_back(data[i]);
            newData.push_back(255);
        }
        image.addData(width, height, newData);
    }
    else {
        errExit("Invalid channels argument for creating bmp.");
    }

    image.write(fileName);
}

std::vector<uint8_t> decodeBmpData(const uint8_t *input, int width, int height,
                                   int channels)
{
    // Calculate row_size for the BMP image; it may be padded if it is not a
    // multiple of 4 bytes.
    const int row_size = (8 * channels * width + 31) / 32 * 4;
    // Data layout is top down if height is negative.
    bool top_down = (height < 0);
    if (top_down)
        height *= -1;

    std::vector<uint8_t> output(height * width * channels);
    for (int i = 0; i < height; i++) {
        int src_pos;
        int dst_pos;

        for (int j = 0; j < width; j++) {
            if (!top_down) {
                src_pos = ((height - 1 - i) * row_size) + j * channels;
            }
            else {
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
                errExit("Unexpected number of channels: " +
                        to_string(channels));
                break;
            }
        }
    }
    return output;
}

std::vector<uint8_t> readBmp(const char *fileName, int *width, int *height,
                             int *channels)
{
    BMP image(fileName);
    if (width)
        *width = image.getWidth();
    if (height)
        *height = image.getHeight();
    if (channels)
        *channels = image.getChannels();
    return decodeBmpData(image.getData().data(), image.getWidth(),
                         image.getHeight(), image.getChannels());
}
