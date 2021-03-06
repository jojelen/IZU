#include "TfLite.h"
#include "bmp.h"
#include "utils.h"

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"

#include <iostream>

using namespace std;

TfLite::TfLite() {}

TfLite::~TfLite() {}

void TfLite::loadModel(const char *modelFile)
{
    TIMER

    mModel = tflite::FlatBufferModel::BuildFromFile(modelFile);
    if (!mModel)
        errExit("Couldn't build model from " + string(modelFile));

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*mModel, resolver)(&mInterpreter);
    if (!mInterpreter)
        errExit("Couldn't build interpreter.");

    const TfLiteGpuDelegateOptions options = {
        .metadata = NULL,
        .compile_options =
            {
                .precision_loss_allowed = 1,
                .preferred_gl_object_type = TFLITE_GL_OBJECT_TYPE_FASTEST,
                .dynamic_batch_enabled = 0,
                .inline_parameters = 0,
            },
    };
    mDelegate.reset(TfLiteGpuDelegateCreate(&options));
    if (mInterpreter->ModifyGraphWithDelegate(mDelegate.get()) != kTfLiteOk)
        errExit("Couldn't modify graph with gpu delegate.");

    // Increases performance on x86 to half the inference time.
    mInterpreter->SetNumThreads(4);
    printInterpreterInfo();
}

void TfLite::printInterpreterInfo() const
{
    cout << "Interpreter info:\n";
    cout << "tensors size: " << mInterpreter->tensors_size() << "\n";
    cout << "nodes size: " << mInterpreter->nodes_size() << "\n";
    cout << "inputs: " << mInterpreter->inputs().size() << "\n";
    cout << "input(0) name: " << mInterpreter->GetInputName(0) << "\n";
}

void TfLite::runInference(const char *inputFile)
{
    TIMER

    if (mInterpreter->AllocateTensors() != kTfLiteOk)
        errExit("Failed allocating tensors.");

    loadBmpImage(inputFile);

    // Running inference
    if (mInterpreter->Invoke() != kTfLiteOk)
        errExit("Failed to invoke tflite.");

    printTopResults();
}

void TfLite::printOps() const
{
    printf("Loading model\n");
    printf("Getting opCodes\n");
    auto opCodes = mModel.get()->GetModel()->operator_codes();
    if (opCodes) {
        printf("Found %lu opCodes!\n", (*opCodes).size());

        for (const auto &opcode : *opCodes) {
            cout << "Builtin operator nr: " << opcode->builtin_code() << endl;
            if (opcode->custom_code())
                cout << "Custom operator: " << opcode->custom_code()->str()
                     << std::endl;
        }
    }
    else {
        cout << "Couldn't get operator_codes()\n";
    }
}

void TfLite::runInference(const cv::Mat &frame)
{
        if (mInterpreter->AllocateTensors() != kTfLiteOk)
            errExit("Failed allocating tensors.");

    loadFrame(frame);

    // Running inference.
#ifdef TIME
    {
        Timer timer("Invoke()");
#endif
        if (mInterpreter->Invoke() != kTfLiteOk)
            errExit("Failed to invoke tflite.");
#ifdef TIME
    }
#endif

    // Used for image classification.
    // printTopResults();
}

std::vector<TfLiteTensor *> TfLite::getOutputs() const
{
    const vector<int> outputs = mInterpreter->outputs();
    vector<TfLiteTensor *> outputTensors;

    for (auto &o : outputs)
        outputTensors.push_back(mInterpreter->tensor(o));

    return outputTensors;
}

void TfLite::printInputOutputInfo() const
{
    const vector<int> inputs = mInterpreter->inputs();
    const vector<int> outputs = mInterpreter->outputs();
    cout << "Nr of inputs = " << inputs.size() << "\n";
    cout << "Nr of outputs = " << outputs.size() << "\n";

    for (size_t i = 0; i < inputs.size(); ++i) {
        TfLiteIntArray *dims = mInterpreter->tensor(inputs[i])->dims;
        cout << "Input tensor " + to_string(i + 1) + " dims = [";
        if (dims->size >= 1) {
            cout << dims->data[0];
            for (int i = 1; i < dims->size; ++i)
                cout << ", " << dims->data[i];
        }
        cout << "]\n";
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        TfLiteIntArray *dims = mInterpreter->tensor(outputs[i])->dims;
        cout << "Output tensor " + to_string(i + 1) + " dims = [";
        if (dims->size >= 1) {
            cout << dims->data[0];
            for (int i = 1; i < dims->size; ++i)
                cout << ", " << dims->data[i];
        }
        cout << "]\n";
    }
}

void TfLite::loadFrame(const cv::Mat &frame)
{
    TIMER

    const vector<int> inputs = mInterpreter->inputs();
    const vector<int> outputs = mInterpreter->outputs();

    int input = inputs[0]; // Index of input tensor;

    size_t frameSize = frame.total() * frame.elemSize();
    size_t inputSize = mInterpreter->tensor(input)->bytes;
    if (frameSize != inputSize) {
        errExit("Frame's byte size doesn't match the models input.");
    }
    uint8_t *inputDataPtr = mInterpreter->typed_tensor<uint8_t>(input);

    // Assuming same layout:
    for (size_t i = 0; i < inputSize; ++i) {
        *(inputDataPtr + i) = *(frame.data + i);
    }

    if (mWriteInputBmp) {
        TfLiteIntArray *dims = mInterpreter->tensor(input)->dims;

        int height = dims->data[1];
        int width = dims->data[2];
        int channels = dims->data[3];
        writeBmp(width, height, channels,
                 mInterpreter->typed_tensor<uint8_t>(input), "temp.bmp");
    }
}

void TfLite::loadBmpImage(const char *bmpFile)
{
    const vector<int> inputs = mInterpreter->inputs();
    const vector<int> outputs = mInterpreter->outputs();

    int input = inputs[0]; // Index of input tensor;

    printInputOutputInfo();
    TfLiteIntArray *dims = mInterpreter->tensor(input)->dims;

    int wanted_height = dims->data[1];
    int wanted_width = dims->data[2];
    int wanted_channels = dims->data[3];

    // Loading bmp image into buffer
    int image_width = 0;
    int image_height = 0;
    int image_channels = 0;
    auto in = readBmp(bmpFile, &image_width, &image_height, &image_channels);

    switch (mInterpreter->tensor(input)->type) {
    case kTfLiteFloat32:
        resize<float>(mInterpreter->typed_tensor<float>(input), in.data(),
                      image_height, image_width, image_channels, wanted_height,
                      wanted_width, wanted_channels);
        break;
    case kTfLiteUInt8:
        resize<uint8_t>(mInterpreter->typed_tensor<uint8_t>(input), in.data(),
                        image_height, image_width, image_channels,
                        wanted_height, wanted_width, wanted_channels);
        break;
    default:
        cout << "[ERROR]: Cannot handle input type "
             << mInterpreter->tensor(input)->type << " yet";
        exit(-1);
    }
    if (mWriteInputBmp)
        writeBmp(wanted_width, wanted_height, wanted_channels,
                 mInterpreter->typed_tensor<uint8_t>(input), "temp.bmp");
}

void TfLite::printTopResults() const
{
    std::vector<std::pair<float, int>> top_results;

    int output = mInterpreter->outputs()[0];
    TfLiteIntArray *output_dims = mInterpreter->tensor(output)->dims;
    // assume output dims to be something like (1, 1, ... ,size)
    auto output_size = output_dims->data[output_dims->size - 1];
    int results = 10;
    float threshold = 0.001;
    switch (mInterpreter->tensor(output)->type) {
    case kTfLiteFloat32:
        get_top_n<float>(mInterpreter->typed_output_tensor<float>(0),
                         output_size, results, threshold, &top_results, true);
        break;
    case kTfLiteUInt8:
        get_top_n<uint8_t>(mInterpreter->typed_output_tensor<uint8_t>(0),
                           output_size, results, threshold, &top_results,
                           false);
        break;
    default:
        errExit("cannot handle output type " +
                to_string(mInterpreter->tensor(output)->type) + " yet");
        exit(-1);
    }

    std::vector<string> labels;
    size_t label_count;

    ReadLabelsFile("res/imageClass/labels_mobilenet_quant_v1_224.txt", &labels,
                   &label_count);

    // Print top results
    cout << "\nObject detection results:\n";
    for (const auto &result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        cout << confidence << ": " << index << " " << labels[index] << "\n";
    }
}
