#include "TfLite.h"
#include "utils.h"

using namespace std;

void runInference(const char *modelFile, const char *inputFile) {
  TfLite tfLite;
  tfLite.loadModel(modelFile);

  tfLite.runInference(inputFile);
}

int main(int argc, char *argv[]) {
  if (argc != 3)
    errExit("usage: <tflite model>\n");

  const char *modelFile = argv[1];
  const char *inputFile = argv[2];

  runInference(modelFile, inputFile);

  return 0;
}