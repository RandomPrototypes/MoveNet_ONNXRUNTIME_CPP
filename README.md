# MoveNet_ONNXRUNTIME_CPP
Sample code using MoveNet with onnxruntime and openCV in cpp 

Based on [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/115_MoveNet)  
[onnxruntime official repository](https://github.com/microsoft/onnxruntime)

# How to build

First, install [onnxruntime](https://github.com/microsoft/onnxruntime).  
Then, run  
```
mkdir build
cd build
cmake ..
make
```
You may have to set the variables `ONNX_RUNTIME_SESSION_INCLUDE_DIRS` and `ONNX_RUNTIME_LIB` if onnxruntime is not detected by cmake.

# How to download the onnx models

Use the scripts from [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/115_MoveNet)

# Command-line options

To change the model file, use --model followed by the model path  
To change the dnn input size, use --input_size followed by the input size (one number)  
To change the keypoint score threshold, use --keypoint_score followed by the score threshold (between 0 and 1)  
To run onnxruntime with cuda, use --cuda  
To change the device id of the camera, use --device_id followed by the camera id (one number)  


