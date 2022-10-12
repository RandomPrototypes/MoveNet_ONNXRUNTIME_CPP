#include <onnxruntime_cxx_api.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <chrono>

void run_inference(Ort::Session& onnx_session, Ort::IoBinding& io_binding, const char *input_name, Ort::Value& src_tensor, float *src_tensor_data, int input_size, const cv::Mat& image, std::vector<cv::Point>* keypoints, std::vector<float>* scores)
{
	int image_width = image.cols;
	int image_height = image.rows;

	//Pre process:Resize, BGR->RGB, Reshape, float32 cast
	cv::Mat input_image;
	cv::resize(image, input_image, cv::Size(input_size, input_size));
	cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);
	const unsigned char *src = input_image.ptr<unsigned char>(0);
	float *dst = src_tensor_data;
	for(int i = input_size * input_size * 3; i > 0; i--)
		*dst++ = *src++;
	
	//Inference
	io_binding.BindInput(input_name, src_tensor);
	onnx_session.Run(Ort::RunOptions{nullptr}, io_binding);
	std::vector<Ort::Value> outputValues = io_binding.GetOutputValues();
	const cv::Mat keypoints_with_scores(17, 3, CV_32FC1, const_cast<float*>(outputValues[0].GetTensorData<float>()));
	
	//Postprocess:Calc Keypoint
	keypoints->clear();
	scores->clear();
	for(int index = 0; index < 17; index++) {
		const float *data = keypoints_with_scores.ptr<float>(index);
		int keypoint_x = static_cast<int>(image_width*data[1]);
		int keypoint_y = static_cast<int>(image_height*data[0]);
		float score = data[2];

		keypoints->push_back(cv::Point(keypoint_x, keypoint_y));
		scores->push_back(score);
	}
}

cv::Mat draw_debug(const cv::Mat& image, double elapsed_time, float keypoint_score_th, const std::vector<cv::Point>& keypoints, const std::vector<float>& scores)
{
	cv::Mat debug_image = image.clone();

	std::vector<std::tuple<int, int, cv::Scalar> > connect_list = {
		std::make_tuple(0, 1, cv::Scalar(255, 0, 0)),  // nose → left eye
		std::make_tuple(0, 2, cv::Scalar(0, 0, 255)),  // nose → right eye
		std::make_tuple(1, 3, cv::Scalar(255, 0, 0)),  // left eye → left ear
		std::make_tuple(2, 4, cv::Scalar(0, 0, 255)),  // right eye → right ear
		std::make_tuple(0, 5, cv::Scalar(255, 0, 0)),  // nose → left shoulder
		std::make_tuple(0, 6, cv::Scalar(0, 0, 255)),  // nose → right shoulder
		std::make_tuple(5, 6, cv::Scalar(0, 255, 0)),  // left shoulder → right shoulder
		std::make_tuple(5, 7, cv::Scalar(255, 0, 0)),  // left shoulder → left elbow
		std::make_tuple(7, 9, cv::Scalar(255, 0, 0)),  // left elbow → left wrist
		std::make_tuple(6, 8, cv::Scalar(0, 0, 255)),  // right shoulder → right elbow
		std::make_tuple(8, 10, cv::Scalar(0, 0, 255)),  // right elbow → right wrist
		std::make_tuple(11, 12, cv::Scalar(0, 255, 0)),  // left hip → right hip
		std::make_tuple(5, 11, cv::Scalar(255, 0, 0)),  // left shoulder → left hip
		std::make_tuple(11, 13, cv::Scalar(255, 0, 0)),  // left hip → left knee
		std::make_tuple(13, 15, cv::Scalar(255, 0, 0)),  // left knee → left ankle
		std::make_tuple(6, 12, cv::Scalar(0, 0, 255)),  // right shoulder → right hip
		std::make_tuple(12, 14, cv::Scalar(0, 0, 255)),  // right hip → right knee
		std::make_tuple(14, 16, cv::Scalar(0, 0, 255)),  // right knee → right ankle
	};

	//Connect Line
	for(auto& connection : connect_list) {
		int index01 = std::get<0>(connection);
		int index02 = std::get<1>(connection);
		cv::Scalar color = std::get<2>(connection);
		if(scores[index01] > keypoint_score_th && scores[index02] > keypoint_score_th) {
			cv::Point point01 = keypoints[index01];
			cv::Point point02 = keypoints[index02];
			cv::line(debug_image, point01, point02, color, 2);
		}
	}

	//Keypoint circle
	for(size_t i = 0; i < keypoints.size(); i++) {
		cv::Point keypoint = keypoints[i];
		float score = scores[i];
		if(score > keypoint_score_th)
			cv::circle(debug_image, keypoint, 3, cv::Scalar(0, 255, 0), -1);
	}

	//Inference elapsed time
	std::string str = "Elapsed Time : "+std::to_string(elapsed_time*1000)+"ms";
	cv::putText(debug_image, str, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

	return debug_image;
}

int main(int argc, char **argv) 
{
	const std::vector<std::string> args(argv + 1, argv + argc);

	std::string onnxModelFilename = "model_float32.onnx";
	int input_size = 192;
	float keypoint_score_th = 0.3;
	bool use_CUDA = false;
	int deviceID = 0;
	cv::Size size = cv::Size(640,480);//cv::Size(1280,720);

	for(size_t i = 0; i < args.size(); i++)
	{
		if(args[i] == "--model") {
			if(i + 1 < args.size()) {
				onnxModelFilename = args[i+1];
				i++;
			} else {
				printf("error : expected filename after --model\n");
				return 0;
			}
		} else if(args[i] == "--input_size") {
			if(i + 1 < args.size()) {
				try {
					input_size = std::stoi(args[i+1]);
					i++;
				} catch(const std::exception&) {
					printf("error : invalid argument after --input_size\n");
					return 0;
				}
			} else {
				printf("error : expected integer after --input_size\n");
				return 0;
			}
		} else if(args[i] == "--keypoint_score") {
			if(i + 1 < args.size()) {
				try {
					keypoint_score_th = std::stof(args[i+1]);
					i++;
				} catch(const std::exception&) {
					printf("error : invalid argument after --keypoint_score\n");
					return 0;
				}
			} else {
				printf("error : expected float after --keypoint_score\n");
				return 0;
			}
		} else if(args[i] == "--cuda") {
			use_CUDA = true;
		} else if(args[i] == "--device_id") {
			if(i + 1 < args.size()) {
				try {
					deviceID = std::stoi(args[i+1]);
					i++;
				} catch(const std::exception&) {
					printf("error : invalid argument after --device_id\n");
					return 0;
				}
			} else {
				printf("error : expected integer after --device_id\n");
				return 0;
			}
		} else {
			printf("unknown argument: %s\n", args[i].c_str());
			return 0;
		}
	}

	//Initialize video capture
	cv::VideoCapture cap;
	cap.open(deviceID);
	if (!cap.isOpened()) {
		printf("can not open device %d\n", deviceID);
		return 0;
	}
	
	//Creates the onnx runtime environment
	Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "movenet");
	Ort::SessionOptions sessionOptions;
	sessionOptions.SetIntraOpNumThreads(1);
	
	//Activates the CUDA backend
	if(use_CUDA) {
		OrtCUDAProviderOptions cuda_options;
		sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
	}
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	//Load model
	#ifdef _WIN32
		size_t filename_size = onnxModelFilename.size();
		wchar_t *wc_filename = new wchar_t[filename_size+1];
		size_t convertedChars = 0;
		mbstowcs_s(&convertedChars, wc_filename, filename_size+1, onnxModelFilename.c_str(), _TRUNCATE);
		Ort::Session onnx_session(env, wc_filename, sessionOptions);
		delete [] wc_filename;
	#else
		Ort::Session onnx_session(env, onnxModelFilename.c_str(), sessionOptions);
	#endif
	Ort::IoBinding io_binding(onnx_session);
	
	Ort::AllocatorWithDefaultOptions allocator;
	
	cap.set(cv::CAP_PROP_FRAME_WIDTH, size.width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, size.height);
	
	// Create tensors
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	
	std::vector<float> src_data(input_size * input_size * 3);
	std::vector<int64_t> src_dims = {1, input_size, input_size, 3};
	Ort::Value src_tensor = Ort::Value::CreateTensor<float>(memoryInfo, src_data.data(), src_data.size(), src_dims.data(), 4);

	char *input_name = onnx_session.GetInputName(0, allocator);
	char *output_name = onnx_session.GetOutputName(0, allocator);
	
	io_binding.BindOutput(output_name, memoryInfo);
	
	cv::Mat frame;
	while(true) 
	{
		auto start_time = std::chrono::steady_clock::now();
		//Capture read
		cap.read(frame);
		if (frame.empty()) {
			printf("error : empty frame grabbed");
			break;
		}

		cv::Mat debug_image = frame.clone();

		//Inference execution
		std::vector<cv::Point> keypoints;
		std::vector<float> scores;
		run_inference(onnx_session, io_binding, input_name, src_tensor, &src_data[0], input_size, frame, &keypoints, &scores);

		auto end_time = std::chrono::steady_clock::now();

		double elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()/1000000.0;

		//Draw 
		debug_image = draw_debug(debug_image, elapsed_time, keypoint_score_th, keypoints, scores);

		cv::imshow("MoveNet(singlepose) Demo", debug_image);
		int key = cv::waitKey(10);
		if(key == 27 || key == 'q')
			break;
	}
	return 0;
}
