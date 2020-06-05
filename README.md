# Computer Pointer Controller

The Computer Pointer Controller utilizes Computer Vision and Intel OpenVINO to control the position of your mouse based solely by the direction of your eyes and the position of your head. Using 4 models and 4 scripts, the Computer Pointer Controller can perform inference on video inputs and more.

## Project Set Up and Installation
The first and foremost thing required is the installation and setup of OpenVINO Toolkit and all of its dependencies. This will require Docker which CAN'T be installed on Windows Home. If you cannot install docker please migrate to another OS such as Ubuntu. The link to the installation guide can be found here.

Then to recieve the code you can do a git clone of this repository: https://github.com/ajaybehuria/Computer-Pointer-Controller

Then go to the Computer-Pointer-Controller directory

Then you have to start a virtual environment. You can do this with: 
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```
Then you can download the pretrained models with the following commands:
```
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
```
Use the -o argument to specify an output directory to store models if desired, however this is optional
## Project Structure
```bash
|--bin
    |--demo.mp4
|--src
    |--face_detection.py
    |--facial_landmarks_detection.py
    |--gaze_estimation.py
    |--head_pose_estimation.py
    |--input_feeder.py
    |--main.py
    |--mouse_controller.py
    |--__pycache__
|--README.md
|--requirements.txt
```
## Demo
After the setup is finished, you can run the model
Make sure you are in the correct directory and then run this command:
```
python starter/src/main.py -fl <Facial Landmarks Model Path> -hp <Head Pose Estimation Model Path> -g <Gaze Estimation Model Path> -i <Input Path(Can use videos or webcam)> -f <Face Detection Model Path> -d <Device Name(Supports CPU, GPU, FPGA)>
```
Here is an example command:
```
python starter/src/main.py -fl landmarks-regression-retail-0009.xml -hp head-pose-estimation-adas-0001.xml -g gaze-estimation-adas-0002.xml -i starter/bin/demo.mp4 -f face-detection-adas-binary-0001.xml -d 'CPU' 
```
## Documentation
You can see more about OpenVINO Python [here](https://docs.openvinotoolkit.org/latest/_inference_engine_ie_bridges_python_docs_api_overview.html)
- [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
- [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
- [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
- [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

These are the arguments for running:  
```
-fl: Specify the path of Face Detection model's xml file  
-hp: Specify the path of Head Pose Estimation model's xml file  
-g: Specify the path of Gaze Estimation model's xml file  
-i: Specify the path of input video file or enter cam for taking input video from webcam  
-d: Specify the target device to infer the video file on the model. Suppoerted devices are: CPU, GPU, FPGA (For running on FPGA used HETERO:FPGA,CPU), MYRIAD.  
-l: Specify the absolute path of cpu extension if some layers of models are not supported on the device.  
-prob (optional) : Specify the probability threshold for face detection model to detect the face accurately from video frame.  
-flags (optional) : Specify the flags from fd, fld, hp, ge if you want to visualize the output of corresponding models of each frame (write flags with space seperation. Ex:- -flags fd fld hp).  
```
## Benchmarks
For different accuracies I got different results:
  
- FP32:
  - The total model loading time is : 821.239 ms
  - The total inference time is : 23.553 sec
  - The total FPS is : 9.505 fps
  
- FP16:
  - The total model loading time is : 593.893 ms
  - The total inference time is : 23.562 sec
  - The total FPS is : 9.504 fps
  
- INT8:
  - The total model loading time is : 807.322ms
  - The total inference time is : 23.525 sec
  - The total FPS is : 9.508 fps
  
## Results
There is very little difference in my results for each precision level. In terms of accuracy, results were extremely similar as well. I can't come to a conclusion based off of results only a millisecond apart. The only real outlier I found was the loading time for the FP16 precision. It was lower than both INT8 and FP32 by a big margin. This is extremely odd seeing as the FP16 accuracy requires more memory and computational power as INT8 and FP16. My results show that statistics can differ greatly sometimes and don't always stick to the rules.


# Computer-Pointer-Controller
