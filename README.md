# Mask Detection with OpenCV and TensorFlow

This project is a real-time face mask detection system using OpenCV for face detection and a pre-trained TensorFlow model for classification.

## Features
- Detects faces in a video stream using OpenCV's Haar Cascade.
- Classifies faces as "Wearing Mask" or "No Mask" using a TensorFlow deep learning model.
- Displays real-time classification results with bounding boxes.

## Dataset
This project uses the **Face Mask Detection Dataset** from Kaggle:
[Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install opencv-python numpy tensorflow
```

## How It Works
- The script captures video from the webcam.
- OpenCV detects faces in the frame.
- The detected face is resized and preprocessed for the TensorFlow model.
- The model predicts whether the person is wearing a mask or not.
- The result is displayed on the video feed with colored bounding boxes.

## Controls
- Press `q` to exit the program.

## Contributing
Feel free to fork this repository and improve the detection accuracy or add new features.

## Additional Information
If you have any suggestions or want to add new features, feel free to contribute or open an issue!
