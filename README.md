# Face Recognition System

## Description
This project implements a state-of-the-art Face Recognition System using the Inception ResNet architecture. The system is designed to recognize faces from images by training a model on a dataset of known faces.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone <https://github.com/zeeshaan28/face-recognition>
   cd face-recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Training the Model**:
   To train the face recognition model, run the following command:
   ```bash
   python train.py
   ```

2. **Testing the Model**:
   To test the trained model on new images, use:
   ```bash
   python test.py
   ```

3. **Docker Setup**:
   If you prefer to use Docker, you can build and run the Docker container:
   ```bash
   docker build -t face-recognition .
   docker run --gpus all face-recognition
   ```

## Requirements
The following packages are required for this project:
- TensorFlow==2.3.0
- Keras==2.4.3
- scikit-learn
- pickle-mixin
- pillow
- opencv-contrib-python-headless
- mtcnn
- numpy==1.16.0

## Training Data
If you would like to train the model and need access to the same training and test data, please contact me directly.

## Acknowledgments
- The Inception ResNet architecture is inspired by the work of Google on face recognition.
- Special thanks to the contributors and the open-source community for their invaluable resources.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.