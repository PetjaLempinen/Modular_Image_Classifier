**Modular CIFAR-10 Image Classifier**

A clean, modular deep learning project built with PyTorch that implements a Convolutional Neural Network (CNN) from scratch to classify the CIFAR-10 dataset. This project includes fully separated components for model architecture, dataset management, training, single-image inference, and real-time webcam classification.

**Project Structure**
```
modular-image-classifier/
├── data/                    # Automatically downloads/stores the CIFAR-10 dataset
├── saved_models/            # Stores trained model checkpoint weights (.pth)
└── src/
    ├── model.py             # Custom CNN architecture with BatchNorm & Dropout
    ├── dataset.py           # CIFAR-10 data loaders with training augmentations
    ├── train.py             # Training loop, loss tracking, and checkpoint saver
    ├── predict.py           # Single-image inference script
    └── webcam_predict.py    # Real-time webcam inference script using OpenCV
```

**Features**

Modular Architecture: Clear separation of concerns across dedicated modules for data pipelines, model definition, training, and inference.

Data Augmentation: Incorporates random cropping and horizontal flipping during training to improve generalization.

Regularization: Employs Batch Normalization (BatchNorm2d) and Dropout layers to stabilize learning and prevent overfitting.

Real-Time Inference: Includes an OpenCV-powered webcam script to test the trained model on live video streams.

**How to Use**

***Download requirements***

***Train the Model***
Trains the CNN on the CIFAR-10 dataset and saves the trained weights to saved_models/cifar10_cnn.pth:

```
python src/train.py
```

***Predict a Single Image***
Loads a saved model checkpoint to classify a local image file update the end of predict.py file to the path of your image and run:

```
python src/predict.py
```

***Run Live Webcam Classification***
Opens your computer's webcam, feeds live frames into the model in real-time, and overlays the predicted label and confidence score on screen. Press q to exit:

```
python src/webcam_predict.py
```