It is a Project to detect the disteraction of driver during driving
# Distracted Driver Detection using MobileNet

## Overview
This project aims to detect distracted drivers using a pre-trained MobileNet convolutional neural network. The dataset used for training and testing contains images of drivers engaged in various activities such as texting, talking on the phone, eating, and more. The MobileNet model, pre-trained on the ImageNet dataset, is fine-tuned on the distracted driver dataset to classify images into different categories of driver distraction.

## Dependencies
- TensorFlow
- NumPy
- pandas
- Matplotlib
- OpenDatasets
- keras

## Usage
1. Download the distracted driver dataset from the Kaggle competition [here](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data).
2. Run the provided Python script to preprocess the data, build the MobileNet model, and train the model on the distracted driver dataset.
3. Optionally, fine-tune the model or experiment with different hyperparameters for better performance.

## Steps
1. **Data Download**: Download the distracted driver dataset from the Kaggle competition using the `opendatasets` library.
2. **Model Selection**: Use the MobileNet architecture, pre-trained on the ImageNet dataset, as the base convolutional neural network for feature extraction.
3. **Model Fine-tuning**: Fine-tune the MobileNet model on the distracted driver dataset by freezing the convolutional base and training the added classification layers.
4. **Data Augmentation**: Use image data generators to perform data augmentation, including zooming, rotation, and horizontal flipping, to increase the diversity of the training dataset.
5. **Model Training**: Train the model on the augmented training data and evaluate its performance on the validation data.

## Results
- The trained model achieves a certain level of accuracy in detecting distracted drivers based on the provided dataset.
- Performance metrics and visualizations can be used to analyze the model's performance and identify areas for improvement.

## Future Work
- Explore other pre-trained models or custom architectures for better performance.
- Experiment with different data augmentation techniques and hyperparameters to improve model generalization and robustness.
- Deploy the trained model for real-time inference or integrate it into a larger system for driver safety applications.
