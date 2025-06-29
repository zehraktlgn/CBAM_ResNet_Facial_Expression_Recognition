# CBAM_ResNet_Facial_Expression_Recognition

This project implements a facial expression recognition system using a ResNet50 backbone enhanced with the CBAM (Convolutional Block Attention Module) attention mechanism. The model is trained on the RAF-DB dataset to classify images into seven basic emotion categories: Surprise, Fear, Disgust, Happy, Sad, Angry, and Neutral.

Features
ResNet50 base model with CBAM attention module

Trained on RAF-DB dataset with 7 emotion classes

Evaluation metrics include accuracy, weighted F1-score, precision, and recall

Real-time emotion prediction from images

Training, validation, and test performance visualization

Dataset
RAF-DB (Real-world Affective Faces Database) is used for training and evaluation.

Model Architecture
Base model: ResNet50 with ImageNet pretrained weights

Attention module: CBAM (Convolutional Block Attention Module)

Output layer: Softmax classifier for 7 emotion classes

Training Details
Data augmentation: rotation, shift, zoom, horizontal flip

Optimizer: Adam

Loss function: Categorical crossentropy

Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

Evaluation Metrics
Accuracy

Weighted F1-Score

Precision

Recall

Confusion matrix and classification report

Prediction
The model supports emotion prediction on single or multiple images.
