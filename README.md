[//]: # (Image References)

[image1]: ./images/center_2018_12_30_15_53_59_752.jpg "Center Images Train 1"
[image2]: ./images/left_2018_12_30_15_53_59_752.jpg "Left Images Train 1"
[image3]: ./images/right_2018_12_30_15_53_59_752.jpg "Right Images Train 1"

[image4]: ./images/center_2018_12_30_15_53_59_752.jpg "Center Images Train 2"
[image5]: ./images/left_2018_12_30_15_53_59_752.jpg "Left Images Train 2"
[image6]: ./images/right_2018_12_30_15_53_59_752.jpg "Right Images Train 2"

[image7]: ./images/center_2018_12_30_17_05_31_284.jpg "Center Images Train 3"
[image8]: ./images/left_2018_12_30_17_05_31_284.jpg "Left Images Train 3"
[image9]: ./images/right_2018_12_30_17_05_31_284.jpg "Right Images Train 3"

## Project Description

In this project, I use a neural network to clone car driving behavior.  It is a supervised regression problem between the car steering angles and the road images in front of a car.  

Those images were taken from three different camera angles (from the center, the left and the right of the car).  

The network is based on [The NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which has been proven to work in this problem domain.



As image processing is involved, the model is using convolutional layers for automated feature engineering.  

### Files included

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

### Run the pretrained model

Start up [the Udacity self-driving simulator](https://github.com/udacity/self-driving-car-sim), choose a scene and press the Autonomous Mode button.  Then, run the model as follows:

```python
python drive.py model.h5
```

### To train the model

You'll need the data folder which contains the training images.

```python
python model.py
```

This will generate a file `model.h5` 

## Model Architecture Design

#### 1. An appropriate model architecture has been employed

The design of the network is based on [the NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which has been used by NVIDIA for the end-to-end self driving test.  As such, it is well suited for the project.  

It is a deep convolution network which works well with supervised image classification / regression problems.  As the NVIDIA model is well documented, I was able to focus how to adjust the training images to produce the best result with some adjustments to the model to avoid overfitting and adding non-linearity to improve the prediction.

I've added the following adjustments to the model. 

- I used Lambda layer to normalized input images to avoid saturation and make gradients work better.
- I've also included ELU for activation function for every layer except for the output layer to introduce non-linearity.

#### 2. Attempts to reduce overfitting in the model

I tried using a drop out layer, but found it didn't do anything substantial.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.  The easiest method to reduce overfitting is just watching the loss for training and validation and making sure they converge and stopping at that point.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

For training, started by driving 2 laps clockwise and then drove counter-clockwise. Also would some times recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to move back to center.

To augment the data set, I also flipped images and angles thinking that this would first increase the number of data samples I had, but also strengthen my distributions of left, right, and straight angles. For example here is a rouge break down of angles with left being -.15 and right being .15 less than or greater than.

Here are a few random images from the training set left, center, and then right. 

![alt text][image2] ![alt text][image1] ![alt text][image3]

![alt text][image5] ![alt text][image4] ![alt text][image6]

![alt text][image9] ![alt text][image7] ![alt text][image9]

#### 5. Final Model Architecture

In the end, the model looks like as follows:

- Image normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Fully connected: neurons: 100, activation: RELU
- Fully connected: neurons:  50, activation: RELU
- Fully connected: neurons:  10, activation: RELU
- Fully connected: neurons:   1 (output)

The below is an model structure output from the Keras which gives more details on the shapes and the number of parameters.

Layer (type)                     | Output Shape        | Param #  | Connected to          |
| -------------------------------- |:-------------------:| --------:| --------------------: |
| lambda_1 (Lambda)                | (None, 160, 320, 3) | 0        | lambda_input_1[0][0]  |
| cropping2d_1 (Cropping2D)        | (None, 65, 320, 3)  | 0        | lambda_1[0][0]        |
| convolution2d_1 (Convolution2D)  | (None, 31, 158, 24) | 1824     | cropping2d_1[0][0]    |
| convolution2d_2 (Convolution2D)  | (None, 14, 77, 36)  | 21636    | convolution2d_1[0][0] |
| convolution2d_3 (Convolution2D)  | (None, 5, 37, 48)   | 43248    | convolution2d_2[0][0] |
| convolution2d_4 (Convolution2D)  | (None, 3, 35, 64)   | 27712    | convolution2d_3[0][0] |
| convolution2d_5 (Convolution2D)  | (None, 1, 33, 64)   | 36928    | convolution2d_4[0][0] |
| flatten_1 (Flatten)              | (None, 2112)        | 0        | convolution2d_5[0][0] |
| dense_1 (Dense)                  | (None, 100)         | 211300   | flatten_1[0][0]       |
| dense_2 (Dense)                  | (None, 50)          | 5050     | dense_1[0][0]         |
| dense_3 (Dense)                  | (None, 10)          | 510      | dense_2[0][0]         |
| dense_4 (Dense)                  | (None, 1)           | 11       | dense_3[0][0]         |

**Total params**: 348,219

**Trainable params**: 348,219

**Non-trainable params**: 0

## References
- NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
- Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim
