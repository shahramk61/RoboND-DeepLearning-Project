[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)


[image_0]: ./docs/misc/fcn.png
[image_1]: ./docs/misc/1.PNG
[image_2]: ./docs/misc/2.PNG
[image_3]: ./docs/misc/3.PNG
[image_4]: ./docs/misc/4.PNG

## Deep Learning project writeup
This document is prepared as a report on the deep learning segmentation project.



## Network Architecture

In this project a Fully convolution neural network is used for segmentation and object detection. the network is trained using the images captured from the simulator. 


* Encoder section:
    
    1-  separable convolution layer with 64 kernel size of 3x3, , stride of 2 , padding = 'SAME' + batch normalization
    
    2-  separable convolution layer with 126 kernel size of 3x3, stride of 2 , padding = 'SAME' + batch normalization
    
    3-  1x1 convolution layer with  264 kernal size of 1x1, stride of 1 , padding = 'SAME' + batch normalization
    
    
* Decoder section:
    
    4- bilinear upsampeling concatenated with layer 1
    
    5- bilinear upsampeling concatenated with input layer



Network code:

```
def fcn_model(inputs, num_classes):
    
    #Add Encoder Blocks. 
    l1 = encoder_block(inputs, 64, 2)
    l2 = encoder_block(l1, 128, 2)
    #Add 1x1 Convolution layer using conv2d_batchnorm().
    l3 = conv2d_batchnorm(l2, 256, kernel_size=1, strides=1)
    
    #Add the same number of Decoder Blocks as the number of Encoder Blocks
    l4 = decoder_block(l3,l1,128)
    l5 = decoder_block(l4, inputs, 64)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(l5)
```

## Reason for layer choice 
For this exercise in the encoder section the separable convolution layer is used instead of the traditional CNN layer for feature extraction in order to decease the number of parameters required to be learned. The 1x1 convolution layer is used to increase the number of feature map since its computation is less expensive in sense of computation. In decoder section instead of regular deconvolution layer(convolution transpose), bilinear upsampling is used again for reduction in computation. 

## Encoder
The convolution layer is used as a encoder since it can reduce the amount of information in the image while keeping the spatial information. The convolution layer can extract information such as edges and abstract information in the few first layers and as the depth of the network increase the network can extract more complicated information and shapes such as curves, squares and ect. 

## 1x1 convolution
Even-though the fully connected layer is a good choice for classification in this exercise it is not useful since the spacial information will be lost. In order to preserve the spacial information we use 1x1 convolution. By doing this the output of the network will be 4D instead of flattening to 2D.

## Decoder
the decoder network which usually is a transpose convolution but for the reason explained above in this exersise bilinear upsampling is to take the feature representation that are extracted using the convolution network and 1x1 convolution as input, process it and make its decision, and produce an output. The two network combined is called an encoder-decoder network.

Note: The same network architecture can be used to follow other objects such as car, dog,... with proper data and retraining the network. The network does not care what kind of object it is segmenting.

over view of network architecture:

![alt text][image_0] 
 

## Training and hyper parameters

Defined and tune your hyper parameters.

batch_size: number of training samples/images that get propagated through the network in a single pass.

num_epochs: number of times the entire training data set gets propagated through the network.

steps_per_epoch: number of batches of training images that go through the network in 1 epoch. We have provided you with a default value. One recommended value to try would be based on the total number of images in training data set divided by the batch_size.

```
learning_rate = .0015
batch_size = 64
num_epochs = 45
steps_per_epoch = 65
```

## Results:


![alt text][image_4] 


Scores for while the quad is following behind the target:
```
number of validation samples intersection over the union evaluated on 542
average intersection over union for background is 0.9935423067526095
average intersection over union for other people is 0.3035931738100663
average intersection over union for the hero is 0.8587927224582236
number true positives: 539, number false positives: 1, number false negatives: 0 

```
Sample result:

![alt text][image_1] 




Scores for images while the quad is on patrol and the target is not visible:

```
number of validation samples intersection over the union evaluated on 270
average intersection over union for background is 0.9800547089776422
average intersection over union for other people is 0.5803947400467847
average intersection over union for the hero is 0.0
number true positives: 0, number false positives: 175, number false negatives: 0
```

Sample result:

![alt text][image_2] 



This score measures how well the neural network can detect the target from far away:

```
number of validation samples intersection over the union evaulated on 322
average intersection over union for background is 0.9955228006381434
average intersection over union for other people is 0.40048507876490835
average intersection over union for the hero is 0.28051583103379496
number true positives: 191, number false positives: 9, number false negatives: 110
```

Sample result:

![alt text][image_3]





Sum all the true positives, etc from the three datasets to get a weight for the score: 0.7121951219512195

The IoU for the dataset that never includes the hero is excluded from grading: 0.5696542767460093

The final grade score is: 0.40570499709715785


## Some of unsuccsesful architecture:

network # 1


* Encoder section:
    
    1-  separable convolution layer with 32 kernel size of 3x3, , stride of 2 , padding = 'SAME' + batch normalization
    
    2-  separable convolution layer with 64 kernel size of 3x3, stride of 2 , padding = 'SAME' + batch normalization
    
    3-  separable convolution layer with 128 kernel size of 3x3, stride of 2 , padding = 'SAME' + batch normalization
    
    4-  1x1 convolution layer with  264 kernal size of 1x1, stride of 1 , padding = 'SAME' + batch normalization
    
    
* Decoder section:
    
    5- bilinear upsampeling concatenated with layer 2
    
    6- bilinear upsampeling concatenated with layer 1
    
    7- bilinear upsampeling concatenated with input layer
    
    

network # 2


* Encoder section:
    
    1-  separable convolution layer with 64 kernel size of 3x3, , stride of 2 , padding = 'SAME' + batch normalization
    
    2-  separable convolution layer with 128 kernel size of 3x3, stride of 2 , padding = 'SAME' + batch normalization
    
    3-  separable convolution layer with 128 kernel size of 3x3, stride of 2 , padding = 'SAME' + batch normalization
    
    4-  1x1 convolution layer with  264 kernal size of 1x1, stride of 1 , padding = 'SAME' + batch normalization
    
    
* Decoder section:
    
    5- bilinear upsampeling concatenated with layer 2
    
    6- bilinear upsampeling concatenated with layer 1
    
    7- bilinear upsampeling concatenated with input layer
    
    
## Limitation and Improvement

Since the data for training was limited and I was not able to collect enough data i had to make use small network. If enough data is available we can use a deeper network with CNN layer for encoder layer and regular deconvolution layer as decoder. It is even possible to use transfer learning and use already trained network such as resnet for the encoder section.

