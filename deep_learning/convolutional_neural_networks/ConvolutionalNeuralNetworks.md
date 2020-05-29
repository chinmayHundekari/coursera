---
title: "Convolutional Neural Networks"
author: Chinmay Hundekari
date: May 16, 2020
---

# To execute:
~~~~
pandoc ConvolutionalNeuralNetworks.md -f markdown+tex_math_dollars -s -o ConvolutionalNeuralNetworks.html --mathjax --toc -V toc-title:"Table of Contents"
~~~~

# Week 1
## Computer Vision
* Rapid advances are happening in Computer vision. Great creative community.
* Object Detection - Identify object and put a box around it.
* Neural Style transfer - Transfer image of a particular style to another.
* Even a simple image of 1 megapxel requires 1000*1000*3 neurals in the first layer. This is infeasible. Convolution helps fix this.

## Edge Detection
* Convolution symbol - $*$
$$ Image[m,n] * filter[x,y] = Output[m-x+1,n-y+1] $$
$$ \rightarrow Image_{ij}[x,y] \cdot filter[x,y] = Output_{ij} $$
$$\begin{bmatrix}
 &_{00}&_{01}&_{02}&_{03}&_{04}&_{05} \\ 
 &_{10}&_{11}&_{12}&_{13}&_{14}&_{25} \\ 
 &_{20}&_{21}&_{22}&_{23}&_{24}&_{35} \\ 
 &_{03}&_{31}&_{32}&_{33}&_{34}&_{35} \\ 
 &_{40}&_{41}&_{42}&_{43}&_{44}&_{45} \\ 
 &_{50}&_{51}&_{52}&_{53}&_{54}&_{55} \\ 
\end{bmatrix} * \begin{bmatrix}
 &_{00}&_{01}&_{02} \\ 
 &_{10}&_{11}&_{12} \\ 
 &_{20}&_{21}&_{22} \\ 
\end{bmatrix} = \begin{bmatrix}
 &_{00}&_{01}&_{02}&_{03}&_{04} \\ 
 &_{10}&_{11}&_{12}&_{13}&_{14} \\ 
 &_{20}&_{21}&_{22}&_{23}&_{24} \\ 
 &_{03}&_{31}&_{32}&_{33}&_{34} \\ 
\end{bmatrix} $$
* python - convforward();tensorflow - tf.nn.conv2d
### filters
$$ Vertical Edge detection filter = \begin{bmatrix}
 &1&0&-1 \\ 
 &1&0&-1 \\ 
 &1&0&-1 \\ 
\end{bmatrix}  $$
$$ Horizontal Edge detection filter = \begin{bmatrix}
 &1&1&1 \\ 
 &0&0&0 \\ 
 &-1&-1&-1 \\ 
\end{bmatrix}  $$
$$ Sobel filter = \begin{bmatrix}
 &1&0&-1 \\ 
 &2&0&-2 \\ 
 &1&0&-1 \\ 
\end{bmatrix}  $$
$$ Scharr filter = \begin{bmatrix}
 &&0&-3 \\ 
 &10&0&-10 \\ 
 &3&0&-3 \\ 
\end{bmatrix}  $$
* Too many filters. Maybe we can use deeplearning and treat the filter as parameters :-)

## Padding
* The corner edges of input are used in convolution less than Inner data.
* Deep neural networks need maximum information preserved. It might be better if convolution did not reduce input data size.
* Padding by p pixel on each side. This results should be a convolutional output of same size as input.
$$ n \times n * f \times f = (n - f + 1) \times (n - f + 1) $$
$$ (n + 2p) \times (n + 2p) * f \times f = (n + 2p - f + 1) \times (n +2p - f + 1) $$
For output size to be same as input size 
$$ n + 2p - f + 1 = n$$
$$ p =\frac{f - 1}{2} $$
* f is normally odd, to avoid asymmetric padding.

## Strided Convolutions
* For Stride = 2 -
$$ \rightarrow Image_{ij}[2x,2y] \cdot filter[x,y] = Output_{ij} $$
$$  Output Size = \lfloor \frac{n + 2p - f}{S} + 1 \rfloor
* Ideal to round down if size is fractional.
* Sometime, the filter matrix may be flipped across both axis. As we are skipping this step, this is better called cross correlation. But it Deep learning, it is just called convolution. We have sacrificed associativity property. But this does not affect Deep learning.

## Convolutions over volumes
### Convolutions on RGB images
* Image data is Height * Width * Number of Channels(RGB). Filter also has same dimensions.
* Strides are taken only across width and height, not channels!

## One layer of convolutional network
### Forward propagation
* Convolution replaces weight - 
$$ a^{[l]} = ReLU(X * f + b),  b \in \mathbb{R^{3}}, X \in R^{3}, f in \mathbb{R^{3}} $$
* Number of parameters for a filter remains at (filter height * filter width + 1 for bias ) multiplied by number of channels.
* So number of parameters to learn remains small. As number of parameters are small, the chances of overfitting are less.
$$  filter size = f^{[l]}, padding size = p^{[l]}, stride = ^{[l]}, n_{c}^{[l]} = number of filters/channels $$
$$ Input: n_{h}^{[l-1]} \times n_{w}^{[l-1]} \times n_{c}^{[l-1]} $$
$$ Output: n_{h}^{[l]} \times n_{w}^{[l]} \times n_{c}^{[l]} $$
$$ n_{w}^{[l]} =  \lfloor \frac{n_{w}^{[l-1]} + 2p^{[l]}- f^{[l]}}{S^{[l]}} + 1 \rfloor $$
$$ n_{h}^{[l]} =  \lfloor \frac{n_{h}^{[l-1]} + 2p^{[l]}- f^{[l]}}{S^{[l]}} + 1 \rfloor $$
$$ Input data : h*w*c $$
$$ Output layer : (n+2p-f)/s +1 * (n+2p-f)/s * Number of filters $$
* Unroll last layer output and use logistic regression/softmax to obtain the result
* Channels generally increase and height and width reduces
* Types of layers:
    1. Convolution
    2. Pooling
    3. Fully connected

## Pooling layers
* Max pooling: For a filter size of and stride of s, max pooling is maximum in each of those matrix, instead of convolution operation. Mostly only maximum value may be important.
* Padding is usually avoided.
$$ n_{w}^{[l]} =  \lfloor \frac{n_{w}^{[l-1]} + 2p^{[l]}- f^{[l]}}{S^{[l]}} + 1 \rfloor $$
$$ Output layer : (n+2p-f)/s +1 \times (n+2p-f)/s \times Number of channels $$
* Average pooling : Take average instead of max.
* No learning in this layer

## Example - LeNet -5
* 32x32x3 --> Layer1 [--Conv(f=5,s=1)--> Conv1 (28x28x6) --Maxpool(f=2,s=20)--> 14x14x6 ] -->Layer 2 [--Conv(f=5,s=1)-->10x10x16 --Maxpool(f=2,s=2)--> 5x5x16] --> 400x1 --> FC3(120,400) --> FC4(84,1)-->10 outputs
* Parameter Sharing: Filters are reused in the image
* Sparsity of connections: In each layer, each output value only depends on few inputs. So, overfitting a label reduces.
* If a cat image is shifted, this can still be detected, because of convolution filter!

# Week 2
## Neural Network Architectures
### LeNet-5
* Image of 32x32x1--Conv(5x5,s=1)-->28x28x6--AvgPool(f=2,s=2)-->14x14x6--Conv(5x5,s=1)-->10x10x16--AvgPool(f=2,s=2)-->5x5x16--FC()-->120--FC()-->84--Softmax-->y
### AlexNet
* Image of 227x227x3--Conv(11x11,s=4,c=96)-->55x55x96--MaxPool(3x3,s=2)-->27x27x96--Conv(5x5,s=1)-->27x27x256--MaxPool(f=3,s=2)-->13x13x256--Conv(3x3)-->13x13x384--Conv(3x3)-->13x13x384--Conv(3x3)-->13x13x384--MaxPool(f=3,s=2)-->6x6x256--FC()-->9216--FC()-->4096--FC()-->4096--Softmax-->y[1000]
### VGG-16
* All filters --> Conv(3x3,s=1), MaxPool(2x2,s=2)
* 224x224x3--Conv(c=64)-->224x224x64--Conv(c=64)-->224x224x64--Maxpool()-->112x112x64--Conv(c=128)-->224x224x128--Conv(c=128)-->224x224x128--Maxpool()-->112x112x128--Conv(c=256)-->224x224x256--Conv(c=256)-->224x224x256--Maxpool()-->112x112x256--Conv(c=512)-->224x224x512--Conv(c=512)-->224x224x512--Maxpool()-->112x112x512
### ResNet
#### Residual Block
$$ z^{[l+1]} = W^{[l+1]}a^{[l]} + b^{[l+1]} $$
$$ a^{[l+1]} = ReLU(z^{[l+1]}) $$
$$ z^{[l+2]} = W^{[l+2]}a^{[l+1]} + b^{[l+2]} $$
$$ a^{[l+2]} = ReLU(z^{[l+2]} \mathbf{+ a^{[l]}}) $$
* Training error get deeper as layers increase in Plain networks. ResNet fix this issue.
* Intermediate Layers which zero out activation are reduced, as some activations skip the layer.
### 1x1 Convolution - Network in Network(2013)
* XxYxC2 = ReLU(XxYxC1 * 1x1xC2)
### Inception Network - [Szegedy et al., 2014, Going Deeper with Convolutions]
* Apply multiple separate filters and concatenate outputs. Let back propagation find the best combination.
* Next layers add up a massive computation cost, as the size is much larger. 1x1 Convolutions reduce the massive computation cost, as they reduce channels.
'''
X -|-------------------------Conv(1x1x64)--|
   |-----------Conv(1x1x16)--Conv(5x5x32)--|
   |----------Conv(1x1x96)--Conv(3x3x128)--> ChannelConcat() --> 28x28x256
   |--MaxPool(3x3,s=1,same)--Conv(3x3x32)--|
'''
* Inception Network also has a softmax output branched in the middle of network. This also helps to avoid over fitting.
## Practical advice
### Open-Source Implementations
* Look for an open-source implementations instead of implementing papers. As implementation detail  may be too finicky.

### Transfer learning
* Using pre-trained networks help to replicate papers, as some of the networks may take weeks.
* For smaller datasets - Use same network, except for softmax layer and retrain the network.
* For larger dataset - Then number of layers you freeze could be reduced.
* For much larger dataset - Use weights only for initialization.
* Very good for Computer Vision applications.
### Data Augmentation
* Reduces the chances that you overfit to hidden features.
* Mirroring - Mirror images to double your data now!
* Random cropping - Randomly crop your data to multiply it many times!
* Color shifting - Add color distortions to create more images. Refer PCA Color augmentation (AlexNet).
* Image on Hard Disk --> CPU Thread distorts image --> Another thread does the training.

### State of Computer vision
* Lots of data is present for Speech recognition. But lesser for Image recognition.
* Either Labelled data or Hand engineer data for Computer Vision.
* Ensembling - Train lots of networks(3-15) and average their outputs. Good for Competitions. Not as great for real world. 
* Multi-crop - Crop images in multiple ways and average the output.
* Not really used for Production.
