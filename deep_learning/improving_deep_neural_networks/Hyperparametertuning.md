---
title: "Hyperparameter Tuning"
author: Chinmay Hundekari
date: May 07, 2020
---

# To execute:
~~~~
pandoc HyperparameterTuning.md -f markdown+tex_math_dollars -s -o HyperparameterTuning.html --mathjax --toc -V toc-title:"Table of Contents"
~~~~

# Hyperparameter Tuning 
## Normalizing activations in a network
### Batch Normalization
* Easily train deep networks.
Normalizing inputs helps to speed up learning. So, can we normalize z() or a() to speed up that layer.

* Normalize each layer of z separately
$$ \mu = \frac{1}{m} \sum\limits_{i} z^i  $$
$$ \sigma^{2} = \frac{1}{m} \sum\limits_{i} (z^i - \mu)^{2} $$
$$ z^{i}_{norm} = \frac{z^{i} - \mu}{\sqrt{\sigma + \epsilon}}  $$
$$ \widetilde{z}^{i} = \gamma z^{i}_{norm} + \beta  $$, where $\gamma$ and $\beta$ are learnable parameters.

* This would help in taking advantage of the non-linear region of sigmoid function, or any other activation function.

### Fitting Batch Norm into a neural network
#### Feedforward Propagation
1. Calculate $z^{[l]}$ for a layer. 
2. Calculate batch norm $\widetilde{z}^{[l]}$ for $z^{[l]}$, with $\beta^{[l]}$ and $\gamma^{[l]}$.
3. Use $\widetilde{z}^{[l]}$ to calclulate $a^{[l]}$.
4. Repeat for all layers.

#### BackPropagation
1. Back propagate $w^{[l]}$ changes with equation $w^{[l]} = w^{[l]} - \alpha \mathrm{d}w^{[l]}$.
2. Back propagate $\beta^{[l]}$ changes with equation $\beta^{[l]} = \beta^{[l]} - \alpha \mathrm{d}\beta^{[l]}$.
3. Back propagate $\gamma^{[l]}$ changes with equation $\gamma^{[l]} = \gamma^{[l]} - \alpha \mathrm{d}\gamma^{[l]}$.

* Equation $z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$ changes to $z^{[l]} = W^{[l]}a^{[l-1]}$. This is because $z$ is normalized, and the averaging leads to loss of $b$. This role is now taken by $\beta$.

### Why does Batch Norm work?
* Works because we normalize weights.
* Classifiers may not work well if data is shifted. e.g. Trained on black cats, and test on colored cats. This is called as covariance shift. Batch normalization reduces this, as each layer is independent, and common features may still persist in the data.
* Effect of earlier layers on later layers is reduced due to normalizing weights.  
* Batch norm on mini batches adds noise after every batch to $z^{[l]}$, forcing the layers to not rely on individual units. This adds slightly to regularization, just like dropout. This is not the main intent of batch norm. Smaller batches increase regularization effect.

### Batch norm at test time
* During test time, when we are running single test cases, it wont be possible to get meaningful values of $\gamma$ and $\beta$.
* This can be resolved by calculating a running exponential moving average of $\gamma$ and $\beta$ after every mini-batch training, and using the end result in test time.


## Multi-class Classification
### Softmax Regression
* The last layer has size equal to the number of classification categories (c).  
* For last layer, we calculate as: 
$$ z^{[L]} = w^{[L]}a^{[L-1]} + b^{[L]} $$
$$ t = e^{z{[L]}} $$
$$ a^{[L]} = \frac{t_{i}}{\sum\limits_{i}t_{i}} $$
* The final layer has normalized values which can be compared with each other and should sum up to 1.
* Hardmax gets a single max value. Softmax is generalized logistic regression(c=2).
### Training a softmax classifier
#### Loss function
For class selected by setting x = 1: 
$$ y^{i} = \begin{bmatrix}\\0\\...\\x \\... \\c \end{bmatrix} $$
$$ \unicode{163} (\hat{y}, y) = \sum\limits_{j}{y_{j} log(\hat{y_{j}})} $$
gets reduced to
$$ \unicode{163} (\hat{y}, y) = y_{x} log(\hat{y_{x}}) $$
$$ \unicode{163} (\hat{y}, y) = log(\hat{y_{x}}) $$
#### Backpropogation
$$ \mathrm{d}z^{[L]} = \hat{y} - y $$

## Intoduction to deeplearning frameworks - Tensorflow
* Always prefer opensource networks, where ownership is also open and not an organization.
### Tensorflow
1. Create a computational graph describing the network.
  1. Create placeholders for data
~~~~
      X = tf.constant(np.random.randn(3,1), name = "X")
      y = tf.constant(39, name='y')
      x = tf.placeholder(tf.float32, name = "x")
~~~~
2. Run the graph with input data.
* Basic structure
~~~~
    y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
    y = tf.constant(39, name='y')                    # Define y. Set to 39

    loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss

    init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                                     # the loss variable will be initialized and ready to be computed
    with tf.Session() as session:                    # Create a session and print the output
        session.run(init)                            # Initializes the variables
        print(session.run(loss))                     # Prints the loss
~~~~
