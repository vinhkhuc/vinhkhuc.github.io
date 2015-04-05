---
layout: post
mathjax: true
comments: true
title:  "Backpropagation In Neural Networks"
excerpt: "We will show how a simple Multilayer Perceptron network is trained using backpropagation."
date:   2015-03-29 21:26:10
---
### Introduction
**Backpropagation** is the most popular method to train an Artificial Neural Network (**ANN**). Training an ANN is about
minimizing the loss function which measures the discrepancy between the network's prediction and the desired output. 

In backpropagation, the errors (i.e. the differences between the prediction and the desired output) are backward propagated
from the output layer to the input layer through the hidden layers. Weights in the network are adjusted to minimize these errors.

### A Mathematical View
Mathematically speakinng, backpropagation is mainly about taking derivative using chain rule. Let's take a look at 
a simple Multilayer Perceptron (**MLP**) network with one hidden layer (as shown in Figure 1 -- 
<span style="color: red; font-weight: bold;">TODO</span>).

$$
\begin{align}
h &= \sigma(x W_1 + b_1) \\
\hat{y} &= \varphi(h W_2 + b_2)
\end{align}
$$

where $$\sigma$$ and $$\varphi$$ are activiation functions for the hidden layer and output layer respectively.

Given a training data set $$\{(x_i, y_i)\}_{i=1}^N$$, the loss function $$L(Y, \hat{Y})$$ is defined based on the learning task. 
For regression task, the loss function is usually defined as Mean Squared Error (**MSE**) while Cross Entropy (**CE**) 
is often used for classification task.  

$$
\begin{align}
MSE(Y, \hat{Y}) &= \frac{1}{N}\sum_{i=1}^N(y_i - \hat{y_i})^2 (1)\\
CE(Y, \hat{Y})  &= -\sum_{i=1}^Ny_i \log\hat{y_i} (2)
\end{align}
$$

Training MLP networks is about finding values for the weight matrices $$W_1$$, $$W_2$$ and bias vectors $$b_1$$, $$b_2$$ to minimize 
the loss function $$L$$. Since $$L$$ is a non-convex function w.r.t. $$W_1$$, $$W_2$$, $$b_1$$, $$b_2$$, we need to use
Stochastic Gradient Descent (**SGD**) to get avoid of local minima and saddle points. 

In SGD, at each iteration we randomly pick a training data point $$(x_i, y_i)$$ to update the parameters using the following formula:

$$
W = W - \mu \frac{\partial L}{\partial W}, b = b - \mu \frac{\partial L}{\partial b}
$$

note that now $$N=1$$ in (1) and (2).


### Implementation Notes
TODO