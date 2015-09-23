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

<div style="text-align:center;">
    <img src="/assets/2015-03-29-backpropagation/mlp-simple.png" width="85%" height="85%">
    <p><em>Figure 1: A simple MLP network</em></p>
</div>

### A Mathematical View
Most computation for backpropagation is actually computing derivative of the loss function using chain rule. 
Let's take a look at a simple Multilayer Perceptron (**MLP**) network with one hidden layer (as shown in Figure 1).

$$
\begin{align}
h &= \sigma(x W_1 + b_1) \\
\hat{y} &= \varphi(h W_2 + b_2)
\end{align}
$$

where $$\sigma$$ and $$\varphi$$ are activation functions for the hidden layer and output layer respectively.

Given a training data set $$D = \{(x_i, y_i)\}_{i=1}^N$$, the loss function $$L(Y, \hat{Y})$$ is defined based on the learning task. 
For regression task, the loss function is usually defined as Mean Squared Error (**MSE**) while Cross Entropy (**CE**) 
is often used for classification task.  

$$
\begin{alignat} {2}
L(Y, \hat{Y}) &= MSE(Y, \hat{Y}) &&= \frac{1}{N}\sum_{i=1}^N(y_i - \hat{y}_i)^2 \quad (1)\\
L(Y, \hat{Y}) &= \ \ CE(Y, \hat{Y})  &&= \frac{1}{N}\sum_{i=1}^N CE(y_i, \hat{y}_i) \qquad \ (2)
\end{alignat}
$$

where $$CE(y_i, \hat{y_i}) = -\sum_{l=1}^L y_{il} \log\hat{y}_{il} $$ ($$L$$ is the number of labels and $$y_i$$ is encoded
as one-hot vector, i.e. its components are one at the target index and zeros everywhere else).

Training MLP networks is about finding values for the weight matrices $$W_1$$, $$W_2$$ and bias vectors $$b_1$$, $$b_2$$ to minimize 
the loss function $$L$$. Since $$L$$ is a non-convex function w.r.t. $$W_1$$, $$W_2$$, $$b_1$$, $$b_2$$, we need to train the ANN model
using Stochastic Gradient Descent (**SGD**) to get avoid of local minima and saddle points. 

In SGD, at each iteration we randomly pick a training data point $$(x_i, y_i)$$ to update the parameters using the following formula:

$$
\begin{alignat} {2}
W_1 &= W_1 - \mu \frac{\partial L}{\partial W_1},\quad b_1 &&= b_1 - \mu \frac{\partial L}{\partial b_1} \\
W_2 &= W_2 - \mu \frac{\partial L}{\partial W_2},\quad b_2 &&= b_2 - \mu \frac{\partial L}{\partial b_2}
\end{alignat}
$$

where $$\mu$$ is the learning rate, and now $$N=1$$ in $$(1)$$ and $$(2)$$.

We will apply the chain rule to derive the derivatives:

$$
\begin{align}
\frac{\partial L}{\partial W_1} 
    &= \frac{\partial L}{\partial \hat{y}} \ \frac{\partial \hat{y}}{\partial h} \ \frac{\partial h}{\partial W_1} \\ 

\frac{\partial L}{\partial W_2} 
    &= \frac{\partial L}{\partial \hat{y}} \ \frac{\partial \hat{y}}{\partial W_2}
\end{align}
$$

So,

$$
\begin{align}
\frac{\partial L}{\partial W_1} 
   &= \frac{\partial L}{\partial \hat{y}} \ \frac{\partial \varphi}{\partial (h W_2 + b_2)} \ 
       \frac{\partial (h W_2 + b_2)}{\partial h} \ \frac{\partial \sigma}{\partial (x W_1 + b_1)} \ \frac{\partial (x W_1 + b_1)}{\partial W_1} \\

\frac{\partial L}{\partial W_2}
   &= \frac{\partial L}{\partial \hat{y}} \ \frac{\partial \varphi}{\partial (h W_2 + b_2)}
       \frac{\partial (h W_2 + b_2)}{\partial W_2}
\end{align}
$$

similarly for $$b_1$$ and $$b_2$$.

Depending on the forms of the activations functions $$\sigma$$ and $$\varphi$$, we can fully derive the above derivatives.

### Implementation
The most important part of implementing an ANN is deriving the derivatives of the loss function w.r.t. the network parameters. 
As we have seen, the chain rule allows us to break the giant derivative of the loss function into derivatives of individual functions.
A very nice implementation trick is explained in [Backprop in practice: Staged computation](//cs231n.github.io/optimization-2). 

**Good news** is that we can skip deriving the loss function's derivative by utilizing [**Theano**](//deeplearning.net/software/theano/) 
to do the heavy lifting for us. A huge bonus of using Theano to implement neural networks is that the code can be greatly 
accelerated (without any changes) when running on the **GPU**.

Here is an example of using Theano to calculate the value of the derivative of $$y = f(x) = x^2$$ at $$x = 4$$:

{% gist fb9cd4d551045b726689 %}

<br>
Finally, this is a **Theano-based implementation** of the simple MLP with one hidden layer.

{% gist 20089f4abf9706d0cdc4 %}