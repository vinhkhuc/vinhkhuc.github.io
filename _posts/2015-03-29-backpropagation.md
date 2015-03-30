---
layout: post
mathjax: true
comments: true
title:  "Backpropagation In Neural Networks"
excerpt: "We will show how a simple Multilayer Perceptron network is trained using backpropagation."
date:   2015-03-29 21:26:10
---
### Introduction
**Backpropagation** is the most popular method to train an Artificial Neural Network (**ANN**). Training an ANN is just about
minimizing the loss function which measures the discrepancy between the network's prediction and the desired output. 

In backpropagation, the errors (i.e. the differences between the prediction and the desired output) are backward propagated
from the output layer to the input layer through the hidden layers. Weights in the network are adjusted to minimize these errors.

### A Mathematical View
For simplicity, we will use a simple Multilayer Perceptron network (as shown in Figure 1). 

### Implementation Notes
TODO