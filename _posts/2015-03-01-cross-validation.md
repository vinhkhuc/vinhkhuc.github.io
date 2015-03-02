---
layout: post
title:  "How Many Folds for Cross-Validation"
excerpt: "Cross-Validation is a method for estimating the performance of a classifier for unseen data. In this post, we will try to figure out how many folds that are should be used for cross-validation."
date:   2015-03-01 14:47:10
---
### Introduction
**[Cross-validation](//en.wikipedia.org/wiki/Cross-validation_%28statistics%29)** (**CV**)
is a method for estimating the performance of a classifier for unseen data. With K-folds, the whole labeled data set is
randomly split into K equal partitions. For each partition, the classifier is trained on the remaining K-1 partitions and
is tested on data from that partition. The final accuracy is the average of all K accuracies.

<div style="text-align:center;">
    <img src="/assets/2015-03-01-cross-validation/5-fold-cv.png" width="85%" height="85%">
    <p><em>Figure 1: Example of 5-fold cross-validation.</em></p>
</div>

### How many folds should we pick?
The question is which value of K we should pick. Higher values of K give less biased estimation. In theory, the best value
of K is N, where N is the total number of training data points in the data set. N-fold CV is also called
**Leave-One-Out Cross-Validation** (**LOOCV**). Although LOOCV gives unbiased estimate of the true accuracy, it is very
costly to compute.

In practice, we usually use **K = 5, 10 or 20** since these K-fold CVs give approximately the same accuracy
estimation as LOOCV but without costly computation. Experiments show that K = 10 is the best choice as we will see below.

The Figure 2 shows Ron Kohavi's experiment with CV using different values of K for the decision tree classifier C4.5
on different data sets. The gray regions indicate confidence intervals for the true accuracies. The negative K stands
for Leave-K-Out.

<div style="text-align:center;">
    <img src="/assets/2015-03-01-cross-validation/ron-kohavi-experiment.png" width="65%" height="65%">
    <p>
        <em>Figure 2: Ron Kohavi. A study of cross-validation and bootstrap for accuracy estimation and model selection. IJCAI '95.</em>
        <a href="http://robotics.stanford.edu/~ronnyk/accEst.pdf">PDF</a>
    </p>
</div>

### My experiment
Here is the Python code to replicate Ron Kohavi's experiment on the [Iris data set](https://archive.ics.uci.edu/ml/datasets/Iris):

{% gist 671cc1cbb9ccc16b456b %}

As we can see in the Figure 3, the good value of K is 10 since it gives the best trade-off between the running speed
and the accuracy estimation.

<div style="text-align:center;">
    <img src="/assets/2015-03-01-cross-validation/cross-validation-experiment-iris.png" width="75%" height="75%">
    <p><em>Figure 3: Cross-validation experiment on the Iris data set.</p></em>
</div>
