---
title: Driving Tasks review
date: 2019-11-10
tags: 
- Driving Tasks
- milestone
categories:
- Driving Tasks
---



# Driving Tasks review

MIT—AVT

## 王飞跃等的相关工作

1.**End-to-End Driving Activities and Secondary Tasks Recognition Using Deep Convolutional Neural Network and Transfer Learning**

- Specifically, seven common driving activities are identified, which are **normal driving, right mirror checking, rear mirror checking, left mirror checking, using in-vehicle video device, texting, and answering mobile phone**. Among these, the first four activities are regarded as normal driving tasks, while the rest three are divided into distraction group.
- using a **Gaussian mixture model (GMM)** to extract the driver region from the background
- **AlexNet**：directly takes the processed RGB images as the input and outputs the identified label.
- to reduce the training cost, the **transfer learning** mechanism is applied
- An average of **79%** detection accuracy

2.**Identification and Analysis of Driver Postures for In-Vehicle Driving Activities and Secondary Tasks Recognition**

- the importance of these features to behaviour recognition is evaluated using **Random Forests (RF) andMaximal Information Coefficient (MIC)** methods.
- **Feedforward Neural Network (FFNN)** is used to identify the seven tasks
- <img src="Driving Tasks review/image-20191204225003730.png" alt="image-20191204225003730" style="zoom: 67%;" />

3.

<img src="C:/Users/闵晨阳1998/AppData/Roaming/Typora/typora-user-images/image-20191205095652323.png" alt="image-20191205095652323" style="zoom:50%;" />

