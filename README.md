# TensorFlow mnist
Using TensorFlow to build a neural network for analyzing handwritten numbers  based on mnist samples.

Use of TensorFlow greatly simplifies creating a complex neural network, but since it is designed to handle complex neural networks (multiple layers and large number of nodes), it introduces challenges for defining and processing a neural network.  In this somewhat simple neural network, there are four layers defined using distinct configurations: Flatten, Dense and Dropout which need to be better understood.  This sample also incorporates implementation for using logits/log-odds which get converted to probabilities for the untrained model which needs to be further explored as to how these are used to define the loss function.
