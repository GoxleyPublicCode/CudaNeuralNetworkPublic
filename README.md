# CudaNueralNetworkPublic

I created this originally in 2022, but now want to make it public as I don't plan any future development on it and I figured it might be useful to someone else.

This project implements a simple Neural Network in C++ and CUDA from first principles. Originally I was using CUBLAS for mat mult but ultimately decided to just implement my own simple mat mult kernel/function in CUDA.

The network can classify the MNIST dataset to an accuracy of around 96% (ok-ish). See Tests.h for more details on the various configurations I tried for MNIST.
