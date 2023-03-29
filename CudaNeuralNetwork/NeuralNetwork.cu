#include "NeuralNetwork.cuh"


//RELU
ReluActivation::ReluActivation(int tSizeOutput)
{
    this->tSizeOutput = tSizeOutput;
    fakeRowsCols = ceil(sqrt(tSizeOutput));
    nBlocks = (tSizeOutput + TILE_DIM - 1) / TILE_DIM;
    initDeviceVectorManaged(lastInput, 0.0f, tSizeOutput); //input size should always equal output size for activations. They dont do any matmult.
}

float* ReluActivation::forward(float* input, float* output)
{
    //lastInput = input;
    //int nBlockRows = (fakeRowsCols + blockSize - 1) / blockSize;
    //int nBlockCols = (fakeRowsCols + blockSize - 1) / blockSize;#

    dim3 fnBlocks; //(nBlockRows, nBlockCols);
    dim3 fnThreads; // (blockSize, blockSize);
    setGridBlockDims(fnBlocks, fnThreads, fakeRowsCols, fakeRowsCols);
    cuda_setVectorsEqualFAST <<< fnBlocks, fnThreads >>> (lastInput, input, fakeRowsCols, tSizeOutput);
    reluForwardPass <<<nBlocks, TILE_DIM >>> (input, output, tSizeOutput);
    return output;
}

float* ReluActivation::backward(float* output__deltaLoss_deltaInput)
{
    reluBackwardPass <<<nBlocks, TILE_DIM >>> (lastInput, output__deltaLoss_deltaInput, tSizeOutput);
    return output__deltaLoss_deltaInput;
}


//SIGMOID
SigmoidActivation::SigmoidActivation(int tSizeOutput)
{
    this->tSizeOutput = tSizeOutput;
    fakeRowsCols = ceil(sqrt(tSizeOutput));
    nBlocks = (tSizeOutput + TILE_DIM - 1) / TILE_DIM;
    initDeviceVectorManaged(lastInput, 0.0f, tSizeOutput);
}

float* SigmoidActivation::forward(float* input, float* output)
{
    //lastInput = input;
    //int nBlockRows = (fakeRowsCols + blockSize - 1) / blockSize;
    //int nBlockCols = (fakeRowsCols + blockSize - 1) / blockSize;
    dim3 fnBlocks;// (nBlockRows, nBlockCols);
    dim3 fnThreads;// (blockSize, blockSize);
    setGridBlockDims(fnBlocks, fnThreads, fakeRowsCols, fakeRowsCols);
    cuda_setVectorsEqualFAST <<< fnBlocks, fnThreads >>> (lastInput, input, fakeRowsCols, tSizeOutput);
    sigmoidForwardPass <<<nBlocks, TILE_DIM >>> (input, output, tSizeOutput);
    return output;
}

float* SigmoidActivation::backward(float* output__deltaLoss_deltaInput)
{
    sigmoidBackwardPass <<<nBlocks, TILE_DIM >>> (lastInput, output__deltaLoss_deltaInput, tSizeOutput);
    return output__deltaLoss_deltaInput;
}


//TANH
TanhActivation::TanhActivation(int tSizeOutput)
{
    this->tSizeOutput = tSizeOutput;
    fakeRowsCols = ceil(sqrt(tSizeOutput));
    nBlocks = (tSizeOutput + TILE_DIM - 1) / TILE_DIM;
    initDeviceVectorManaged(lastInput, 0.0f, tSizeOutput);
}

float* TanhActivation::forward(float* input, float* output)
{
    /*
    if (tSizeOutput <= 10)
    {
        cudaDeviceSynchronize();
        std::cout << "\nTAN IN: ";
        for (int x = 0; x < 10; x++)std::cout << input[x] << ", ";
        std::cout << "\n";
    }
    */
    //lastInput = input;
    //int nBlockRows = (fakeRowsCols + blockSize - 1) / blockSize;
    //int nBlockCols = (fakeRowsCols + blockSize - 1) / blockSize;
    dim3 fnBlocks;// (nBlockRows, nBlockCols);
    dim3 fnThreads;// (blockSize, blockSize);
    setGridBlockDims(fnBlocks, fnThreads, fakeRowsCols, fakeRowsCols);
    cuda_setVectorsEqualFAST <<< fnBlocks, fnThreads >>> (lastInput, input, fakeRowsCols, tSizeOutput);
    tanhForwardPass <<<nBlocks, TILE_DIM >>> (input, output, tSizeOutput);
    return output;
}

float* TanhActivation::backward(float* output__deltaLoss_deltaInput)
{
    /*
    if (tSizeOutput <= 10)
    {
        cudaDeviceSynchronize();
        std::cout << "\nTAN LASTIN: ";
        for (int x = 0; x < 10; x++)std::cout << lastInput[x] << ", ";
        std::cout << "\n";
    }
    */
    tanhBackwardPass <<<nBlocks, TILE_DIM >>> (lastInput, output__deltaLoss_deltaInput, tSizeOutput);
    return output__deltaLoss_deltaInput;
}


//SOFTMAX
SoftmaxActivation::SoftmaxActivation(int tSizeOutput)
{
    this->tSizeOutput = tSizeOutput;
    fakeRowsCols = ceil(sqrt(tSizeOutput));
    nBlocks = (tSizeOutput + TILE_DIM - 1) / TILE_DIM;
    initDeviceVectorManaged(sum, 0.0f, 1);
    initDeviceVectorManaged(max, 0.0f, 1);
    initDeviceVectorManaged(lastInput, 0.0f, tSizeOutput);
    initDeviceVectorManaged(lastOutput, 0.0f, tSizeOutput);
    cudaMallocManaged(&S_ST, tSizeOutput * tSizeOutput * sizeof(float));
    cudaMallocManaged(&diag_S, tSizeOutput * tSizeOutput * sizeof(float));
    cudaDeviceSynchronize();
}

float* SoftmaxActivation::forward(float* input, float* output)
{
    //lastInput = input;
    int nBlockRows = (fakeRowsCols + TILE_DIM - 1) / TILE_DIM;
    int nBlockCols = (fakeRowsCols + TILE_DIM - 1) / TILE_DIM;
    dim3 fnBlocks;// (nBlockRows, nBlockCols);
    dim3 fnThreads;// (blockSize, blockSize);
    setGridBlockDims(fnBlocks, fnThreads, fakeRowsCols, fakeRowsCols);
    cuda_setVectorsEqualFAST <<< fnBlocks, fnThreads >>> (lastInput, input, fakeRowsCols, tSizeOutput);
    calcSoftmaxMax <<<nBlocks, TILE_DIM >>> (input, max, tSizeOutput);
    calcSoftmaxSum <<<nBlocks, TILE_DIM >>> (input, sum, max, tSizeOutput);
    softmaxForwardPass <<<nBlocks, TILE_DIM >>> (input, output, sum, max, tSizeOutput);
    cuda_setVectorsEqualFAST <<< fnBlocks, fnThreads >>> (lastOutput, output, fakeRowsCols, tSizeOutput);
    return output;
}

float* SoftmaxActivation::backward(float* output__deltaLoss_deltaInput)
{
    //sigmoidBackwardPass <<<nBlocks, blockSize >>> (lastInput, output__deltaLoss_deltaInput, tSizeOutput);
    softmaxBackwardPass(lastInput, lastOutput, output__deltaLoss_deltaInput, sum, max, S_ST, diag_S, tSizeOutput);
    return output__deltaLoss_deltaInput;
}


//MEAN SQUARED ERROR
MeanSquaredError::MeanSquaredError(int tSizeOutput)
{
    this->tSizeOutput = tSizeOutput;
    nBlocks = (tSizeOutput + TILE_DIM - 1) / TILE_DIM;
}

float* MeanSquaredError::lossForward(float* output, float* target, float* totalError)
{
    meanSquaredErrorForwardPass <<<nBlocks, TILE_DIM >>> (output, target, totalError, tSizeOutput); //here target will have a new element at the very end which will be the computed loss
    //cudaDeviceSynchronize();
    return totalError;
}

float* MeanSquaredError::lossBackward(float* output, float* target)
{
    meanSquaredErrorBackwardPass <<<nBlocks, TILE_DIM >>> (output, target, tSizeOutput); //here output is mutated to the gradient of the loss function, can then be passed down and multiplied backwards for backprop
    //cudaDeviceSynchronize();
    return output;
}


//CROSS ENTROPY ERROR
CrossEntropyError::CrossEntropyError(int tSizeOutput)
{
    this->tSizeOutput = tSizeOutput;
    nBlocks = (tSizeOutput + TILE_DIM - 1) / TILE_DIM;
}

float* CrossEntropyError::lossForward(float* output, float* target, float* totalError)
{
    crossEntropyErrorForwardPass <<<nBlocks, TILE_DIM >>> (output, target, totalError, tSizeOutput); //here target will have a new element at the very end which will be the computed loss
    //cudaDeviceSynchronize();
    return totalError;
}

float* CrossEntropyError::lossBackward(float* output, float* target)
{
    crossEntropyErrorBackwardPass <<<nBlocks, TILE_DIM >>> (output, target, tSizeOutput); //here output is mutated to the gradient of the loss function, can then be passed down and multiplied backwards for backprop
    //cudaDeviceSynchronize();
    return output;
}


//LINEAR
LinearLayer::LinearLayer(int inputCols, int outputCols, int batchSize, int activationFunction, float learningRate, int weightInitMode)
    :batchSize(batchSize), inputCols(inputCols), outputCols(outputCols), learningRate(learningRate)
{

    tSizeWeights = inputCols * outputCols; //tailor the weights shape to suit the desired input vs output
    tSizeOutput = batchSize * outputCols;
    tSizeInput = batchSize * inputCols;

    fakeRowsCols = ceil(sqrt(tSizeInput));

    nBlocks = (tSizeInput + TILE_DIM - 1) / TILE_DIM;

    cudaMallocManaged(&weights, tSizeWeights * sizeof(float));
    initDeviceVectorManaged(bias, 0.0f, outputCols);
    //initDeviceVectorManaged(deltaWeights, 0.0f, tSizeOutput);
    initDeviceVectorManaged(lastInput, 0.0f, tSizeInput);

    //If A is an m × n matrix and B is an n × p matrix, C = AB -> m × p matrix
    //weights error must be same size as weights
    cudaMallocManaged(&weightsError, tSizeWeights * sizeof(float));
    //input error is to be same size as input
    //outputError(batchSize*outputCols) . weightsT(outputCols, inputCols) = inputError(batchSize*inputCols)
    initDeviceVectorManaged(inputError, 0.0f, tSizeInput);

    //cudaDeviceSynchronize();

    switch (weightInitMode)
    {
    case(KAIMING): { kaimingInit(weights, inputCols, outputCols); break; }
    case(XAVIER): { xavierInit(weights, inputCols, outputCols); break; }
    case(RANDFLOAT): { randFloatInit(weights, inputCols, outputCols); break; }
    case(HE): { HEInit(weights, inputCols, outputCols); break; }
    }

    switch (activationFunction)
    {
    case(RELU): {activationLayer = new ReluActivation(tSizeOutput); break; } //use HE
    case(SIGMOID): {activationLayer = new SigmoidActivation(tSizeOutput); break; } //use xavier
    case(SOFTMAX): {activationLayer = new SoftmaxActivation(tSizeOutput); break; } //use xavier
    case(TANH): {activationLayer = new TanhActivation(tSizeOutput); break; } //use xavier
    }

    //cudaDeviceSynchronize();
}

//https://mlfromscratch.com/neural-network-tutorial/#/
float* LinearLayer::forward(float* input, float* output)
{
    
    //int nBlockRows = (fakeRowsCols + blockSize - 1) / blockSize;
    //int nBlockCols = (fakeRowsCols + blockSize - 1) / blockSize;
    dim3 fnBlocks;// (nBlockRows, nBlockCols);
    dim3 fnThreads;// (blockSize, blockSize);
    setGridBlockDims(fnBlocks, fnThreads, fakeRowsCols, fakeRowsCols);
    cuda_setVectorsEqualFAST <<< fnBlocks, fnThreads >>> (lastInput, input, fakeRowsCols, tSizeInput); //must be OUTPUT SIZE HERE DUMMY!
    
    //cuda_setVectorsEqual <<< nBlocks, blockSize >>> (lastInput, input, tSizeInput);

    /*
    if (tSizeOutput <= 10)
    {
        cudaDeviceSynchronize();
        std::cout << "\nLINEAR IN: ";
        for (int x = 0; x < 50; x++)std::cout << input[x] << ",";
        std::cout << "\n";
    }
    */
    /*
    cudaDeviceSynchronize();
    std::cout << "\nLIN: ";
    for(int x = 0; x < 2; x++)std::cout << lastInput[x] << ",";
    std::cout << "\n";
    */

    linearForwardPass(input, weights, bias, output, batchSize, inputCols, outputCols);
    //cudaDeviceSynchronize();
    if (activationLayer)activationLayer->forward(output, output); //activation layer essentially mutates the linears output
    return output;
}

float* LinearLayer::backwardAndUpdate(float* backproppedError)
{
    if (activationLayer)activationLayer->backward(backproppedError);
    //cudaDeviceSynchronize();
    //linearBackwardPass(output__backproppedError, weights, batchSize, inputCols, outputCols); //applies gradient of this layer to the backpropped error
    //return output__backproppedError;
    linearBackwardAndUpdatePass(lastInput, backproppedError, weights, bias, inputError, weightsError, batchSize, inputCols, outputCols, learningRate);
    return inputError; //pass this back down the layers
}

/*
void LinearLayer::update(float* input__backproppedError)
{
    if (activationLayer)activationLayer->backward(input__backproppedError); //activation layer applies its gradient first, then linear updates
    linearUpdate(lastInput, input__backproppedError, weights, bias, batchSize, inputCols, outputCols, learningRate);
}
*/
