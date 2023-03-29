#include "Resources.cuh"

//ACTIVATIONS
class AbstractActivationLayer
{
public:
    int tSizeOutput;
    int fakeRowsCols; //for fast vector equal setting

    virtual float* forward(float* input, float* output) { return nullptr; };
    virtual float* backward(float* output__deltaLoss_deltaInput) { return nullptr; };
};

class ReluActivation : public AbstractActivationLayer
{
public:
    ReluActivation(int tSizeOutput);
    ~ReluActivation() { cudaFree(lastInput); }

    int nBlocks;
    float* lastInput; //just a ref to last thing it got as input in forward()

    float* forward(float* input, float* output);
    float* backward(float* output__deltaLoss_deltaInput); //gradient (for relu at least) is dependent on the last input it received
};

class SigmoidActivation : public AbstractActivationLayer
{
public:
    SigmoidActivation(int tSizeOutput);
    ~SigmoidActivation() { cudaFree(lastInput); }

    int nBlocks;
    float* lastInput; //just a ref to last thing it got as input in forward()

    float* forward(float* input, float* output);
    float* backward(float* output__deltaLoss_deltaInput); //gradient (for relu at least) is dependent on the last input it received
};

class TanhActivation : public AbstractActivationLayer
{
public:
    TanhActivation(int tSizeOutput);
    ~TanhActivation() { cudaFree(lastInput); }

    int nBlocks;
    float* lastInput; //just a ref to last thing it got as input in forward()

    float* forward(float* input, float* output);
    float* backward(float* output__deltaLoss_deltaInput); //gradient (for relu at least) is dependent on the last input it received
};

class SoftmaxActivation : public AbstractActivationLayer
{
public:
    SoftmaxActivation(int tSizeOutput);
    ~SoftmaxActivation() { 
        cudaFree(lastInput);
        cudaFree(lastOutput);
        cudaFree(max);
        cudaFree(sum);
        cudaFree(S_ST);
        cudaFree(diag_S);
    }

    int nBlocks;
    float* lastInput; //just a ref to last thing it got as input in forward()
    //you need memory for these
    float* lastOutput; //must be copy of last output, we cannot gaurentee that output is not modified
    float* max;
    float* sum;
    float* S_ST; //softmax vector output * itself transpose, is a matrix
    float* diag_S; //zero matrix, but with S along the diag (last softmax output on diag)

    float* forward(float* input, float* output);
    float* backward(float* output__deltaLoss_deltaInput); //gradient (for relu at least) is dependent on the last input it received
};


//LOSS FUNCTIONS
class AbstractLossFunction
{
public:
    int tSizeOutput;
    int nBlocks;

    virtual float* lossForward(float* output, float* target, float* totalError) { return nullptr; };
    virtual float* lossBackward(float* output, float* target) { return nullptr; };
};

class MeanSquaredError : public AbstractLossFunction
{
public:
    MeanSquaredError(int tSizeOutput);

    float* lossForward(float* output, float* target, float* totalError);
    float* lossBackward(float* output, float* target);
};


class CrossEntropyError : public AbstractLossFunction
{
public:
    CrossEntropyError(int tSizeOutput);

    float* lossForward(float* output, float* target, float* totalError);
    float* lossBackward(float* output, float* target);
};




//LAYERS

//INPUT: rows = 1 SAMPLE per row, cols = features for each sample!
//WEIGHTS: rows = number of input cols, cols = desired output cols (i.e neuron count)
class LinearLayer
{
public:
    LinearLayer(int inputCols, int outputCols, int batchSize, int activationFunction, float learningRate = 0.01f, int weightInitMode = KAIMING); //batch size == input rows
    ~LinearLayer() { cudaFree(lastInput); cudaFree(weights); cudaFree(bias); cudaFree(inputError); cudaFree(weightsError);} // cudaFree(deltaWeights); }

    float* weights;
    float* bias;
    AbstractActivationLayer* activationLayer = nullptr;

    //you need memory for this
    float* lastInput;
    float* inputError;
    float* weightsError;
    //float* deltaWeights; //store the weight delta. Size should be tSizeOutput

    int inputCols;
    int outputCols;
    int batchSize;

    int tSizeWeights;
    int tSizeOutput;
    int tSizeInput;
    int fakeRowsCols;
    int nBlocks;

    float learningRate;

    float* forward(float* input, float* output);
    float* backwardAndUpdate(float* backproppedError);
    //void update(float* backproppedError);
};


//NEURAL NET
class NeuralNetwork
{
public:
    NeuralNetwork(std::vector<LinearLayer*> layers, int lossFunctionEnum)
        :layers(layers)
    {
        tSizeOutput = layers.back()->tSizeOutput;
        initDeviceVectorManaged(totalError, 0.0f, 1);

        switch (lossFunctionEnum)
        {
        case(MEAN_SQURED_ERROR): {lossFunction = new MeanSquaredError(tSizeOutput); break; }
        case(CROSS_ENTROPY_ERROR): {lossFunction = new CrossEntropyError(tSizeOutput); break; }
        }
    }
    ~NeuralNetwork() { cudaFree(totalError); }

    std::vector<LinearLayer*> layers;
    AbstractLossFunction* lossFunction;
    float* totalError; //read and manage this from higher scope, write to it from within here GPU side, saves unessary synch call spam
    int tSizeOutput;

    inline float* networkForward(float* input, float* output)
    {
        //std::cout << "\n";
        float* curInput = input;
        for (int x = 0; x < layers.size(); x++)
        {
            curInput = layers[x]->forward(curInput, output);
            //cudaDeviceSynchronize();
            
            /*
            cudaDeviceSynchronize();
            std::cout << "LAYER OUT: " << x << ": ";
            for(int y = 0; y < layers[x]->tSizeOutput; y++)std::cout << output[y] << ", ";
            std::cout << "\n";
            */
        }
        //std::cout << "\n";

        return output;
    }

    inline void networkUpdate(float* backproppedError)
    {
        float* inputError = backproppedError;
        for (int x = layers.size()-1; x >=0 ; x--)
        {
            //layers[x]->update(backproppedError); //you do update first because you dont want the gradient of the layers own weights to apply before it has updated
            //layers[x]->backward(backproppedError); //layers weight gradient will then hit the backpropped error.
            inputError = layers[x]->backwardAndUpdate(inputError);
            //cudaDeviceSynchronize();
        }
    }

    inline float networkTrain(float* input, float* target, float* output, int nIterations)
    {
        float error = 0.0f;
        for (int x = 0; x < nIterations; x++)
        {
            //std::cout << "DEBUG\n";

            networkForward(input, output);
            lossFunction->lossForward(output, target, totalError); //t loss now added to totalError //mainly just used for tracking/meta purposes
            float* backproppedError = lossFunction->lossBackward(output, target);

            //cudaDeviceSynchronize();
            //error += target[layers.back()->tSizeOutput]; //now manage this via totalError in higher scope.
            
            /*
            std::cout << " BP ERROR:\n";
            for (int x = 0; x < 1; x++)std::cout << backproppedError[x] << ", ";
            std::cout << "\n";
            */

            /*
            cudaDeviceSynchronize();
            std::cout << " loss: " << backproppedError[0] << "\n";
            std::cout << "Weights Prior: \n";
            for (int x = 0; x < 2; x++)std::cout << layers[0]->weights[x] << ", ";
            std::cout << "\n";
            
            cudaDeviceSynchronize();
            std::cout << "\n targ: " << target[0] << " out: " << output[0] << "\n";
            */

            networkUpdate(backproppedError);

            /*
            cudaDeviceSynchronize();
            std::cout << "Weights POST: \n";
            for (int x = 0; x < layers.back()->inputCols; x++)
            {
                std::cout << "\n";
                for (int y = 0; y < layers.back()->outputCols; y++)std::cout << layers.back()->weights[x* layers.back()->outputCols+y] << ", ";
            }
            std::cout << "\n";
            std::cout << "DEBUG\n";
            std::cout << "\n";
            */
        }

        return error / float(nIterations);
    }

    inline float networkTest(float* input, float* target, float* output, int tSizeOutput, bool categorical)
    {
        networkForward(input, output);
        //for categorical mode, you can just take the maximum prediction and see if it matches target
        cudaDeviceSynchronize();

        float thisAcc = 0.0f;

        if (categorical)
        {
            float maxPred = output[0];
            int predVal = 0;
            for (int x = 1; x < tSizeOutput; x++)
            {
                if (output[x] > maxPred)
                {
                    maxPred = output[x];
                    predVal = x;
                }
            }

            if (target[predVal] == 1.0f)thisAcc = 1.0f; //assumes target is 1 hot encoded vec
        }
        else {
            float tErr = 0.0f;
            for (int x = 0; x < tSizeOutput; x++)
            {
                tErr += output[x] - target[x];
            }
            thisAcc = tErr / float(tSizeOutput);
        }

        return thisAcc;
    }
};