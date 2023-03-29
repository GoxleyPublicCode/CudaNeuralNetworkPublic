#pragma once
#include "NeuralNetwork.cuh"
#include <assert.h>
#include <chrono>

class TestCase
{
private:

    void _assertEqual(float v1, float v2)
    {
        assert(v1 == v2);
    }

    void _assertAproxEqual(float v1, float v2, float margin = 0.01f)
    {
        assert((v1 > v2 - margin) && (v1 < v2 + margin));
    }

    void _assertNotEqual(float v1, float v2)
    {
        assert((v1 != v2));
    }

    float getMatrixMultValue(float* input, float* weights, int col, int nCols, int nRows) //works where you have column vector for input, row will iterate
    {
        float sum = 0.0f;
        int rowCount = 0;
        for (int x = col; x < nCols * nRows; x += nRows)
        {
            sum += input[col] * weights[x];
        }
        return sum;
    }


public:

    void testMatMult()
    {
        int Arows = 4;
        int Acols = 2;
        int Brows = 2;
        int Bcols = 4;

        //B =
        //[1, 2, 3, 4,
        // 5, 6, 7, 8]

        //A
        //[1, 5
        // 3, 4
        // 5, 6
        // 7, 8]

        //C = A * B = C(4*4)
        //

        float* A;
        float* B;
        float* C;
        initDeviceVectorManaged(A, 0.0f, Arows * Acols);
        initDeviceVectorManaged(B, 0.0f, Brows * Bcols);
        initDeviceVectorManaged(C, 0.0f, Arows * Bcols); 

        for (int x = 0; x < Arows; x++)
        {
            for (int y = 0; y < Acols; y++)
            {
                //int val = y * Arows + x + 1;
                //A[x * Acols + y] = val;
                A[x * Acols + y] = x * Acols + y + 1;
            }
        }

        for (int x = 0; x < Brows; x++)
        {
            for (int y = 0; y < Bcols; y++)
            {
                B[x * Bcols + y] = x * Bcols + y + 1;
            }
        }

        dim3 dim_grid;// (ceilf(Arows / (float)blockSize), ceilf(Bcols / (float)blockSize), 1);
        dim3 dim_block;// (blockSize, blockSize, 1);
        setGridBlockDims(dim_grid, dim_block, Arows, Bcols);
        //for (int x = 0; x < 8; x++)std::cout << A[x] << "\n";
        cuda_matMult << <dim_grid, dim_block >> > (A, B, C, Arows, Acols, Bcols);
        //cublas_matrixMult(globalCublasHandle, A, B, C, Arows, Acols, Bcols, CUBLAS_OP_N, CUBLAS_OP_N);
        cudaDeviceSynchronize();

        std::vector<int> answers = { 11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68, 47, 62, 77, 92 };
        for (int x = 0; x < Arows * Bcols; x++)
        {
            _assertAproxEqual(C[x], answers[x]);
        }

    }


    void testMatMultTransposed()
    {
        int Arows = 2;
        int Acols = 4;
        int Brows = 2;
        int Bcols = 4;

        //A = B =
        //[1, 2, 3, 4,
        // 5, 6, 7, 8]

        //A^T = 
        //[1, 5
        // 2, 6
        // 3, 7
        // 4, 8]

        //C = A^T * B = C(4*4)
        //

        float* A;
        float* B;
        float* C;
        initDeviceVectorManaged(A, 0.0f, Arows * Acols);
        initDeviceVectorManaged(B, 0.0f, Brows * Bcols);
        initDeviceVectorManaged(C, 0.0f, Acols * Bcols); //A will be transposed

        //both identical
        for (int x = 0; x < Arows; x++)
        {
            for (int y = 0; y < Acols; y++)
            {
                A[x * Acols + y] = x * Acols + y + 1;
                B[x * Acols + y] = x * Acols + y + 1;
            }
        }

        dim3 dim_grid;// (ceilf(Acols / (float)blockSize), ceilf(Bcols / (float)blockSize), 1);
        dim3 dim_block;// (blockSize, blockSize, 1);
        setGridBlockDims(dim_grid, dim_block, Acols, Bcols);
        //for (int x = 0; x < 8; x++)std::cout << A[x] << "\n";
        cuda_matMult<<<dim_grid, dim_block >>>(A, B, C, Acols, Arows, Bcols, TRANSPOSE);
        cudaDeviceSynchronize();

        std::vector<int> answers = { 26, 32, 38, 44, 32, 40, 48, 56, 38, 48, 58, 68, 44, 56, 68, 80 };
        for (int x = 0; x < Acols * Bcols; x++)
        {
            _assertAproxEqual(C[x], answers[x]);
        }

    }

    void testMatMultTransposed2()
    {
        int Arows = 4;
        int Acols = 2;
        int Brows = 4;
        int Bcols = 2;


        //A = B
        //[1, 5
        // 2, 6
        // 3, 7
        // 4, 8]

        //B^T =
        //[1, 2, 3, 4,
        // 5, 6, 7, 8]


        //C = A^T * B = C(4*4)
        //

        float* A;
        float* B;
        float* C;
        initDeviceVectorManaged(A, 0.0f, Arows * Acols);
        initDeviceVectorManaged(B, 0.0f, Brows * Bcols);
        initDeviceVectorManaged(C, 0.0f, Arows * Brows); //B will be transposed

        //both identical
        for (int x = 0; x < Arows; x++)
        {
            for (int y = 0; y < Acols; y++)
            {
                int val = y * Arows + x + 1;
                A[x * Acols + y] = val;
                B[x * Acols + y] = val;
            }
        }

        dim3 dim_grid;// (ceilf(Arows / (float)blockSize), ceilf(Brows / (float)blockSize), 1);
        dim3 dim_block;// (blockSize, blockSize, 1);
        setGridBlockDims(dim_grid, dim_block, Arows, Brows);

        //for (int x = 0; x < 8; x++)std::cout << A[x] << "\n";
        cuda_matMult << <dim_grid, dim_block >> > (A, B, C, Arows, Acols, Brows, NONE, TRANSPOSE);
        cudaDeviceSynchronize();

        std::vector<int> answers = { 26, 32, 38, 44, 32, 40, 48, 56, 38, 48, 58, 68, 44, 56, 68, 80 };
        for (int x = 0; x < Acols * Bcols; x++)
        {
            _assertAproxEqual(C[x], answers[x]);
        }
    }


    void testMatMultTransposed3()
    {
        int Arows = 2;
        int Acols = 4;
        int Brows = 4;
        int Bcols = 2;


        //A =
        //[1, 2, 3, 4,
        // 5, 6, 7, 8]

        //B = 
        //[1, 5
        // 2, 6
        // 3, 7
        // 4, 8]


        //C = A^T * B^T = C(4*4)
        float* A;
        float* B;
        float* C;
        initDeviceVectorManaged(A, 0.0f, Arows * Acols);
        initDeviceVectorManaged(B, 0.0f, Brows * Bcols);
        initDeviceVectorManaged(C, 0.0f, Acols * Brows); //B will be transposed

        for (int x = 0; x < Arows; x++)
        {
            for (int y = 0; y < Acols; y++)
            {
                A[x * Acols + y] = x * Acols + y + 1;
            }
        }

        for (int x = 0; x < Brows; x++)
        {
            for (int y = 0; y < Bcols; y++)
            {
                int val = y * Brows + x + 1;
                B[x * Bcols + y] = val;
            }
        }

        dim3 dim_grid;// (ceilf(Acols / (float)blockSize), ceilf(Brows / (float)blockSize), 1);
        dim3 dim_block;// (blockSize, blockSize, 1);
        setGridBlockDims(dim_grid, dim_block, Acols, Brows);

        //for (int x = 0; x < 8; x++)std::cout << A[x] << "\n";
        cuda_matMult << <dim_grid, dim_block >> > (A, B, C, Acols, Arows, Brows, TRANSPOSE, TRANSPOSE);
        cudaDeviceSynchronize();

        std::vector<int> answers = { 26, 32, 38, 44, 32, 40, 48, 56, 38, 48, 58, 68, 44, 56, 68, 80 };
        for (int x = 0; x < Acols * Bcols; x++)
        {
            _assertAproxEqual(C[x], answers[x]);
        }
    }

    void testMatMultTransposed4()
    {
        int Arows = 1;
        int Acols = 5;
        int Brows = 2;
        int Bcols = 5;

        //A =
        //[1, 2, 3, 4, 5]
        //

        //B = 
        //[1, 2, 3, 4, 5
        // 6, 7, 8, 9, 10]


        //C = A * B^T = C(1*2)
        float* A;
        float* B;
        float* C;
        initDeviceVectorManaged(A, 0.0f, Arows * Acols);
        initDeviceVectorManaged(B, 0.0f, Brows * Bcols);
        initDeviceVectorManaged(C, 0.0f, Arows * Brows); //B will be transposed

        for (int x = 0; x < Arows; x++)
        {
            for (int y = 0; y < Acols; y++)
            {
                A[x * Acols + y] = x * Acols + y + 1;
            }
        }

        for (int x = 0; x < Brows; x++)
        {
            for (int y = 0; y < Bcols; y++)
            {
                B[x * Bcols + y] = x * Bcols + y + 1;
            }
        }

        dim3 dim_grid; // (ceilf(Arows / (float)TILE_DIM), ceilf(Brows / (float)TILE_DIM), 1);
        dim3 dim_block; // (blockSize, blockSize, 1);
        setGridBlockDims(dim_grid, dim_block, Arows, Brows);
        //for (int x = 0; x < 8; x++)std::cout << A[x] << "\n";
        cuda_matMult <<<dim_grid, dim_block >>> (A, B, C, Arows, Acols, Brows, NONE, TRANSPOSE);
        cudaDeviceSynchronize();

        std::vector<int> answers = { 55, 130 };
        for (int x = 0; x < Arows * Brows; x++)
        {
            _assertAproxEqual(C[x], answers[x]);
        }
    }

    void testLinearLayerForward() //currently setup for only one row, so bear that in mind.
    {
        int inRows = 1;
        int inCols = 10;
        int outCols = 10;
        LinearLayer linear(inCols, outCols, inRows, NONE, 1.0f);
        float* input;
        float* output;
        //initVal_Float(input, 1.0f, 10);
        initDeviceVectorManaged(output, 0.0f, inRows * outCols);
        initDeviceVectorManaged(input, 1.0f, inRows * inCols);
        //set some arbitrary bias
        float* bias;
        float* weights;
        //cudaMallocManaged(&bias, 10 * sizeof(float));
        cudaFree(linear.bias);
        cudaFree(linear.weights);
        initDeviceVectorManaged(bias, 1.0f, inRows * outCols);
        initDeviceVectorManaged(weights, 2.0f, inCols * outCols);
        linear.bias = bias;
        linear.weights = weights;

        cudaDeviceSynchronize();
        linear.forward(input, output);
        cudaDeviceSynchronize();

        for (int x = 0; x < outCols; x++)
        {
            _assertAproxEqual(output[x], getMatrixMultValue(input, linear.weights, x, inCols, outCols) + bias[x]);
        }

        /*
        for (int x = 0; x < outCols; x++)
        {
            std::cout << "\n";
            for (int y = 0; y < inRows; y++)
            {
                std::cout << output[y * outCols + x] << ", ";
            }
        }
        */
    }

    void testLinearLayerBackwardAndUpdate() //currently setup for only one row, so bear that in mind.
    {
        int inRows = 1;
        int inCols = 10;
        int outCols = 10;
        float* input;
        float* output;
        float* error;
        LinearLayer linear(inCols, outCols, inRows, NONE, 1.0f);
        //initVal_Float(input, 1.0f, 10);
        initDeviceVectorManaged(output, 0.0f, inRows * outCols);
        initDeviceVectorManaged(input, 3.0f, inRows * inCols);
        initDeviceVectorManaged(error, 0.5f, inRows * inCols);
        //set some arbitrary bias
        float* bias;
        float* weights;
        //cudaMallocManaged(&bias, 10 * sizeof(float));
        cudaFree(linear.bias);
        cudaFree(linear.weights);
        initDeviceVectorManaged(bias, 1.0f, inRows * outCols);
        initDeviceVectorManaged(weights, 2.0f, inCols * outCols);
        linear.bias = bias;
        linear.weights = weights;

        /*
        std::cout << "PRIOR\n";
        for (int x = 0; x < inCols; x++)
        {
            std::cout << "\n";
            for (int y = 0; y < outCols; y++)
            {
                std::cout << linear.weights[x * outCols + y] << ",";
            }
        }
        std::cout << "BIAS: ";
        for (int x = 0; x < outCols; x++)std::cout << linear.bias[x] << ", ";
        std::cout << "\n";
        */

        cudaDeviceSynchronize();
        linear.forward(input, output);
        error = linear.backwardAndUpdate(error);
        cudaDeviceSynchronize();

        /*
        std::cout << "POST UPDATE\n";
        for (int x = 0; x < inCols; x++)
        {
            std::cout << "\n";
            for (int y = 0; y < outCols; y++)
            {
                std::cout << linear.weights[x * outCols + y] << ",";
            }
        }
        std::cout << "BIAS: ";
        for (int x = 0; x < outCols; x++)std::cout << linear.bias[x] << ", ";
        std::cout << "\n";
        */

        for (int x = 0; x < inRows * outCols; x++)
        {
            _assertAproxEqual(linear.bias[x], 0.5f);
        }

        for (int x = 0; x < inCols * outCols; x++)
        {
            _assertAproxEqual(linear.weights[x], 0.5f);
        }

        for (int x = 0; x < inRows * inCols; x++)
        {
            _assertAproxEqual(error[x], 1.0f*inCols); //the origional error gets matMulted to transpose of weights
        }
    }

    void testXOR()
    {
        std::vector<float*> trainData;
        std::vector<float*> trainTarget; //(labels)
        float* input;
        float* output;
        float* target;
        //initDeviceVectorManaged(trainData, 0.0f, 8);
        //initDeviceVectorManaged(trainTarget, 0.0f, 4);
        initDeviceVectorManaged(output, 0.0f, 1);
        initDeviceVectorManaged(input, 0.0f, 2);
        initDeviceVectorManaged(target, 0.0f, 1);
        std::vector<std::pair<float, float>> data = { {0,0}, {0,1}, {1,0}, {1,1} };
        std::vector<float> targets = { 0, 1, 1, 0 };
        //std::vector<std::pair<float, float>> data = { {1,1}, {1,1}, {1,1}, {1,1} };
        //std::vector<float> targets = { 1, 1, 1, 1 };
        for (auto& d : data)
        {
            float* newRow;
            initDeviceVectorManaged(newRow, 0.0f, 2);
            newRow[0] = d.first;
            newRow[1] = d.second;
            trainData.push_back(newRow);
        }

        for (auto& t : targets)
        {
            float* newRow;
            initDeviceVectorManaged(newRow, t, 1);
            trainTarget.push_back(newRow);
        }
        
        cudaDeviceSynchronize();

        int inRows = 1;
        int inCols = 2;
        int hiddenCols = 4; //10;
        int outCols = 1;
        //with 0.01f learning and 10k iterations it can learn this. That seems overkill?
        //also 0.1f lr, 1k iterations, hidden cols=10, 4 layers works pretty damn well //edit: reaaally well, tried it 10 times and worked every time.
        float learningRate = 0.1f;
        int trainIterations = 1000;

        /*
        LinearLayer linear1(inCols, hiddenCols, inRows, TANH, learningRate, RANDFLOAT);
        LinearLayer linear2(hiddenCols, hiddenCols, inRows, TANH, learningRate, RANDFLOAT);
        LinearLayer linear3(hiddenCols, hiddenCols, inRows, TANH, learningRate, RANDFLOAT);
        LinearLayer linear4(hiddenCols, outCols, inRows, TANH, learningRate, RANDFLOAT);
        */

        //https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65 uses 2 layers total, 3 hidden neurons //smallest Ive gotten is 2 layers with 4 hidden...
        LinearLayer linear1(inCols, hiddenCols, inRows, TANH, learningRate, RANDFLOAT);
        //LinearLayer linear2(hiddenCols, hiddenCols, inRows, TANH, learningRate, RANDFLOAT);
        LinearLayer linear2(hiddenCols, outCols, inRows, TANH, learningRate, RANDFLOAT);

        NeuralNetwork nn(std::vector<LinearLayer*>{&linear1, &linear2}, MEAN_SQURED_ERROR);

        for (int x = 0; x < trainIterations; x++)
        {
            for(int y = 0; y < trainData.size(); y++)
            {
                nn.networkTrain(trainData[y], trainTarget[y], output, 1);
            }
        }

        float tAcc = 0.0f;
        for (int y = 0; y < trainData.size(); y++)
        {
            tAcc += nn.networkTest(trainData[y], trainTarget[y], output, 1, false);
            std::cout << " targ: " << trainTarget[y][0] << " out: " << output[0] << "\n";
        }

        std::cout << "TACC: " << tAcc << "\n";

    }

    void testCategoricalLearning() //does work for all activiations, assuming that you go with "highest output is the value it picked" methodology
    {
        const int tTrainSamples = 3;
        std::vector<float*> trainData(tTrainSamples);
        std::vector<float*> trainLabel(tTrainSamples);
        float* output;

        int inRows = 1;
        int inCols = 500;
        int outCols = 10;
        float learningRate = 0.1f;

        //input is a single vector with a kind of "spot" in the middle of it, learn this to be categorical value of choice "0"
        initDeviceVectorManaged(output, 0.0f, inRows * outCols);


        for (int s = 0; s < tTrainSamples; s++)
        {
            initDeviceVectorManaged(trainData[s], 0.0f, inRows * inCols);
            initDeviceVectorManaged(trainLabel[s], 0.0f, inRows * outCols);

            for (int x = 0; x < inCols; x++)
            {
                int start = s * 100;
                if (x > start && x < start + 100)
                {
                    trainData[s][x] = 1.0f - (abs(100.0f - (x - 200)) / 100.0f);
                }
            }
        }
        trainLabel[0][0] = 1.0f;
        trainLabel[1][5] = 1.0f;
        trainLabel[2][9] = 1.0f;

        LinearLayer linear1(inCols, 250, inRows, SIGMOID, learningRate, RANDFLOAT);
        LinearLayer linear2(250, 100, inRows, SIGMOID, learningRate, RANDFLOAT);
        LinearLayer linear3(100, outCols, inRows, SIGMOID, learningRate, RANDFLOAT);
        NeuralNetwork nn(std::vector<LinearLayer*>{&linear1, &linear2, &linear3}, MEAN_SQURED_ERROR);

        int epochs = 5;
        int trainIters = 100;
        int sampleIters = 1;
        for (int e = 0; e < epochs; e++)
        {
            std::cout << "EPOCH: " << e << "\n";
            float tErr = 0.0f;
            for (int x = 0; x < trainIters; x++)
            {
                for (int s = 0; s < tTrainSamples; s++)
                {
                    nn.networkTrain(trainData[s], trainLabel[s], output, sampleIters);
                }
            }

            cudaDeviceSynchronize();
            tErr = nn.totalError[0];
            nn.totalError[0] = 0.0f;

            std::cout << " err: " << tErr / float(trainIters*tTrainSamples) << "\n";
        }

        std::cout << "TESTING:\n";
        for (int s = 0; s < tTrainSamples; s++)
        {
            std::cout << " sample: " << s << "\n";
            nn.networkForward(trainData[s], output);
            cudaDeviceSynchronize();
            for (int y = 0; y < outCols; y++)
            {
                std::cout << "  targ: " << trainLabel[s][y] << " out: " << output[y] << "\n";
            }
        }


    }


    void testMNIST()
    {
        std::cout << "\n MNIST:\n";
        //might be useful https://mlfromscratch.com/neural-network-tutorial/#/

        std::string base = "D:/C++ Work/CUDAwork/CUDAwork/data/";
        std::string trainingImagesPath = base + "train-images.idx3-ubyte";
        std::string trainingLabelsPath = base + "train-labels.idx1-ubyte";
        std::string imagesPath = base + "t10k-images.idx3-ubyte";
        std::string labelsPath = base + "t10k-labels.idx1-ubyte";

        //Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
        //The labels values are 0 to 9.
        std::vector<std::vector<float>> trainImages;
        std::vector<float> trainLabelsRaw;
        readMnistImages(trainImages, trainingImagesPath);
        readMnistLabels(trainLabelsRaw, trainingLabelsPath);

        std::vector<std::vector<float>> images;
        std::vector<float> labelsRaw;
        readMnistImages(images, imagesPath);
        readMnistLabels(labelsRaw, labelsPath);

        //for ease of training, transform the labels from a numeric 0->9 into a vector of probabilities, 0.0f->100.0f. First in vector is 0, last in vector is 9
        std::vector<std::vector<float>> trainingLabels = transformMnistLabelsToProbVector(trainLabelsRaw);
        std::vector<std::vector<float>> labels = transformMnistLabelsToProbVector(labelsRaw);

        int batchSize = 1; //number of images per batch to train from //inRows
        int inCols = 784;
        int hiddenCols1 = 100;
        int hiddenCols2 = 50;
        int outCols = 10;
        int epochs = 20; //number of training epochs
        int iterationsPerEpoch = 1; //per batch
        int iterationsPerImage = 1;
        float learningRate = 0.03f;

        //relu n.b: if you arent using it in conjunction with smax out (which helpfully makes the backprop error more manageable) you may need to crank lr down a lot.

        //LinearLayer linear1(inCols, hiddenCols1, batchSize, TANH, learningRate, RANDFLOAT);
        //LinearLayer linear2(hiddenCols1, hiddenCols2, batchSize, TANH, learningRate, RANDFLOAT);
        //LinearLayer linear3(hiddenCols2, outCols, batchSize, TANH, learningRate, RANDFLOAT);

        //this somehow works for sigmoid? //but sigmoid only likes this lol??? //sigmoid seems to hate larger layer sizes //increasing learning rate helps sig if it gets stuck!
        /*
        LinearLayer linear1(inCols, 100, batchSize, SIGMOID, learningRate, RANDFLOAT);
        LinearLayer linear2(100, 50, batchSize, SIGMOID, learningRate, RANDFLOAT);
        LinearLayer linear3(50, outCols, batchSize, SIGMOID, learningRate, RANDFLOAT);
        */

        //for sigmoid, best ive seen so far is {in, 512}, {512, 512}, {512, out} with lr=0.1f, XAVIER init, 95.5% acc.
        //for relu you need tiny tiny learning rate, else gradient explodes
        //for relu, this actually works: {in, 512}, {512, 512}, {512, 512}, {512, 512}, {256, out}, lr=0.001f, HE init. Acc = 95%
        //highest acc so far: relu - {in, in}, {in, in}, {in, in}, {in, in}, {in, out}, lr=0.01f, HE init. Acc = 96.8%
        //for softmax it seems you need tiny learning rate to avoid gradient explosion, with sigmoid at least...
        //5 layer sigmoid currently requiring lr=0.000000001f, suggests either incorrect weight init, or an error somewhere
        //smax + tanh works with lr = 0.001f on 5 layer. Works pretty darn well in fact
        //over 6 layers with relu + smax and the gradients start to explode
    
        LinearLayer l1(inCols, 1000, batchSize, RELU, learningRate, HE);
        LinearLayer l2(1000, outCols, batchSize, SOFTMAX, learningRate, XAVIER);
        //LinearLayer l3(300, 300, batchSize, RELU, learningRate, HE);
        //LinearLayer l4(300, 300, batchSize, RELU, learningRate, HE);
        //LinearLayer l5(300, 300, batchSize, RELU, learningRate, HE);
        //LinearLayer l6(300, 300, batchSize, RELU, learningRate, HE);
        //LinearLayer l7(300, outCols, batchSize, SOFTMAX, learningRate, XAVIER);

        //too many epochs lead to overfitting, provide very minimal final acc increase even if train acc decreases.
        //At a certain point the neural network is no longer reducing train acc because it is learning the problem, instead it is now learning
            //the dataset, i.e memorising the dataset.


        //2 layer tests, 50 epochs:
        // {in, 1000} tanh, {1000, out} smax, lr = 0.01f //0.9635 //600s
        // {in, 1000} tanh, {1000, out} smax, lr = 0.02f //0.9579 //647s
        // {in, 1000} tanh, {1000, out} smax, lr = 0.005f //0.9591 //1080s
        // {in, 1000} tanh, {1000, out} smax, lr = 0.01f //100e //0.9605 //1200s
        // {in, 1000} tanh, {1000, out} smax, lr = 0.01f //10e //0.9617 //167s ??????? more acc with 10e than 100e? wtf?
        // {in, 1000} tanh, {1000, out} smax, lr = 0.01f //10e //0.9594 //167s //relatively low variance
        // {in, 1000} tanh, {1000, out} smax, lr = 0.01f //10e //0.9637 //167s //KAIMING
        // {in, 1000} tanh, {1000, out} smax, lr = 0.001f //10e //0.9217 //167s
        // {in, 1000} tanh, {1000, out} smax, lr = 0.1f //10e //0.9497 //167s
        // {in, 1000} tanh, {1000, out} smax, lr = 0.01f //30e //0.9614 //400s
        // 
        // {in, 2000} tanh, {2000, out} smax, lr = 0.01f //0.9592 //1074s //less final acc required than above...
        // {in, 2000} tanh, {2000, out} smax, lr = 0.005f //0.9591 //1080s
        // {in, 2000} tanh, {2000, out} smax, lr = 0.02f //0.9546 //1080s
        // 
        // {in, inCols} tanh, {inCols, out} smax, lr = 0.01f //0.9605 //500s

        // {in, 1000} relu, {1000, out} smax, lr = 0.01f //10e //0.9637 //162s 
        // {in, 1000} relu, {1000, out} smax, lr = 0.01f //15e //0.9640 //162s 
        // {in, 1000} relu, {1000, out} smax, lr = 0.01f //20e //0.9680 //162s 
        // {in, 1000} relu, {1000, out} smax, lr = 0.01f //20e //0.9670 //162s 
        // {in, 1000} relu, {1000, out} smax, lr = 0.01f //30e //0.9675 //412s 

        //20 epochs seems best //relu seems to break 2 layer above 1000, even with small lr
        //{in, in} relu, {in, in} relu, { in, out } smax, lr = 0.01f //20e //0.9664 //400s 
        //{in, in} relu, {in, in} relu, { in, out } smax, lr = 0.01f //10e //0.9664 //400s //exactly the same, odd eh? seems to reach max learning after ~10epochs with 3 layers
        //{in, in} relu, {in, in} relu, { in, out } smax, lr = 0.01f //5e //0.9477 //400s 

        //finally managed some fat relus to work
        // {in, 2000} relu, { 2000, 2000 } relu, { 2000, out } smax, lr = 0.01f //20e //0.9658 //1162s 
        // relus {in, 300} , {300, 300} x 5, {300, out} smax, lr = 0.01f //30e //0.9023
        // {in, 5000} relu, {5000, out} smax, lr = 0.01f //10e //0.9664 //492.704
        // {in, 5000} relu, {5000, out} smax, lr = 0.02f //10e //0.9636 //492.704
        // {in, 10000} relu, {10000, out} smax, lr = 0.01f //20e //0.9665 //492.704

        //doesnt work with 4/3/2000 1 relu, 1 smax? for whatever reason // works with 1 and 5k //consistently. Why it doesnt work for 2/3/4k i dunno.
        //works with 10k.

        //{in, in}, {in, in}, {in, in}, {in, in}, relu {in, out} smax, lr = 0.01f //20e //0.9644

        NeuralNetwork nn(std::vector<LinearLayer*>{
            &l1, &l2, // &l3, &l4, &l5, //&l6, //&l7
        }, CROSS_ENTROPY_ERROR);

        float* output;
        initDeviceVectorManaged(output, 0.0f, batchSize * outCols);

        std::chrono::steady_clock::time_point begin, end;
        begin = std::chrono::steady_clock::now();

        int trainSize = trainImages.size(); //the smaller this is, the more efficient the training. Almost as if the amount of mem you have alloced on the GPU drags on perf a lot...
        int trainsPerEpoch = 1;
        int validateSize = 5;
        int trainsPerImage = 1;

        //prep data
        /*
        std::vector<float*> trainImagesDevice;
        std::vector<float*> trainlabelsDevice;
        for (int x = 0; x < trainSize; x++)
        {
            float* image;
            float* label;
            initDeviceVectorManagedFromVec(image, trainImages[x], trainImages[x].size());
            initDeviceVectorManagedFromVec(label, trainingLabels[x], trainingLabels[x].size());
            trainImagesDevice.push_back(image);
            trainlabelsDevice.push_back(label);
        }
        */

        std::vector<float*> testImagesDevice;
        std::vector<float*> testlabelsDevice;
        for (int x = 0; x < validateSize; x++)
        {
            float* image;
            float* label;
            initDeviceVectorManagedFromVec(image, images[x], images[x].size());
            initDeviceVectorManagedFromVec(label, labels[x], labels[x].size());
            testImagesDevice.push_back(image);
            testlabelsDevice.push_back(label);
        }

        //this allocation now accounting for 50% cpu overhead, may be best to do it more in bulk. But GPU usage already up to 98% so, probs is ok...
        //just preload them all, too much overhead on the loading
        std::vector<float*> trainImagesDevice;
        std::vector<float*> trainlabelsDevice;
        for (int x = 0; x < trainImages.size(); x++)
        {
            float* image;
            float* label;
            initDeviceVectorManagedFromVec(image, trainImages[x], trainImages[x].size());
            initDeviceVectorManagedFromVec(label, trainingLabels[x], trainingLabels[x].size());
            trainImagesDevice.push_back(image);
            trainlabelsDevice.push_back(label);
        }

        //training, validation
        int offset = 0;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            std::cout << "\nEPOCH " << epoch << "\n";

            float tError = 0.0f;

            std::vector<int> thisEpochTrainIndices;
            for (int x = 0; x < trainImages.size(); x++)thisEpochTrainIndices.push_back(x);

            for (int iter = 0; iter < trainsPerEpoch; iter++)
            {
                int start = epoch * trainsPerEpoch * trainSize + (iter * trainSize) + offset;
                if (start + trainSize>= trainImages.size())
                {
                    offset -= trainImages.size();
                    start = 0;
                    //break;
                }

                //train
                for (int x = start; x < start+trainSize-1; x++)
                {
                    //for SGD pick an image at random from the ones not yet tested, then remove it from the available indices
                    if (!thisEpochTrainIndices.size())break;
                    int randEind = randomInt(0, thisEpochTrainIndices.size() - 1);
                    int randIndex = thisEpochTrainIndices[randEind];
                    thisEpochTrainIndices.erase(thisEpochTrainIndices.begin() + randEind);

                    //tError += 
                    nn.networkTrain(trainImagesDevice[randIndex], trainlabelsDevice[randIndex], output, trainsPerImage);
                    //std::cout << "\n\n";
                    //for (int i = 0; i < 10; i++)std::cout << " out: " << output[i] << " targ:" << trainlabelsDevice[x][i] << "\n";
                }
            }

            //read total error, and reset it
            cudaDeviceSynchronize();
            tError = nn.totalError[0];
            nn.totalError[0] = 0.0f;
          
            std::cout << "  error: " << tError / float(trainSize * trainsPerEpoch) << "\n";
            //validate
            /*
            float tAcc = 0.0f;
            for (int x = 0; x < validateSize; x++)
            {
                tAcc += nn.networkTest(testImagesDevice[x], testlabelsDevice[x], output, batchSize * outCols, true);
            }
            std::cout << "Avg PredAcc: " << tAcc / float(validateSize) << "\n";
            */
        }

        //dealloc
        for (int x = 0; x < trainImages.size(); x++)
        {
            cudaFree(trainImagesDevice[x]);
            cudaFree(trainlabelsDevice[x]);
        }
        trainlabelsDevice.clear();
        trainImagesDevice.clear();

        //test out
        std::cout << "TESTING SAMPLE:\n";
        for (int x = 0; x < validateSize; x++)
        {
            nn.networkForward(testImagesDevice[x], output);
            cudaDeviceSynchronize();
            std::cout << " TEST IMAGE " << x << "\n";
            for (int y = 0; y < outCols; y++)
            {
                std::cout << "  targ: " << testlabelsDevice[x][y] << " out: " << output[y] << "\n";
            }
            std::cout << "\n";
        }

        testImagesDevice.clear();
        testlabelsDevice.clear();

        std::cout << "\nFULL TEST:\n";
        int testBatchSize = 100;
        int correct = 0;
        int tTested = 0;

        for (int b = 0; b < labels.size() / testBatchSize; b++)
        {
            int start = b * testBatchSize;
            for (int x = start; x < start + testBatchSize; x++)
            {
                float* image;
                float* label;
                initDeviceVectorManagedFromVec(image, images[x], images[x].size());
                initDeviceVectorManagedFromVec(label, labels[x], labels[x].size());
                testImagesDevice.push_back(image);
                testlabelsDevice.push_back(label);
            }

            for (int x = 0; x < testBatchSize; x++)
            {
                nn.networkForward(testImagesDevice[x], output);
                cudaDeviceSynchronize();
                float largestPred = -1000.0f;
                int predIndex = 0;
                for (int ot = 0; ot < outCols; ot++)
                {
                    if (output[ot] > largestPred)
                    {
                        largestPred = output[ot];
                        predIndex = ot;
                    }
                }
                if (testlabelsDevice[x][predIndex] > 0.0f)correct++;
                tTested++;
            }

            for (int x = 0; x < testBatchSize; x++)
            {
                cudaFree(testImagesDevice[x]);
                cudaFree(testlabelsDevice[x]);
            }
            testImagesDevice.clear();
            testlabelsDevice.clear();
        }
     
        std::cout << " correct: " << correct << " / " << tTested << "\n";
        std::cout << "FINAL ACC: " << float(correct) / float(tTested) << "\n";


        end = std::chrono::steady_clock::now();
        std::cout << "Training time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0f << std::endl;
    }


    void runAllTests()
    {
        std::cout << "RUNNING ALL TESTS...\n";

        testMatMult();
        testMatMultTransposed();
        testMatMultTransposed2();
        testMatMultTransposed3();
        testMatMultTransposed4();
        testLinearLayerForward();
        testLinearLayerBackwardAndUpdate();
        testXOR();
        testCategoricalLearning();
        testMNIST();

        std::cout << "ALL TESTS COMPLETED SUCCESSFULLY\n";
    }
};
