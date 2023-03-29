#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <device_double_functions.h>
#include <random>
#include <fstream>

//GLOBALS
//global handle
extern cublasHandle_t globalCublasHandle;

//DEFS
#define TILE_DIM 32
//#define BLOCK_ROWS 8
#define NONE -1
#define LEAKY_RELU_ALPHA 0.1f //setting this higher seemed to reduce the effects of gradient explosion

//ENUMS
enum WEIGHT_INIT_MODE
{
    KAIMING,
    XAVIER,
    RANDFLOAT,
    HE,
};

enum LOSS_FUNCTIONS
{
    MEAN_SQURED_ERROR, //mean squared error
    CROSS_ENTROPY_ERROR, //cross entropy loss
};

enum ACTIVATION_FUNCTIONS
{
    RELU,
    SIGMOID,
    SOFTMAX,
    TANH
};

enum MATRIX_OPERATIONS
{
    TRANSPOSE
};

//CPU FUNCS/UTILS
static void cudaErr(cudaError_t in)
{
    if (in != cudaSuccess)
    {
        std::cout << "CUDA ERROR!\n";
    }
}

static void cublasCheck(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS ERROR! Status was " << status << "\n";
    }
}

static void setGridBlockDims(dim3& dimGrid, dim3& dimBlock, int nRows, int nCols)
{
    dimGrid = dim3((nRows + TILE_DIM - 1) / TILE_DIM, (nCols + TILE_DIM - 1) / TILE_DIM);
    dimBlock = dim3(TILE_DIM, TILE_DIM);
}

//C = A * B
//uses column major, think it has caused issues when using it as I assume matrices to be row major. Best to avoid.
/*
static void cublas_matrixMult(
    cublasHandle_t& cublasHandle,
    float* A,
    float* B,
    float* C,
    int A_rows,
    int A_cols,
    int B_cols,
    cublasOperation_t opA = CUBLAS_OP_N,
    cublasOperation_t opB = CUBLAS_OP_N,
    float alpha = 1.0f,
    float beta = 0.0f //adds const to all outputs of C (resulting matrix)
)
{
    //static float alpha = 1.0f;
    //static float beta = 0.0f;
    //static ; //A if CUBLAS_OP_N, A^T if CUBLAS_OP_T, A^H if CUBLAS_OP_C
    //static ; //B if CUBLAS_OP_N, B^T if CUBLAS_OP_T, B^H if CUBLAS_OP_C

    //cudaDeviceSynchronize();
    //General matrix multiply -> gemm
    cublasCheck(cublasSgemm(
        cublasHandle,
        opA,
        opB,
        A_rows,
        B_cols,
        A_cols,
        &alpha,
        A,
        A_rows,
        B,
        A_cols,
        &beta,
        C,
        A_rows
    ));
    cudaDeviceSynchronize(); //this seems to be required for release
    //cudaDeviceSynchronize();
}
*/

//https://stats.stackexchange.com/questions/373136/softmax-weights-initialization
static void kaimingInit(float* w, int n_in, int n_out)
{
    float std = sqrt(1.0f / (float)n_in);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std);

    for (int i = 0; i < n_in * n_out; i++)
    {
        w[i] = dist(gen);
    }
}

static float truncatedNormalDistribution(float mean, float sigma, float min, float max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, sigma);

    float val = dist(gen);
    while(val <= min || val >= max)val = dist(gen); //in theory, this could go on forever.

    return val;
}

static void xavierInit(float* w, int n_in, int n_out)
{
    //float std = sqrt(2.0f / ((float)n_in + (float)n_out));
    //float upper = 6.0f / sqrt(n_in);
    float upper = sqrt(6.0f) / sqrt(n_in + n_out);
    float lower = -upper;

    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<float> dist(lower, upper);

    for (int i = 0; i < n_in * n_out; i++)
    {
        w[i] = dist(gen);
    }
}

static void randFloatInit(float* w, int n_in, int n_out)
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n_in * n_out; i++)
    {
        w[i] = dist(gen) - 0.5f;
    }
}

//this works for relu, but only up to around 1000 out layers...for whatever bloody reason...
static void HEInit(float* w, int n_in, int n_out)
{
    /*
    float sigma = sqrt(1.55f / float(n_in));
    for (int i = 0; i < n_in * n_out; i++)
    {
        w[i] = truncatedNormalDistribution(0.0f, sigma, -1.0f, 1.0f);
    }
    */

    //for {in, 2000}, {2000, 2000}, {2000, out} all relu, this works but it needs a 0.001f LR...

    //this actually works pretty well with smax even with 0.01f lr
    float sigma2 = 1.0f / float(n_in*0.5f+n_out*0.5f);
    float lim = sqrt(3.0f * sigma2);
    std::random_device rd;
    std::default_random_engine gen(rd());
    //std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist(-lim, lim); //initial performance improved on 2000 out relu when allowing negatives

    for (int i = 0; i < n_in * n_out; i++)
    {
        w[i] = dist(gen); //*sigma;
    }

    /*

    float sigma = sqrt(2.0f / ((float)n_in + (float)n_out));
    //float sigma = 2.0f / sqrt(n_in); //above 2000 this seems to work with like 6.0f? but the starting error is massive and gradients die

    //sigma too low: vanishing gradient //sigma too high: exploding gradient up to NaN
    //In theory for a given layer with n_in, n_out, (ASSUMING that the overall inputs are limited/normalized to 0->1) there is a critical point
    //this critical point is where say you have a full row (or whatever) of pure 1.0s in the input.
    //if you multiply that input across the entire layer and get the output, if weights are too high then at backprop the gradient might be sifficiently large
        //to cause oscilation around 0 in the weights
    //Hence the point at which the output is large enough to cause this oscialtion - grad explosion - can be seen as the "critial value"
    //Ideally you want to get as close to the critical value as possible with weight init, but never go over it.
    //One thing is for sure -> if inputs lim(0,1) then setting weights between (-1,1) with a multiplier of 1 / (tSize + 1) will NEVER exceed critical value
    //So "sigma = 1.0f / float(n_in + n_out + 1);" can be taken as a base case where we know for sure we are safe -> the initial output will never exceed 1
    //In reality we are probably quite far below the crit point here though, we probs have some room to move this up and save some dead epochs
    //Usually what you see is EPOCH 0: 0.23, EPOCH 1: 0.225, EPOCH 2: 0.224, then it may jump to EPOCH 3: 0.09 and start learning, so first 2 epochs are dead 

    //hmmm sigma = 1.0f / float(n_in + n_out + 1) failed for 2 layer 3000w
    //actually it does seem to work, but needs at LEAST 2 layers to be RELU, the above was 1 relu, 1 smax (out)

    //float sigma = 1.0f / float(n_in + n_out + 1); //this actually worked with a large 3 layer relu, 2000 weights each...took a few dead epochs but did start learning

    std::random_device rd;
    std::default_random_engine gen(rd());
    //std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f); //initial performance improved on 2000 out relu when allowing negatives

    for (int i = 0; i < n_in * n_out; i++)
    {
        w[i] = dist(gen) * sigma;
    }
    */
}

static int randomInt(int min, int max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}

static void readTestCSV(float* inp, std::string name)
{
    std::ifstream file(name);
    std::string line;

    while (std::getline(file, line, '\n'))
    {
        *inp = std::stof(line);
        inp++;
    }
}


//mnist

static int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
static void readMnistImages(std::vector<std::vector<float>>& images, std::string fullFilePath)
{
    std::ifstream file(fullFilePath, std::ios::binary); //+"t10k-images-idx3-ubyte"
    //std::vector<std::vector<float>> images;
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
        for (int i = 0; i < number_of_images; ++i)
        {
            std::vector<float> image;
            for (int r = 0; r < n_rows; ++r)
            {
                for (int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    image.push_back(float(temp) / 255.0f); //divide by 255 so you get a value between 0 and 1
                }
            }
            images.push_back(image);
        }
    }
}
static void readMnistLabels(std::vector<float>& labels, std::string fullFilePath)
{
    std::ifstream file(fullFilePath, std::ios::binary); //+"t10k-images-idx3-ubyte"
//std::vector<std::vector<float>> images;
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_labels = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);
        for (int i = 0; i < number_of_labels; ++i)
        {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            labels.push_back(float(temp));
        }
    }
}

static std::vector<std::vector<float>> transformMnistLabelsToProbVector(std::vector<float>& rawLabels)
{
    std::vector<std::vector<float>> probData;
    for (auto& val : rawLabels)
    {
        std::vector<float> thisProbs;
        for (int x = 0; x < 10; x++)
        {
            if (val == float(x))
            {
                thisProbs.push_back(1.0f);
            }
            else {
                thisProbs.push_back(0.0f);
            }
        }
        probData.push_back(thisProbs);
    }

    return probData;
}

//managed version, cpu and gpu readible (at cost to performance)
static void initDeviceVectorManaged(float*& deviceVec, float initialVal, int size)
{
    //float* initVec;
    cudaErr(cudaMallocManaged(&deviceVec, size * sizeof(float)));
    for (int x = 0; x < size; x++)deviceVec[x] = initialVal; //can be made faster via cuda kernel
    //deviceVec = initVec;
}

static void initDeviceVectorManagedFromVec(float*& deviceVec, std::vector<float> initialValues, int size)
{
    //float* initVec;
    cudaErr(cudaMallocManaged(&deviceVec, size * sizeof(float)));
    cudaDeviceSynchronize();
    for (int x = 0; x < size; x++)deviceVec[x] = initialValues[x]; //can be made faster via cuda kernel
    //deviceVec = initVec;
}

static void zeroDeviceVector(float*& deviceVec, int size)
{
    cudaErr(cudaMemset(&deviceVec, 0, size * sizeof(float)));
}

static void zeroManagedVector(float*& managedVec, int size)
{
    for (int x = 0; x < size; x++)managedVec[x] = 0.0f;
}

static float randFloatLinear(float start, float end)
{
    float evalMin = std::min(start, end); //just in case
    float evalMax = std::max(start, end);
    static std::random_device rd;
    static std::default_random_engine eng(rd());
    std::uniform_real_distribution<> distr(evalMin, evalMax);

    return distr(eng);
}


//CUDA FUNCS/UTILS
//https://medium.com/analytics-vidhya/matrix-multiplication-in-cuda-a-simple-guide-bab44bc1f8ab
//https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu
//A(m*n) * B(n*p) = C(m*p)
__global__ static void cuda_matMult(float* A, float* B, float* C, int Arows, int width, int Bcols, int matOpA = NONE, int matOpB = NONE)
{
    //maybe try some stuff in here https://benvanwerkhoven.github.io/kernel_tuner/matrix_multiplication.html
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // check boundry conditions
    if (row < Arows && col < Bcols)
    {
        float thisVal = 0.0f;
        for (int i = 0; i < width; i++)
        {
            int Aindex = (matOpA == TRANSPOSE) ? row + i * Arows : row * width + i;
            int Bindex = (matOpB == TRANSPOSE) ? col * width + i : i * Bcols + col;
            //if (matOpA == TRANSPOSE)Aindex = row + i * Arows;
            //if (matOpB == TRANSPOSE)Bindex = col * width + i;

            thisVal += A[Aindex] * B[Bindex];
        }
        C[row * Bcols + col] = thisVal;
    }
    

    //not working
    /*
    int y = blockIdx.x * TILE_DIM + threadIdx.x; //col //swapped these around and the matmult tests all passed so...yeah technically totally gooberish but it works
    int x = blockIdx.y * TILE_DIM + threadIdx.y; //row
    //int width = gridDim.x * TILE_DIM;
    int aSize = Arows * width;
    int bSize = width * Bcols;
    int cSize = Arows * Bcols;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        float thisVal = 0.0f;
        for (int i = 0; i < width; i++)
        {
            //int index = (y + j) * width + x;

            int Aindex = (matOpA == TRANSPOSE) ? (y+j) + i * Arows : (y+j) * width + i;
            int Bindex = (matOpB == TRANSPOSE) ? x * width + i : i * Bcols + x;
            //if (matOpA == TRANSPOSE)Aindex = row + i * Arows;
            //if (matOpB == TRANSPOSE)Bindex = col * width + i;

            if (Aindex < aSize && Bindex < bSize)
            {
                thisVal += A[Aindex] * B[Bindex];
            }
        }
        int cIndex = (y+j) * Bcols + x;
        if (cIndex < cSize) //you defo want this
        {
            C[cIndex] = thisVal;
        }
    }
    */
}

/*
__device__ static void cuda_matMultFAST(float* A, float* B, float* C, int Arows, int width, int Bcols, int matOpA = NONE, int matOpB = NONE)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // check boundry conditions
    if (row < Arows && col < Bcols)
    {
        float thisVal = 0.0f;
        for (int i = 0; i < width; i++)
        {
            int Aindex = row * width + i;
            int Bindex = i * Bcols + col;
            if (matOpA == TRANSPOSE)Aindex = row + i * Arows;
            if (matOpB == TRANSPOSE)Bindex = col * width + i;

            thisVal += A[Aindex] * B[Bindex];
        }
        C[row * Bcols + col] = thisVal;
    }
}
*/


//SLOWWWW maybe just use set matrices equal instead?
/*
__global__ static void cuda_setVectorsEqual(float* toSet, float* setTo, int tSizeVector)
{
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    if (ind < tSizeVector)
    {
        toSet[ind] = setTo[ind];
    }
}
*/

__global__ static void cuda_setVectorsEqualFAST(float* toSet, float* setTo, int cols, int tSize)
{
    
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    //if ((row < fakeRows) && (col < fakeCols))
   // {
    int ind = row * cols + col;
    if(ind < tSize)toSet[ind] = setTo[ind];
    //}
    

    //not working i dont think //not massive speed up anyways
    /*
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int index = (y + j) * width + x;
        if (index < tSize)
        {
            toSet[index] = setTo[index];
        }
    }
    */
}


__global__ static void cuda_addVectors(float* input, float* toAdd, int tSize)
{
    
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    if (ind < tSize)
    {
        input[ind] += toAdd[ind];
    }
    

    //borked, not fast
    /*
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        int index = (y + j) * width + x;
        if (index < tSize)
        {
            input[index] += toAdd[index];
        }
    }
    */
}

/*
__global__ static void cuda_setMatricesEqual(float* toSet, float* setTo, int rows, int cols)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if ((row < rows) && (col < cols))
    {
        int ind = row * cols + col;
        toSet[ind] = setTo[ind];
    }
}
*/

static void linearForwardPass(float* input, float* weights, float* bias, float* output, int inputRows, int inputCols, int outputCols)
{
    //cublas_matrixMult(globalCublasHandle, input, weights, output, inputRows, inputCols, outputCols);
    dim3 dim_grid; //(ceilf(inputRows / (float)blockSize), ceilf(outputCols / (float)blockSize), 1);
    dim3 dim_block;// (blockSize, blockSize, 1);
    setGridBlockDims(dim_grid, dim_block, inputRows, outputCols);
    int width = inputCols;
    //assert(inputCols == outputCols);
    cuda_matMult <<<dim_grid, dim_block >>> (input, weights, output, inputRows, width, outputCols, NONE, NONE); //input_error = np.dot(output_error, self.weights.T)

    int tSize = inputRows * outputCols;
    //int inputBlocks = (tSize + blockSize - 1) / blockSize;
    cuda_addVectors<<<dim_grid, dim_block >>>(output, bias, tSize);
    //cudaDeviceSynchronize();

    /*
    cudaDeviceSynchronize();
    std::cout << "\nIN: ";
    for (int x = 0; x < outputCols; x++)std::cout << input[x] << ", ";
    std::cout << "\nW: \n";
    for (int x = 0; x < inputCols; x++)
    {
        std::cout << "\n";
        for (int y = 0; y < outputCols; y++)
        {
            std::cout << weights[x * outputCols + y] << ", ";
        }
    }
    std::cout << "\nOUT: ";
    for (int x = 0; x < outputCols; x++)std::cout << output[x] << ", ";
    std::cout << "\n";
    */
}

__global__ static void cuda_linearUpdateWeights(float* weights, float* weightsError, int inputCols, int outputCols, float learningRate) // float* bias, float* inputError
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if ((row < inputCols) && (col < outputCols))
    {
        int index = row * outputCols + col;
        //gradient clipping...I think?
        //float grad = fmaxf(-0.5f, weightsError[index]);
        //grad = fminf(0.5f, grad);

        weights[index] -= learningRate * weightsError[index];
    }
}

__global__ static void cuda_linearUpdateBias(float* bias, float* backproppedError, int inputRows, int outputCols, float learningRate)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if ((row < inputRows) && (col < outputCols))
    {
        int index = row * outputCols + col;
        bias[index] -= learningRate * backproppedError[index];
    }
}

//weights_error = np.dot(self.input.T, output_error)
/*
__global__ static void cuda_calcWeightsError(float* priorInput, float* backproppedError, float* output__weightsError, int inputRows, int inputCols, int outputCols)
{
    //backproppedError (inputRows, outputCols)
    //priorInput (inputRows, inputCols)
    //WeightErros (inputCols, outputCols)

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (ROW < inputRows && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}
*/

//__global__ static void cuda_generateWeightsErrors(float* weightsError, float* priorInput, float* )


// //http://cs231n.stanford.edu/handouts/linear-backprop.pdf
// //https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
//here computing dL/dX = (dL/dY) * W^T
static void linearBackwardAndUpdatePass(float* input__priorInput, float* input__backproppedError, float* input__weights, float* input__bias, float* output__inputError, float* output__weightsError, int inputRows, int inputCols, int outputCols, float learningRate)
{
    /*
    cudaDeviceSynchronize();
    std::cout << "BP\n";
    for (int x = 0; x < 10; x++)std::cout << input__backproppedError[x] << ", ";
    std::cout << "\n";
    //cublas_matrixMult(globalCublasHandle, input__backproppedError, input__weights, output__inputError, inputRows, inputCols, outputCols, CUBLAS_OP_N, CUBLAS_OP_T); //input_error = np.dot(output_error, self.weights.T)
    */

    dim3 dim_grid; // (ceilf(inputRows / (float)blockSize), ceilf(inputCols / (float)blockSize), 1);
    dim3 dim_block; // (blockSize, blockSize, 1);
    setGridBlockDims(dim_grid, dim_block, inputRows, inputCols);

    int width = outputCols;
    //assert(inputCols == outputCols);
    cuda_matMult<<<dim_grid, dim_block >>>(input__backproppedError, input__weights, output__inputError, inputRows, width, inputCols, NONE, TRANSPOSE); //input_error = np.dot(output_error, self.weights.T)

    //dim3 dim_grid2; // (ceilf(inputCols / (float)blockSize), ceilf(outputCols / (float)blockSize), 1);
    setGridBlockDims(dim_grid, dim_block, inputCols, outputCols);
    int width2 = inputRows;
    cuda_matMult <<<dim_grid, dim_block >>> (input__priorInput, input__backproppedError, output__weightsError, inputCols, width2, outputCols, TRANSPOSE, NONE); //weights_error = np.dot(self.input.T, output_error)

    //I've pulled a sneaky one here, manually transposed the thing via swapping inputRows, inputCols. Think that should work fine? tested with multiple rows and it does work
    //cublas_matrixMult(globalCublasHandle, input__priorInput, input__backproppedError, output__weightsError, inputCols, inputRows, outputCols, CUBLAS_OP_N, CUBLAS_OP_T); //weights_error = np.dot(self.input.T, output_error)
    //cudaDeviceSynchronize();
    // update parameters
    //self.weights -= learning_rate * weights_error
    //self.bias -= learning_rate * output_error
    //cudaDeviceSynchronize();


    int WnBlockRows = (inputCols + TILE_DIM - 1) / TILE_DIM;
    int WnBlockCols = (outputCols + TILE_DIM - 1) / TILE_DIM;
    dim3 WnBlocks(WnBlockRows, WnBlockCols);
    dim3 WnThreads(TILE_DIM, TILE_DIM);
    cuda_linearUpdateWeights <<<WnBlocks, WnThreads >>> (input__weights, output__weightsError, inputCols, outputCols, learningRate);

    int BnBlockRows = (inputRows + TILE_DIM - 1) / TILE_DIM;
    int BnBlockCols = (outputCols + TILE_DIM - 1) / TILE_DIM;
    dim3 BnBlocks(BnBlockRows, BnBlockCols);
    dim3 BnThreads(TILE_DIM, TILE_DIM);
    cuda_linearUpdateBias <<<BnBlocks, BnThreads >>> (input__bias, input__backproppedError, inputRows, outputCols, learningRate);

    //then once you are done, output__inputError will be the input__backproppedError for any other layers you are backpropping
    //cudaDeviceSynchronize();
    
    /*
    cudaDeviceSynchronize();
    if (outputCols <= 10)
    {
        std::cout << "\nTESTDEBUG\n";
        std::cout << "PRIOR IN: ";
        for (int x = 0; x < inputCols; x++)std::cout << input__priorInput[x] << ",";
        std::cout << "\n BACK ERROR IN: ";
        for (int x = 0; x < outputCols; x++)std::cout << input__backproppedError[x] << ",";
        std::cout << "\n BACK ERROR OUT: ";
        for (int x = 0; x < inputCols; x++)std::cout << output__inputError[x] << ",";
        std::cout << "\n WEIGHTS ERROR: ";
        for (int x = 0; x < inputCols; x++)
        {
            std::cout << "\n";
            for (int y = 0; y < outputCols; y++)
            {
                std::cout << output__weightsError[x * outputCols + y] << ",";
            }
        }
        
        std::cout << "\n WEIGHTS: ";
        for (int x = 0; x < inputCols; x++)
        {
            std::cout << "\n";
            for (int y = 0; y < outputCols; y++)
            {
                std::cout << input__weights[x * outputCols + y] << ",";
            }
        }
        
        std::cout << "\nTESTDEBUG\n";
    }
    */
    

}

//dLdW = X^T * dLdY -> dLdW = 
/*
__global__ static void cuda_linearUpdate(float* input, float* weights, float* bias, float* input__backproppedError, int inputRows, int inputCols, int outputCols, float learningRate)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int indexInput;
    int indexWeights;
    int indexOutput;

    if ((row < inputRows) && (col < outputCols))
    {
        indexOutput = row * outputCols + col;
        atomicAdd(&bias[col], -learningRate * input__backproppedError[indexOutput]);

        for (int i = 0; i < outputCols; i++)
        {
            indexInput = row * inputCols + i;
            indexWeights = i * outputCols + col;

            atomicAdd(&weights[indexWeights], -learningRate * input[indexInput] * input__backproppedError[indexOutput]);
        }
    }
}
*/

/*
static void linearUpdate(float* input, float* input__backproppedError, float* weights, float* bias, int inputRows, int inputCols, int outputCols, float learningRate)
{
    int nBlockRows = (inputRows + blockSize - 1) / blockSize;
    int nBlockCols = (outputCols + blockSize - 1) / blockSize;
    dim3 nBlocks(nBlockRows, nBlockCols);
    dim3 nThreads(blockSize, blockSize);

    cuda_linearUpdate <<<nBlocks, nThreads >>> (input, weights, bias, input__backproppedError, inputRows, inputCols, outputCols, learningRate);
}
*/

__global__ static void reluForwardPass(float* input, float* output, int tSizeOutput)
{
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
    float alpha = LEAKY_RELU_ALPHA; //0->1
    if (ind < tSizeOutput)
    {
        output[ind] = fmaxf(alpha*input[ind], input[ind]); //technically leaky relu, because normal relu seems garbo - you often get infinite/very high error values, then gradients become unresponsive
    }
}

__global__ static void reluBackwardPass(float* input, float* output__deltaLoss_deltaInput, int tSizeOutput)
{
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
    float alpha = 0.01f;
    if (ind < tSizeOutput)
    {
        float grad = (input[ind] > 0) ? 1.0f : alpha;
        output__deltaLoss_deltaInput[ind] = grad *output__deltaLoss_deltaInput[ind]; //take the value thats there, multiply it by gradient, thats the new value
    }
}

__global__ static void sigmoidForwardPass(float* input, float* output, int tSizeOutput)
{
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    if (ind < tSizeOutput)
    {
        output[ind] = fdividef(1.0f, (1.0f + expf(-input[ind]))); //fmaxf(0, input[ind]);
    }
}

__global__ static void sigmoidBackwardPass(float* input, float* output__deltaLoss_deltaInput, int tSizeOutput)
{
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    if (ind < tSizeOutput)
    {
        float sigIn = fdividef(1.0f, (1.0f + expf(-input[ind])));
        output__deltaLoss_deltaInput[ind] = (sigIn * (1.0f - sigIn)) * output__deltaLoss_deltaInput[ind];
    }
}

__global__ static void tanhForwardPass(float* input, float* output, int tSizeOutput)
{
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    if (ind < tSizeOutput)
    {
        output[ind] = tanh(input[ind]); //fmaxf(0, input[ind]);
    }
}

__global__ static void tanhBackwardPass(float* input, float* output__deltaLoss_deltaInput, int tSizeOutput)
{
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    if (ind < tSizeOutput)
    {
        //ok this looks sus to me.
        //Firstly say top layer, output__deltaLoss_deltaInput will be a 1x10 vec for mnist,
        //input will be a 1x700+ vec, so using the same index there is meaningless.
        float th = tanh(input[ind]);
        output__deltaLoss_deltaInput[ind] = (1.0f - th * th) * output__deltaLoss_deltaInput[ind];
    }
}

//where max is single element vector
__global__ static void calcSoftmaxMax(float* input, float* max, int tSizeInput)
{
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    if (ind == 0)max[0] = input[ind];

    if (ind < tSizeInput)
    {
        if (input[ind] > max[0])max[0] = input[ind];
    }
}


//where sum is single element vector
__global__ static void calcSoftmaxSum(float* input, float* sum, float* max, int tSizeInput)
{
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    if (ind == 0)sum[0] = 0.0f;

    if (ind < tSizeInput)
    {
        atomicAdd(&sum[0], expf(input[ind] - max[0]));
    }
}

//https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
__global__ static void softmaxForwardPass(float* input, float* output, float* sum, float* max, int tSizeOutput)
{
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    if (ind < tSizeOutput)
    {
        //output[ind] = fdividef(expf(input[ind] - max[0]), sum[0]);
        float out = fdividef(expf(input[ind] - max[0]), sum[0]);
        output[ind] = fminf(0.999999f, fmaxf(out, 0.000000001f));// +0.000001f; //add very very small constant to the cost function, prevent it from being zero
    }
}

/*
__global__ static void softmaxBackwardPass(float* input, float* output__deltaLoss_deltaInput, float* sum, float* max, int tSizeOutput)
{
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    if (ind < tSizeOutput)
    {
        float smaxIn = fdividef(expf(input[ind] - max[0]), sum[0]);
        float smaxDij = smaxIn * (1.0f - smaxIn);
        //must also consider i != j

        output__deltaLoss_deltaInput[ind] = (sigIn * (1.0f - sigIn)) * output__deltaLoss_deltaInput[ind];
    }
}
*/

__global__ static void _softmaxCalcS_ST(float* S_ST, float* lastSoftmaxOutput, int tSizeOutput)
{
    int ind = blockDim.x * blockIdx.x + threadIdx.x; //ind of S

    if (ind < tSizeOutput)
    {
        for (int x = 0; x < tSizeOutput; x++) //ind of ST
        {
            S_ST[ind * tSizeOutput + x] = lastSoftmaxOutput[ind] * lastSoftmaxOutput[x];
        }
    }
}

__global__ static void _softmaxCalcDiag_S(float* diag_S, float* S_ST, float* lastSoftmaxOutput, int tSizeOutput)
{
    int ind = blockDim.x * blockIdx.x + threadIdx.x; //ind of S

    if (ind < tSizeOutput)
    {
        for (int x = 0; x < tSizeOutput; x++) //ind of ST
        {
            //WARNING! terniary operator seems to produce nan's on release?
            //diag_S[ind * tSizeOutput + x] = (x==ind) ? lastSoftmaxOutput[ind] : 0.0f; //along the diag set it to S (last sofmax output)
            int thisIndex = ind * tSizeOutput + x;
            diag_S[thisIndex] = float(x == ind) * lastSoftmaxOutput[ind] + 0.0f;
            diag_S[thisIndex] -= S_ST[thisIndex]; //do the minusing in here, may as well
        }
    }
}


//def softmax_grad(softmax):
//  s = softmax.reshape(-1, 1) //reshape as column vector
//  return np.diagflat(s) - np.dot(s, s.T) //diagflat -> creates a 2d matrix with the vector as the diagonal, n*n matrix

static void softmaxBackwardPass(float* lastInput, float* lastOutput, float* output__deltaLoss_deltaInput, float* sum, float* max, float* S_ST, float* diag_S, int tSizeOutput)
{
    /*
    cudaDeviceSynchronize();
    std::cout << "\n";
    for (int x = 0; x < tSizeOutput; x++)
    {
        std::cout << output__deltaLoss_deltaInput[x] << ",";
    }
    std::cout << "\n";
    */

    //firstly we want the result of S * S^T which will be a matrix of shape tSizeOutput * tSizeOutput, should work fine via cublas?
    //secondly we need a matrix of zeros other than the diagonal which is S
    //thirdly we want the resulting matrix of diagS - S.S^T
    //once we have that gradient matrix, we can matmult output__deltaLoss_deltaInput by the gradient matrix, should then output the required deltas
    int nBlocks = (tSizeOutput + TILE_DIM - 1) / TILE_DIM;
    _softmaxCalcS_ST <<<nBlocks, TILE_DIM >>> (S_ST, lastOutput, tSizeOutput);
    _softmaxCalcDiag_S <<<nBlocks, TILE_DIM >>> (diag_S, S_ST, lastOutput, tSizeOutput);

    //now diag_S is the gradient matrix. We can now do mat mult using it to compute new output__deltaLoss_deltaInput
    //cublas_matrixMult(globalCublasHandle, output__deltaLoss_deltaInput, diag_S, output__deltaLoss_deltaInput, 1, tSizeOutput, tSizeOutput);
    dim3 dim_grid; // (ceilf(inputRows / (float)blockSize), ceilf(inputCols / (float)blockSize), 1);
    dim3 dim_block; // (blockSize, blockSize, 1);
    setGridBlockDims(dim_grid, dim_block, 1, tSizeOutput);

    //assert(inputCols == outputCols);
    cuda_matMult <<<dim_grid, dim_block >>> (output__deltaLoss_deltaInput, diag_S, output__deltaLoss_deltaInput, 1, tSizeOutput, tSizeOutput, NONE, NONE); //here treating the entire calculation as a single row vector mult on output, this may fail for more than 1 row output??

    /*
    cudaDeviceSynchronize();
    std::cout << "\nOUT: ";
    for (int x = 0; x < tSizeOutput; x++)
    {
        std::cout << output__deltaLoss_deltaInput[x] << ",";
    }
    std::cout << "\n";

    std::cout << "\nDIAG_S:\n";
    for (int x = 0; x < tSizeOutput; x++)
    {
        std::cout << "\n";
        for (int y = 0; y < tSizeOutput; y++)
        {
            std::cout << diag_S[x * tSizeOutput + y] << ", ";
        }
    }
    std::cout << "\n";
    */
}


__global__ static void meanSquaredErrorForwardPass(float* output, float* target, float* totalError, int tSizeOutput)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < tSizeOutput)
    {
        //if (index == 0)target[tSizeOutput] = 0.0f;
        atomicAdd(&totalError[0], fdividef(powf(output[index] - target[index], 2.0f), tSizeOutput));
    }
}

__global__ static void meanSquaredErrorBackwardPass(float* output, float* target, int tSizeOutput)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < tSizeOutput)
    {
        //output[index] = fdividef(2.0f * (output[index] - target[index]), tSizeOutput);
        output[index] = fdividef(2.0f * (output[index] - target[index]), tSizeOutput);
    }
}


__global__ static void crossEntropyErrorForwardPass(float* output, float* target, float* totalError, int tSizeOutput)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < tSizeOutput)
    {
        //if (index == 0)target[tSizeOutput] = 0.0f;
        atomicAdd(&totalError[0], - fdividef(target[index] * logf(output[index] + 0.001f), tSizeOutput)); //add a very very small constant to that logf to prevent nans from cascading
    }
}

__global__ static void crossEntropyErrorBackwardPass(float* output, float* target, int tSizeOutput)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < tSizeOutput)
    {
        //output[index] = fdividef(2.0f * (output[index] - target[index]), tSizeOutput);
        output[index] = (output[index] - target[index]); //valid when used with softmax
    }
}
