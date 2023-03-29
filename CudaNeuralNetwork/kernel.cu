#include "Tests.h"


//DESIGN PRINCIPLES:
//1 file for generic functions/cuda functions
//proper file structure for the neural network
//the network layers PASS IN/OUTPUT AROUND do not store unless absolutely neccasary! avoid confusion of what is going where!

//NEURAL NETWORK:
//INPUT: rows = 1 SAMPLE per row, cols = features for each sample!
//WEIGHTS: rows = number of input cols, cols = desired output cols (i.e neuron count)

int main()
{
    cublasCreate(&globalCublasHandle);

    TestCase tc;
    tc.runAllTests();

    cublasDestroy(globalCublasHandle);
    return 0;
}