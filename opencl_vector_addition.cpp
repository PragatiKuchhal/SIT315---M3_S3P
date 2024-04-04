#include <iostream>
#include <vector>
#include <CL/cl.h>

using namespace std;

void addVectors_OpenCL(vector<int>& a, vector<int>& b, vector<int>& c, int n) {
    // Get available platforms
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    // Get available devices
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create a context and command queue for the device
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Create buffers for the input and output vectors
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * n, a.data(), NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * n, b.data(), NULL);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * n, NULL, NULL);

    // Create a program from the kernel source code
    const char* kernelSourceCode =
        "__kernel void vector_add_ocl(__global const int* a, __global const int* b, __global int* c, int n) {\n"
        "   int i = get_global_id(0);\n"
        "   if (i < n) {\n"
        "       c[i] = a[i] + b[i];\n"
        "   }\n"
        "}\n";
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourceCode, NULL, NULL);

    // Build the program
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create a kernel object from the program
    cl_kernel kernel = clCreateKernel(program, "vector_add_ocl", NULL);

    // Set the kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &n);

    // Execute the kernel on the device
    size_t globalSize = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    // Read the result back into the output vector
    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(int) * n, c.data(), 0, NULL, NULL);

    // Release OpenCL resources
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main() {
    int n = 5; // Vector size

    // Create input vectors
    vector<int> a = {1, 2, 3, 4, 5};
    vector<int> b = {6, 7, 8, 9, 10};
    vector<int> c(n);

    // Perform vector addition using OpenCL
    addVectors_OpenCL(a, b, c, n);

    // Print the result
    cout << "Result of vector addition:" << endl;
    for (int i = 0; i < n; ++i) {
        cout << c[i] << " ";
    }
    cout << endl;

    return 0;
}
