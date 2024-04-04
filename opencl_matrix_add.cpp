#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>  // Include OpenCL header file
#include <chrono>   // Include chrono for time measurements

#define PRINT 1     // Macro for print control

int SZ = 100000000; // Default size of vectors

int *v1, *v2, *v_out; // Pointers for input and output vectors

cl_mem bufV1, bufV2, bufV_out; // OpenCL memory objects for vectors

cl_device_id device_id;  // OpenCL device id
cl_context context;      // OpenCL context
cl_program program;      // OpenCL program
cl_kernel kernel;        // OpenCL kernel
cl_command_queue queue;  // OpenCL command queue
cl_event event = NULL;   // OpenCL event object
int err;                 // OpenCL error variable

cl_device_id create_device(); // Function declaration for creating OpenCL device
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname); // Function declaration for setting up OpenCL context, device, queue, and kernel
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename); // Function declaration for building OpenCL program from source
void setup_kernel_memory(); // Function declaration for setting up OpenCL memory buffers
void copy_kernel_args();    // Function declaration for copying kernel arguments
void free_memory();         // Function declaration for freeing allocated memory
void init(int *&A, int size); // Function declaration for initializing vectors with random values
void print(int *A, int size); // Function declaration for printing vectors

int main(int argc, char **argv) {
    if (argc > 1) {
        SZ = atoi(argv[1]); // Set size of vectors from command line argument
    }

    init(v1, SZ); // Initialize vector v1
    init(v2, SZ); // Initialize vector v2
    init(v_out, SZ); // Initialize output vector v_out

    size_t global[1] = {(size_t)SZ}; // Global work size for OpenCL kernel

    print(v1, SZ); // Print vector v1
    print(v2, SZ); // Print vector v2
   
    // Setup OpenCL device, context, queue, and kernel
    setup_openCL_device_context_queue_kernel((char *)"./vector_ops_ocl.cl", (char *)"vector_add_ocl");
    setup_kernel_memory(); // Setup OpenCL memory buffers
    copy_kernel_args();    // Copy kernel arguments

    auto start = std::chrono::high_resolution_clock::now(); // Start time measurement

    // Enqueue OpenCL kernel for execution
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, &event);
    clWaitForEvents(1, &event); // Wait for kernel execution to finish
    
    // Read output vector v_out from OpenCL memory buffer
    clEnqueueReadBuffer(queue, bufV_out, CL_TRUE, 0, SZ * sizeof(int), &v_out[0], 0, NULL, NULL);
    print(v_out, SZ); // Print output vector v_out

    auto stop = std::chrono::high_resolution_clock::now(); // Stop time measurement
    std::chrono::duration<double, std::milli> elapsed_time = stop - start; // Calculate elapsed time

    printf("Kernel Execution Time: %f ms\n", elapsed_time.count()); // Print kernel execution time
    free_memory(); // Free allocated memory
}

// Function definition for initializing vectors with random values
void init(int *&A, int size) {
    A = (int *)malloc(sizeof(int) * size); // Allocate memory for vector A

    for (long i = 0; i < size; i++) {
        A[i] = rand() % 100; // Initialize each element of A with a random value
    }
}

// Function definition for printing vectors
void print(int *A, int size) {
    if (PRINT == 0) {
        return; // If print control is set to 0, return without printing
    }

    if (PRINT == 1 && size > 15) {
        for (long i = 0; i < 5; i++) {
            printf("%d ", A[i]); // Print first 5 elements of A
        }
        printf(" ..... "); // Print ellipsis
        for (long i = size - 5; i < size; i++) {
            printf("%d ", A[i]); // Print last 5 elements of A
        }
    } else {
        for (long i = 0; i < size; i++) {
            printf("%d ", A[i]); // Print all elements of A
        }
    }
    printf("\n----------------------------\n"); // Print separator
}

// Function definition for freeing allocated memory
void free_memory() {
    // Release OpenCL memory objects
    clReleaseMemObject(bufV1);
    clReleaseMemObject(bufV2);
    clReleaseMemObject(bufV_out);

    // Release OpenCL kernel, command queue, program, and context
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    free(v1);  // Free memory allocated for v1
    free(v2);  // Free memory allocated for v2
    free(v_out); // Free memory allocated for v_out
}

// Function definition for copying kernel arguments
void copy_kernel_args() {
    clSetKernelArg(kernel, 0, sizeof(int), (void *)&SZ); // Set kernel argument 0 (size)
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufV1); // Set kernel argument 1 (bufV1)
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufV2); // Set kernel argument 2 (bufV2)
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufV_out); // Set kernel argument 3 (bufV_out)

    if (err < 0) {
        perror("Couldn't create a kernel argument"); // Print error message if failed to create kernel argument
        printf("error = %d", err); // Print error code
        exit(1); // Exit program with error code 1
    }
}

// Function definition for setting up OpenCL memory buffers
void setup_kernel_memory() {
    // Create OpenCL memory buffers for v1, v2, and v_out
    bufV1 = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    bufV2 = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    bufV_out = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);

    // Write data from host to device memory buffers
    clEnqueueWriteBuffer(queue, bufV1, CL_TRUE, 0, SZ * sizeof(int), &v1[0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufV2, CL_TRUE, 0, SZ * sizeof(int), &v2[0], 0, NULL, NULL);
}

// Function definition for setting up OpenCL device, context, queue, and kernel
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname) {
    device_id = create_device(); // Create OpenCL device
    cl_int err;

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err); // Create OpenCL context
    if (err < 0) {
        perror("Couldn't create a context"); // Print error message if failed to create context
        exit(1); // Exit program with error code 1
    }

    program = build_program(context, device_id, filename); // Build OpenCL program from source

    // Create OpenCL command queue
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err < 0) {
        perror("Couldn't create a command queue"); // Print error message if failed to create command queue
        exit(1); // Exit program with error code 1
    }

    kernel = clCreateKernel(program, kernelname, &err); // Create OpenCL kernel
    if (err < 0) {
        perror("Couldn't create a kernel"); // Print error message if failed to create kernel
        printf("error =%d", err); // Print error code
        exit(1); // Exit program with error code 1
    }
}

// Function definition for building OpenCL program from source
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename) {
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;

    program_handle = fopen(filename, "r"); // Open OpenCL kernel file for reading
    if (program_handle == NULL) {
        perror("Couldn't find the program file"); // Print error message if failed to find program file
        exit(1); // Exit program with error code 1
    }
    fseek(program_handle, 0, SEEK_END); // Move file pointer to end
    program_size = ftell(program_handle); // Get size of file
    rewind(program_handle); // Rewind file pointer to beginning
    program_buffer = (char *)malloc(program_size + 1); // Allocate memory for program buffer
    program_buffer[program_size] = '\0'; // Null-terminate program buffer
    fread(program_buffer, sizeof(char), program_size, program_handle); // Read program file into program buffer
    fclose(program_handle); // Close program file

    // Create OpenCL program from source
    program = clCreateProgramWithSource(ctx, 1, (const char **)&program_buffer, &program_size, &err);
    if (err < 0) {
        perror("Couldn't create the program"); // Print error message if failed to create program
        exit(1); // Exit program with error code 1
    }
    free(program_buffer); // Free program buffer memory

    // Build OpenCL program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size); // Get program build log size
        program_log = (char *)malloc(log_size + 1); // Allocate memory for program build log
        program_log[log_size] = '\0'; // Null-terminate program build log
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL); // Get program build log
        printf("%s\n", program_log); // Print program build log
        free(program_log); // Free program build log memory
        exit(1); // Exit program with error code 1
    }

    return program; // Return OpenCL program
}

// Function definition for creating OpenCL device
cl_device_id create_device() {
   cl_platform_id platform;
   cl_device_id dev;
   int err;

   err = clGetPlatformIDs(1, &platform, NULL); // Get OpenCL platform ID
   if(err < 0) {
      perror("Couldn't identify a platform"); // Print error message if failed to identify platform
      exit(1); // Exit program with error code 1
   } 

   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL); // Get OpenCL GPU device ID
   if(err == CL_DEVICE_NOT_FOUND) {
      printf("GPU not found\n"); // Print message if GPU not found
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL); // Get OpenCL CPU device ID
   }
   if(err < 0) {
      perror("Couldn't access any devices"); // Print error message if failed to access any devices
      exit(1); // Exit program with error code 1
   }

   return dev; // Return OpenCL device ID
}
