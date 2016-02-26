rm cuda_kernel_image_processing.d cuda_kernel_image_processing.o cuda_kernel_image_processing

# NVCC Compiler
/usr/local/cuda-7.5/bin/nvcc -I/usr/include/opencv -G -g -O0 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -m64 -M -o "cuda_kernel_image_processing.d" "cuda_kernel_image_processing.cu"
/usr/local/cuda-7.5/bin/nvcc -I/usr/include/opencv -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21 -m64  -x cu -o  "cuda_kernel_image_processing.o" "cuda_kernel_image_processing.cu"

# NVCC Linker
/usr/local/cuda-7.5/bin/nvcc --cudart static -L/usr/lib/x86_64-linux-gnu --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21 -m64 -link -o  "cuda_kernel_image_processing"  ./cuda_kernel_image_processing.o   -lopencv_core -lopencv_highgui -lopencv_imgproc

./cuda_kernel_image_processing landscape.jpg 5 16 16 false