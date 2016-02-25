rm openmp_kernel_image_processing

g++ openmp_kernel_image_processing.cpp -o openmp_kernel_image_processing -L /usr/lib/x86_64-linux-gnu/libopencv_core.so /usr/lib/x86_64-linux-gnu/libopencv_highgui.so /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so  -lopencv_core -lopencv_highgui -lopencv_imgproc -fopenmp

./openmp_kernel_image_processing lena.jpg 3 4 32 true