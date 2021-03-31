@echo off

cd opencv
mkdir build
cd build

cmake .. -G "Visual Studio 15 Win64" ^
-DCMAKE_INSTALL_PREFIX=..\..\lib ^
-DBUILD_WITH_STATIC_CRT=OFF ^
-DBUILD_IPP_IW=OFF ^
-DBUILD_ITT=OFF ^
-DBUILD_JAVA=OFF ^
-DBUILD_SHARED_LIBS=OFF ^
-DBUILD_TESTS=OFF ^
-DBUILD_opencv_calib3d=OFF ^
-DBUILD_opencv_dnn=OFF ^
-DBUILD_opencv_features2d=OFF ^
-DBUILD_opencv_flann=OFF ^
-DBUILD_opencv_highgui=OFF ^
-DBUILD_opencv_ml=OFF ^
-DBUILD_opencv_objdetect=OFF ^
-DBUILD_opencv_photo=OFF ^
-DBUILD_opencv_video=OFF ^
-DBUILD_opencv_videoio=OFF ^
-DBUILD_opencv_videostab=OFF ^
-DBUILD_opencv_java_bindings_generator=OFF ^
-DBUILD_opencv_python_bindings_generator=OFF ^
-DWITH_1394=OFF ^
-DWITH_CUDA=OFF ^
-DWITH_CUFFT=OFF ^
-DWITH_DIRECTX=OFF ^
-DWITH_DSHOW=OFF ^
-DWITH_EIGEN=OFF ^
-DWITH_FFMPEG=OFF ^
-DWITH_GSTREAMER=OFF ^
-DWITH_OPENCL=OFF ^
-DWITH_OPENCAMDBALSL=OFF ^
-DWITH_OPENCLAMDFFT=OFF ^
-DWITH_OPENCL_SVM=OFF ^
-DWITH_PROTOBUF=OFF ^
-DWITH_VFW=OFF ^
-DWITH_VTK=OFF ^
-DWITH_WIN32UI=OFF

rem cmake --build . --config Debug --target install
cmake --build . --config Release --target install
