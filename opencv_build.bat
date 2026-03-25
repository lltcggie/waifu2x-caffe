@echo off

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64

cd /d "%~dp0"

rem wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.12.0.zip
rem opencv_contrib.zip を解凍します。
rem cmake -D OPENCV_EXTRA_MODULES_PATH=%~dp0\opencv_contrib-4.12.0\modules

cd opencv
mkdir build
cd build

set CFLAGS=/Zc:preprocessor
set CXXFLAGS=/Zc:preprocessor

cmake .. -G Ninja ^
-DCMAKE_INSTALL_PREFIX=..\..\lib ^
-DBUILD_WITH_STATIC_CRT=OFF ^
-DBUILD_IPP_IW=OFF ^
-DBUILD_ITT=OFF ^
-DBUILD_JAVA=OFF ^
-DBUILD_SHARED_LIBS=OFF ^
-DBUILD_TESTS=OFF ^
-DBUILD_PERF_TESTS=OFF ^
-DBUILD_opencv_calib3d=OFF ^
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
-DBUILD_opencv_apps=OFF ^
-DBUILD_opencv_aruco=OFF ^
-DBUILD_opencv_bgsegm=OFF ^
-DBUILD_opencv_bioinspired=OFF ^
-DBUILD_opencv_ccalib=OFF ^
-DBUILD_opencv_cudaarithm=OFF ^
-DBUILD_opencv_cudabgsegm=OFF ^
-DBUILD_opencv_cudacodec=OFF ^
-DBUILD_opencv_cudafeatures2d=OFF ^
-DBUILD_opencv_cudafilters=OFF ^
-DBUILD_opencv_cudaimgproc=OFF ^
-DBUILD_opencv_cudalegacy=OFF ^
-DBUILD_opencv_cudaobjdetect=OFF ^
-DBUILD_opencv_cudaoptflow=OFF ^
-DBUILD_opencv_cudastereo=OFF ^
-DBUILD_opencv_cudawarping=OFF ^
-DBUILD_opencv_cudev=OFF ^
-DBUILD_opencv_datasets=OFF ^
-DBUILD_opencv_face=OFF ^
-DBUILD_opencv_freetype=OFF ^
-DBUILD_opencv_fuzzy=OFF ^
-DBUILD_opencv_hfs=OFF ^
-DBUILD_opencv_img_hash=OFF ^
-DBUILD_opencv_line_descriptor=OFF ^
-DBUILD_opencv_mcc=OFF ^
-DBUILD_opencv_objc_bindings_generator=OFF ^
-DBUILD_opencv_optflow=OFF ^
-DBUILD_opencv_phase_unwrapping=OFF ^
-DBUILD_opencv_plot=OFF ^
-DBUILD_opencv_reg=OFF ^
-DBUILD_opencv_rgbd=OFF ^
-DBUILD_opencv_saliency=OFF ^
-DBUILD_opencv_shape=OFF ^
-DBUILD_opencv_stereo=OFF ^
-DBUILD_opencv_structured_light=OFF ^
-DBUILD_opencv_surface_matching=OFF ^
-DBUILD_opencv_text=OFF ^
-DBUILD_opencv_tracking=OFF ^
-DBUILD_opencv_xfeatures2d=OFF ^
-DBUILD_opencv_ximgproc=OFF ^
-DBUILD_opencv_xobjdetect=OFF ^
-DBUILD_opencv_xphoto=OFF ^
-DBUILD_opencv_python3=OFF ^
-DBUILD_opencv_python_tests=OFF ^
-DBUILD_opencv_quality=OFF ^
-DBUILD_opencv_rapid=OFF ^
-DBUILD_opencv_signal=OFF ^
-DBUILD_opencv_stitching=OFF ^
-DBUILD_opencv_wechat_qrcode=OFF ^
-DBUILD_opencv_js_bindings_generator=OFF ^
-DWITH_1394=OFF ^
-DWITH_CUDA=ON ^
-DWITH_CUDNN=ON ^
-DWITH_CUFFT=ON ^
-DWITH_DIRECTX=OFF ^
-DWITH_DSHOW=OFF ^
-DWITH_EIGEN=OFF ^
-DWITH_FFMPEG=OFF ^
-DWITH_GSTREAMER=OFF ^
-DWITH_OPENCL=OFF ^
-DWITH_OPENCAMDBALSL=OFF ^
-DWITH_OPENCLAMDFFT=OFF ^
-DWITH_OPENCL_SVM=OFF ^
-DWITH_ADE=OFF ^
-DWITH_ARITH_DEC=OFF ^
-DWITH_ARITH_ENC=OFF ^
-DWITH_IPP=OFF ^
-DWITH_ITT=OFF ^
-DWITH_VFW=OFF ^
-DWITH_VTK=OFF ^
-DWITH_TESSERACT=OFF ^
-DWITH_WIN32UI=OFF ^
-DBUILD_opencv_dnn=ON ^
-DBUILD_opencv_cudev=ON ^
-DWITH_PROTOBUF=ON ^
-DOPENCV_DNN_CUDA=ON ^
-DCUDA_ARCH_BIN=8.9 ^
-DCUDA_ARCH_PTX=12.0 ^
-DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ^
-DCMAKE_C_FLAGS="/DWIN32 /D_WINDOWS /W3 /Zc:preprocessor" ^
-DCMAKE_CXX_FLAGS="/DWIN32 /D_WINDOWS /W3 /GR /EHsc /Zc:preprocessor" ^
-DOPENCV_EXTRA_MODULES_PATH=%~dp0\opencv_contrib-4.12.0\modules ^
-DCMAKE_BUILD_TYPE=Debug


cmake --build . --config Debug --target install
rem cmake --build . --config Release --target install
