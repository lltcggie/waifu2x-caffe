@echo off

cd "%~dp0"

set CUDA_VISIBLE_DEVICES=-1
start waifu2x-caffe-gui.exe
