waifu2x-caffe (for Windows)
----------

 Author: lltcggie

This software uses only the conversion function of the image conversion software "[waifu2x](https://github.com/nagadomi/waifu2x)"
This software was rewritten using [Caffe](http://caffe.berkeleyvision.org/) and built for Windows.
It can be converted using CPU, but it can be converted faster than CPU by using CUDA (or cuDNN).

GUI supports English, Japanese, Simplified Chinese, Traditional Chinese, Korean, Turkish, Spanish, Russian, and French.

You can download the software from [the releases page](https://github.com/lltcggie/waifu2x-caffe/releases).


 Requirements
----------

At least the following environment is required to run this software.

 * OS: Windows Vista or later 64bit (There is no exe for 32bit)
 * Memory: 1GB or more of free memory (however, this depends on the image size to be converted)
 * Microsoft Visual C++ 2015 Redistributable Package Update 3 (x64 version) must be installed (Required)
    - The above package is [here.](https://www.microsoft.com/ja-jp/download/details.aspx?id=53587)
    - After pressing the `Download` button, select `vcredist_x64.exe` to download and install.
    - If you can't find it, try searching with "Visual C++ 2015 Redistributable Package Update 3".

When converting with cuDNN

 * GPU: NVIDIA GPU with Compute Capability 3.0 or higher

If you want to know the Compute Capability of your GPU, check it out on [this page](https://developer.nvidia.com/cuda-gpus).


 How to use (GUI version)
--------

"Waifu2x-caffe.exe" is GUI software. Start by double-clicking.
Or, drag and drop the file or folder to "waifu2x-caffe.exe" with Explorer and the conversion will be performed with the settings at the last startup.
In that case, depending on the settings, if the conversion is successful, the dialog will be closed automatically.
You can also use the GUI to set options on the command line.
For details, see the section on command line options (common) and command line options (GUI version).

After starting, drag and drop an image or folder into the "Input path" field to automatically set the "Output path" field.
If you want to change the output destination, change the "Output path" column.

You can change the conversion settings to your liking.


## Input/output setting
    Settings related to file input/output.

### Input path
    Specify the path of the file you want to convert.
    If you specify a folder, the files with the "extension to be converted in the folder"
    including subfolders will be converted.
    
    You can specify multiple files and folders by dragging.
    In that case, the files are output in the new folder while maintaining the folder structure.
    (In the input path column, "(Multi Files)" is displayed. The output folder name is generated
    from the file and folder name that the mouse holds.)
    When you click the browse button to select a file, you can select a single file, a folder, or multiple files.

### Output path
    Specify the path to save the converted image.
    When a folder is specified in "Input path", the converted file is saved in the specified folder (without changing the folder structure). If the specified folder does not exist, it will be created automatically.

### Extension to be converted in the folder
    If the "Input path" is a folder, specify the extension of the image to be converted in the folder.
    The default value is `png:jpg:jpeg:tif:tiff:bmp:tga`.
    The delimiter is `:`.
    Case does not matter.
    Example.png:jpg:jpeg:tif:tiff:bmp:tga

### Output extension
    Specify the format of the converted image.
    The values ​​that can be set for "Output image quality setting" and "Output depth bit rate" differ depending on the format specified here.

### Output quality setting
    Specify the quality of the converted image.
    The value that can be set is an integer.
    The range and meaning of the values ​​that can be specified depend on the format set in "Output extension".
     * .jpg: Value range (0 to 100) The higher the number, the higher the image quality
     * .webp: Range of values ​​(1 to 100) The higher the number, the higher the image quality
     * .tga: Value range (0 to 1) 0 means no compression, 1 means RLE compression

### Output depth bits
    Specify the number of bits per channel of the converted image.
    The value that can be specified depends on the format set in "Output extension".

## Conversion image quality/processing settings
    Settings related to the file conversion processing method and image quality.

### Conversion mode
    Specify the conversion mode.
     * Noise reduction and magnification: Noise reduction and magnification are performed.
     * Enlarge: Enlarge
     * Noise removal: Performs noise removal
     * Noise removal (automatic detection) and enlargement: Enlarge. Noise removal is also performed only when the input is a JPEG image

### JPEG noise removal level
    Specify the noise reduction level. Higher levels remove noise more powerfully, but may result in a flatter picture.

### Enlarged size
    Set the size after enlargement.
     * Specified by enlargement ratio: Enlarges the image at the specified enlargement ratio
     * Specified width after conversion: Enlarges to the specified width while maintaining the aspect ratio of the image (pixels)
     * Specified height after conversion: Enlarges to the specified height while maintaining the aspect ratio of the image (pixels)
     * Specify the width and width after conversion: Enlarges to the specified height and width. Specify as "1920x1080" (Unit is pixel)
    For magnifications greater than 2x (only one time if removing noise is done the first time) Enlarges by 2x until the specified magnification is exceeded, and if the magnification is not a power of 2, shrinks last The process of doing is performed. Therefore, the conversion result may be a flat picture.

### Model
    Specify the model to use.
    The best model depends on the image to be converted, so we recommend that you try various models.
     * 2D illustration (RGB model): 2D illustration model that converts all RGB of the image
     * Photo/Anime (Photo model): Model for photo/animation
     * 2D illustration (UpRGB model): A model that converts faster than 2D illustration (RGB model) with the same or higher image quality. However, the amount of memory (VRAM) that is consumed is larger than that of the RGB model, so adjust the division size if forced termination during conversion.
     * Photo/animation (Up Photo model): A model that converts at higher speed than the photo/animation (Photo model) with the same or higher image quality. However, the amount of memory (VRAM) that is consumed is larger than that of the Photo model, so adjust the division size if it is forcibly terminated during conversion.
     * 2D illustration (Y model): A model for 2D illustration that converts only the brightness of the image
     * 2D illustration (UpResNet10 model): A model that converts with higher image quality than 2D illustration (UpRGB model). Note that this model will change the output result if the division size is different.
     * 2D illustration (CUnet model): The model that can convert the 2D illustration with the highest image quality with the included model. Note that this model will change the output result if the division size is different.

### Use TTA mode
    Specify whether to use TTA (Test-Time Augmentation) mode.
    Using TTA mode, conversion is 8 times slower, but PSNR (one of the image evaluation indexes) is about 0.15.

## Processing speed setting
    It is a group of setting items that affect the processing speed of image conversion.

### division size
    Specify the width (pixel unit) when dividing and processing internally.
    How to decide the optimum number (the process ends at the fastest) is explained in the section "Split size".
    Separated by "-------", the upper one is the divisor of the vertical and horizontal size of the input image,
    The lower one is a general division size read from "crop_size_list.txt".
    If the partition size is too large, the amount of memory required (VRAM amount when using GPU) exceeds the memory available on the PC, and the software will be killed, so be careful.
    Since it will affect the processing speed to some extent, when converting a large number of images with the same image size by specifying a folder, it is recommended to check the optimal division size before conversion.
    However, be aware that depending on the model, the output result may change when the division size is changed.
    (In that case, use the default split size and adjust the batch size to speed up processing.)

### Batch size
    Specify the size when processing all at once internally.
    Increasing the batch size may increase the processing speed.
    Make sure that the amount of memory required, as well as the partition size, does not exceed the memory available on your PC.

## Operation setting
    Settings that summarize the operation settings which are unlikely to be changed.

### Automatic conversion start setting at file input
    Set whether to start the conversion automatically when the input file is specified by the reference button or drag and drop.
    If the input file is given to exe as an argument, the setting contents of this item have no effect.
     * Do not start automatically: Do not start conversion automatically when inputting a file
     * Start after inputting one file: Start conversion automatically after inputting one file
     * Start after inputting a folder or multiple files: Start conversion automatically when inputting a folder or multiple files. Single image file Convert files only when adjusting conversion settings

### Processor used
    Specifies the processor that does the conversion.
     * CUDA (cuDNN if available): CUDA(GPU) is used for conversion (cuDNN is used when cuDNN is available)
     * CPU: Only CPU is used for conversion

### Do not overwrite output file
    If this setting is ON, conversion will not be performed if a file with the same name exists in the image writing destination.

### Startup settings with arguments
    Set the operation when an input file is given to exe as an argument.
     * Convert at startup: Start conversion automatically at startup
     * Exit on success: auto exit if not failed at the end of conversion

### Used GPU No.
    You can specify the device number to use when there are multiple GPUs. Ignored when in CPU mode or when an invalid device number is specified.

### Fixed folder for input reference
    The folder that is first displayed when you click the input reference button is fixed to the folder set here.

### Fixed folder when referencing output
    The output destination folder of the converted image is fixed to the folder set here.
    Also, the folder that is first displayed when you click the output reference button is fixed to the folder set here.

## Other
    It is a group of other setting items.

### UI language
    Set the UI language.
    When starting up for the first time, the same language as the PC language setting is selected. (English if not present)

### cuDNN check
    You can check if you can use cuDNN by clicking the "cuDNN check" button.
    If cuDNN is not available, the reason will be displayed.

Click "Run" button to start conversion.
If you want to cancel while it is converting, click the "Cancel" button.
However, there is a time lag before it actually stops.
The progress bar shows the progress when changing multiple images.
The log shows the estimated remaining time, which is an estimate when processing multiple files with the same height and width.
Therefore, it is not useful when the size of the file is different, and when the number of images to be processed is 2 or less, only "Unknown" is displayed.


 How to use (CUI version)
--------

"Waifu2x-caffe-cui.exe" is a command line tool.
Start a `command prompt`, type the command as follows, and press enter.


The following command prints usage information to the screen.
```
waifu2x-caffe-cui.exe --help
```

The following command is an example of commands that perform image conversion.
```
waifu2x-caffe-cui.exe -i mywaifu.png -m noise_scale --scale_ratio 1.6 --noise_level 2
```
After executing the above, the conversion result is saved in `mywaifu(noise_scale)(Level2)(x1.600000).png`.

For the command list and details of each command, refer to the section on command line options (common) and command line options (CUI version).


 Command line options (common)
--------

With this software, the following options can be specified.
In the GUI version, if the command line option other than the input file is specified and started, the option file is not currently saved.
For the options not specified in the GUI version, the options at the time of the previous termination will be used.

### -l <string>, --input_extention_list <string>
    When input_file is a folder, specify the extension of the image to be converted in the folder.
    The default value is `png:jpg:jpeg:tif:tiff:bmp:tga`.
    The delimiter is `:`.
    Example.png:jpg:jpeg:tif:tiff:bmp:tga

### -e <string>, --output_extention <string>
    Specifies the extension of the output image when input_file is a folder.
    The default value is `png`.

### -m <noise|scale|noise_scale>, --mode <noise|scale|noise_scale>
    Specify the conversion mode. If not specified, `noise_scale` is selected.
    * noise: Performs noise reduction (to be exact, performs image conversion using a noise reduction model)
    * scale: Enlarge (to be exact, after enlarging with the existing algorithm, perform image conversion using the model for enlarged image complement)
    * noise_scale: Performs noise reduction and enlargement (after noise reduction, enlargement processing continues)
    * auto_scale: Scales. Noise removal is also performed only when the input is a JPEG image

### -s <number with decimal point>, --scale_ratio <number with decimal point>
    Specify how many times to enlarge the image. The default value is `2.0`, but you can specify a value other than 2.0.
    If scale_width or scale_height is specified, that one has priority.
    If you specify a number other than 2.0, the following processing is performed.
    * First, repeat 2x enlargement to cover the specified magnification as necessary and sufficient.
    * If a value other than a power of 2 is specified, the enlarged image will be reduced to the specified magnification.

### -w <integer>, --scale_width <integer>
    Enlarges to the specified width while maintaining the aspect ratio of the image (in pixels).
    If specified at the same time as scale_height, the image will be enlarged to have the specified width and height.

### -h <integer>, --scale_height <integer>
    Enlarges to the specified height while maintaining the aspect ratio of the image (pixels).
    If specified at the same time as scale_width, the image will be enlarged to have the specified width and height.

### -n <0|1|2|3>, --noise_level <0|1|2|3>
    Specify the noise reduction level. As for the model for noise removal, only levels 0-3 are prepared, so
    Please specify 0, 1 or 2 or 3.
    The default value is `0`.

### -p <cpu|gpu|cudnn>, --process <cpu|gpu|cudnn>
    Specifies the processor used for processing. The default value is `gpu`.
    * cpu: Perform conversion using CPU.
    * gpu: Convert using CUDA (GPU). For Windows version only, if cuDNN is available, use cuDNN.
    * cudnn: Convert using cuDNN.

### -c <integer>, --crop_size <integer>
    Specify the division size. The default value is `128`.

### -q <integer>, --output_quality <integer>
    Set the image quality of the converted image. The default value is `-1`
    The values ​​that can be specified and their meanings depend on the format set in "Output extension".
    If -1, the default value for each image format will be used.

### -d <integer>, --output_depth <integer>
    Specify the number of bits per channel of the converted image. The default value is `8`.
    The value that can be specified depends on the format set in "Output extension".

### -b <integer>, --batch_size <integer>
    Specify the mini-batch size. The default value is `1`.
    The mini-batch size is the number of blocks that the image is divided into by "division size" and processed at the same time. For example, if you specify `2`, it will be converted every 2 blocks.
    When the mini-batch size is increased, the GPU usage rate increases as well as when the split size is increased, but if you feel it is measured, it is more effective to increase the split size.
    (For example, if the split size is `64` and the mini-batch size is `4`, the split size is `128` and the mini-batch size is `1`.

### --gpu <int>
    Specify the GPU device number used for processing. The default value is `0`.
    Note that GPU device numbers start at 0.
    Ignored if no GPU is used for processing.
    If a GPU device number that does not exist is specified, it will be executed on the default GPU.

### -t <0|1>, --tta <0|1>
    If you specify `1`, TTA mode is used. The default value is `0`.

### --, --ignore_rest
    Ignores all options after this option is specified.
    For script batch files.


 Command line option (GUI version)
--------

In the GUI version, arguments that do not apply to the option specification are recognized as an input file.
Input files can be specified as files, folders, multiple files and folders at the same time.

### -o <string>, --output_folder <string>
    Set the path to the folder where you want to save the converted images.
    Save the converted file in the specified folder.
    The conversion file naming convention is the same as the output file name that is automatically determined when the input file is set in the GUI.
    If not specified, it will be saved in the same folder as the first input file.

### --auto_start <0|1>
    If you specify `1`, the conversion will start automatically at startup.

### --auto_exit <0|1>
    If you specify `1`, if conversion is successful at startup, it will end automatically if conversion is successful.

### --no_overwrite <0|1>
    If you specify `1`, if there is a file with the same name in the image write destination, conversion will not be performed.

### -y <upconv_7_anime_style_art_rgb|upconv_7_photo|anime_style_art_rgb|photo|anime_style_art_y|upresnet10|cunet>, --model_type <upconv_7_anime_style_art_rgb|upconv_7_photo|anime_style_art_net_net_up_art_rg_net_style_art_rg_net_art_rg_
    Specify the model to use.
    The setting item “Model” on the GUI and the following we respond.
    * upconv_7_anime_style_art_rgb: 2D illustration (UpRGB model)
    * upconv_7_photo :Photo/Anime (UpPhoto model)
    * anime_style_art_rgb: 2D illustration (RGB model)
    * photo: Photo/animation (Photo model)
    * anime_style_art_y: 2D illustration (Y model)
    * upresnet10: 2D illustration (UpResNet10 model)
    * cunet: 2D illustration (CUnet model)


 Command line option (CUI version)
--------

### --version
    Print version information and exit.

### -?, --help
    Display usage information and exit.
    Please use when you want to check how to use it easily.

### -i <string>, --input_file <string>
    (Required) Path to the image to convert
    When a folder is specified, all image files under that folder will be converted and output to the folder specified by output_file.

### -o <string>, --output_file <string>
    The path to the file where you want to save the converted image
    (when input_file is a folder) Path to the folder to save the converted image
    (When input_file is an image file) Be sure to enter the extension (such as .png at the end).
    If not specified, the file name is automatically determined and saved in that file.
    The file name determination rule is
    `[original image file name] ``(model name) ``(mode name) ``(noise reduction level (in noise reduction mode)) ``(magnification ratio (in magnification mode))'' (output (Number of bits (other than 8 bits)) ``.output extension`
    It looks like.
    The location to save is basically the same directory as the input image.

### --model_dir <string>
    Specify the path to the directory where the model is stored. The default value is `models/cunet`.
    The following models are included as standard.
     * `models/anime_style_art_rgb`: 2D illustration (RGB model)
    * `models/anime_style_art`: 2D illustration (Y model)
    * `models/photo`: Photo/animation (Photo model)
    * `models/upconv_7_anime_style_art_rgb`: 2D illustration (UpRGB model)
    * `models/upconv_7_photo` :Photo/Anime (UpPhoto model)
    * `models/upresnet10`: 2D illustration (UpResNet10 model)
    * `models/cunet`: 2D illustration (CUnet model)
    * `models/ukbench`: Old-fashioned photographic model (only the enlarged model is included, noise removal is not possible)
    Basically, you don't have to specify it. Please specify it when using a model other than the default model or your own model.

### --crop_w <int>
    Specify the division size (width). If not set, the value of crop_size will be used.
    If you specify the divisor of the width of the input image, conversion may be faster.

### --crop_h <integer>
    Specify the division size (vertical width). If not set, the value of crop_size will be used.
    If you specify a divisor of the height of the input image, conversion may be faster.


 Division size
--------

waifu2x-caffe (also waifu2x) converts images.
The image is divided into pieces of a certain size, converted one by one, and finally combined into a single image.
The division size (crop_size) is the width (in pixels) when dividing this image.

If the GPU is not exhausted even if it is converted with CUDA (the usage of the GPU is not close to 100%),
If you increase this number, the process may end sooner. (Because the GPU can be used up)
Please adjust it while seeing GPU Load (GPU usage rate) and Memory Used (VRAM usage rate) with [GPU-Z](http://www.techpowerup.com/gpuz/).
Also, refer to the following characteristics.

 * Larger numbers do not necessarily mean faster
 * If the division size is a divisor (or a number with a small remainder when divided) of the vertical and horizontal size of the image, the amount of wasteful calculation decreases and the speed increases. (In some cases, it seems that the numerical value that does not apply to this condition is the fastest.)
 * If you double the number, theoretically the amount of memory used will be 4 times (actually it is 3 to 4 times), so be careful not to drop the software. Especially, CUDA consumes much more memory than cuDNN, so be careful.


 About images with alpha channel
--------

This software also supports enlargement of images with alpha channel.
However, please note that it takes about twice as long as the image without alpha channel is enlarged because the process is to enlarge the alpha channel by itself.
However, if the alpha channel is composed of a single color, it can be expanded in about the same time as it was without.


 The format of language files
--------

Language files format is JSON.
If you create new language file, add language setting to'lang/LangList.txt'.
'lang/LangList.txt' format is TSV(Tab-Separated Values).

  * LangName: Language name
  * LangID: Primary language [See MSDN](https://msdn.microsoft.com/en-us/library/windows/desktop/dd318693.aspx)
  * SubLangID :Sublanguage [See MSDN](https://msdn.microsoft.com/en-us/library/windows/desktop/dd318693.aspx)
  * FileName: Language file name

ex.

  * Japanese LangID: 0x11(LANG_JAPANESE), SubLangID: 0x01(SUBLANG_JAPANESE_JAPAN)
  * English(US) LangID: 0x09(LANG_ENGLISH), SubLangID: 0x01(SUBLANG_ENGLISH_US)
  * English(UK) LangID: 0x09(LANG_ENGLISH), SubLangID: 0x02(SUBLANG_ENGLISH_UK)


Note
------------

This software is not guaranteed safe.
Please use it at the discretion of the user.
The creator does not assume any obligation.


Acknowledgment
------
The original [waifu2x](https://github.com/nagadomi/waifu2x) and model were produced and published under the MIT license [ultraist](https://twitter.com/ultraistter) Mr.
Created [waifu2x-converter](https://github.com/WL-Amigo/waifu2x-converter-cpp) based on the original waifu2x [Amigo](https://twitter.com/WL_Amigo) Mr. (I used to refer to how to write README and LICENSE.txt, how to use OpenCV)
[waifu2x-chainer](https://github.com/tsurumeso/waifu2x-chainer) was created to create the original model, and it was published under the MIT license [tsurumeso](https:// github.com/tsurumeso)
will be grateful to.
Also, @paul70078 for translating the message into English, @yoonhakcher for translating the message into Simplified Chinese, @mzhboy for the pull request for Simplified Chinese translation,
@Kenin0726 for translating the message into Korean, @aruhirin for suggesting improvements to the Korean translation,
@Lizardon1995 for translating messages in Traditional Chinese, @yoonhakcher, @Scharynche for pull request for Turkish translation, @Serized for pull request for French translation, Brazilian Portuguese @Simrafael for pull request for translation, @AndreasWebdev for pull request for German translation, @07pepa for pull request for Czech translation,
Thanks to JYUNYA for providing the GUI version of the icon.