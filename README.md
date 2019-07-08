# Object Detection using YOLO with [Darkflow](https://github.com/thtrieu/darkflow)

### Introduction
This notebook is not about training on your own data. Instead, it is a guide about how to use YOLOv2 with your own data. It shows the process of using an image as input then outputing the pre-defined labels for the identified object on the image. After completing this notebook, I will go over on how the training process can be done in a separate notebook.

### Contents
0. __Installing Darkflow with GPU support on Windows 10__
1. __References for YOLO implementation__
2. __Importing Dependences__
3. __Build the model__
4. __Gain the results of detected objects__
5. __Boxing around the objects__
6. __Boxing with Video and output the video__

## 0. Installing Darkflow with GPU support on Windows 10

**IMPORTANT:** Please read all the Notes and pay attention to the **IMPORTANT** / **WARNING** tags to help your future self save some time of debugging!

### Let's get started!

**__WARNING:__** I've tested this only on NVIDIA GPUs. This installation guide **won't work** if you have any other different than that. Also, look at [this NVIDIA post](https://developer.nvidia.com/cuda-gpus) to check if your graphics card is CUDA supported. Otherwise, it **won't work** also. To check your graphics card model, from the Device Manager look for the Display Adapter setting, open it and read the name of your adapter. 

First of all, you'll need to clone the [original Darkflow repository](https://github.com/thtrieu/darkflow) and install Anaconda on your system. After that, download the desired weights from https://pjreddie.com/darknet/yolo/ or the [Google Drive directory](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU) from the original Darkflow repo that correspond to the `.cfg` that will be used and save them in a new `/bin` folder inside darkflow-master.

Run the following commands from Anaconda Prompt with administrator privileges on the root folder darkflow-master:
- `conda update conda`
- `conda install tensorflow-gpu` (if for some reason it doesn't work for you, try using pip instead. I recommend using conda as the main packet manager because I had a hard time installing tensorflow-gpu using pip and it didn't work out.)
- `conda install cython numpy`
- `conda config --add channels conda-forge`
- `conda install opencv`
- `python setup.py build_ext --inplace`

If everything went OK, open the python editor with the `python` command and run: `import tensorflow-gpu as tf`. Next, we'll deal with the errors that showed up.


### Installing CUDA drivers:

You'll get an error message about CUDA drivers like cudart64_XX.dll fails, where ‘XX’ is a version number. For example, if it shows cudart_90.dll, this means that you need version 9.0 of the CUDA drivers.
  
Download the version indicated on the warning on the [official NVIDIA CUDA download site](https://developer.nvidia.com/cuda-toolkit-archive) and install it by following the on-screen commands.
  
After that, try importing the `tensorflow-gpu` package again. Two things (1. **OR** 2.) may happen:
  1. You'll get the CUDA error message again with a different version. Repeat the CUDA installation step again with the version informed.
  2. You'll get an error message about the wrong or missing cuDNN drivers, something like cudnn64_X.dll is missing, where X is a version number. Take note of that for the next step.


### Installing cuDNN:

Download the cuDNN from here: https://developer.nvidia.com/cudnn. You’ll see a variety of cuDNN downloads. Here’s where you’ll have to match to the version of CUDA that you downloaded in the previous section. So, for example, I used CUDA 10.0, so made sure I used a cuDNN that matches both this and the required version you saw in the last step, so I chose cuDNN v7.6.1 for CUDA 10.0.

This will download a ZIP file with several folders, each containing the cuDNN files (one DLL, one header and one library). Find your CUDA installation (should be at something like: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0`).

You’ll see the directories from the ZIP file are also in this directory — i.e. there’s a bin, an include, a lib etc. Copy the files from the zip to the relevant directory. So, for example, drag `cudnn64_7.dll` that is in the `\bin` directory of the zip to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin`, and do the same for the others.
 
 
### Back to Python!

After all that installation, try importing the `tensorflow-gpu` package one more time. If you have the right versions of the libraries, you should see no errors at all. Otherwise, go back to the previous steps and make sure you’ve downloaded and installed the correct versions of the drivers that cause these errors.
  
*NOTE:* If you encounter any problem like Microsoft Visual C++ Build Tools is required, then just download from Microsoft website along with the SDK. For more information about this problem you may refer to [this issue](https://github.com/thtrieu/darkflow/issues/788).

**__IMPORTANT:__** When I installed the CUDA drivers, my NVIDIA driver was overwritten. To solve this, just open the GeForce Experience or any other software that allows you to update the driver again and do so. You'll need it installed on your PC to make the GPU TensorFlow work.
  
To test if the TensorFlow installed is really the GPU version, open the python editor again and run:
```
>> import tensorflow as tf
>> tf.test.gpu_device_name()
```
If you get ' ' then no GPU device was found. Try reinstalling the CUDA drivers and cuDNN too, also the default NVIDIA driver for your graphics card.


### Testing, at last!

Finally, after all that it's time for some testing! Run the following command from the root directory of darkflow-master:

`python flow --model cfg/yolov2.cfg --load bin/yolov2.weights --demo camera --gpu 0.7`

You can alter the `.cfg` and the `.weights` files at your will. Note that this can cause an error like `AssertionError: expect 44948596 bytes, found 44948600`. A workarround I found was to subtract both values and change on line 121 from the file `darkflow-master\darkflow\utils\loader.py` by adding to the `self.offset` value the result of the subtraction. Check [this issue](https://github.com/thtrieu/darkflow/issues/223) for more info.
  - On this example, on the line 121 there is `self.offset = 16`, so 44948600 - 44948596 = 4. My new line 121 becomes: `self.offset = 16 + 4`

Another error might appear like `AssertionError: labels.txt and cfg/yolov2.cfg indicate inconsistent class numbers`. My workarround was to change de `labels.txt` on the root directory and overwrite it with all the content from `\cfg\coco.names`. Don't forget to check if the number on `classes=` field in the `.cfg` file you're using matches the number of lines on the `coco.names` and `labels.txt`, like stated on [this other issue](https://github.com/thtrieu/darkflow/issues/859).

Wait and you'll see the camera output video in a new window with the boxing and labels of the objects identified. (On my laptop with GTX 1060 graphic card, the rendering speed is around 12.5 FPS).

*NOTE:* For more information about the command above and the output information on the terminal, look at the [documentation on the original Darkflow repository](https://github.com/thtrieu/darkflow/blob/master/README.md). I'm using the `--gpu 0.7` argument because using it at 1.0 overflows the available memory on my graphics card.

That's it! If everything went OK, check the Jupyter Notebook on this repository for other applications!


### Disclaimer and references:

This guide was created with the intention of being a quick step-by-step and not getting too deep into the concepts behind it. Please visit the following sites that were used as references:
- [https://expschoolwork.blogspot.com/2018/11/using-yolo-in-anaconda.html](https://expschoolwork.blogspot.com/2018/11/using-yolo-in-anaconda.html) (much similar to this guide but with no instructions on how to install CUDA/cuDNN);
- [https://medium.com/@lmoroney_40129/installing-tensorflow-with-gpu-on-windows-10-3309fec55a00](https://medium.com/@lmoroney_40129/installing-tensorflow-with-gpu-on-windows-10-3309fec55a00) (has a lot of information and details regarding all the installation process on Windows 10 but doesn't use Anaconda).
