## Installation

Before installing, please take notice of the following:

**The code uses tensorflow and CUDA that can use the GPU if its NVIDIA. If you're running a GPU that is non-NVIDA, you can still run the program, but it will be using your CPU instead.
Please comment out the following lines if you are unable to use your GPU: lines 8 - 14**

1. Check your versions to make sure that tensorflow, CUDA and cudnn will work together.
    To figure out what versions you need, you can reference this website: [NVIDIA gpus](https://developer.nvidia.com/cuda-gpus)

    Once you figure out your CC (Compute Capability) for your GPU, you can reference this chart in this post: [Stackoverflow question](https://stackoverflow.com/questions/28932864/which-compute-capability-is-supported-by-which-cuda-versions)
    
    - After you know which one you need to download, visit this site to see what versions are going to be compatible: [Tensorflow website](https://www.tensorflow.org/install/source#gpu)
        
    - Tensorflow version downloads: [Pypi tensorflow versions](https://pypi.org/project/tensorflow/#history)

    - CUDA version downloads: [CUDA downloads](https://developer.nvidia.com/cuda-toolkit-archive)

    - cuDNN version downloads: [cuDNN downloads](https://developer.nvidia.com/rdp/cudnn-archive)


    For my case, I am running a 1050 TI, so I am using tensorflow 2.5.2, with CUDA version 11.2 and cudnn version 8.1


2. Install the required libraries. If you have the .txt file downloaded, you may use "pip install -r requirements.txt".
    NOTE: You will most likely end up needing to change the version of tensorflow in your project to the one you downloaded.

## Usage

Now that you have the required tools to run the program (or plan on running it off the CPU), here is how you can set up the program.

Required parameters to change before using:
- Line 22: cat_image_directory. Change this to where your dataset is located
- Line 23: SavedFolder. Change this to where you want your output pictures to be saved.

  After changing these required parameters, you're all set up to run the program:
    - If you have it in pycharm, you can simply just run the code
    - If you want to run it using command prompt, navigate to the file directory with the python file and type "python main.py"

Optional parameters to change based off preference:
- Line 18: num_epochs. The number of epochs you will train the model for
- Line 21: image_size. This changes the resoulation of the generated files
- Line 94: 25. This number is how often you want the training model to save the files generated.
- Line 101: examples, rows, cols. These arugments change how the output files are viewed. Examples changes how many photos are generated per epoch displayed. Rows is how many rows are displayed in the plot. Cols is the same as previously mentioned. 
- Line 129: cat_generator. The name of the saved model.


## Results

-Filler text

## Credit

Thanks to https://github.com/TrojanPinata for providing the data set of 64x64 cat faces
