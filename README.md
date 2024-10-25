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

Instead of hardcoding the values into the program, you can use command line arugments to run it. Below are the list of avaiable arugments:

### Arguments

| Argument               | Description                                                                            | Required | Default |
|------------------------|----------------------------------------------------------------------------------------|----------|---------|
| `--cat_image_directory`| Directory path to training data. Required UNLESS using a pre-trained model.            | Yes      | None    |
| `--saved_folder`       | Directory path where generated images and models will be saved. Always Required.       | Yes      | None    |
| `--num_epochs`         | Number of epochs for training.                                                         | No       | 501     |
| `--batch_size`         | Batch size for training.                                                               | No       | 256     |
| `--noise_dim`          | Max range for the random noise generated.                                              | No       | 100     |
| `--image_size`         | Size of the generated images (for example, 64x64).                                     | No       | 64      |
| `--save_interval`      | How often (in epochs) the model will save and output images.                           | No       | 5       |
| `--pretrained_model`   | Directory path to saved model. This will skip the training and just produce an output. | No       | None    |

For example, here is what I would type for it to run:

-----

## Results

Using the dataset of 64x64 cat faces and setting batch size to 256 and the image dimension to 64, I was able to produce these results after 350 epochs:

![Amazing](/Results/gan_generated_image_epoch_350.png)

These results are not perfect, but they are definitely heading towards the right direction. Due to my computer's limiations, it would take a while for me to produce more than 500 epochs, even using the GPU. But the longer the program is running, the better the results _should_ turn out.

## How does it work?

-Filler text

## Credit

Thanks to https://github.com/TrojanPinata for providing the data set of 64x64 cat faces
