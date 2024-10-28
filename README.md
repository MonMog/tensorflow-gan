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

| Argument | Description                                                                                       | Required | Default |
|----------|---------------------------------------------------------------------------------------------------|----------|---------|
| `--f`    | Directory path where generated images and models will be saved. Always Required.                  | Yes      | None    |
| `--d`    | Directory path to training data. Required if not using a pre-trained model.                       | Depends  | None    |
| `--p`    | Directory path to saved model. This will skip the training and just produce an output.            | Depends  | None    |
| `--a`    | Number of images to generate from the pre-trained model.                                          | No       | 20      |
| `--e`    | Number of epochs to train                                                                         | No       | 5001    |
| `--b`    | Batch size for training                                                                           | No       | 256     |
| `--n`    | Max range for the random noise generated.                                                         | No       | 100     |
| `--i`    | Size of the generated images (for example, 64x64).                                                | No       | 64      |
| `--s`    | How often (in epochs) the model will save and output images.                                      | No       | 5       |
| `--ng`   | How many neurons to start in the first layer for the generator.                                   | No       | 256     |
| `--nd`   | How many neurons to start in the second layer for the discriminator.                              | No       | 256     |



### Example 1: Training a model with custom values
For example, here is what I would type for it to run:

`python main.py --d "D:\cats" --f "D:\Programming\Projects\Output" --e 1000 --b 128 --s 10 --n 50 --i 32`
   - This will generate and train a model with the custom values for epochs, batch size, save interval, noise dim and image size.

### Example 2: Training a model with default values   
Due to having default values, this command also works:

`python main.py --d "D:\cats" --f "D:\Programming\Projects\Output"`
   - This will generate and train a model with the default values.

### Example 3: Loading a pretrained model  
If you want to load a previously saved model, I would do:

`python main.py --p "D:\Programming\Projects\LatestCatsOutput\gan_generator.tf" --f "D:\Programming\Projects\Output" --a 100`
   - This will load the pre-trained model and generate 100 photos using the model.

### NOTE:
If you're going to load a previously saved model, you need to make sure to specify the same parameters used for making the generator for loading it. That means if you trained a generate with a custom noise_dim, you will need load the pretrained model with the same custom noise_dim. If you do not do this, it will try to load the pretrained model with the default values assigned and you will receive an error:

`python main.py --p "D:\Programming\Projects\LatestCatsOutput\gan_generator.tf" --f "D:\Programming\Projects\LatestCatsOutput" --n 50 --a 100`



## Results

Using the dataset of 64x64 cat faces and setting batch size to 256 and the image dimension to 64, I was able to produce these results after 350 epochs:

![Amazing](/Results/gan_generated_image_epoch_350.png)

These results are not perfect, but they are definitely heading towards the right direction. Due to my computer's limiations, it would take a while for me to produce more than 500 epochs, even using the GPU. But the longer the program is running, the better the results _should_ turn out.

## How does it work?

I am going to _attempt_ an explaination on how GAN works

There are two major componets in a GAN, the discrimator and the generator. The generators job is to produce images that can "fool" the discrimator and get pass it. The discrimators job is to block any images that don't match the dataset, as in, judge the work of the generator. The discrimator and the generator are two different neural networks that are constantly trying to one-up each other. The generator's goal is to keep improving its output until it gets accepted by the discrimator. The discrimator's goal is to only allow images that belong with the rest of the dataset. 

### Building the generator

Since we just mentioned that a GAN is two neural networks that are competing with each other, we first need to make the neural network for the generator. The generator takes in random noise, called Z, and tries to mimic an image from the real data set. That is a basic overview of what the generator does so lets use tensorflow to build the generator:
     - In order to build a simple neural network or model, we would need to use `tf.keras.Sequential`. What this does is takes layers as an input, in order, and then returns a model using the layers created. 
     - Next, we need to determine what type of layer we will be using.


## Credit

Thanks to https://github.com/TrojanPinata for providing the data set of 64x64 cat faces
