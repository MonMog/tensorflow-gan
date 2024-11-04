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

There are two major componets in a GAN, the discriminator and the generator. The generators job is to produce images that can "fool" the discrimator and get pass it. The discrimators job is to block any images that don't match the dataset, as in, judge the work of the generator. The discrimator and the generator are two different neural networks that are constantly trying to one-up each other. The generator's goal is to keep improving its output until it gets accepted by the discrimator. The discrimator's goal is to only allow images that belong with the rest of the dataset. 

### Building the generator

Since we just mentioned that a GAN is two neural networks that are competing with each other, we first need to make the neural network for the generator. The generator takes in random noise, called Z, and tries to mimic an image from the real data set. That is a basic overview of what the generator does so let's use tensorflow to build the generator:
 - In order to build a simple neural network system or model, we would need to use `tf.keras.Sequential`. What this does is takes layers as an input, in order, and then returns a model using the layers created. Next, we need to determine what type of layer we will be using. If you head over to the tensorflow documentation and look under the layers tab, you will see that we have MANY options to choose from. The one that is best suited for our current situation would be a Dense layer. To normally use the Dense layer, we would only need one input, the amount of neurons we would want that layer to have. But since this is our first layer in the model, we need to specify what these layers are being connected to, which will be the random noise. Which will give us `tf.keras.layers.Dense(neuron_gen, input_dim=noise_dim),` where neuron_gen is the amount of neurons you would want the layer to have and noise_dim is the random noise.
 - We now have our generator with our first layer connected to the random noise. For our next layer in our generator, we are going to use a LeakyReLU. In the documentation, its description is "Leaky version of a Rectified Linear Unit activation layer.". This is going to be needed because when our neural network is training and it receives a negative input, it basically doesn't know how to handle it and will output a zero, which means it will stop learning. That is not a favorable outcome and we want to prevent the neurons from freaking out. From my understanding, the LeakyReLU layer will multiply the negative inputs by the factor to allow it to not output a zero, allowing them to still output a non-zero, making them still able to learn.
 - The next layer we will be adding is another Dense layer. We need to do this because the generator needs more than just one layer to turn random noise into fake images from the data set. The first dense layer takes in the random noise and attempts to mold something useful out of it. The second dense layer is going to pick up what the first dense layer attempted and try to refine it and fix its flaws. Knowing this, it doesn't mean that the more layers in a neural network, the better it will perform. If you add too many layers in your neural network, you may run into problems like overfitting, where the NN basically learns exactly what the dataset is and produces no variation in its output. Sadly, there is no perfect formula to determine how many layers are needed in a model, you need to consider your factors and test them and observe the results. Now let's try to understand the parameters in this layer. The total neurons in this layer will be equal to `image_size * image_size * 3`. Why? Because `image_size * image_size` will give us the total amount of pixels in the picture we are trying to generate. So why do we need the `* 3`? This is because we want to have the 3 color channels, RGB. This way, each pixel in the generated image would fall between the range of [R, G, B], or [0-255, 0-255, 0-255]. But then the second parameter is the activation function, and in this case we will be using tanh. The tanh function is going to normalize us RGB values to fall within -1 to 1, something something so that we can later convert it back to a color image when we are doing image processing and also something to do with making our neural network train faster (I think).
 - The final layer of the generator will be the reshape layer. It will take the previous layer, the dense layer with the image * image * 3, and it will reshape it into ` tf.keras.layers.Reshape((image_size, image_size, 3)`. What this does is transforms the previous 1D layer into a 3D layer. We want 3 dimensions because we are trying to have it represented into an image format, the height, the width and the number of channels for the color (RGB). This layer is needed because it will be fed into the discriminator's initial layer and if it's not the right shape, the discriminator will not do its job properly since it doesn't know what it's looking at.


That concludes the generator. At the end of the function, you may add this line `print(generator.summary())` (if you named the Sequential generator) and it will output the information about each layer and how many neurons it has and if its trainable. 

### Building the discriminator

This is the other major half of a GAN, the discriminator. The discriminator's job is to decide if the given input is a real or fake image. The process of building the discriminator is very similiar to building the generator, we will be using the same layers here. 
 - Same thing for the start of the generator, we will be needing to use `tf.keras.Sequential` to build the model for the discriminator. The first layer of the discriminator is going to be a flatten layer. Currently, our input data is a 3D vector (an image) and we need to flatten it into a 1D. We do this because Dense layers can only take 1D vectors as inputs.
 - The second layer of the discriminator is going to be the Dense layer. This first Dense layer of the discrimiantor is going to try to learn group of features or patterns within the image to determine if its real or fake. If its going to study the image to tell if its real or fake, shouldn't the amount of neurons in the dense layer be equal to `image_size * image_size * 3` or at least `image_size * image_size`? While I thought this is what we would want to do for the layer, we actually want to have less than `image_size * image_size` neurons in this dense layer. This is because, as previosuly mentioned, this layer is trying to learn patterns within the image. If we give it a smaller amount of neurons to start with, its going to be forced to generalize the image and attempt to pick up on features and look at group of pixels as a whole. If we do this, it should make the discriminator able to learn patterns (Hopefully of course). At the time of writing this, It has come to me that `tf.keras.layers.Dense(neuron_dis)` isn't exactly the best amount of neurons to have in the layer since it actually should be dependent on the image_size. I believe that a good range would be 4% - 16% of `image_size * image_size * 3`. So if you're using image_size = 64, you would want around the range of 4% - 16% of 12,288 to be your first dense layer for the discriminator. After the dense layer, we are going to add a LeakyReLU layer like we previosuly did in the generator. After this layer, some might choose to add `tf.keras.layers.Dropout(0.4)` but when I tested it, the results were as favorable.
 - The final layer is going to be `tf.keras.layers.Dense(1, activation='sigmoid')`. Since the discriminator is trying to output if the image is real or fake, we need a way for it to inform us. We can do this by giving it a single neuron and then giving it an activation function of sigmoid. This way the range is from 0-1, with 0 being fake and 1 being real.

### Building the GAN together

Now that we have both the generator and discriminator models built, its time to compile them together and set some parameters for them.
 - We need to compile the discriminator with its own settings before we can build the GAN. Since the build_gan() function takes the two already built models, we can use discriminator.compile() to compile it with the parameters. We are going to be concered with two of them, the loss function and the learning rate. They are a few loss functions to choose from (Wasserstein, Least Squares and Hinge), but we are going to be looking at the binary_crossentropy. In short, from what I can understand about this loss function, is that it calculates the difference between the true image and the predicted probability for that image. I am not sure if I am confusing myself or just having a hard time understanding it. I believe it works something like this: If the model is looking at a real image, lets say it gives it a score of 0.3 (it can give scores from 0, being fake, to 1, being real) meaning it thinks its a fake image. The loss function will be high to show that this was an error. The same thing happens in the other case, where if its looking at a fake image, but it gives it a score of 0.9, the loss function will be high to let the model know that "hey pal, did you just blow in from stupid town?", hopefully to let it learn from its mistakes. Now that we have our loss function, we need to determine the learning rate. The optimizer that we are going to be using is Adam because I don't understand it and there are probably other options but this one is commonly used so it has to be good, right? After ignoring that part, we need to figure out the learning rate. This part is very sensitive and just a small change will have noticable results. The learning rate is how fast the model is able to pick up things and updates in its work. When I first saw this, I thought to myself, shouldn't I just make this a very high number so it learns much quicker? The answer is mostly no, we want a low learning rate so the model is able to understand the fine details and take the time for it to understand what its looking at. If we make the rate higher, its possible its going to make drastic jumps in its work, overlooking important details. Another reason we need to be delicate with choosing the learning rate is because if we have one model, suppose the generator, learn and excel at a faster rate compared to the discriminator, the generator is going to know how to fool the discriminator every single time while the discriminator isn't going to be able to keep up, resulting in work that will not be favorable.
 - The next part is something I am still trying to understand myself, its for setting the discriminator's trainable parameter.




## Credit

Thanks to https://github.com/TrojanPinata for providing the data set of 64x64 cat faces
