import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os

# If its not detecting your GPU, you need to download CUDA or just comment all this line out and it will use CPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Change all of this as needed
one = 1
num_epochs = 20000 + one
batch_size = 256
noise_dim = 100
image_size = 128  # might want to change this depending on how beefy your PC Is
cat_image_directory = 'D:/Cat/cats/CAT_00'  # yah, change this 100% to where your training data is stored
SavedFolder = 'SaveItHere08'  # could change it to a directory but if you dont, just makes a folder in the same folder as the program

if not os.path.exists(SavedFolder):
    os.makedirs(SavedFolder)
    # ''' just makes the folder if its not already there'''


def generate_noise(batch_size, noise_dim):
    return np.random.normal(0, 1, size=(batch_size, noise_dim))
    # '''random noise is good for making the pictures'''

def build_generator():
    generator = tf.keras.Sequential([
        tf.keras.layers.Dense(256, input_dim=noise_dim),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(image_size * image_size * 3, activation='tanh'),
        tf.keras.layers.Reshape((image_size, image_size, 3))
    ])
    return generator
    # ''' defining the layers needed, already put in the image_size so just change the variable instead of messing here'''

def build_discriminator():
    discriminator = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(image_size, image_size, 3)),
        tf.keras.layers.Dense(256),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return discriminator
    # need a discriminator, the person detecting the fake images if its legit or not

def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002))
    discriminator.trainable = False

    gan = tf.keras.Sequential([generator, discriminator])
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002))
    return gan
    # need to build it together


def load_real_cats(dataset_path):
    cat_images = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            image = Image.open(os.path.join(dataset_path, filename))
            image = image.resize((image_size, image_size))
            image = image.convert('RGB')
            cat_images.append(np.array(image))
    cat_images = np.array(cat_images)
    cat_images = (cat_images.astype(np.float32) - 127.5) / 127.5
    return cat_images
    # some dumb data processing so it can load the images and train them
def train_gan(generator, discriminator, gan, X_train, num_epochs, batch_size, noise_dim):
    for epoch in range(num_epochs):
        for _ in range(X_train.shape[0] // batch_size):
            real_cats = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
            fake_cats = generator.predict(generate_noise(batch_size, noise_dim))

            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            discriminator_loss_real = discriminator.train_on_batch(real_cats, real_labels)
            discriminator_loss_fake = discriminator.train_on_batch(fake_cats, fake_labels)
            discriminator_loss = 0.5 * (discriminator_loss_real + discriminator_loss_fake)

            noise = generate_noise(batch_size, noise_dim)
            generator_loss = gan.train_on_batch(noise, real_labels)

        print(f"Epoch {epoch}/{num_epochs}, D Loss: {discriminator_loss}, G Loss: {generator_loss}")

        if epoch % 25 == 0:
            save_generated_images(generator, epoch)

    # really wish I could explain whats going on here, something something making the discriminator know whats a real imainge
    # and whats a fake imainge so when its training on the dataset, it can sort through it and allow the real photos to go
    # and ban the fake photos

def save_generated_images(generator, epoch, examples=20, rows=2, cols=10, figsize=(10, 4)):
    ## you can change the parameters in this function to how many pictures you want to loan and row and col for it
    noise = generate_noise(examples, noise_dim)
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(generated_images[i, :, :, :], cmap='jet')  # You can change the cmap as needed
        plt.axis('off')
    plt.tight_layout()

    image_filename = os.path.join(SavedFolder, f'gan_generated_image_epoch_{epoch}.png')
    plt.savefig(image_filename)

if __name__ == '__main__':
    X_train = load_real_cats(cat_image_directory)

    # Build the generator, discriminator, and GAN
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)

    # Train the GAN
    train_gan(generator, discriminator, gan, X_train, num_epochs, batch_size, noise_dim)

    # Save the generator model
    generator.save('cat_generator.h5')
    # No idear what a h5 file is, but probably something useful?
