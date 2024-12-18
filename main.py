import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
import argparse

# If its not detecting your GPU, you need to download CUDA or just comment all this line out and it will use CPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# end of line if you chose to use your CPU instead

def generate_noise(batch_size, noise_dim):
    return np.random.normal(0, 1, size=(batch_size, noise_dim))


def build_generator(noise_dim, image_size, neuron_gen):
    generator = tf.keras.Sequential([
        tf.keras.layers.Dense(neuron_gen, input_dim=noise_dim),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(image_size * image_size * 3, activation='tanh'),
        tf.keras.layers.Reshape((image_size, image_size, 3))
    ])
    return generator


def build_discriminator(image_size, neuron_dis):
    discriminator = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(image_size, image_size, 3)),
        tf.keras.layers.Dense(neuron_dis),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return discriminator


def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002))
    discriminator.trainable = False

    gan = tf.keras.Sequential([generator, discriminator])
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002))
    return gan


def load_real_cats(dataset_path, image_size):
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


def train_gan(generator, discriminator, gan, X_train, num_epochs, batch_size, noise_dim, save_interval, saved_folder):
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

        if epoch % save_interval == 0:
            save_generated_images(generator,noise_dim, epoch, saved_folder)
            generator.save(os.path.join(saved_folder, 'gan_generator.tf'),
                           overwrite=True,
                           include_optimizer=True,
                           save_format='tf')

def save_generated_images(generator,noise_dim, epoch, saved_folder, examples=20, rows=2, cols=10, figsize=(10, 4)):
    noise = generate_noise(examples, noise_dim)
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(generated_images[i, :, :, :], cmap='jet')
        plt.axis('off')
    plt.tight_layout()

    image_filename = os.path.join(saved_folder, f'gan_generated_image_epoch_{epoch}.png')
    plt.savefig(image_filename)
    plt.close()  # ...


def save_individual_generated_images(generator, noise_dim, amount, saved_folder):
    for i in range(amount):
        noise = generate_noise(1, noise_dim)
        generated_image = generator.predict(noise)
        generated_image = 0.5 * generated_image + 0.5

        image_filename = os.path.join(saved_folder, f'generated_image_{i + 1}.png')
        plt.imsave(image_filename, generated_image[0])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or generate images using GAN')

    parser.add_argument('--f', type=str, help='Directory path where generated images and models will be saved. Always Required. ', required=True)
    parser.add_argument('--p', type=str, help=' Directory path to saved model. This will skip the training and just produce an output.', default=None)
    parser.add_argument('--a', type=int, help='Number of images to generate from the pre-trained model.', default=20)
    parser.add_argument('--d', type=str, help='Directory path to training data. Required if not using a pre-trained model.', default=None)
    parser.add_argument('--e', type=int, help='Number of epochs to train', default=5001)
    parser.add_argument('--b', type=int, help='Batch size for training', default=256)
    parser.add_argument('--n', type=int, help='Max range for the random noise generated. ', default=100)
    parser.add_argument('--i', type=int, help='Size of the generated images (for example, 64x64).', default=64)
    parser.add_argument('--s', type=int, help='How often (in epochs) the model will save and output images.', default=5)
    parser.add_argument('--ng', type=int, help="How many neurons to start in the first layer for the generator.", default=256)
    parser.add_argument('--nd', type=int, help="How many neurons to start in the first layer for the discriminator.", default=256)



    args = parser.parse_args()

    if args.p:
        if not os.path.exists(args.p):
            print(f"Error: Pre-trained model '{args.p}' does not exist.")
            exit(1)
        generator = tf.keras.models.load_model(args.p)
        save_individual_generated_images(generator, args.n, args.a, args.f)
        print("Completed generating files using pretrained model.")

    else:
        if not args.d:
            print("Error: The --d argument is required when not using a pre-trained model.")
            exit(1)

        if not os.path.exists(args.d):
            print(f"Error: The specified cat image directory '{args.d}' does not exist.")
            exit(1)

        X_train = load_real_cats(args.d, args.i)

        generator = build_generator(args.n, args.i, args.ng)
        discriminator = build_discriminator(args.i, args.nd)
        gan = build_gan(generator, discriminator)

        train_gan(generator, discriminator, gan, X_train, args.e, args.b, args.n,
                  args.s, args.f)

        generator.save(os.path.join(args.f, 'final-generator.tf'),
                       overwrite=True,
                       include_optimizer=True,
                       save_format='tf')
