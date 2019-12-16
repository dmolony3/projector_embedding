import tensorflow as tf
import os
import numpy as np
from PIL import Image

class DataReader():
    """Reads images from text file"""
    def __init__(self, directory, batch_size):
        self.directory = directory
        self.batch_size = batch_size

    def read_files(self, file_path):
        file_list = []
        f = open(file_path, 'r')
        for line in f:
            if line:
                file_list.append(os.path.join(self.directory, line.strip()))
        f.close()
        return file_list

    def decode_image(self, image):
        image = tf.io.read_file(image)
        image = tf.io.decode_png(image, channels=1)
        return image

    def read_batch(self, file_path):
        file_list = self.read_files(file_path)
        data = tf.data.Dataset.from_tensor_slices((file_list))
        #file_list = tf.data.Dataset.list_files(str(file_path/'*/*'))
        data = data.map(self.decode_image)
        data = data.batch(batch_size=self.batch_size, drop_remainder=False)
        return data


directory = '/home/microway/Documents/SPADE/results/IVUS_45MHz/inference_latest/images/synthesized_image'
data_file = '/home/microway/Documents/SPADE/results/IVUS_45MHz/inference_latest/images/synthesized_image/generated.txt'
num_image_rows = 64 # must be divisble by batch_size
embedding_dim = 64
image_dim = 128

sprite_file = 'sprite_image' + '_' + str(embedding_dim) + 'x' + str(embedding_dim) +'.jpg'
vecs_file = 'vecs' + '_' + str(embedding_dim) + 'x' + str(embedding_dim) +'.tsv'
meta_file = 'metadata' + '_' + str(embedding_dim) + 'x' + str(embedding_dim) +'.tsv'

batch_size = 1
num_images = num_image_rows**2

data = DataReader(directory, batch_size)
data_iterator = iter(data.read_batch(data_file))

image_list = []
f1 = open(vecs_file, 'a', encoding='utf-8')
f2 = open(meta_file, 'a', encoding='utf-8')
for i in range(num_images):
    batch = next(data_iterator)
    batch_downsampled = tf.image.resize(batch, [embedding_dim, embedding_dim])
    batch_vectorized = tf.reshape(batch_downsampled, shape=[batch_size, embedding_dim*embedding_dim])
    batch_vectorized = tf.squeeze(batch_vectorized, 0)
    dim = batch_vectorized.shape
    f1.write('\t'.join([str(batch_vectorized.numpy()[j]) for j in range(dim[0])]) + "\n")
    f2.write('{}\n'.format(1)) # set label to 1 for all images
    image_list.append(tf.squeeze(tf.squeeze(tf.image.resize(batch, [image_dim, image_dim]), 0), -1))
f1.close()
f2.close()

# generate the sprite image
sprite_image = np.zeros((image_dim*num_image_rows, image_dim*num_image_rows))
idx = 0
for i in range(num_image_rows):
    for j in range(num_image_rows):
        sprite_image[i*image_dim:(i+1)*image_dim, j*image_dim:(j+1)*image_dim] = image_list[idx]
        idx += 1
im = Image.fromarray(sprite_image.astype(np.uint8))
im.save(sprite_file)

