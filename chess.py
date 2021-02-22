# SETUP
from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pylab as plt
import tensorflow as tf
import tensorflowjs as tfjs
import tensorflow_hub as hub

from tensorflow.keras import layers
import numpy as np
import PIL.Image as Image

DATA_PATH = 'batch-1'

#MODEL

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from tensorflow.keras.preprocessing import image
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import PIL.Image as Image
from os import path
from IPython.display import SVG, display, Image

BATCH_SIZE = 32
MODEL_INCEPTION_V3 = {
    "shape": (299, 299),
    "url": "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
    "preprocessor": inception_v3_preprocess_input
}
MODEL_MOBILENET_V2 = {
    "shape": (224, 224),
    "url": "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
    "preprocessor": mobilenet_preprocess_input
}
def create_model(arch=MODEL_MOBILENET_V2):
  image_input = tf.keras.Input(shape=(*arch["shape"],3), name='img')
  nn = hub.KerasLayer(arch["url"],
                      input_shape=(*arch["shape"],3),
                      trainable=True)(image_input)
  nn = layers.Dense(2050, activation='relu')(nn)
  outputs = []
  for i in range(64):
    out = layers.Dense(13, activation='softmax')(nn)
    outputs.append(out)
  model = tf.keras.models.Model(inputs=image_input, outputs=outputs)
  model.compile(optimizer='adam', loss=["categorical_crossentropy"] * 64, loss_weights=[1.0]*64)
  return model

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, labels, batch_size=BATCH_SIZE, shuffle=True, arch=MODEL_MOBILENET_V2):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.shuffle = shuffle
        self.arch = arch
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        start = index*self.batch_size
        end = (index+1)*self.batch_size
        idx_labels = self.labels[start:end]
        X = np.zeros((self.batch_size, *self.arch["shape"], 3))
        for i, label in enumerate(idx_labels):
          # make X
          img = image.load_img(DATA_PATH + "/" + label, target_size=self.arch["shape"])
          a = image.img_to_array(img)
          a = self.arch["preprocessor"](a)
          X[i,] = a
        y = []
        for sq in range(64):
          out = np.zeros((self.batch_size,13))
          for i, label in enumerate(idx_labels):
            fen = path.splitext(label)[0]
            rows = self.fill_ones(fen).split("-")
            rows.reverse()
            c = rows[sq // 8][sq % 8]
            idx = self.fen_char_to_idx(c)
            out[i,idx] = 1.0
          y.append(out)
        return X, y

    def fen_char_to_idx(self, c):
      s = "KQRBNPkqrbnp1"
      return s.find(c)

    def fill_ones(self, fen):
      for i in range(8,1,-1):
        fen = fen.replace(str(i), "1"*i)
      return fen

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
          random.shuffle(self.labels)


#TRAINING

import glob

labels = glob.glob("batch-1/*.jpg")
labels = list(map(lambda l: path.basename(l), labels))
print("Number of labels " + str(len(labels)))
labels_train, labels_val = train_test_split(labels)
training_generator = DataGenerator(labels_train)
validation_generator = DataGenerator(labels_train)

# tensorboard callback
tbCallBack = TensorBoard(log_dir='./log', histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         batch_size=BATCH_SIZE,
                         write_images=True)
checkpoint = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             mode='auto',
                             save_freq=1)

# Train model on dataset
model = create_model()
model.summary()
history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              verbose=2,
                              epochs=40,
                              callbacks=[tbCallBack, checkpoint])

# season to taste
history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              verbose=2,
                              epochs=5,
                              callbacks=[tbCallBack, checkpoint])



model.save('model/saved_model')

piece_lookup = {
    0 : "K",
    1 : "Q",
    2 : "R",
    3 : "B",
    4 : "N",
    5 : "P",
    6 : "k",
    7 : "q",
    8 : "r",
    9 : "b",
    10 : "n",
    11 : "p",
    12 : "1",
}
def y_to_fens(y):
  results = []
  for n in range(BATCH_SIZE):
    fen = ""
    for sq in range(64):
      piece_idx = np.argmax(y[sq][n,])
      fen += piece_lookup[piece_idx]
    a = [fen[i:i+8] for i in range(0, len(fen), 8)]
    a = a[::-1]
    fen = "/".join(a)
    for i in range(8,1,-1):
      old_str = "1" * i
      new_str = str(i)
      fen = fen.replace(old_str, new_str)
    results.append(fen)
  return results

test_X, test_y = validation_generator.__getitem__(0)
batch_y = model.predict(test_X)
true_fens = y_to_fens(test_y)
pred_fens = y_to_fens(batch_y)

index_to_show = 1
file_name = DATA_PATH + "/" + validation_generator.labels[index_to_show]
print("3D Image")
display(Image(filename=file_name, width=400))
BASE_URL = "https://us-central1-spearsx.cloudfunctions.net/chesspic-fen-image/"
print("2D Ground Truth " + true_fens[index_to_show])
display(SVG(url=BASE_URL+true_fens[index_to_show]))
print("2D Prediction " + pred_fens[index_to_show])
display(SVG(url=BASE_URL+pred_fens[index_to_show]))


# print("Save h5")
# model.save('model/keras_model/my_model.h5') # save keras format

# print("SAVE JSON")
# tfjs.converters.save_keras_model(model, 'model/json_model')

# print("SAVE EXPERIMENTAL")
# tf.keras.experimental.export_saved_model(model, 'model/experimental_model')

# new_model = tf.keras.experimental.load_from_saved_model(path)
# new_model.summary()
