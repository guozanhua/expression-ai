import os
import numpy as np
import pandas as pd
from keras.utils import np_utils, plot_model
from keras.models import Sequential
from keras.layers.core import Dropout, Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import tensorflow as tf
import pickle
# from sklearn.preprocessing import LabelEncoder 
os.environ['TP_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_ITERATIONS = 1001
REGULARIZATION = 1e-2
IMAGE_SIZE = 48
NUM_LABELS = 7
VERBOSE = True
VALIDATION_PERCENT = 0.1
OPTIM = SGD()
IMAGE_LOCATION_NORM = IMAGE_SIZE / 2
NB_EPOCH = 200

np.random.seed(0)
emotion = {0:'anger', 1:'disgust',\
           2:'fear',3:'happy',\
           4:'sad',5:'surprise',6:'neutral'}

# preparing the data
def read_data(data_dir, force=False):
    def create_onehot_label(x):
        label = np.zeros((1, NUM_LABELS), dtype=np.float32)
        label[:, int(x)] = 1
        return label

    pickle_file = os.path.join(data_dir, "EmotionDetectorData.pickle")
    if force or not os.path.exists(pickle_file):
        train_filename = os.path.join(data_dir, "train.csv")
        data_frame = pd.read_csv(train_filename)
        data_frame['Pixels'] = data_frame['Pixels'].apply(lambda x: np.fromstring(x, sep=" ") / 255.0)
        data_frame = data_frame.dropna()
        print("Reading train.csv ...")

        train_images = np.vstack(data_frame['Pixels']).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        print(train_images.shape)
        train_labels = np.array([i for i in map(create_onehot_label, data_frame['Emotion'].values)]).reshape(-1, NUM_LABELS)
        print(train_labels.shape)

        permutations = np.random.permutation(train_images.shape[0])
        train_images = train_images[permutations]
        train_labels = train_labels[permutations]
        validation_percent = int(train_images.shape[0] * VALIDATION_PERCENT)
        validation_images = train_images[:validation_percent]
        validation_labels = train_labels[:validation_percent]
        train_images = train_images[validation_percent:]
        train_labels = train_labels[validation_percent:]

        print("Reading test.csv ...")
        test_filename = os.path.join(data_dir, "test.csv")
        data_frame = pd.read_csv(test_filename)
        data_frame['Pixels'] = data_frame['Pixels'].apply(lambda x: np.fromstring(x, sep=" ") / 255.0)
        data_frame = data_frame.dropna()
        test_images = np.vstack(data_frame['Pixels']).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

        with open(pickle_file, "wb") as file:
            try:
                print('Picking ...')
                save = {
                    "train_images": train_images,
                    "train_labels": train_labels,
                    "validation_images": validation_images,
                    "validation_labels": validation_labels,
                    "test_images": test_images,
                }
                pickle.dump(save, file, pickle.HIGHEST_PROTOCOL)
                print("Successfully pickled!")

            except:
                print("Unable to pickle file :/")

    with open(pickle_file, "rb") as file:
        save = pickle.load(file)
        train_images = save["train_images"]
        train_labels = save["train_labels"]
        validation_images = save["validation_images"]
        validation_labels = save["validation_labels"]
        test_images = save["test_images"]

    return train_images, train_labels, validation_images, validation_labels, test_images

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", ".", "Path to data files")
tf.flags.DEFINE_string("logs_dir", "EmotionDetector_logs/", "Path to where log files are to be saved")
tf.flags.DEFINE_string("mode", "train", "model: train (DEFAULT)/ test")

train_images, train_labels, valid_images, valid_labels, test_images = read_data(FLAGS.data_dir)
print("train images shape = ", train_images.shape)
print("test labels shape = ", test_images.shape)

# model
model = Sequential()
model.add(Conv2D(64, (5,5), padding='same',
	input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Conv2D(64, (5,5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Conv2D(128, (4,4), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(3072))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(NUM_LABELS))
model.add(Activation('softmax'))
# model.summary()

# train
model.compile(loss='categorical_crossentropy', optimizer=OPTIM,
              metrics=['accuracy'])
model.fit(train_images, train_labels,
          batch_size=BATCH_SIZE, epochs=NB_EPOCH,
          verbose=VERBOSE, validation_split=VALIDATION_PERCENT)
score = model.evaluate(valid_images, valid_labels, verbose=VERBOSE)
print("Test Score:", score[0])
print("Test accuracy:", score[1])

# plot_model(model, to_file='model.png')
model_json = model.to_json()
open('cifar10_architecture.json', 'w').write(model_json)
model.save_weights('cifar10_weights.h5', overwrite=True)