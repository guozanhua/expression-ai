import os
import cv2
import numpy as np
import scipy.misc
from keras.models import model_from_json
from keras.optimizers import SGD
from utils import format_image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
emotion = {0: 'anger', 1: 'disgust',
           2: 'fear', 3: 'happy',
           4: 'sad', 5: 'surprise', 6: 'neutral'}

SIZE_FACE = 48

face_cascade = cv2.CascadeClassifier(
    'E:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    'E:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')
eye_glasses_cascade = cv2.CascadeClassifier(
    'E:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml')

# load model
model_architecture = 'emotion_recog.json'
model_weights = 'emotion_recog_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

# load images
img_names = ['sjobs.jpg']
imgs = [np.resize(np.array([format_image(face_cascade, cv2.imread(img_name))]),
                  (1, SIZE_FACE, SIZE_FACE, 1)) for img_name in img_names]
# train
optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optim,
              metrics=['accuracy'])

# predict
predictions = model.predict_classes(imgs)
print("Emotion: {}".format(emotion[predictions[0]]))
