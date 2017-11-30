import os
import cv2
import numpy as np
import cv2
from keras.models import model_from_json
from keras.optimizers import SGD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

face_cascade = cv2.CascadeClassifier(
    'E:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    'E:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')
eye_glasses_cascade = cv2.CascadeClassifier(
    'E:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml')

SIZE_FACE = 48
optim = SGD()

emotion = {0: 'anger', 1: 'disgust',
           2: 'fear', 3: 'happy',
           4: 'sad', 5: 'surprise', 6: 'neutral'}


def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  faces = face_cascade.detectMultiScale(
      image,
      scaleFactor=1.3,
      minNeighbors=5
  )
  # None is we don't found an image
  if not len(faces) > 0:
    return None
  max_area_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
      max_area_face = face
  # Chop image to face
  face = max_area_face
  image = image[face[1]:(face[1] + face[2]),
                face[0]:(face[0] + face[3])]

  # Resize image to network size
  try:
    image = cv2.resize(image, (SIZE_FACE, SIZE_FACE),
                       interpolation=cv2.INTER_CUBIC) / 255.
    # while True:
    #   cv2.imshow("frame", image)
    #   if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
  except Exception:
    print("[+] Problem during resize")
    return None
  # cv2.imshow("Lol", image)
  # cv2.waitKey(0)
  return image

# LOAD MODEL
model_architecture = './emotion_recog.json'
model_weights = './emotion_recog_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

cap = cv2.VideoCapture(0)

while cap.isOpened():
  ret, img = cap.read()
  # img = cv2.imread(img)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  feed = np.resize(np.array([format_image(img)]),
                   (1, SIZE_FACE, SIZE_FACE, 1))
  model.compile(loss='categorical_crossentropy', optimizer=optim,
                metrics=['accuracy'])
  predictions = model.predict_classes(feed)
  print(predictions)
  # let's detect multiscale (some images may be closer to camera than
  # others) images
  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=1.1, minNeighbors=5)

  for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    if len(eyes) == 0:
      eyes = eye_glasses_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
      cv2.rectangle(roi_color, (ex, ey),
                    (ex + ew, ey + eh), (0, 255, 0), 2)

  cv2.imshow('img', img)
  if cv2.waitKey(10) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()
