import keras
from keras.preprocessing.image import img_to_array
import numpy as np
import json
import operator
from PIL import Image
from deep_learning_object_detection import detect


def get_pic(file):
    img = Image.Image.resize(file, (299, 299))
    x = img_to_array(img)
    x = np.array(x).astype('float32') / 255
    x = np.expand_dims(x, axis=0)
    return x


with open('static/labels.txt', 'r') as f:
    distros_dict = json.load(f)
labels_old = list(distros_dict)

labels = []
for l in labels_old:
    l = l[10:].replace("_", " ")
    l = l.replace("-", " ")
    labels.append(l)
    print(l)

model = keras.models.load_model("dl_models/transfer_learning_weights.01-0.36.hdf5")
model.summary()


def classify(img):
    img = Image.open(img)
    img.save("test.jpg")
    label_detect = detect("test.jpg")
    if not label_detect is None:
        img = get_pic(img)
        prediction = model.predict(img)
        max_index, max_value = max(enumerate(prediction[0]), key=operator.itemgetter(1))
        print(labels[max_index], max_value)
        label, score = labels[max_index], max_value
    else:
        label = "Dog is not detected in image, kindly use other image"
        max_value = 0
    return [label, max_value]
