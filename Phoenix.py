import pickle
import cv2
import tensorflow as tf
import numpy as np
def evaluate():
    path = (
        "/Users/aviral/Documents/JIIT/6th Semester/WTCS/Project WTCS/Express App/uploads/testpnimage.jpeg"
    )
    lm = tf.keras.models.load_model(
        "/Users/aviral/Documents/JIIT/6th Semester/WTCS/Project WTCS/Express App/my_model.h5"
    )
    y = pickle.load(
        open(
            "/Users/aviral/Documents/JIIT/6th Semester/WTCS/Project WTCS/Express App/y.pkl",
            "rb",
        )
    )
    img = cv2.imread(path)
    img = cv2.resize(img, (100, 100))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = lm.predict(img_array, verbose=0)
    finalans = y[np.argmax(predictions)]
    if finalans == 0:
        finalans = "The model diagnoses it as Pneumonia."
    if finalans == 1:
        finalans = "The model diagnoses it as not Pneumonia."
    print(finalans)
evaluate()