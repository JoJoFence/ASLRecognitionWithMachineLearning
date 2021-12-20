import keras
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
#import imutils
import os

alphabet=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
model_path = "/Users/JonasHansen/Library/Mobile Documents/com~apple~CloudDocs/Sophomore Year/ENG-SCI 111/Final Project/ASL_model_2"
model = keras.models.load_model(model_path)

def classify(image):
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    proba = model.predict(image)
    pred_proba = proba[0, proba.argmax()]
    idx = np.argmax(proba)
    return alphabet[idx], pred_proba
    
cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
#    image = cv2.imread('amer_sign2.png')
#    cv2.imshow("image", image)
    img = cv2.flip(img, 1)
    top, right, bottom, left = 75, 750, 400, 1090
    roi = img[top:bottom, right:left]
    roi=cv2.flip(roi,1)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow('roi',gray)
    alpha, perc_prob = classify(gray)
    cv2.rectangle(img, (left, top), (right, bottom), (0,255,0), 2)
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,str(alpha) + " - " + str(np.round(perc_prob*100, 3)) + "%",(50,530),font,3,(0,0,255),2)
#    cv2.resize(img,(1000,1000))
    cv2.imshow('img',img)
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()

