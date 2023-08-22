import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")

#Baca gambar di img dan ubah menjadi skala abu-abu dan simpan dalam warna abu-abu.
#Gambar diubah menjadi skala abu-abu, karena kaskade wajah tidak perlu beroperasi pada gambar berwarna.
img = cv.imread('images/septian.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Deteksi semua wajah dalam gambar.
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#Gambar persegi panjang di atas wajah, dan deteksi mata di wajah
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    #ROI adalah region of interest dengan area yang memiliki wajah di dalamnya.
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    #Detect eyes in face
    eyes = eye_cascade.detectMultiScale(roi_gray)

    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
