#Impor perpustakaan yang diperlukan
from scipy.spatial import distance
import tkinter as tk
from imutils import face_utils
import numpy as np
import pygame #Untuk memutar suara
import time
import dlib
import cv2

#Inisialisasi Pygame dan muat musik
pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')

# Ambang minimum rasio aspek mata di bawah alarm yang dipicu
EYE_ASPECT_RATIO_THRESHOLD = 0.28

#Minimum bingkai berturut-turut dengan rasio mata di bawah ambang batas untuk memicu alarm
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50

#COunts no. of consecutuve frames below threshold value
COUNTER = 0

# Muat kaskade wajah yang akan digunakan untuk menggambar persegi panjang di sekitar wajah yang terdeteksi.
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

#Fungsi ini menghitung dan mengembalikan rasio aspek mata
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear

def start_detection():
    global COUNTER
    # Memuat detektor dan prediktor wajah, menggunakan file prediktor bentuk dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Ekstrak indeks landmark wajah untuk mata kiri dan kanan
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

    #Mulai perekaman video webcam
    video_capture = cv2.VideoCapture(0)
    # Berikan waktu untuk menginisialisasi kamera (tidak diperlukan)
    time.sleep(2)

    while(True):
        # Baca setiap bingkai dan balikkan, dan ubah ke skala abu-abu
        ret, frame = video_capture.read()
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mendeteksi titik wajah melalui fungsi detektor
        faces = detector(gray, 0)

        # Deteksi wajah melalui haarcascade_frontalface_default.xml
        face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Gambar persegi panjang di sekitar setiap wajah yang terdeteksi
        for (x,y,w,h) in face_rectangle:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        # Mendeteksi titik wajah
        for face in faces:

            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # Dapatkan susunan koordinat mata kiri dan mata kanan
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # Hitung rasio aspek kedua mata
            leftEyeAspectRatio = eye_aspect_ratio(leftEye)
            rightEyeAspectRatio = eye_aspect_ratio(rightEye)

            eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

            # Gunakan lambung untuk menghilangkan perbedaan kontur cembung dan menggambar bentuk mata di sekitar mata
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Mendeteksi jika rasio aspek mata kurang dari ambang batas
            if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
                COUNTER += 1
                # Jika tidak. bingkai lebih besar dari bingkai ambang,
                if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                    pygame.mixer.music.play(-1)
                    cv2.putText(frame, "Anda Mengantuk!!", (150,200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            else:
                pygame.mixer.music.stop()
                COUNTER = 0

        #Tampilkan umpan video
        cv2.imshow('Video', frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    #Akhirnya saat pengambilan video selesai, lepaskan pengambilan video dan hancurkan SemuaWindows
    video_capture.release()
    cv2.destroyAllWindows()
pass

# Membuat jendela utama
window = tk.Tk()
window.title("Pendeteksi Kantuk")
window.geometry("400x300")  # Mengatur lebar menjadi 400 dan tinggi menjadi 300
window.configure(bg="black")

# Membuat tombol "Start"
start_button = tk.Button(window, text="Mulai", command=start_detection)
start_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Memulai main loop jendela
window.mainloop()