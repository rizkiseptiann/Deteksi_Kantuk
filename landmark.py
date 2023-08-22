import dlib
import cv2

face_detector = dlib.get_frontal_face_detector()

landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  

image = cv2.imread('images/hasil.png') 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_detector(gray)

for face in faces:
    landmarks = landmark_predictor(gray, face)

    for point in landmarks.parts():
        x, y = point.x, point.y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  

cv2.imshow('Landmark Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
