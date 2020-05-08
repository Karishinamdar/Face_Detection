# import the necessary packages
import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('/home/hydro/Documents/face_detection/cascades/haarcascade_frontalface_default.xml')

# load the image and convert it to grayscale
image = cv2.imread('messi.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faceRects = face_cascade.detectMultiScale(gray, 1.1, 5)
print("I found {} face(s)".format(len(faceRects)))

# loop over the faces and draw a rectangle around each
for (x, y, w, h) in faceRects:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the detected faces
cv2.imshow('image', image)
cv2.waitKey(0)
