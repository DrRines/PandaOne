import cv2
import sys
import numpy as np
import time

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces & save face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        sub_face = frame[y:y+h, x:x+w]

        write_ok = int(time.time()) % 2 == 0

        if write_ok:
            face_file_name = "faces/face_" + str(int(time.time())) + ".jpg"
            cv2.imwrite(face_file_name, sub_face)


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()