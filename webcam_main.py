import cv2

# Importing the pre-trained data
cascade_path = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascade_path)

# Capturing the video using cv2
video_capture = cv2.VideoCapture(0)

while True:
    # Capturing the video frame-by-frame
    ret, frame = video_capture.read()

    # Recognition method works only on grayscale images, so we convert the rgb to grayscale
    # **cv2 processes images only in BRG, instead of RGB.
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Drawing a square around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Showing the detected faces in the video frames
    cv2.imshow("Detected Faces", frame)

    # Closing the window when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the captured video
video_capture.release()
cv2.destroyAllWindows()