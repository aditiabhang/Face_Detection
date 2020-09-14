import cv2

# Loading the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# reading the image
img = cv2.imread('friends.jpg')
print(img)
# If frame is not found then..
if img is not None:
    # NOTES:
    # Recognition method works only on grayscale images, so we convert the rgb to grayscale
    # **cv2 processes images only in BRG, instead of RGB.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.5, 5)

    # Drawing a rectangle around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Displaying the output
    cv2.imshow('Detected Faces', img)
    cv2.imwrite('Detected_Faces.jpg', img)
    cv2.waitKey()
else:
    print("empty frame")
    exit(1)
