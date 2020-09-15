import cv2

# Loading the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detecting a face from a webcam
capture = cv2.VideoCapture(0)

while True:
    # return 2 variable :
    #   _ = flag to indicate correctly read frame.
    #   img = frame itself
    _, img = capture.read()

    if img is not None:

        # Recognition method works only on grayscale images, so we convert the rgb to grayscale
        # **cv2 processes images only in BRG, instead of RGB.
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.5, 5)

        # Drawing a rectangle around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # showing the detected face
        cv2.imshow('Detected Face', img)
        cv2.waitKey()
    else:
        print("empty frame")
        # exit(1)

    # Let the loop break when esc key is pressed
    break_loop = cv2.waitKey(30) & 0xff
    if break_loop == 27:
        break

    # Release the video
    capture.release()


#-----------------------------------------------#
# # This is for detecting a face from an image-----#
#
# # reading the image
# img = cv2.imread('friends.jpg')
#
# # If frame is not found then..
# if img is not None:
#
#     # NOTES:
#     # Recognition method works only on grayscale images, so we convert the rgb to grayscale
#     # **cv2 processes images only in BRG, instead of RGB.
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray_img, 1.5, 5)
#
#     # Drawing a rectangle around detected faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#
#     # Displaying the output
#     cv2.imshow('Detected Faces', img)
#     cv2.imwrite('Detected_Faces.jpg', img)
#     cv2.waitKey()
# else:
#     print("empty frame")
#     exit(1)
# #-----------------------------------------------#
