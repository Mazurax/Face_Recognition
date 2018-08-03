import face_recognition
from imutils.video import WebcamVideoStream
import numpy as np
import imutils
import time
import cv2

from image_to_array import ImageToArray

net = cv2.dnn.readNetFromCaffe("../../resources/caffe/deploy.prototxt.txt",
                               "../../resources/caffe/res10_300x300_ssd_iter_140000.caffemodel")

vs = WebcamVideoStream(src=0).start()
time.sleep(1.0)

while True:
    if vs.sleeping:
        continue
    frame = vs.read()

    # grab the frame dimensions and convert it to a blob
    resized = cv2.resize(frame, (300, 300))
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(resized, 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        if confidence < 0.3:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        ita = ImageToArray(frame, box)
        ita.get_array()
        # vs.drawRec(startX, startY, endX, endY)
        # draw the bounding box of the face along with the associated
        # probability
        # text = "{:.2f}%".format(confidence * 100)
        # y = startY - 10 if startY - 10 > 10 else startY + 10
        #
        # cv2.rectangle(frame, (startX, startY), (endX, endY),
        #               (0, 0, 255), 2)
        # cv2.putText(frame, text, (startX, y),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # cv2.imshow("Frame", frame)
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord("q"):
    #     break
