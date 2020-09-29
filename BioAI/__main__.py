import argparse

import cv2
from face_detector import FaceDetector
def main(model):
    webcam = cv2.VideoCapture(0)
    face_detector = FaceDetector(model)

    while True:
        _, frame = webcam.read()
        face_detector.refresh(frame)
        frame = face_detector.annotate_frame()
        cv2.imshow("FaceDetector", frame)
        key = cv2.waitKey(1) & 0xFF ## I have no fucking idea why but frame is refusing to show unless this line is present


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--detection-method", type=str, default="hog",
                help="face detection model to use: either 'hog' or 'cnn'")
    args = vars(ap.parse_args())
    try:
        main(args["detection_method"])
    except KeyboardInterrupt:
        pass
