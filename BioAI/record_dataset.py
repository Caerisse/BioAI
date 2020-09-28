import os
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True, help="name of the person to save the dataset")
try:
    args = vars(ap.parse_args())
    name = args["name"]

    dataset_dir = os.path.sep.join(["..", "FacesDataset"])
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    dataset_dir = os.path.sep.join([dataset_dir, name])
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    total = 0

    webcam = cv2.VideoCapture(0)

    red = (0, 0, 255)

    while True:
        _, frame = webcam.read()

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        # if the `k` key was pressed, write the frame to disk
        # so we can later process it and use it for face recognition
        if key == ord("k"):
            p = os.path.sep.join([dataset_dir, "{}.png".format(
                    str(total).zfill(5))])
            cv2.imwrite(p, frame)
            total += 1
        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break


except KeyboardInterrupt:
    pass

