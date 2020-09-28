
import pickle
import cv2
import face_recognition


class FaceDetector:
    def __init__(self, model = "hog", encodings_file = "../FaceEncodings.pickle"):
        """
        :param model: 'hog' or 'cnn'
        """
        self.frame = None
        self.faces = None
        self.landmarks = None
        self.model = model
        self.encodings_file = encodings_file
        self.data = pickle.loads(open(encodings_file, "rb").read())
        self.r = 1

    def _analyze(self):
        # convert the input frame from BGR to RGB then
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        # resize it to have a width of 750px (to speedup processing)
        # rgb = imutils.resize(rgb, width=750)
        self.r = self.frame.shape[1] / float(rgb.shape[1])

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb, model=self.model)
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(self.data["encodings"], encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        self.faces = zip(boxes, names)

    def refresh(self, frame):
        """Refreshes the frame and analyzes it."""
        self.frame = frame
        self._analyze()

    def annotate_frame(self):
        frame = self.frame.copy()
        # print(frame)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        # loop over the recognized faces
        for i, ((top, right, bottom, left), name) in enumerate(self.faces):
            color = colors[i%len(colors)]
            # rescale the face coordinates
            top = int(top * self.r)
            right = int(right * self.r)
            bottom = int(bottom * self.r)
            left = int(left * self.r)

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            y = top - 15 if top - 15 > 15 else bottom + 20
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        return frame
