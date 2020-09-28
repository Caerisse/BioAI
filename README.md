# BioAI

## Requirements
Run :
```
pip install -r requirements.txt
pip install git+https://github.com/ageitgey/face_recognition_models
```

## Create dataset of known faces
There are two ways to do it:
 1. Create a FacesDataset folder and inside of it folders with the name of the person as name and put photos of their face inside
 2. Sit the person in front of the pc webcam and run `python BioAI/record_dataset.py -n <name>` press 'k' to save a picture and 'q' to end the script

Around 10 pictures seems to be enough, more may be needed for different illuminations and face positions to make the program more precise. There can be more than one face per picture, given they are all of the same person.

## Encode faces
Run:
```
python BioAI/encode_faces.py -d <method>
```
\<method> is the face detection model to use, either 'hog' or 'cnn', cnn if better but much slower, 'hog' is the default

## Run
Run:
```
python BioAI -d <method>
```
\<method> is the face detection model to use, either 'hog' or 'cnn', cnn if better but much slower, 'hog' is the default