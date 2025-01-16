
"""
Created on Thu May 27 21:00:30 2021

@author: Mohammed Zeeshaan

"""
import os
import pickle
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


# Load the saved model trained at train.py
print('Loading saved Facenet model.......')
model = load_model('pretrained_facenet_model/facenet.h5', compile=False)


print('Printing Inception model inputs and outputs..........')
# summarize input and output shape
print(model.inputs)
print(model.outputs)


# Load the classifier model back from pickle file
print('Loading the SVM classifier pickle file.......')
with open('SVM_Classifier.pkl', 'rb') as file:  
    classifier_model = pickle.load(file)
    
# Initalising MTCNN face detector
print('Initalising MTCNN face detector..........')
detector = MTCNN()

print('Scanning known faces from the dataset....')
print('...............')
print('........................')
def scan_known_people():
    known_names = []
    known_face_encodings = []

    known_dir = os.listdir('train_test_data/test/')

# Loop through each person in the training directory
    for person in known_dir:
        pix = os.listdir('train_test_data/test/'+ person)

    # Loop through each training image for the current person
        for img in pix:
        # Get the face encodings for the face in each image file
            img_raw = image.load_img(('train_test_data/test/' + person + '/' + img), target_size=(160, 160))

            image_raw = image.img_to_array(img_raw)

            image_raw = np.expand_dims(image_raw, axis=0)
            final_img = image_raw/255
            face_enc = model.predict(final_img)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            known_face_encodings.append(face_enc)
            known_names.append(person)
    known_face_encodings=np.array(known_face_encodings)
    return known_names, known_face_encodings

known_names, known_face_encodings= scan_known_people()


def face_distance(face_encodings, face_to_compare, tolerance=2.0):
    """
    This function is defined to detect the unknown faces which are not from the database.
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)/10


def test_image(images_to_check, known_face_encodings, tolerance=2.0, show_distance=False):
    print(f'Predicting the face on the image:------ {images_to_check.split("/")[1]} ') 
    # load image from file
    img_raw = image.load_img(images_to_check)
    # convert to array
    img_array = image.img_to_array(img_raw)
            
    results = detector.detect_faces(img_array)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
            
    # extract the face
    face = img_array[y1:y2, x1:x2]
    
    face_img = image.array_to_img(face)
    face_img= face_img.resize(size=(160,160))

    
    # Preprocess according to our inception model
    img_raw = image.img_to_array(face_img)

    img_raw = np.expand_dims(img_raw, axis=0)
    unknown_image = img_raw/255

    unknown_face_encoding = model.predict(unknown_image)[0]
    unknown_face_encoding= np.array(unknown_face_encoding).reshape(1,-1)

    distances = face_distance(known_face_encodings, unknown_face_encoding)
    result = list(distances <= tolerance)

    if True in result:
        y_pred=classifier_model.predict(unknown_face_encoding)
        ypred=list(y_pred)[0]
        print(f'The Face in the image {images_to_check.split("/")[1]} is recognised as: {ypred.upper()}')
    else:
        print("The Face in the image {} is {}".format(images_to_check.split('/')[1],'UNKNOWN'))

    if not unknown_face_encoding.all():
        # print out fact that no faces were found in image
        print("{},{}".format(images_to_check, 'No face found'))
    print('...............................')   
print('Predictions of five different faces in which two faces are known to the model and other three faces are out of this dataset')
print('..................')
print('.............................')
print('Loading Predictions..........')
print('...................')

unseen_photos=[]   
for entry in os.scandir('unseen_photos'):
    if entry.is_file():
        unseen_photos.append(entry.name)
for img in unseen_photos:
    test_image(('unseen_photos/'+ img ) , known_face_encodings)

