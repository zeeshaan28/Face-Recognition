"""
Created on Thu May 27 12:09:42 2021

@author: Mohammed Zeeshaan
"""
# Importing all the required libraries
import os
import numpy as np
import shutil
import random
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import re
import sys
import pickle
from inception_resnet_v1 import *
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Creating Train / Test folders
print('Splitting the dataset into train and test and saving in the directory(train_test_data)...... ')
root_dir = 'cropped_images/' # data root path
classes_dir = ['chris_evans', 'chris_hemsworth','mark_ruffalo','robert_downey_jr','scarlett_johansson'] #total labels

#Deleting the directory if it already exists
try:
  shutil.rmtree('train_test_data')
except FileNotFoundError:
  print("Directory not available, Creating a new directory.........")
except:
  print("Something else went wrong")

test_ratio = 0.2

for cls in classes_dir:
    os.makedirs('train_test_data/train/' + cls)
    os.makedirs('train_test_data/test/' + cls)


# Creating partitions of the data after shuffeling
    src = root_dir + cls   # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                               [int(len(allFileNames)* (1 - test_ratio))])


    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, 'train_test_data/train/' + cls)


    for name in test_FileNames:
        shutil.copy(name, 'train_test_data/test/' + cls)
        
        
# Initialising the InceptionV1 architecture from  inception_resnet_v1 
print('Intialising the InceptionV1 architecture........')
model = InceptionResNetV1(classes=512)
#model.summary()

# Loading the pre_trained weights 
# Using the state-of-the-art Facenet model by google
weights_dir= 'pretrained_facenet_model/weights/facenet_keras_weights.h5'
model.load_weights(weights_dir)


# Defining a loss function which is specially built to train face_recognition models
print('Defining the Triplet_loss function........ ')
def triplet_loss(y_true, y_pred, alpha=0.3):
    """
    Implementation of the triplet loss as defined by formula (3)
    Arguments:
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss

# Compiling the Inception model with the newly defined triplet_loss function
model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])


# Extracting the face encodings from the images in the training dataset 

# Face encodings on image directories
def encoding(folder_path):

    encodings = []
    names = []
    dir = os.listdir(folder_path)

    # Loop through each person in the directory
    for person in dir:
        pix = os.listdir(folder_path + person)

        # Loop through each training image for the current person
        for img in pix:
            
            img_raw = image.load_img((folder_path + person + "/" + img), target_size=(160, 160)) # Input Size for the model is 160*160
        
            # Preprocessing the image
            image_raw = image.img_to_array(img_raw)
    
            image_raw = np.expand_dims(image_raw, axis=0)
            img = image_raw/255
            
            # Get the face encodings for the face in each image file by passing through the facenet model created
            face_encodings = model.predict(img)[0]
            
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_encodings)
            names.append(person)
            
    encodings=np.array(encodings)
    names=np.array(names)
            
    return encodings,names

print('Extracting the face encodings for the training data..........')
print('............................')
X_train,y_train= encoding('train_test_data/train/')

# Initialising the SVM Classifier 
# Selecting this SVM classifier because it performs very well scaled encoding variables 

print('Trainig the SVM classifier using the extracted face encodings.....')
classifier_model= SVC(kernel='linear',gamma='scale')
# Fitting the model with the extracted face encodings and variables
classifier_model.fit(X_train,y_train)

# Prediction on training set
print('Predicting the classifier model on training data.....')
ytrain_pred=classifier_model.predict(X_train)

#Printing the training Accuracy
print(f'The Training accuracy of this Classifier model is {accuracy_score(y_train,ytrain_pred)*100}%')
print('.................................')

# Creating the X_test, y_test for testing the Classifier Model
print('Extracting the face encodings for the test data..........')
print('............................')
X_test,y_test= encoding('train_test_data/test/') 

print('Predicting the classifier model on test data.....')
y_pred=classifier_model.predict(X_test)
# Printing the testing Accuracy
print(f'The Testing accuracy of this Classifier model is {accuracy_score(y_test,y_pred)*100}%')
print('..............................')

print('Saving the SVM Classifier Model as pickle file......')
model_name = "SVM_Classifier.pkl"  
with open(model_name, 'wb') as file:  
    pickle.dump(classifier_model, file)
print('Model Saved')
print('..............................')

print('Saving the state-of-the-art Facenet model....... ')
model.save('pretrained_facenet_model/facenet.h5')
print('Model Saved')




 





        
        

        
    
    
    