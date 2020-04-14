# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mtcnn
from PIL import Image
from os import listdir

# example of loading the keras facenet model
from keras.models import load_model
# load the model
model = load_model('facenet_keras.h5')
# summarize input and output shape
print(model.inputs)
print(model.outputs)





# function for face detection with mtcnn
from numpy import asarray
from mtcnn.mtcnn import MTCNN

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    
    # extract the bounding box from the first face
    try:
        results = detector.detect_faces(pixels)
    except:
        return None
        
    
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array



# extract a single face from a given photograph
def extract_face_rt(image, required_size=(160, 160)):
    # load image from file

    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    if(len(results) == 0):
        return []
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


"""
#5 celeb project
# specify folder to plot
folder = 'personal_data/train/Imran/'
i = 1

# enumerate files
for filename in listdir(folder):
    # path
    path = folder + filename
    # get face
    face = extract_face(path)
    print(i, face.shape)
    # plot
    plt.subplot(2, 6, i)
    plt.axis('off')
    plt.imshow(face)
    i += 1
plt.show()

"""






# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path)
        # store
        faces.append(face)
    return faces


import os
# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not os.path.isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)


from numpy import savez_compressed








#DON'T RUN IF DATA LOADED
# load train dataset
trainX, trainy = load_dataset('personal_data/train/')
print(trainX.shape, trainy.shape)

# save arrays to one file in compressed format
savez_compressed('personal_data.npz', trainX, trainy)












# calculate a face embedding for each face in the dataset using facenet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model

# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]







#DON'T RUN IF DATA LOADED
# load the face dataset
data = load('personal_data.npz')
trainX, trainy = data['arr_0'], data['arr_1']
print('Loaded: ', trainX.shape, trainy.shape)
# load the facenet model
model = load_model('facenet_keras.h5')
print('Loaded Model')
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)

# save arrays to one file in compressed format
savez_compressed('personal_data_embeddings.npz', newTrainX, trainy)





# develop a classifier for the 5 Celebrity Faces Dataset
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
# load dataset
data = load('personal_data_embeddings.npz')
trainX, trainy = data['arr_0'], data['arr_1']
print('Dataset: train=%d' % (trainX.shape[0]))
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
# fit model
classifier = SVC(kernel='linear', probability=True)
classifier.fit(trainX, trainy)















import cv2
import numpy as np
import os 
import matplotlib.pyplot as plt


cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
#names = ['None', 'Anik', 'Araf', 'Imran', 'Z', 'W'] 
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    print('taking image....')
    ret, img_main =cam.read()
    gray = cv2.cvtColor(img_main,cv2.COLOR_BGR2GRAY)
    
    
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img_main, (x,y), (x+w,y+h), (0,255,0), 2)
    
    

        img = gray[y:y+h,x:x+w]
        img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        
        res = cv2.resize(img_rgb, dsize=(160, 160), interpolation=cv2.INTER_CUBIC)
        
        #testX = Image.fromarray(img_rgb.astype('uint8'), 'RGB')
        #testX = extract_face_rt(image)
        
        if(len(res) < 1):
            continue
        
        testX = get_embedding(model, res)
        
        
        testX = asarray(testX)
        
        testX = testX.reshape(1,-1)
        
        id = classifier.predict(testX)
        
        id_text = out_encoder.inverse_transform(id)
        
        print(id_text)
        
        cv2.putText(
                    img_main, 
                    str(id_text), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
         
    
    cv2.imshow('camera',img_main) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


