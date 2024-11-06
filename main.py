# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from tkinter import filedialog
import cv2
import os
import numpy
# from keras.layers import Conv2D,Dropout, Flatten, Dense,MaxPooling2D, MaxPool2D
import keras.layers.normalization
from opt_einsum.backends import tensorflow
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D, MaxPool2D
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import pandas as pd
import random
from tensorflow.python.keras.utils.np_utils import to_categorical
import pickle

count = 0
images = []
classNo = []
labelFile = 'signnames.csv'
classes = 43
testRatio = 0.2  # if 1000 images split will 200 for testing
validationRatio = 0.2  # if 1000 images 20% from remaining 800 will be 160 for valid
path_current = os.getcwd()
imageDim = (32, 32, 3)

####IMPORTING THE IMAGES FROM TRAIN FOLDER

for j in range(classes):
    path = os.path.join(path_current, 'train', str(j))
    imagesList = os.listdir(path)
    for i in imagesList:
        image = cv2.imread(path + '\\' + i)
        imageResized = cv2.resize(image, (32, 32))
        imageResized = numpy.array(imageResized)
        images.append(imageResized)
        classNo.append(count)
    count += 1

images = numpy.array(images)
classNo = numpy.array(classNo)

print(images.shape, classNo.shape)

##### Split Data - make the train
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

# X - images and Y - ids
# X_train - array of images to train
# y_train - class id of images(correspond)


# Validation
print("Data shapes")
print("Train")
print(X_train.shape, y_train.shape)
print("Validation ")
print(X_validation.shape, y_validation.shape)
print("Test")
print(X_test.shape, y_test.shape)

####Read sign names from csv files

data = pd.read_csv(labelFile)
print("Data with sign names:", data.shape)


####Preprocessing the images

def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)  # standard the light in image
    img = img / 255  # to normalize values between 0 and 1 not 0 to 255
    return img


#####processing all the images from train, test, validation
X_train = numpy.array(list(map(preprocessing, X_train)))  # for all the images
X_validation = numpy.array(list(map(preprocessing, X_validation)))
X_test = numpy.array(list(map(preprocessing, X_test)))
#cv2.imshow("GrayScale Images", X_train[random.randint(0, len(X_train) - 1)])  # just to verify the tain
# cv2.waitKey(5000)


##### add a depth of 1 - for better lines
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

####augmentation of images : to make from some images more images, making it more generic, creating various similar images
dataGen = ImageDataGenerator(width_shift_range=0.1,  # 10%
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,  # distorted along an axis(aplecata)
                             rotation_range=10)  # degrees

dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)  # generate 20 images when it s called
X_batch, y_batch = next(batches)

#######from label to one encoding(making matrix with 0 and 1 based on classes number)
y_test = to_categorical(y_test, classes)
y_train = to_categorical(y_train, classes)
y_validation = to_categorical(y_validation, classes)


###########convolution neural network model
def myModel():
    nodesNr = 500
    filterNr = 60  ##to dont remove pixels based on filter size
    filterSize = (5, 5)  ##the kernel that move around the image to get the features
    # making padding
    filterSize2 = (3, 3)
    poolSize = (
    2, 2)  # for more generalize, to reduce overfitting(when detail and noise in training and go to negative result)

    model = Sequential()
    model.add(Conv2D(filterNr, filterSize, activation='relu', input_shape=X_train.shape[1:])) #adding 1st filter on the data from x_train shape
    model.add(Conv2D(filterNr, filterSize, activation='relu'))
    model.add(MaxPooling2D(pool_size=poolSize)) #selecting the maxim value from pool_size

    model.add(Conv2D(filterNr // 2, filterSize2, activation='relu')) #adding another filter(for better accuracy)
    model.add(Conv2D(filterNr // 2, filterSize2, activation='relu'))
    model.add(MaxPool2D(pool_size=poolSize))
    model.add(Dropout(0.5))

    model.add(Flatten())#redimension shapes if it s necessary
    model.add(Dense(nodesNr, activation='relu'))
    model.add(Dropout(0.5)) # to hide some neurons from my model
    model.add(Dense(classes, activation='softmax'))  # output layer

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #compile the model
    return model


####TRAIN
model = myModel()
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test)) #final validation and accuracy of the model, based on train data and test images
model.save("traffic_classifier")



