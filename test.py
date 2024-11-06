from tkinter import filedialog
import cv2
import tensorflow
from keras.models import model_from_json
from tensorflow.keras.utils import CustomObjectScope

def getModel():
    global model
    model = tensorflow.keras.models.load_model('traffic_classifier', custom_objects=None)
    model.summary()
    print("AM IESIT DIN GET MODEL")
    return model


getModel()
print("AM INCARCAT")


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


def getClassName(classNo):
    if classNo == 0:
        return 'Speed Limit (20 km/h)'
    elif classNo == 1:
        return 'Speed Limit (30 km/h)'
    elif classNo == 2:
        return 'Speed Limit (50 km/h)'
    elif classNo == 3:
        return 'Speed Limit (60 km/h)'
    elif classNo == 4:
        return 'Speed Limit (70 km/h)'
    elif classNo == 5:
        return 'Speed Limit (80 km/h)'
    elif classNo == 6:
        return 'End of Speed Limit (80 km/h)'
    elif classNo == 7:
        return 'Speed Limit (100 km/h)'
    elif classNo == 8:
        return 'Speed Limit (120 km/h)'
    elif classNo == 9:
        return 'No passing'
    elif classNo == 10:
        return 'No passing for vehicles over 3.5 metric tons'
    elif classNo == 11:
        return 'Right-of-way at the next intersection'
    elif classNo == 12:
        return 'Priority road'
    elif classNo == 13:
        return 'Yield'
    elif classNo == 14:
        return 'Stop'
    elif classNo == 15:
        return 'No vehicles'
    elif classNo == 16:
        return 'Vehicles over 3.5 metric tons prohibited'
    elif classNo == 17:
        return 'No entry'
    elif classNo == 18:
        return 'General Caution'
    elif classNo == 19:
        return 'Dangerous curve to the left'
    elif classNo == 20:
        return 'Dangerous curve to the right'
    elif classNo == 21:
        return 'Double curve'
    elif classNo == 22:
        return 'Bumpy road'
    elif classNo == 23:
        return 'Slippery Road'
    elif classNo == 24:
        return 'Road narrows on the right'
    elif classNo == 25:
        return 'Road Work'
    elif classNo == 26:
        return 'Traffic signals'
    elif classNo == 27:
        return 'Pedestrians'
    elif classNo == 28:
        return 'Children crossing'
    elif classNo == 29:
        return 'Bicycle crossing'
    elif classNo == 30:
        return 'Beware of ice/snow'
    elif classNo == 31:
        return 'Wild animals crossing'
    elif classNo == 32:
        return 'End of all speed and passing limits'
    elif classNo == 33:
        return 'Turn right ahead'
    elif classNo == 34:
        return 'Turn left ahead'
    elif classNo == 35:
        return 'Ahead only'
    elif classNo == 36:
        return 'Go straight or right'
    elif classNo == 37:
        return 'Go straight or left'
    elif classNo == 38:
        return 'Keep right'
    elif classNo == 39:
        return 'Keep left'
    elif classNo == 40:
        return 'Roundabout mandatory'
    elif classNo == 41:
        return 'End of no passing'
    elif classNo == 42:
        return 'End of no passing by vehicles over 3.5 metric tons'


def classify(file_path):
    image = cv2.imread(file_path)
    imageResized = cv2.resize(image, (32, 32))
    imageResized = preprocessing(imageResized)
    imageResized = imageResized.reshape(1,32,32,1)
    pred = model.predict(imageResized)
    classIndex = model.predict_classes(imageResized)
    sign = getClassName(classIndex)
    print(sign)


def upload_image():
    try:
        while filedialog.ACTIVE:
            file_path = filedialog.askopenfilename()
            classify(file_path)
    except:
        pass




upload_image()
cv2.waitKey(4000)
