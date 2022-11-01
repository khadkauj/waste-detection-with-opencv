from flask import Flask, render_template, url_for
import cv2
from cvzone.ClassificationModule import Classifier
app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

    
@app.route('/identify')
def identify():
    cap = cv2.VideoCapture(0)
    classifier = Classifier('Tensorflow-Model/keras_model.h5', 'Tensorflow-Model/labels.txt')

    while True:
        _, img = cap.read()
        prediction = classifier.getPrediction(img)
        print(prediction)
        cv2.imshow('The identified image is', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    app.run(debug=True)