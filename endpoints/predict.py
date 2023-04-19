import os
import numpy as np
import cv2
from flask import request, jsonify
from flask_restx import Namespace, Resource, reqparse
from utils.load_classifier import get_classifier
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
# from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import Model
from tensorflow import convert_to_tensor, float32, expand_dims

from config import UPLOAD_FOLDER
from utils.file_validations import allowed_file

api = Namespace('Prediction APIs', description='Predict and classify the object')

classifier:Model = get_classifier()

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

CATEGORIES = ['Benign', 'Malignant']

upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', type=FileStorage, location='files', required=True)

@api.route('/prediction')
class predict(Resource):
    @api.expect(upload_parser)
    def post(self):
        if 'file' not in request.files:
            resp = jsonify({'message' : 'No file part in the request'})
            resp.status_code = 400
            return resp
        file = request.files['file']
        if file.filename == '':
            resp = jsonify({'message' : 'No file selected for prediction'})
            resp.status_code = 400
            return resp
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            file.close()
            img=cv2.imread(file_path)
            img = cv2.resize(img, (299,299), interpolation = cv2.INTER_AREA)
            cv2.imshow('abc', img)
            cv2.waitKey(5000)
            imgData = convert_to_tensor(img, dtype=float32)
            imgData = expand_dims(imgData , 0)
            os.remove(file_path)
            prediction = classifier.predict(imgData)
            predicted_class = CATEGORIES[np.argmax(prediction[0])]
            resp = jsonify({
                'predicted_class' : f'{predicted_class}',
                'prediction' : f'{prediction[0]}',
                'file_name' : filename,
                'categories' : CATEGORIES
                })
            resp.status_code = 200
            return resp
        else:
            resp = jsonify({'message' : f'Allowed file types are {ALLOWED_EXTENSIONS}'})
            resp.status_code = 400
            return resp
    