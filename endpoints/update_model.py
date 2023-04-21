import os
from flask_restx import Namespace, Resource, reqparse
from flask import request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from flask_cors import cross_origin

from utils import load_classifier
from utils.file_validations import allowed_file
from config import RESOURCES

api = Namespace('Update Model Files APIs', description='Update the model files to be used.')

ALLOWED_EXTENSIONS = {'h5'}

upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', type=FileStorage, location='files', required=True)

@api.route('/update_classifier_file')
# @cross_origin()
class predict(Resource):
    @api.expect(upload_parser)
    def post(self):
        if 'file' not in request.files:
            resp = jsonify({'message' : 'No file part in the request'})
            resp.status_code = 400
            return resp
        file = request.files['file']
        if file.filename == '':
            resp = jsonify({'message' : 'No file selected for updating classifier.'})
            resp.status_code = 400
            return resp
        if file.filename == 'classifier.h5':
            resp = jsonify({'message' : """File name can not be 'classifier.h5'. It is already reserved. Please rename and upload the file again"""})
            resp.status_code = 400
            return resp
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
            filename = secure_filename(file.filename)
            file.save(os.path.join(RESOURCES, filename))
            load_classifier.update_classifier(filename)
            file.close()
            resp = jsonify({'message' : 'Classifier updated successfully.'})
            resp.status_code = 200
            return resp
        else:
            resp = jsonify({'message' : f'Allowed file types are {ALLOWED_EXTENSIONS}'})
            resp.status_code = 400
            return resp