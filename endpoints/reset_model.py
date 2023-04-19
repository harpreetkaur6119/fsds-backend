import os
from flask_restx import Namespace, Resource
from flask import jsonify

from utils import load_classifier

api = Namespace('Reset Model APIs', description='Reset the model to the default.')

@api.route('/reset_classifier')
class predict(Resource):
    def get(self):
        load_classifier.reset_classifier()
        resp = jsonify({'message' : 'Model weights reset to default successfully.'})
        resp.status_code = 200
        return resp