import os
from flask import Blueprint
from flask_restx import Api

from .predict import api as prediction_ns
from .update_model import api as upload_ns
from .reset_model import api as reset_ns

blueprint = Blueprint('api', __name__, url_prefix=os.getenv('SWAGGER_URL', default='/swagger/'))

api = Api(
    blueprint,
    title='Backend API Contract',
    version='1.0',
    description='Backend APIs only accessible by frontend.',
)

api.add_namespace(prediction_ns, path='/dmz')
api.add_namespace(upload_ns, path='/dmz')
api.add_namespace(reset_ns, path='/dmz')