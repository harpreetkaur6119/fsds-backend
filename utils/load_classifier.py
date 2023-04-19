import os
from tensorflow import float32, Variable, reduce_mean, abs
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda, Dropout, Activation, Dense
from tensorflow.keras.models import load_model as lm
from tensorflow.keras import Model
from tensorflow import Variable, float32
import json

from config import RESOURCES

classifier = None

gm_exp = Variable(3., dtype=float32)

def load_model(file_name) -> Model:

    gm_exp = Variable(3., dtype=float32)
    loaded_model =lm(os.path.join(RESOURCES,file_name), 
                   custom_objects={
                    'gm_exp': gm_exp
                    }
                    )
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # loaded_model.load_weights(os.path.join(RESOURCES,file_name))
    print("Loaded model from disk")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return loaded_model

def get_classifier() -> Model:
    global classifier
    if classifier is None:
        classifier = load_model('classifier.h5')
    return classifier

def update_classifier(file_name):
    global classifier
    classifier = load_model(file_name)
    return classifier

def reset_classifier():
    global classifier
    classifier = load_model('classifier.h5')
    return classifier