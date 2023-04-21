from flask import Flask
from flask_cors import CORS

from config import load_env
from endpoints import blueprint

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.register_blueprint(blueprint)


if __name__ == "__main__":
    load_env()
    app.run(host='0.0.0.0')
