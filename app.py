from flask import Flask

from config import load_env
from endpoints import blueprint

app = Flask(__name__)
app.register_blueprint(blueprint)

if __name__ == "__main__":
    load_env()
    app.run()
