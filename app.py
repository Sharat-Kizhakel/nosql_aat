from flask import Flask, jsonify
from config import IP, PORT, DEBUG

app = Flask(__name__)

@app.get('/')
def home():
    return jsonify(
        dict(
            data='Testing stuff'
        )
    )

if __name__=='__main__':
    app.run(host=IP, port=PORT, debug=DEBUG)