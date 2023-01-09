from flask import Flask, jsonify, render_template
from config import IP, PORT, DEBUG, DB_UNAME, DB_URI
from neo4j import GraphDatabase
import os

app = Flask(__name__)
graphdb = GraphDatabase.driver(DB_URI, auth=(DB_UNAME, os.environ['DB_PSWD']))
session=graphdb.session()

@app.get('/')
def home():
    return render_template('home.html')

@app.get('/data')
def data():
    return jsonify(
        dict(
            data='Testing stuff'
        )
    )

if __name__=='__main__':
    app.run(host=IP, port=PORT, debug=DEBUG)