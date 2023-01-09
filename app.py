from flask import Flask, jsonify, render_template
from config import IP, PORT, DEBUG, DB_UNAME, DB_URI
from neo4j import GraphDatabase
import os

app = Flask(__name__)
graphdb = GraphDatabase.driver(DB_URI, auth=(DB_UNAME,"SPK@2000"))
session1=graphdb.session()
q5="LOAD CSV FROM 'file:///covid_19_clean_complete.csv' AS line CREATE (:PATIENT {Province:line[0],Country:line[1],Lat:line[2],Long:line[3],Date:line[4],Confirmed:line[5],Deaths:line[6],Recovered:line[7],Active:line[8],WHO_Region:line[9]})"
nodes=session1.run(q5)

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