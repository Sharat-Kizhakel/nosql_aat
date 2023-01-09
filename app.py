from flask import Flask, jsonify, render_template
from config import IP, PORT, DEBUG, DB_UNAME, DB_URI
from neo4j import GraphDatabase
import os

app = Flask(__name__)
graphdb = GraphDatabase.driver(DB_URI, auth=(DB_UNAME,"SPK@2000"))
session1=graphdb.session()
q1="LOAD CSV FROM 'file:///covid_19_clean_complete.csv' AS line CREATE (:PATIENT {Province:line[0],Country:line[1],Lat:line[2],Long:line[3],Date:line[4],Confirmed:line[5],Deaths:line[6],Recovered:line[7],Active:line[8],WHO_Region:line[9]})"
q2="MATCH(N) DELETE N"
q3='MATCH(n:PATIENT) RETURN n.Country,collect(n.Active)'
# LOAD CSV FROM 'file:///covid_19_clean_complete.csv' AS line CREATE (:PATIENT {country:line[1],Lat:line[2],Long:line[3],Date:line[4],Confirmed:toInteger(line[5]),Deaths:toInteger(line[6]),Recovered:toInteger(line[7]),Active:toInteger(line[8]),WHO_Region:line[9]}), (:COUNTRY{name:line[1]}),(:Province{name:line[0]})






#routes relevance pending


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