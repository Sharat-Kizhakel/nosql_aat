from flask import Flask, jsonify
from config import IP, PORT, DEBUG, DB_UNAME, DB_URI
from neo4j import GraphDatabase
import os

app = Flask(__name__)
graphdb = GraphDatabase.driver(DB_URI,
                              auth=(DB_UNAME, os.environ['DB_PSWD']))
session=graphdb.session()
query="LOAD CSV FROM 'file:///covid_data.csv' AS line CREATE (:PATIENT {Province:line[0],Country:line[1],Lat:line[2],Long:line[3],Date:line[4],Confirmed:line[5],Deaths:line[6],Recovered:line[7],Active:line[8],WHO_Region:line[9]})" 
print(session.run(query))

@app.get('/')
def home():
    return jsonify(
        dict(
            data='Testing stuff'
        )
    )

if __name__=='__main__':
    app.run(host=IP, port=PORT, debug=DEBUG)