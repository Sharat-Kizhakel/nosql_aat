from config import IP, PORT, DEBUG, DB_UNAME, DB_URI
import app as ap
import json
from neo4j import GraphDatabase
import streamlit as st
import pandas as pd
import csv
import base64
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.svm import SVC
st.title("COVID-19 ANALYSIS:")
st.write("")
st.write("")

#database connection again to run queries
dataset = st.sidebar.file_uploader(
    label="Upload your dataset", type=["csv", "txt"])
 #from app.py file





def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background('covid_19_bg.jpg')

if dataset is not None:
    dataset_df = pd.read_csv(dataset)
    
    # print(dataset_df)
    st.write(dataset_df)
else:
    
    pass
    # st.write(dataset_name)
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
classifier_name = st.sidebar.selectbox(
    "Select metric by country", ("Active cases", "Deaths", "WHO_region"))
st.sidebar.write("")
st.sidebar.write("")
visualization_name = st.sidebar.selectbox(
    "Select Visualization", ("Count Plot", "Heat Map", "Pair Plot"))
if st.sidebar.selectbox("Select Prediction:", ("Predict", "Predict")):
    st.title('COVID-19 ANALYSIS')
    with st.form(key='form1'):
        # getting the input data from the user
        print("Inside form")

        

       

        # creating a button for Prediction
        

        submit_button = st.form_submit_button(label='Covid-19 Diagnosis')

        if submit_button:

            print(submit_button)
            st.write("Checking for chances of cancer:")
            
            # [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry].reshape())
            
            # if pred == 1:
            #     cancer_diagnosis = 'The patient has cancer'
            #     st.success(cancer_diagnosis)
            # else:
            #     cancer_diagnosis = 'The patient does not have cancer'
            #     st.success(cancer_diagnosis)
# else:
#     pass

if dataset is not None:
    def get_dataset(dataset_name):
        # if dataset_name == "data.csv":
        nodes=ap.session1.run(ap.q1)
        # for node in nodes:
        #     print(node)
        print("inside get dataset", dataset)
        df = pd.read_csv(dataset_name)  # cant access relative path files
       
        X = df.iloc[:, 2:31].values
        y = df.iloc[:, 1].values
        return X, y

    print("In get function", dataset.name)
    X, y = get_dataset(dataset.name)
    st.write("Shape of data:", X.shape)
    st.write("")
    

    def visualization_display(visualization_name):

        if visualization_name == "Count Plot":

            # st.title(visualization_name)
            # fig = plt.figure(figsize=(10, 4))
            # sns.countplot(ml2.df['diagnosis'], label="count")

            # st.pyplot(fig)
            pass
        elif visualization_name == "Heat Map":
            # st.title(visualization_name)
            # fig, ax = plt.subplots()
            # sns.heatmap(ml2.df.iloc[:, 1:10].corr(), annot=True, fmt=".0%")
            # st.write(fig)
            pass
        elif visualization_name == "Pair Plot":

            # st.title(visualization_name)
            # fig = sns.pairplot(ml2.df.iloc[:, 1:5], hue="diagnosis")
            # st.pyplot(fig)
            pass
        elif visualization_name == "DTree Confusion Matrix":

           
            pass

    def accuracy_display(classifier_name):

        if classifier_name == "Active cases":
            nodes=ap.session1.run(ap.q3)
            nodes1=nodes.data()
            print("+++++++++++++++++++++++serialzied data+++++++++++++++++")
            print(nodes1)
            print("overrrrrrrrrrrrrrrrrrrrrrrrrrr+++++++++++++++++++++++serialzied data+++++++++++++++++")
            # for node in nodes1:
            #     print(node)
            ap.session1.run(ap.q2)
        elif classifier_name == "Deaths":

            # st.write("Model:", classifier_name)
            # st.write(" ")
            # st.write("Accuracy:", ml2.accuracy_score(
            #     ml2.Y_test, ml2.model[0].predict(ml2.X_test)))
            pass
        
        elif classifier_name == "WHO_region":
            # st.write("Model:", classifier_name)
            # st.write(" ")
            # st.write("Accuracy:", ml2.accuracy_score(
            #     ml2.Y_test, ml2.model[1].predict(ml2.X_test)))
            pass
       
        
    accuracy_display(classifier_name)
    visualization_display(visualization_name)
else:
    # st.warning("You need to upload a csv file first")
    pass
# ml2.ml2.models(ml2.X_train, ml2.Y_train)
