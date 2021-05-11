import streamlit as st
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("mid_term.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('Dataset.csv')


# Extracting dependent and independent variables:
# Extracting independent variable:
X = dataset.iloc[:, :-1].values
# Extracting dependent variable:
y = dataset.iloc[:, -1].values

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.
imputer = imputer.fit(X[:, :])
#Replacing missing data with the calculated mean value
X[:, :]= imputer.transform(X[:,:])

y=y.reshape(-1,1)
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y[:, 0] = labelencoder_y.fit_transform(y[:, 0])
y=y.astype('int')

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


def predict_note_authentication(meanfreq,sd,median,iqr,skew,kurt,mode,centroid,dfrange):

  output= model.predict(sc.transform([[meanfreq,sd,median,iqr,skew,kurt,mode,centroid,dfrange]]))
  print("Person will ",output)
  if output==[0]:
    prediction="Female"


  if output==[1]:
    prediction="Male"


  print(prediction)
  return prediction
def main():

    html_temp = """
   <div class="" style="background-color:gray;" >
   <div class="clearfix">
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">MID TERM - I</p></center>
   <center><p style="font-size:30px;color:white;margin-top:10px;">NAME: NITIN TILWANI</p></center>
   <center><p style="font-size:25px;color:white;margin-top:0px;">|| PIET18CS104 || Sec: B || Roll No 48 ||</p></center>
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Person Leaving Prediction")
    meanfreq = st.number_input('Insert mean frequency',0,1)
    sd= st.number_input('Insert SD',0,1)
    median = st.number_input('Insert median',0,1)
    iqr = st.number_input('Insert iqr',0,1)
    skew = st.number_input('Insert skew')
    kurt = st.number_input('Insert kurt')
    mode = st.number_input('Insert mode',)
    centroid = st.number_input('Insert centroid')
    dfrange = st.number_input('Insert dfrange')

    # iqr = st.number_input('Insert SD',0,1)
    # skew = st.number_input('Insert skew')
    # kurt = st.number_input('Insert kurt')
    # mode = st.number_input('Insert mode',0,1)
    # centroid = st.number_input('Insert centroid',0,1)
    # dfrange = st.number_input('Insert dfrange')

    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(meanfreq,sd,median,iqr,skew,kurt,mode,centroid,dfrange)
      st.success('Model has predicted that -> {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by NITIN TILWANI")
      st.subheader("B-Section,PIET")

if __name__=='__main__':
  main()
