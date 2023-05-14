import streamlit as st
import pandas as pd
import datetime
import numpy as np
import model
from PIL import Image

st.write(
    '''
    # Simple Weather Prediction App.
    This web application predicts **weather type**.
    '''
)



st.sidebar.header("User Input Parameters")

def user_input_features():
    date = st.sidebar.date_input('Date to predict')
    min_temp = st.sidebar.slider('Minimum temperature ', -50.0, 20.0, 0.5)
    max_temp = st.sidebar.slider('Maximum temperature', -2.0, 40.0, 0.5)
    precipitation = st.sidebar.slider('precipitation', 0.0, 56.0, 0.5)
    wind = st.sidebar.slider('wind speed', 0.0, 10.0, 0.2)
    data={'precipitation':[precipitation],
          'temp_max':[max_temp],
          'temp_min':[min_temp],
          'wind':[wind],
          'date':[date]
    }
    features=pd.DataFrame(data,index=[0])
    return features

data=user_input_features()

st.write('### Input Features : ',data)

weatherModel=model.WeatherPredictionModel()


weatherState=str(weatherModel.predict(data)[0])

st.write('\n ## Predict weather:')
if weatherState=='rain':
    image = Image.open('C:/Users/INTEL/Desktop/DataCamp/WeatherPrediction/images/rain.png')
    st.image(image, caption='Weather will be rainy')
elif weatherState=='snow':
    image = Image.open('C:/Users/INTEL/Desktop/DataCamp/WeatherPrediction/images/snow.png')
    st.image(image, caption='Weather will be snow')
elif  weatherState == 'sun':
    image = Image.open('C:/Users/INTEL/Desktop/DataCamp/WeatherPrediction/images/sun.png')
    st.image(image, caption='Weather will be sunny')
elif  weatherState == 'fog':
    image = Image.open('C:/Users/INTEL/Desktop/DataCamp/WeatherPrediction/images/fog.png')
    st.image(image, caption='Weather will be fog')
elif  weatherState == 'drizzle':
    image = Image.open('C:/Users/INTEL/Desktop/DataCamp/WeatherPrediction/images/drizzle.png')
    st.image(image, caption='Weather will be drizzle')



