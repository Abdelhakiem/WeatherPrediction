import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRFClassifier



class WeatherPredictionModel:
    classifier = XGBRFClassifier()
    def __init__(self):
        # Importing the dataframe:
        self.__df = pd.read_csv("./input/seattle-weather.csv")
        self.__df.dropna(inplace=True)
        self.__df['date'] = pd.to_datetime(self.__df['date'])
        self.__featureEngineering()
        self.__y = self.__df['weather']
        self.__df = self.__df.drop('weather',axis=1)
        self.__x=self.__df[['precipitation','temp_max','temp_min','wind','PC_date']]
        self.__scaler = StandardScaler()
        self.__x = self.__scaler.fit_transform(self.__x)

        self.classifier.fit(self.__x, self.__y)


    def predict(self,pred):
        pred['date'] = pd.to_datetime(pred['date'])
        pred['month'] = pred['date'].dt.month
        pred['day'] = pred['date'].dt.day
        pred['year']=pred['date'].dt.year
        pred['PC_date'] =self.__pca.transform(pred[['year', 'month', 'day']])
        pred.drop(['year', 'month', 'day', 'date'], axis=1,inplace=True)
        pred=pd.DataFrame(self.__scaler.transform(pred[['precipitation','temp_max','temp_min','wind','PC_date']]),columns=['precipitation','temp_max','temp_min','wind','PC_date'])

        res=self.classifier.predict(pred)

        return self.__encoder.classes_[res]
        # return pred

    def __featureEngineering(self):
        self.__encoder=LabelEncoder()
        self.__df['weather'] = self.__encoder.fit_transform(self.__df['weather'])
        self.__df['year'] = self.__df['date'].dt.year
        self.__df['month'] = self.__df['date'].dt.month
        self.__df['day'] = self.__df['date'].dt.day
        self.__pca = PCA(n_components=1)
        self.__df['PC_date'] = self.__pca.fit_transform(self.__df[['year', 'month', 'day']])
        self.__df = self.__df.drop(['year', 'month', 'day', 'date'], axis=1)









