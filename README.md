<H3>ENTER YOUR NAME:RAMYA P</H3>
<H3>ENTER YOUR REGISTER NO:212223230168</H3>
<H3>EX. NO.1</H3>
<H3>DATE:28-1-26</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
~~~
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df= pd.read_csv("Churn_Modelling.csv")
print(df)

X=df.iloc[:,:-1].values
print(X)

y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

print(df.isnull().sum())
y=df.iloc[:,-1].values
print(y)

df.duplicated()
print(df['EstimatedSalary'].describe())

scaler=MinMaxScaler()
df1 = pd.DataFrame(
    scaler.fit_transform(df.select_dtypes(include='number')),
    columns=df.select_dtypes(include='number').columns
)
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

print(X_train)
print(len(X_train))
print(X_test)
print("Lenght of X_test ",len(X_test))
~~~


## OUTPUT:

<img width="1450" height="767" alt="image" src="https://github.com/user-attachments/assets/7e35a730-2e09-49bc-a61c-b6f883a0c9e6" />
<img width="1312" height="328" alt="image" src="https://github.com/user-attachments/assets/8bf154ba-1546-478f-a9ce-9831666c0528" />

## X VALUES:

<img width="1057" height="237" alt="image" src="https://github.com/user-attachments/assets/9b36ad17-092f-46f1-8e05-feebcb7d2d27" />

## Y VALUES:

<img width="1077" height="121" alt="image" src="https://github.com/user-attachments/assets/98c8b26d-e202-4013-a510-25b8623075d3" />

## NULL VALUES:

<img width="1277" height="402" alt="image" src="https://github.com/user-attachments/assets/e4cc14b2-e16b-49ff-9999-722164b9b89e" />

<img width="1312" height="432" alt="image" src="https://github.com/user-attachments/assets/6d0d9838-d39e-45e1-89e5-6d84eb6450b6" />

## DUPLICATED VALUES:

<img width="1305" height="323" alt="image" src="https://github.com/user-attachments/assets/67bdfdf3-3cf6-4dfe-9533-29eb9ef1985a" />

## DESCRIPTION:

<img width="1231" height="262" alt="image" src="https://github.com/user-attachments/assets/4e6f2873-e689-49ee-9ded-05ce4bbe06fb" />

## TRAINING DATA:

<img width="1375" height="676" alt="image" src="https://github.com/user-attachments/assets/3e498849-b76b-4ecb-92ed-cb688efe85f8" />

## TESTING DATA:

<img width="1300" height="496" alt="image" src="https://github.com/user-attachments/assets/09577525-d52e-4ded-ba65-f8fe19f1769b" />













## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


