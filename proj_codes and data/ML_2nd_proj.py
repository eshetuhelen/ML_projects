#!/usr/bin/env python
# coding: utf-8

# In[36]:


#TASK 1 is importing important libraries

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[37]:


#TASK 2 is Data Collection and processing
#importing the dataset
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()


# In[38]:


print(breast_cancer_dataset)


# In[39]:


# Loadint the data to the dataframe
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)


# In[40]:


# printing the first 5 rows of the dataframe
data_frame.head()


# In[41]:


# Adding the 'target' column to the dataframe
data_frame['label'] = breast_cancer_dataset.target


# In[42]:


#Print the last 5 rows of the dataframe
data_frame.tail()


# In[43]:


# Number of rows and columns in the dataset
data_frame.shape


# In[44]:


# Getting some info about the data
data_frame.info()


# In[45]:


#Checking for missing values
data_frame.isnull().sum()


# In[46]:


#Statstical measures of the data
data_frame.describe()


# In[47]:


#Checking the distributons
data_frame['label'].value_counts()


# In[48]:


data_frame.groupby('label').mean()


# In[49]:


#Separating the data and the label
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']


# In[50]:


print(X)


# In[51]:


print(Y)


# In[52]:


#Spliting the data in to training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)


# In[53]:


print(X.shape, X_train.shape, X_test.shape)


# In[54]:


#MODEL TRAINING
#LOGISTIC REGRESSION
model = LogisticRegression()


# In[55]:


model.fit(X_train, Y_train)


# In[56]:


#MODEL EVALUATION
#AAccuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)


# In[57]:


print('Accuracy on training data = ', training_data_accuracy)


# In[58]:


#Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)


# In[59]:


print('Accuracy on test data = ', test_data_accuracy)


# In[60]:


#BUILDING A PREDICTIVE SYSTEM
input_data=(17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)


# In[61]:


# Change the input data in to numpy array
input_data_as_numpy_array = np.asarray(input_data)


# In[62]:


#Reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# In[63]:


prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast Cancer is Benign')


# In[29]:


# Save the trained data
import pickle


# In[30]:


file_name = 'trained_model.sav'
pickle.dump(model, open(file_name,'wb'))


# In[31]:


# Loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# In[32]:


#BUILDING A PREDICTIVE SYSTEM
input_data=(17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)


# In[33]:


# Change the input data in to numpy array
input_data_as_numpy_array = np.asarray(input_data)


# In[34]:


#Reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# In[35]:


prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast Cancer is Benign')


# In[ ]:





# In[ ]:




