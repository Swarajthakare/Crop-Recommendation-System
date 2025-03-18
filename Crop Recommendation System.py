#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing Data

# In[2]:


crop = pd.read_csv(r'B:\2023 data set project march\Crop_recommendation.csv')
crop.shape


# In[3]:


crop.head()


# ## six Quesion 

# In[4]:


crop.shape


# In[5]:


crop.info()


# In[6]:


crop.isnull().sum()


# In[7]:


crop.duplicated().sum()


# In[8]:


crop.describe()


# ## Exploring Data

# In[9]:


correlation = crop.corr()
correlation


# In[10]:


sns.heatmap(correlation, annot= True, cbar=True, cmap='coolwarm')


# In[11]:


crop['label'].value_counts()


# In[12]:


sns.distplot(crop['N'])
plt.show()


# ## Encoding

# In[13]:


crop_dict = {
"rice": 1,
"maize" : 2,
"jute" :  3,
"cotton" :   4,
"coconut":   5,
"papaya" :   6,
"orange" :   7,
"apple"  :   8,
"muskmelon"  :   9,
"watermelon" :   10,
"grapes" :   11,
"mango"  :   12,
"banana" :   13,
"pomegranate":   14,
"lentil"     :   15,
"blackgram"  :   16,
"mungbean"   :   17,
"mothbeans"  :   18,
"pigeonpeas" :   19,
"kidneybeans":   20,
"chickpea"   :   21,
"coffee"     :   22
}
crop['crop_num'] = crop['label'].map(crop_dict)


# In[14]:


crop['crop_num'].value_counts()


# In[15]:


crop.head()


# In[16]:


crop.drop('label',axis = 1,inplace = True)


# In[17]:


crop.head()


# ## Train Test Split

# In[18]:


X = crop.drop('crop_num',axis = 1)
y = crop['crop_num']


# In[19]:


X.shape


# In[20]:


y.shape


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


X_train.shape


# In[24]:


X_test.shape


# ## Scale the features using MinMaxScaler

# In[25]:


X_train


# In[26]:


from sklearn.preprocessing import MinMaxScaler


# In[27]:


ms = MinMaxScaler()


# In[28]:


ms.fit(X_train)
X_train = ms.transform(X_train)
X_test = ms.transform(X_test)


# In[29]:


X_train


# ## Standarization

# In[30]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


# In[31]:


X_train


# ## Training Models

# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


# In[33]:


# create instances of all models
models = {
    'Logistic Regression' : LogisticRegression(),
    'Naive Bayes' : GaussianNB(),
    'Support Vector Machine' : SVC(),
    'K-Nearest Neighbors' : KNeighborsClassifier(),
    'Decision Tree' : DecisionTreeClassifier(),
    'Random Forest' : RandomForestClassifier(),
    'Bagging' : BaggingClassifier(),
    'AdaBoost' : AdaBoostClassifier(),
    'Gradient Boosting' : GradientBoostingClassifier(),
    'Extra Trees' : ExtraTreeClassifier(),
}


for name, md in models.items():
    md.fit(X_train,y_train)
    ypred = md.predict(X_test)
    
    print(f"{name} with accuracy : {accuracy_score(y_test,ypred)}")


# In[34]:


rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
ypred = rfc.predict(X_test)
accuracy_score(y_test,ypred)


# ## Predictive System

# In[35]:


def recommendation(N,P,K,temperature,humidity,ph,rainfall):
    features = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    prediction = rfc.predict(features).reshape(1,-1)
    
    return prediction[0]


# In[36]:


N = 80
P = 120
K = 100
temperature = 40
humidity = 50
ph = 30
rainfall = 120

predict = recommendation(N,P,K,temperature,humidity,ph,rainfall)

crop_dict = {1:"rice", 2:"maize", 3:"jute", 4:"cotton", 5:"coconut",6:"papaya" ,7:"orange", 
             8:"apple", 9:"muskmelon", 10:"watermelon", 11:"grapes", 12:"mango", 13:"banana", 
             14:"pomegranate", 15:"lentil", 16:"blackgram", 17:"mungbean", 18:"mothbeans",
             19:"pigeonpeas", 20:"kidneybeans", 21:"chickpea", 22:"coffee" }

if predict[0] in crop_dict:
    crop = crop_dict[predict[0]]
    print('{} is a best crop to be cultivated'.format(crop))
else:
    print("sorry are not able to recommend a proper crop for this environment")


# ## Model making 

# In[37]:


import pickle
pickle.dump(rfc,open('model.pkl','wb'))


# In[ ]:




