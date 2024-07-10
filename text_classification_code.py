#!/usr/bin/env python
# coding: utf-8

# In[1]:


classes = ["damaged_infra" , "damaged_nature" , "fire" , "flood" , "human_damage" , "non_damage"]
import pandas as pd

df = pd.read_csv("text_data.csv")
df = df[df['CLASSES'] != "yemencrisis_2015-04-22_17-39-59.txt"]
df_non_damage = df[df['CLASSES'] == "non_damage"]
df_non_damage = df_non_damage.sample(1400)
df = df[df['CLASSES'] != "non_damage"]
df_balanced = pd.concat([df , df_non_damage])
df_balanced.groupby('CLASSES').describe()


# In[2]:


df_balanced['label_num'] = df_balanced['CLASSES'].map({
    'damage_infrastructure' : 0, 
    'damaged_nature': 1, 
    'fires': 2, 
    'flood': 3,
    'human_damage': 4,
    'non_damage' : 5
})


# In[3]:


df_balanced


# In[4]:


df1 = pd.read_csv("text_data1.csv")


# In[5]:


df1


# In[6]:


df1.rename(columns={'labeled_num': 'label_num'}, inplace=True)


# In[7]:


df1.loc[df1['label_num'] > 1, 'label_num'] = 3
df1.loc[df1['label_num'] < 1, 'label_num'] = 2


# In[8]:


df1


# In[9]:


df_balanced = pd.concat([df_balanced , df1] , axis = 0)


# In[10]:


df_balanced


# In[11]:


df_balanced.groupby('CLASSES').describe()


# In[12]:


import pandas as pd
import regex as re

# Function to remove emojis and other non-alphanumeric characters
def clean_caption(caption):
    # Remove emojis
    caption = re.sub(r'[^\w\s,]', '', caption)
    return caption

# Apply the function to the dataframe
df_balanced['cleaned_caption'] = df_balanced['CAPTION'].apply(clean_caption)

print(df_balanced)


# In[ ]:





# In[13]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    df_balanced.cleaned_caption, 
    df_balanced.label_num, 
    test_size=0.1, # 20% samples will go to test dataset
    random_state=202,
    stratify=df_balanced.label_num
)


# In[14]:



from sklearn.feature_extraction.text import TfidfVectorizer


# In[15]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

#1. create a pipeline object
clf = Pipeline([
     ('vectorizer_tfidf',TfidfVectorizer()),    
     ('KNN', KNeighborsClassifier())         
])

#2. fit with X_train and y_train
clf.fit(x_train, y_train)


#3. get the predictions for X_test and store it in y_pred
y_pred = clf.predict(x_test)


#4. print the classfication report
print(classification_report(y_test, y_pred))


# In[16]:


import lime
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer()
sample_text = "All these fires around Washington sure do make for some great photos. Thank you to all wildland firefighters. #flyfishing #washington #explore #forrestfire"
exp = explainer.explain_instance(sample_text, clf.predict_proba, num_features=10)


# In[17]:


exp.show_in_notebook()


# In[18]:


import numpy as np


# In[19]:


def text_prediction(sentence):
    max = np.argmax(clf.predict_proba([sentence]))
    print(classes[max])
    


# In[20]:


def lime_explainability(sentence):
    exp = explainer.explain_instance(sentence , clf.predict_proba , num_features = 5)
    print(exp.show_in_notebook())
    


# In[21]:


lime_explainability("All these fires around Washington sure do make for some great photos. Thank you to all wildland firefighters. #flyfishing #washington #explore #forrestfire")


# In[22]:


y_predict = clf.predict(x_test)


# In[23]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)


# In[24]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[25]:


pip install joblib


# In[26]:


import joblib
joblib.dump(clf , 'F_text_model.pkl')

