#!/usr/bin/env python
# coding: utf-8

# In[1]:


classes = ["damaged_infra" , "damaged_nature" , "fire" , "flood" , "human_damage" , "non_damage"]


# In[2]:


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


# In[3]:


import pandas as pd

df = pd.read_csv("text_data.csv")


# In[4]:


df.groupby('CLASSES').describe()


# In[5]:


df = df[df['CLASSES'] != "yemencrisis_2015-04-22_17-39-59.txt"]


# In[6]:


df.groupby('CLASSES').describe()


# In[7]:


df_non_damage = df[df['CLASSES'] == "non_damage"]


# In[8]:


df_non_damage = df_non_damage.sample(1400)


# In[9]:


df = df[df['CLASSES'] != "non_damage"]


# In[10]:


df_balanced = pd.concat([df , df_non_damage])


# In[11]:


df_balanced.groupby('CLASSES').describe()


# In[12]:


df_balanced.shape


# In[13]:


df_balanced


# In[14]:


number = []
for i in range(1390):
    number.append(0)
for i in range(465):
    number.append(1)
for i in range(388):
    number.append(2)
for i in range(385):
    number.append(3)
for i in range(239):
    number.append(4)
for i in range(1400):
    number.append(5)


# In[15]:


number[1389]


# In[16]:


df_balanced['NUMBER'] = number


# In[17]:


df_balanced.sample(1390)


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


x_train , x_test , y_train , y_test = train_test_split(df_balanced['CAPTION'] , df_balanced['NUMBER'] , stratify=df_balanced['NUMBER'])


# In[20]:


bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")


# In[21]:


def get_sentence_embeding(sentences):
    preprocessed_text = bert_preprocess(sentences)
    return bert_encoder(preprocessed_text)['pooled_output']

get_sentence_embeding(["my name is rohit" , "my name is rahul"])


# In[22]:


from sklearn.metrics.pairwise import cosine_similarity


# In[23]:


x_train_array = get_sentence_embeding(x_train[:3200])


# In[24]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense


# In[25]:


x_test_array = get_sentence_embeding(x_test[:1067])


# In[26]:


input_layer = Input(shape=(768,))
Layer_1 = Dense(64, activation="relu")(input_layer)
Layer_2 = Dense(64, activation="relu")(Layer_1)
output_layer= Dense(6, activation="softmax")(Layer_2)
##Defining the model by specifying the input and output layers
model = Model(inputs=input_layer, outputs=output_layer)
## defining the optimiser and loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
             metrics = ['accuracy'])
## training the model
model.fit(x_train_array, y_train,epochs=200)


# In[27]:


model.evaluate(x_test_array , y_test)


# In[28]:


def predict(sentence):
    t = get_sentence_embeding([sentence])
    prediction = model.predict(t)
    predicted_class = np.argmax(prediction, axis=1)
    return classes[predicted_class[0]]

    


# In[29]:


import numpy as np


# In[30]:


predict("Goil filling station on fire #hmm #accrafloods")


# In[31]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[32]:


y_pred_prob = model.predict(x_test_array)


# In[33]:


y_pred = np.argmax(y_pred_prob, axis=1)
conf_matrix = confusion_matrix(y_test, y_pred)


# In[34]:


import matplotlib.pyplot as plt


# In[35]:


disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.show()

