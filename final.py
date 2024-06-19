#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries


# In[2]:


pip install lime


# In[3]:


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'


# In[4]:


from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# In[5]:


from tensorflow import keras


# In[6]:


model = keras.models.load_model("better_classification.h5")


# In[7]:


classes = ["damaged_infrastructure" , "damaged_nature" , "fire" , "flood" , "human_damage" , "non_damage"]


# In[8]:


from tensorflow.keras.preprocessing.image import img_to_array

def classify_image(model, image_path, target_size=(128, 128)):
    try:
        # Open the image file
        img = Image.open(image_path)
        # Resize the image to match the input shape expected by the model
        img = img.resize(target_size)
        # Convert the image to an array
        img_array = img_to_array(img)
        # Normalize the image array
        img_array = img_array / 255.0
        # Expand dimensions to match the input shape (batch_size, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make predictions using the model
        predictions = model.predict(img_array)
        # Get the index of the class with the highest confidence
        predicted_class = np.argmax(predictions, axis=1)
        
        return predicted_class[0]
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


# In[9]:


from tensorflow.keras.preprocessing import image


# In[10]:


def process_image( image_path):
    img = image.load_img(image_path, target_size=(128,128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


# In[11]:


def compute_saliency_map(model, img_array, class_index):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, img_tensor)
    grads = tf.reduce_mean(grads, axis=-1)[0]
    return grads.numpy()


# In[13]:


explainer = lime_image.LimeImageExplainer()


# In[14]:


def predict_fn(images):
    images = np.array(images)
    predictions = model.predict(images)
    return predictions


# In[15]:


def lime(image , ):
    explanation = explainer.explain_instance(
    image[0].astype('double'),  # Convert the image to the format expected by LIME
    predict_fn,
    top_labels=5,
    hide_color=0,
    num_samples=1000
    )
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
    top_label,
    positive_only=True,  # Show only positive regions
    num_features=5,
    hide_rest=False
    )
    # Apply the positive mask and color overlay
    boundary_img = mark_boundaries(image[0], mask, outline_color=(0, 1, 0))  # Green outline for positive regions
    return boundary_img


# In[24]:


def show_lime(image_path):
    img_array = process_image(image_path)
    boundary = lime(img_array)
    # Display the result
    plt.figure(figsize=(4,4))
    plt.imshow(boundary)
    plt.title('Positive Regions , using LIME')
    plt.axis('off')
    plt.show()


# In[34]:


def predict_model(model, image_path):
    x = classify_image(model, image_path, target_size=(128, 128))
    print(classes[x])


# In[26]:


def original_image(image_path):
    img = image.load_img(image_path, target_size=(128,128))
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title('Original Image')
    


# In[28]:


def saliency_map(image_path):
    img_array = process_image(image_path)
    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])
    smap = compute_saliency_map(model, img_array, class_index)
    plt.figure(figsize=(6, 6))
    plt.imshow(smap, cmap='jet')
    plt.colorbar()
    plt.title('Saliency Map')


# In[ ]:




