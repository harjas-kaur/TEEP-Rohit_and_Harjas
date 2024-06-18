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


#pip install lime


# In[2]:


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'


# In[3]:


from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# In[4]:


from tensorflow import keras


# In[5]:


model = keras.models.load_model("better_classification.h5")


# In[6]:


from tensorflow.keras.preprocessing.image import img_to_array
# Function that returns the classification in text format
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


# In[7]:


from tensorflow.keras.preprocessing import image
def process_image( image_path):
    img = image.load_img(image_path, target_size=(128,128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def compute_saliency_map(model, img_array, class_index):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, img_tensor)
    grads = tf.reduce_mean(grads, axis=-1)[0]
    return grads.numpy()

classes = ["damaged_infrastructure" , "damaged_nature" , "fire" , "flood" , "human_damage" , "non_damage"]

def plot(image_path , img , saliency_map , boundary_img):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(saliency_map, cmap='jet')
    plt.colorbar()
    plt.title('Saliency Map')
    plt.show()
    predicted_class = classify_image(model, image_path)
    print("Predicted class:", classes[predicted_class])
    plt.figure(figsize=(4,4))
    plt.imshow(boundary_img)
    plt.title('Positive Regions , using LIME')
    plt.axis('off')
    plt.show()


explainer = lime_image.LimeImageExplainer()


# In[17]:


def predict_fn(images):
    images = np.array(images)
    predictions = model.predict(images)
    return predictions


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


# In[22]:


def show_lime(boundary_img):
    # Display the result
    plt.figure(figsize=(4,4))
    plt.imshow(boundary_img)
    plt.title('Positive Regions , using LIME')
    plt.axis('off')
    plt.show()


def show(image_path):
    img = image.load_img(image_path, target_size=(128,128))
    img_array = process_image(image_path)
    boundary = lime(img_array)
    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])
    smap = compute_saliency_map(model, img_array, class_index)
    plot(image_path , img , smap , boundary)
#show(r"download_2.jpeg")

