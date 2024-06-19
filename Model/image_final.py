import os
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras
import asyncio

model_path = "better_classification.h5"

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

class ImageModel:
    def __init__(self, model_path="better_classification.h5"):
        self.classes = ["damaged_infrastructure", "damaged_nature", "fire", "flood", "human_damage", "non_damage"]
        self.model = keras.models.load_model(model_path)
        self.explainer = lime_image.LimeImageExplainer()

    def classify_image(self, image_path, target_size=(128, 128)):
        try:
            img = Image.open(image_path)
            img = img.resize(target_size)
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            
            return predicted_class
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def process_image(self, image_path, target_size=(128, 128)):
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    def compute_saliency_map(self, img_array, class_index):
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            preds = self.model(img_tensor)
            loss = preds[:, class_index]

        grads = tape.gradient(loss, img_tensor)
        grads = tf.reduce_mean(grads, axis=-1)[0]
        return grads.numpy()

    def lime(self, img_array):
        explanation = self.explainer.explain_instance(
            img_array[0].astype('double'),
            self.predict_fn,
            top_labels=5,
            hide_color=0,
            num_samples=1000
        )
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            top_label,
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        boundary_img = mark_boundaries(img_array[0], mask, outline_color=(0, 1, 0))
        return boundary_img

    def predict_fn(self, images):
        images = np.array(images)
        predictions = self.model.predict(images)
        return predictions

    def show_lime(self, image_path):
        img_array = self.process_image(image_path)
        boundary = self.lime(img_array)
        
        # Plotting LIME figure
        plt.figure(figsize=(6, 6))
        plt.imshow(boundary)
        plt.title('LIME')
        plt.axis('off')
        
        # Convert plot to PNG image in memory
        image_buf = BytesIO()
        plt.savefig(image_buf, format='png')
        image_buf.seek(0)
        
        plt.close()  # Close the plot to free up memory
        
        return image_buf
        

    def original_image(self, image_path):
        img = image.load_img(image_path, target_size=(128, 128))
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        plt.show()

    def saliency_map(self, image_path):
        img_array = self.process_image(image_path)
        preds = self.model.predict(img_array)
        class_index = np.argmax(preds[0])
        smap = self.compute_saliency_map(img_array, class_index)
        plt.figure(figsize=(6, 6))
        plt.imshow(smap, cmap='jet')
        plt.colorbar()
        plt.title('Saliency Map')
        plt.axis('off')
        plt.show()

    def show(self, image_path):
        self.original_image(image_path)
        img_array = self.process_image(image_path)
        boundary = self.lime(img_array)
        preds = self.model.predict(img_array)
        class_index = np.argmax(preds[0])
        smap = self.compute_saliency_map(img_array, class_index)
        plt.figure(figsize=(6, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(boundary)
        plt.title('LIME')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(smap, cmap='jet')
        plt.colorbar()
        plt.title('Saliency Map')
        plt.axis('off')
        plt.show()

    def predict_model(self, image_path):
        predicted_class = self.classify_image(image_path, target_size=(128, 128))
        if predicted_class is not None:
            return self.classes[predicted_class]
        else:
            print("Failed to predict the class.")
            return "failed"



# Example usage:
#if __name__ == "__main__":
#    model_path = "better_classification.h5"
#    image_path = '../images/images_11.jpeg'
#    img_model = ImageModel(model_path)
#    img_model.show_lime(image_path)
#    x= img_model.predict_model(image_path)
#    print(x)
    #img_model.show(image_path)
    #img_model.saliency_map(image_path)
    #img_model.original_image(image_path)
    
    
