import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("./model/tomato_disease_model.h5")

# Class names
class_names = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Prediction function
def predict(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)[0]

    result_dict = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    predicted_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    return predicted_class, result_dict

# Sample image path (replace with your own sample)
examples = [
    ["test_images/test0.jpg"],
    ["test_images/test_1.jpg"],
    ["test_images/test2.jpg"],
    ["test_images/test3.jpg"],
    ["test_images/test4.jpg"],
    ["test_images/test5.jpg"]
]

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Tomato Leaf Image"),
    outputs=[
        gr.Label(num_top_classes=3, label="Top Prediction"),
        gr.Label(label="Confidence Scores")
    ],
    title="🍅 Tomato Disease Classifier",
    description="""
    Upload a tomato leaf image to detect diseases using a deep learning model.
    This app can identify 10 tomato diseases including Yellow Leaf Curl Virus, Mosaic Virus, and Bacterial Spot.
    """,
    theme="soft",
    examples=examples,
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()
