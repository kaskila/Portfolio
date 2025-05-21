# Import required libraries
import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import random

# Load pre-trained Keras models for tomato and maize leaf disease classification
tomato_model = tf.keras.models.load_model("models/tomato_disease_model.h5")
maize_model = tf.keras.models.load_model("models/maize_disease_model.h5")

# Class labels based on the PlantVillage dataset
tomato_classes = [
    "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", "Septoria Leaf Spot",
    "Spider Mites", "Target Spot", "Tomato Mosaic Virus", "Yellow Leaf Curl Virus", "Healthy"
]

maize_classes = [
    "Cercospora Leaf Spot", "Common Rust", "Northern Leaf Blight", "Healthy"
]

# Disease Info: More detailed causes and treatments for tomato diseases
tomato_disease_info = {
    "Bacterial Spot": {"Cause": "Bacterial infection caused by *Xanthomonas* species.",
                       "Treatment": "Use copper-based fungicides, improve drainage, remove affected leaves, and avoid overhead watering."},
    "Early Blight": {"Cause": "Fungal infection caused by *Alternaria solani*, thrives in warm, humid conditions.",
                     "Treatment": "Apply fungicides, remove affected leaves, and rotate crops to reduce soil-borne spores."},
    "Late Blight": {
        "Cause": "Water mold infection caused by *Phytophthora infestans*, thrives in cool, wet conditions.",
        "Treatment": "Use resistant varieties, apply systemic fungicides, and remove infected plants."},
    "Leaf Mold": {"Cause": "Fungal infection caused by *Passalora fulva*, thrives in humid environments.",
                  "Treatment": "Increase air circulation, remove infected leaves, and apply fungicides."},
    "Septoria Leaf Spot": {
        "Cause": "Fungal infection caused by *Septoria lycopersici*, usually appears in the lower leaves.",
        "Treatment": "Prune infected leaves, apply fungicides, and avoid watering from above."},
    "Spider Mites": {"Cause": "Insect infestation by spider mites, which suck sap from leaves, weakening plants.",
                     "Treatment": "Use insecticidal soap or neem oil, increase humidity around plants, and regularly inspect for pests."},
    "Target Spot": {
        "Cause": "Fungal infection, caused by *Corynespora cassiicola*, characterized by round lesions with dark centers.",
        "Treatment": "Use recommended fungicides, practice crop rotation, and improve plant spacing."},
    "Tomato Mosaic Virus": {"Cause": "Viral infection spread by aphids, causing yellowing and stunting of plants.",
                            "Treatment": "Remove infected plants, control aphids, and use virus-free seeds for planting."},
    "Yellow Leaf Curl Virus": {
        "Cause": "Viral infection transmitted by whiteflies, causes yellowing and curling of leaves.",
        "Treatment": "Use virus-free transplants, control whiteflies with insecticides, and remove infected plants."},
    "Healthy": {"Cause": "No disease present.",
                "Treatment": "No treatment needed, continue to provide standard care."}
}

# Disease Info: More detailed causes and treatments for maize diseases
maize_disease_info = {
    "Cercospora Leaf Spot": {
        "Cause": "Fungal infection caused by *Cercospora zeae-maydis*, resulting in dark lesions on leaves.",
        "Treatment": "Apply fungicides, practice crop rotation, and remove infected plant residues."},
    "Common Rust": {
        "Cause": "Fungal infection caused by *Puccinia sorghi*, characterized by reddish-orange pustules on leaves.",
        "Treatment": "Use resistant maize varieties, apply fungicides, and practice crop rotation."},
    "Northern Leaf Blight": {
        "Cause": "Fungal infection caused by *Exserohilum turcicum*, leads to long, dark lesions on leaves.",
        "Treatment": "Use resistant varieties, apply fungicides, and practice crop rotation."},
    "Healthy": {"Cause": "No disease present.",
                "Treatment": "No treatment needed, maintain optimal growing conditions."}
}


# Main prediction function: takes crop type and image input, returns prediction info
def predict_disease(crop, image):
    # Choose model and label set based on selected crop
    if crop == "Tomato":
        model = tomato_model
        class_names = tomato_classes
        info = tomato_disease_info
    elif crop == "Maize":
        model = maize_model
        class_names = maize_classes
        info = maize_disease_info
    else:
        return "Invalid crop selection", 0, "N/A", "N/A", []

    # Preprocess image: resize and normalize
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Make prediction and extract top prediction info
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    confidence = float(prediction[predicted_index]) * 100
    predicted_label = class_names[predicted_index]
    cause = info[predicted_label]["Cause"]
    treatment = info[predicted_label]["Treatment"]

    # Get top 3 predictions with confidence
    top3_indices = prediction.argsort()[-3:][::-1]
    top3_labels = [(class_names[i], f"{prediction[i] * 100:.2f}%") for i in top3_indices]

    return predicted_label, f"{confidence:.2f}%", cause, treatment, top3_labels


# Clear all UI inputs and outputs
def clear_all():
    return "", None, "", "", [], "Tomato"


# Load random image examples from local folders for UI display
def get_random_examples():
    tomato_files = [f for f in os.listdir("tomato_test_images") if f.lower().endswith(".jpg")]
    maize_files = [f for f in os.listdir("maize_test_images") if f.lower().endswith(".jpg")]
    tomato_samples = random.sample(tomato_files, 2)
    maize_samples = random.sample(maize_files, 2)

    # Create list of [crop, file path] pairs for Gradio examples
    examples = [[
        "Tomato", os.path.join("tomato_test_images", f)
    ] for f in tomato_samples] + [[
        "Maize", os.path.join("maize_test_images", f)
    ] for f in maize_samples]

    return examples


# Define the Gradio UI interface
with gr.Blocks(theme=gr.themes.Soft()) as app:
    # Title and subtitle
    gr.Markdown("## 🌾 Intelligent Crop Disease Detection System")
    gr.Markdown("Smart farming made simple. Detect diseases in tomato and maize crops by uploading a leaf image. Get instant diagnosis and expert treatment advice.")
    gr.Markdown("👨‍🔬 Developed by Kasikila Isaac @ **We Speak Data & AI Foundation**")

    with gr.Row():
        with gr.Column(scale=1):
            # Input widgets
            crop_selector = gr.Dropdown(choices=["Tomato", "Maize"], label="Select Crop Type", value="Tomato")
            image_input = gr.Image(type="pil", label="Upload Leaf Image")
            predict_button = gr.Button("Predict")
            clear_button = gr.Button("Clear")

            # Random examples section
            examples = gr.Examples(
                examples=get_random_examples(),
                inputs=[crop_selector, image_input],
                label="Sample Images"
            )

        with gr.Column(scale=2):
            # Output widgets
            output_label = gr.Textbox(label="Predicted Disease")
            confidence_output = gr.Textbox(label="Confidence")
            cause_output = gr.Textbox(label="Cause")
            treatment_output = gr.Textbox(label="Treatment")
            top3_output = gr.Dataframe(headers=["Disease", "Confidence (%)"], label="Top 3 Predictions", type="array")

    # Bind predict and clear functions to UI buttons
    predict_button.click(fn=predict_disease,
                         inputs=[crop_selector, image_input],
                         outputs=[output_label, confidence_output, cause_output, treatment_output, top3_output])

    clear_button.click(fn=clear_all,
                       inputs=[],
                       outputs=[output_label, image_input, confidence_output, cause_output, treatment_output,
                                crop_selector])

# Run the Gradio app
if __name__ == "__main__":
    app.launch()
