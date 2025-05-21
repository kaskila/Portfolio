---
title: Tomato & Maize Disease Classifier 🌾
emoji: 🍅
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
---

# 🌾 Intelligent Crop Disease Detection System

Welcome to the **Tomato & Maize Disease Detection System** powered by TensorFlow and Gradio — developed by **We Speak Data & AI Foundation**.

This app uses deep learning models to detect and classify common tomato and maize leaf diseases from uploaded images. The system also provides confidence scores, likely causes of the disease, and treatment recommendations.

---

## 🧠 How It Works

1. Upload a photo of a **tomato** or **maize** leaf.
2. The model identifies the crop and classifies the image into one of the disease categories.
3. You'll get:
   - The **top prediction**
   - **Confidence score**
   - A **table with the top 3 predictions**
   - The **possible cause**
   - Suggested **treatment**

---

## 🖼️ Sample Images

To try the app quickly, you can click on any of the example images included from the `test_images/` folder.

---

## 📦 Model Details

- **Tomato Model**: `tomato_disease_model.h5`  
  - Trained on the PlantVillage Tomato Dataset
- **Maize Model**: `maize_disease_model.h5`  
  - Trained on the PlantVillage Maize Dataset

Both models are Convolutional Neural Networks (CNNs) optimized for leaf disease classification.

---

## ⚙️ Tech Stack

- 🧠 TensorFlow for model inference  
- 🎨 Gradio for the interactive web interface  
- 🖼️ Pillow for image preprocessing  
- 🐍 Python

---

## 🚀 Try It Out

- Upload your own **tomato** or **maize** leaf image  
- Or pick one of the sample images  
- Get instant diagnosis and treatment tips

---

## 🤝 About the Foundation

**We Speak Data & AI Foundation** is a Zambian initiative focused on creating impact through data science and artificial intelligence — especially among youths and professionals.

---

## 📬 Contact

For questions, feedback, or collaboration:

**Kasikila Isaac**  
Founder @ *We Speak Data & AI Foundation*  
📧 kasikilaisaac24@gmail.com  
🌐 [linkedin.com/in/kasikila-isaac/](https://linkedin.com/in/kasikila-isaac/)

---
