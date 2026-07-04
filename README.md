Challenge Solution
Limited data Used transfer learning with ImageNet‑pretrained ResNet50
Background bias Removed backgrounds and replaced them with 10 different colors for each image
Multiple objects per image Used multi‑label classification with BCEWithLogitsLoss
Large model size Converted to ONNX (~100 MB) for efficient deployment
Deployment memory limit (512 MB) Removed heavy dependencies (torchvision) and used manual preprocessing

---

📈 Future Improvements

· Add more classes (e.g., from ImageNet)
· Train on a larger dataset
· Use YOLO for object detection with bounding boxes
· Add video support for real‑time detection

---

📄 License

MIT

---

🤝 Connect with Me

· GitHub: https://github.com/hatamzadeh86
· LinkedIn: https://www.linkedin.com/in/amir-mohammad-hatemzadeh-44b2a138b/?lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3BvULhas3VTauzoP7WztIUTA%3D%3D
· Email: activedirectoryn@gmail.com

---

⭐ If you find this project useful, please give it a star on GitHub!

---

This README clearly explains:

1. ✅ Why background removal was done
2. ✅ Why 10 different backgrounds were added
3. ✅ The full pipeline from data to deployment
4. ✅ Results and metrics
5. ✅ How to run the project

Let me know if you want any changes! 🚀

Here's a professional English README for your GitHub repository that highlights all the work you've done:

---

🧠 Multi‑Label Object Detection with ResNet50 & ONNX

An end‑to‑end computer vision pipeline for detecting multiple objects in a single image – from dataset preparation to deployment on Render.

---

📌 Overview

This project implements a multi‑label classification system that can detect 21 different object classes (aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, neutral, person, pottedplant, sheep, sofa, train, tvmonitor) in a single image.

The pipeline includes:

· Dataset augmentation with background removal and replacement
· Fine‑tuning ResNet50 for multi‑label classification
· Conversion to ONNX for fast and lightweight inference
· Deployment as a web app using Gradio + Render

---

🎯 Key Features

1. Background Removal & Augmentation

One of the biggest challenges in computer vision is background bias – models often learn the background instead of the object itself.

To solve this, we:

· Removed the background from every image using rembg (a deep learning-based segmentation model).
· Replaced it with 10 different solid colors (white, black, gray, blue, red, green, yellow, orange, pink, cyan) for each original image.
· This forced the model to focus on the object's shape and texture, not the background.

Result: The model generalizes much better to real‑world images (e.g., photos from Google or a phone camera) because it has seen the same object in many different background contexts.

2. Multi‑Label Classification

Unlike standard classification where each image has only one label, this model can predict multiple labels for a single image (e.g., a person + car + dog in one photo).

3. ONNX Conversion

The trained PyTorch model was converted to ONNX format, which:

· Reduces model size (~100 MB)
· Speeds up inference
· Makes deployment easier across different platforms

4. Web Deployment

The ONNX model is served through a Gradio web app deployed on Render (free tier). Users can upload an image and get real‑time predictions with confidence scores.

---

📊 Dataset

· Source: Pascal VOC 2012 (21 classes)
· Total images: ~29,130
· Augmentation: Each image was augmented with 10 different backgrounds (see above)
· Train/Val/Test split: 80/10/10

---

🧠 Model Architecture

· Base model: ResNet50 (pretrained on ImageNet)
· Task: Multi‑label classification
· Loss function: BCEWithLogitsLoss
· Optimizer: Adam (lr=0.0001)
· Epochs: 30 (with early stopping)
· Final metrics:
  · F1‑Score: 0.9947
  · Exact Match: 0.9787
  · Loss: 0.0093

---

🚀 How to Use

Option 1: Try the live demo (if deployed)

👉 Live Demo on Render

Option 2: Run locally

1. Clone the repository:
  
   git clone https://github.com/hatamzadeh86/resnet50_multilabel_detect_all.git
   cd resnet50_multilabel_detect_all
   
2. Install dependencies:
  
   pip install -r requirements.txt
   
3. Download the ONNX model from the Releases section.
4. Run the Gradio app:
  
   python app.py
   
5. Open your browser and go to http://127.0.0.1:7860

---

📁 Project Structure

.
├── app.py                      # Gradio web application
├── requirements.txt            # Python dependencies
├── classes.txt                 # List of 21 class names
├── resnet50_multilabel.onnx    # Trained ONNX model (download from Releases)
├── README.md                   # This file
└── .gitignore
---

🔧 Technologies Used

Tool Purpose
PyTorch Model training & fine‑tuning
ONNX / ONNX Runtime Model conversion & inference
Gradio Web UI for demo
Render Free cloud deployment
rembg Background removal for augmentation
Pillow / NumPy Image processing & data handling

---

🎯 Challenges & Solutions
