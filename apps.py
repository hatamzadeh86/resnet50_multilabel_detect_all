import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import gradio as gr
import os
import onnxruntime as ort
import numpy as np


import os
import requests
import urllib.parse

def download_model_if_not_exists():
    model_path = "resnet50_multilabel.onnx"
    if os.path.exists(model_path):
        print("model as ghbl ")
        return
    
    print("GitHub Release...")
    # لینک دانلود مستقیم از Release (مثل: https://github.com/USER/REPO/releases/download/v1.0.0/model.onnx)
    url = 'https://github.com/hatamzadeh86/resnet50_multilabel_detect_all/releases/download/v1.0.1/model_resnet50_multi.onnx'
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(model_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print("seccuss dowload model")

# قبل از بارگذاری مدل، این تابع رو صدا بزن
download_model_if_not_exists()









# ============================================
# تنظیمات
# ============================================
MODEL_PATH = r"C:\Users\E-PART.iR\Desktop\Detect_shi\model_onnx\model_resnet50_multi.onnx"
CLASSES_PATH = r"C:\Users\E-PART.iR\Desktop\Detect_shi\classis_data.txt"
INPUT_SIZE = 224
DEVICE = torch.device("cpu")

# ============================================
# بارگذاری اسم کلاس‌ها
# ============================================
with open(CLASSES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

NUM_CLASSES = len(class_names)

# ============================================
# تعریف مدل (دقیقاً مثل زمان آموزش)
# ============================================
def get_model(num_classes):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

# ============================================
# بارگذاری مدل
# ============================================
# model = get_model(NUM_CLASSES)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model = model.to(DEVICE)

# model.eval()

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name




# ============================================
# پیش‌پردازش تصویر
# ============================================
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================
# تابع پیش‌بینی
# ============================================
def predict(image, threshold=0.5):
    # پیش‌پردازش
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE).numpy()

    # پیش‌بینی
    with torch.no_grad():
        outputs = session.run(None , {input_name : input_tensor})
        logics = outputs[0][0]

        probs = 1 / (1 + np.exp(-logics))

    # انتخاب کلاس‌هایی که احتمالشان بالاتر از آستانه است
    results = []
    for idx, prob in enumerate(probs):
        if prob > threshold:
            results.append((class_names[idx], float(prob)))

    # مرتب‌سازی بر اساس احتمال (بالاترین اول)
    results = sorted(results, key=lambda x: x[1], reverse=True)


    if not results:
        return 'not'

    output_text = "\n\n"
    for name, prob in results:
        output_text += f"- {name}: {prob*100:.2f}%\n"

    return output_text

# ============================================
# رابط کاربری Gradio
# ============================================
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="📸 عکس خود را آپلود کنید"),
        gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="آستانه اطمینان (Threshold)")
    ],
    outputs=gr.Textbox(label="📊 نتایج", lines=10),
    title="🧠 تشخیص اشیاء چندبرچسبی با ResNet50",
    description="یک عکس آپلود کنید تا مدل همه اشیاء موجود در آن را تشخیص دهد.",
    examples=[
        ["example1.jpg"],
        ["example2.jpg"]
    ]
)

if __name__ == "__main__":
    iface.launch()