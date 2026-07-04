import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import f1_score


import random
import numpy as np
import torch

def set_seed(seed=42):
    """تنظیم Seed برای تکرارپذیری کامل"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # برای چندین GPU
    

    # تنظیمات PyTorch برای قطعی کردن عملیات
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================
# در ابتدای فایل، بلافاصله بعد از importها:
# ============================================
set_seed(42)




class Config:
    CSV_PATH = r"C:\Users\E-PART.iR\Desktop\Detect_shi\train_labels.csv"
    IMAGE_DIR = r"C:\Users\E-PART.iR\Desktop\data_set_train\train"
    MODEL_SAVE_PATH = r"C:\Users\E-PART.iR\Desktop\Detect_shi/resnet50_multilabel.pth"
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 0.0001
    INPUT_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATIENCE = 5

class MultiLabelDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        
        all_labels = set()
        for labels in self.df['labels']:
            for lbl in labels.split(';'):
                all_labels.add(lbl.strip())
        
        self.classes = sorted(all_labels)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        labels_list = row['labels'].split(';')
        label_vector = torch.zeros(self.num_classes)
        for lbl in labels_list:
            lbl = lbl.strip()
            if lbl in self.class_to_idx:
                label_vector[self.class_to_idx[lbl]] = 1
        
        return image, label_vector

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(Config.INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(Config.INPUT_SIZE),
        transforms.CenterCrop(Config.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(Config.DEVICE)
    
    for param in model.parameters():
        param.requires_grad = True
    
    return model

def train_model():
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
    
    train_transform, val_transform = get_transforms()
    
    full_dataset = MultiLabelDataset(Config.CSV_PATH, Config.IMAGE_DIR, train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    model = get_model(full_dataset.num_classes)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.1)
    
    best_f1 = 0.0
    counter = 0
    
    print("="*60)
    print(f"Training ResNet50 Multi-Label")
    print(f"Classes: {full_dataset.num_classes}")
    print(f"Device: {Config.DEVICE}")
    print("="*60)
    
    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
        
        for images, labels in train_bar:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = running_loss / len(train_loader)
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
        exact_match = np.mean(np.all(all_preds == all_labels, axis=1))
        
        epoch_time = time.time() - start_time
        
        if f1 > best_f1:
            best_f1 = f1
            counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Epoch {epoch+1}: Best model saved with F1: {f1:.4f}")
        else:
            counter += 1
            if counter >= Config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - F1: {f1:.4f} - Exact Match: {exact_match:.4f} - Time: {epoch_time:.1f}s")
    
    print(f"Best F1: {best_f1:.4f}")
    return model

def predict_image(image_path, model, class_names, threshold=0.5):
    transform = transforms.Compose([
        transforms.Resize(Config.INPUT_SIZE),
        transforms.CenterCrop(Config.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    predicted_classes = []
    for idx, prob in enumerate(probs):
        if prob > threshold:
            predicted_classes.append((class_names[idx], prob))
    
    return sorted(predicted_classes, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    train_model()
    
    temp_dataset = MultiLabelDataset(Config.CSV_PATH, Config.IMAGE_DIR)
    class_names = temp_dataset.classes
    model = get_model(temp_dataset.num_classes)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.to(Config.DEVICE)
    
    result = predict_image(r"C:\Users\E-PART.iR\Desktop\images\376270_638.jpg" , model, class_names)
    print("Prediction:")
    for cls, prob in result:
        print(f"   {cls}: {prob*100:.2f}%")




def predict_single_image(image_path, model_path, class_names=None, threshold=0.5):
    """
    Predict classes for a single image using saved model

    Args:
        image_path: Path to the image file
        model_path: Path to the saved model (.pth)
        class_names: List of class names (if None, loads from CSV)
        threshold: Confidence threshold (default: 0.5)

    Returns:
        List of (class_name, probability) tuples sorted by probability
    """
    # 1. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(class_names) if class_names else 21
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 2. Load class names if not provided
    if class_names is None:
        df = pd.read_csv(Config.CSV_PATH)
        all_labels = set()
        for labels in df['labels']:
            for lbl in labels.split(';'):
                all_labels.add(lbl.strip())
        class_names = sorted(all_labels)

    # 3. Preprocess image
    transform = transforms.Compose([
        transforms.Resize(Config.INPUT_SIZE),
        transforms.CenterCrop(Config.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return []

    input_tensor = transform(image).unsqueeze(0).to(device)

    # 4. Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    # 5. Extract predictions above threshold
    predicted_classes = []
    for idx, prob in enumerate(probs):
        if prob > threshold:
            predicted_classes.append((class_names[idx], prob))

    return sorted(predicted_classes, key=lambda x: x[1], reverse=True)




# ============================================
# تست پیش‌بینی روی یک عکس جدید
# ============================================
if __name__ == "__main__":
    # 1. ابتدا مدل را آموزش بدهید (یا بارگذاری کنید)
    # train_model()  # اگر مدل قبلاً آموزش داده شده، این خط را کامنت کنید

    # 2. پیش‌بینی روی یک عکس جدید
    result = predict_single_image(
        image_path=r"C:\Users\E-PART.iR\Desktop\images\download (8).jpg",
        model_path=Config.MODEL_SAVE_PATH,
        threshold=0.5
    )

    # 3. نمایش نتایج
    print("\n" + "="*50)
    print("Prediction Results:")
    print("="*50)
    for cls, prob in result:
        print(f"   {cls}: {prob*100:.2f}%")
    print("="*50)













