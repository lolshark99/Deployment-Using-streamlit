import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from model import EmotionModel

device = torch.device("cpu")

model = EmotionModel(7)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

class_names = ['angry','disgust','fear','happy','neutral','sad','surprise']

mean = [0.5456, 0.4975, 0.4794]
std  = [0.1993, 0.1924, 0.1891]

transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.title("Emotion Detector")

img_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if img_file:
    img = Image.open(img_file).convert("RGB")
    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = img_np[y:y+h, x:x+w]
        face = transform(Image.fromarray(face)).unsqueeze(0)

        with torch.no_grad():
            output = model(face)
            probs = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(probs).item()

        label = f"{class_names[pred]} ({probs[pred]*100:.1f}%)"

        cv2.rectangle(img_np, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img_np, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0), 2)

    st.image(img_np)