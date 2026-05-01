import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from model import EmotionModel
import pandas as pd

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = EmotionModel(7)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

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

class EmotionProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # get current frame

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = Image.fromarray(face)
            face = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face)
                pred = torch.argmax(output, dim=1).item()

            label = class_names[pred]

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0), 2)

        return img  # return processed frame continuously

st.title("Emotion Detector")

st.subheader("Image Mode")

img_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if img_file:
    img = Image.open(img_file)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.1, 3)

    if len(faces) == 0:
        faces = [(0, 0, img_np.shape[1], img_np.shape[0])]

    for (x,y,w,h) in faces:
        face = img_np[y:y+h, x:x+w]

        if face is None or face.size == 0:
            continue

        face = Image.fromarray(face).convert("RGB")
        face = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face)
            probs = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(probs).item()

        label = f"{class_names[pred]} ({probs[pred]*100:.1f}%)"

        cv2.rectangle(img_np, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img_np, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0), 2)

    st.image(img_np, channels="RGB")

st.subheader("Real-Time Mode")

webrtc_streamer(
    key="emotion",
    video_transformer_factory=EmotionProcessor
)