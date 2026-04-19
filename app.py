import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from model import EmotionModel
import pandas as pd

device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = EmotionModel(7)
    model.load_state_dict(torch.load("model.pth", map_location=device))
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

st.title("🎭 Emotion Detector")
st.caption("Tip: Keep face clear and centered for best results")

img_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if img_file:
    img = Image.open(img_file)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    img_np = np.array(img)

    h, w = img_np.shape[:2]
    scale = 600 / max(h, w)
    if scale < 1:
        img_np = cv2.resize(img_np, (int(w*scale), int(h*scale)))

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(40, 40)
    )

    st.write(f"Faces detected: {len(faces)}")

    if len(faces) == 0:
        st.warning("No face detected, using full image")
        faces = [(0, 0, img_np.shape[1], img_np.shape[0])]

    all_probs = None

    for (x,y,w,h) in faces:
        face = img_np[y:y+h, x:x+w]

        if face is None or face.size == 0:
            continue

        face = Image.fromarray(face).convert("RGB")
        face = transform(face).unsqueeze(0)

        with torch.no_grad():
            output = model(face)
            probs = torch.softmax(output, dim=1)[0]
            all_probs = probs.cpu().numpy()
            pred = torch.argmax(probs).item()

        label = f"{class_names[pred]} ({probs[pred]*100:.1f}%)"

        cv2.rectangle(img_np, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img_np, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0), 2)

    st.image(img_np, channels="RGB")

    if all_probs is not None:
        df = pd.DataFrame({
            "Emotion": class_names,
            "Confidence": all_probs
        })

        st.subheader("Emotion Confidence")
        st.bar_chart(df.set_index("Emotion"))

        pred_emotion = class_names[np.argmax(all_probs)]
        confidence = np.max(all_probs) * 100
        st.metric("Top Emotion", pred_emotion, f"{confidence:.1f}%")