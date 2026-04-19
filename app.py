import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
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

def draw_distribution(probs, class_names, width=300, height=200):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    bar_width = width // len(class_names)

    for i, p in enumerate(probs):
        x1 = i * bar_width
        x2 = x1 + bar_width - 5
        bar_height = int(p * height)

        y1 = height - bar_height
        y2 = height

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), -1)

        cv2.putText(img, class_names[i][:3],
                    (x1, height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255,255,255), 1)

        cv2.putText(img, f"{p*100:.0f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255,255,255), 1)

    return img

st.title("Emotion Detector")

img_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if img_file:
    img = Image.open(img_file).convert("RGB")
    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    all_probs = None

    if len(faces) == 0:
        st.warning("No face detected")

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
    import pandas as pd

    if all_probs is not None:
        df = pd.DataFrame({
            "Emotions":class_names,
            "Probabilities":all_probs
        })
        st.subheader("Emotion Confidence")
        st.bar_chart(df.set_index("Emotions"))