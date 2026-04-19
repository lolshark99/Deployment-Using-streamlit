import torch
import cv2
import numpy as np
from torchvision import transforms
from model import EmotionModel

mean_list_train = [0.5456, 0.4975, 0.4794]
std_list_train  = [0.1993, 0.1924, 0.1891]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionModel(num_classes=7)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

class_names = ['angry','disgust','fear','happy','neutral','sad','surprise']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(mean_list_train, std_list_train)
])

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)


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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    probs = None 

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred = np.argmax(probs)

        label = f"{class_names[pred]} ({probs[pred]*100:.1f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0), 2)

    if probs is not None:
        graph = draw_distribution(probs, class_names)
        frame[10:210, 10:310] = graph

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()