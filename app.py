import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import os
import json

# --- SETTINGS ---
MODEL_PATH = 'outputs/best_model.pth'
CLASS_NAMES_PATH = 'outputs/class_names.json'  # Saved during training
NUM_CLASSES = 12  # Update if necessary
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD CLASS NAMES ---
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, NUM_CLASSES)
    
    # Load model weights with strict=False for flexibility
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()
    return model

# --- IMAGE TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- INFERENCE FUNCTION ---
def predict(image, model):
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)
        top3_prob, top3_indices = torch.topk(probabilities, 3)

    top_preds = [(class_names[i], float(top3_prob[0][j]) * 100)
                 for j, i in enumerate(top3_indices[0])]
    return class_names[pred.item()], float(confidence) * 100, top_preds

# --- STREAMLIT UI ---
st.title("🔍 Farm Insect Detection")
st.write("Upload an image to predict the insect.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Predicting...")
    predicted_class, confidence_score, top_preds = predict(image, model)


    st.success(f"🎯 Predicted: {predicted_class} ({confidence_score:.2f}% confidence)")

    st.write("Top 3 Predictions:")
    for i, (label, score) in enumerate(top_preds, 1):
        st.write(f"{i}. {label} - {score:.2f}%")
