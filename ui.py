import streamlit as st
from PIL import Image
import torch
import time
import json
import wikipedia
from fpdf import FPDF
import base64
import os
from datetime import datetime
import csv
import pandas as pd
import openai
from dotenv import load_dotenv
from pathlib import Path
import hashlib
import numpy as np
import cv2
from captum.attr import IntegratedGradients
import requests
import random
import smtplib

# Setup UI config first
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Use pathlib for relative path
#img_path = Path(__file__).parent / "background.jpg"  # Assumes background.jpg is in the same folder as this script
#img_base64 = get_base64_image(img_path)

#st.markdown(
    #f"""
    #<style>
    #.stApp {{
     #   background-image: url("data:image/jpg;base64,{img_base64}");
      #  background-size: cover;
     #   background-position: center;
     #   background-attachment: fixed;
    #}}
    #.block-container {{
       # background-color: rgba(0, 0, 0, 0.55);
       # padding: 2rem;
       # border-radius: 10px;
   # }}
    #</style>
    #""", unsafe_allow_html=True
#)

# === Splash screen BEFORE heavy imports ===
startup = st.empty()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def send_otp(email):
    otp = ''.join([str(random.randint(0, 9)) for _ in range(8)])
    try:
        # Set up the SMTP server
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login("plantdoctor@gmail.com", "ninwuj-homJe8-wiqjow")  # Replace with your credentials or use st.secrets
        subject = "Your OTP for Plant Disease Detector Registration"
        body = f"Your OTP is: {otp}"
        message = f"Subject: {subject}\n\n{body}"
        server.sendmail("plantdoctor@gmail.com", email, message)
        server.quit()
        return otp
    except Exception as e:
        st.error(f"Failed to send OTP: {e}")
        return None

def username_exists(username):
    try:
        with open("users.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == username:
                    return True
    except FileNotFoundError:
        return False
    return False

def explain_prediction_with_text(model, image_tensor, device="cpu"):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device).float()
    pred_class = model(image_tensor).argmax(dim=1).item()
    ig = IntegratedGradients(model)
    attributions, _ = ig.attribute(inputs=image_tensor, target=pred_class, return_convergence_delta=True)
    attributions = attributions.squeeze().cpu().detach().numpy()
    importance_score = np.mean(np.abs(attributions))
    explanation_text = f"""
- Predicted class: **{class_names[pred_class]}**
- This class was chosen because important pixel regions contributed strongly to the output score.
- Integrated Gradients (IG) attribution score: **{importance_score:.4f}**
    """
    return explanation_text

def generate_gradcam(model, image_tensor, class_idx=None, device="cpu"):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    gradients = []
    activations = []
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    def forward_hook(module, input, output):
        activations.append(output)
    target_layer = model.layer4[-1].conv2
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item() if class_idx is None else class_idx
    score = output[0, pred_class]
    model.zero_grad()
    score.backward()
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= np.min(cam)
    cam /= np.max(cam)
    forward_handle.remove()
    backward_handle.remove()
    img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img -= img.min()
    img /= img.max()
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + img
    overlay = overlay / np.max(overlay)
    return overlay, cam, pred_class

def show_random_image(disease_name):
    base_dir = Path("CombinedDataset") / "train"
    class_dir = base_dir / disease_name
    if not class_dir.exists():
        st.error(f"Folder not found for {disease_name}")
        return
    image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
    if not image_files:
        st.warning("No images found in this folder!")
        return
    random_image = random.choice(image_files)
    st.image(Image.open(random_image), caption=f"Example of {disease_name}")

def get_geo_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        return data.get("city"), data.get("region"), data.get("country")
    except Exception as e:
        return None, None, None

def register_user(username, password, allow_location, allow_notifications):
    with open("users.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([username, hash_password(password), allow_location, allow_notifications])

def authenticate_user(username, password):
    try:
        with open("users.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == username and row[1] == hash_password(password):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.session_state["allow_location"] = row[2] == "True"
                    st.session_state["allow_notifications"] = row[3] == "True"
                    return True
    except FileNotFoundError:
        return False

def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.to(device)
    return model

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "resnet18_plant_disease_detection.pth"

with startup.container():
    st.title("Initializing Plant Disease Detector")
    with st.spinner("Loading model and preparing environment..."):
        progress = st.progress(0, text="Loading...")
        for percent in range(0, 41, 10):
            time.sleep(0.05)
            progress.progress(percent, text=f"Loading... {percent}%")
        from engine import resnet18, device
        from torchvision import transforms
        # Define test_transform
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        from engine import resnet18
        loaded_resnet18 = load_model(resnet18, MODEL_PATH, device)
        print(f"✅ Model loaded from {MODEL_PATH}")
        model = resnet18
        model.eval().to(device)
        for percent in range(50, 101, 10):
            time.sleep(0.05)
            progress.progress(percent, text=f"Loading... {percent}%")
        time.sleep(0.2)
startup.empty()

# from libretranslatepy import LibreTranslateAPI
# from gtts import gTTS

# Extended language options for translation
languages = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh-cn",
    "Chinese (Traditional)": "zh-tw",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Swedish": "sv",
    "Turkish": "tr",
    "Thai": "th",
    "Vietnamese": "vi",
    "Greek": "el",
    "Hebrew": "he",
    "Polish": "pl",
    "Czech": "cs",
    "Danish": "da",
    "Finnish": "fi",
    "Hungarian": "hu",
    "Norwegian": "no",
    "Malay": "ms",
    "Indonesian": "id",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
    "Punjabi": "pa",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml"
}

# Translator instance
# translator = LibreTranslateAPI()

# Function to translate text
# def translate_text(text, lang_code):
    # try:
        # translated = translator.translate(text, "auto", lang_code)
        # return translated.text
    # except Exception as e:
        # return f"Translation Error: {e}"

# Function to generate and play audio
# def speak_text_gtts(text, lang_code):
    # tts = gTTS(text=text, lang=lang_code)
    # tts.save("output.mp3")
    # os.system("afplay output.mp3")  # For macOS, use `afplay`

# === Set up the sidebar ===
# Sidebar Navigation
st.sidebar.title("Navigation")
menu_option = st.sidebar.radio("Go to", ["Home", "Upload & Predict", "Generate Report", "PlantPedia", "Predict Play", "Plant Saathi", "Help", "Write a Review", "Contact"])

# === Prediction Function ===
import torch.nn.functional as F

import torch.nn.functional as F

def predict_image_streamlit(model, image, transform, device, class_names=None):
    # Preprocess
    image = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(image)  # raw logits
        probs = F.softmax(output, dim=1)  # convert to probabilities
        predicted_idx = probs.argmax(dim=1).item()
        confidence = probs[0][predicted_idx].item()

    if class_names:
        return class_names[predicted_idx], confidence
    else:
        return predicted_idx, confidence


# === Report Generation ===
# Function to get disease information from Wikipedia
def get_disease_info(disease_name):
    try:
        summary = wikipedia.summary(disease_name + " plant disease", sentences=150)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Too many options: {e.options[:3]}"
    except wikipedia.exceptions.PageError:
        return "Disease information not found."
    
# Function to save image and prediction
def save_prediction_image(image):
    if not os.path.exists("predictions"):
        os.makedirs("predictions")
    image_path = f"predictions/{datetime.now().strftime('%Y%m%d_%H%M%S')}_prediction.jpg"
    image.save(image_path)
    return image_path

from fpdf import FPDF
from datetime import datetime
import os

def generate_pdf_report(
    image_path,
    prediction,
    confidence=None,
    wiki_summary=None,
    detailed_info=None,
    custom_filename="plant_disease_report.pdf"
):
    def sanitize_text(text):
        """Remove or replace unsupported characters."""
        return text.replace("•", "-").encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()  # First page: Image only

    # Center the image on the page (A4: 210x297mm)
    img_w, img_h = 160, 160  # Adjust as needed
    page_w, page_h = pdf.w, pdf.h
    img_x = (page_w - img_w) // 2
    img_y = (page_h - img_h) // 2

    if image_path:
        pdf.image(image_path, x=img_x, y=img_y, w=img_w, h=img_h)

    pdf.add_page()  # Second page: Text only
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, txt=sanitize_text("Plant Disease Detection Report"), ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 10, txt=sanitize_text(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), ln=True)
    pdf.cell(0, 10, txt=sanitize_text(f"Prediction: {prediction}"), ln=True)

    if confidence:
        pdf.cell(0, 10, txt=sanitize_text(f"Confidence: {confidence:.2f}%"), ln=True)
        pdf.ln(5)

    # Write summary and info on second page
    if wiki_summary:
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, sanitize_text("Disease Summary:"))
        pdf.multi_cell(0, 10, sanitize_text(wiki_summary))
        pdf.ln(5)

    if detailed_info:
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, sanitize_text(f"Plant: {detailed_info['plant']}"))
        pdf.multi_cell(0, 10, sanitize_text(f"Disease Name: {prediction}"))
        pdf.multi_cell(0, 10, sanitize_text("Symptoms:\n" + "\n".join(f"- {s}" for s in detailed_info["symptoms"])))
        pdf.multi_cell(0, 10, sanitize_text("Causes:\n" + "\n".join(f"- {c}" for c in detailed_info["causes"])))
        pdf.multi_cell(0, 10, sanitize_text(f"Severity:\n  Mild: {detailed_info['severity_mild']}\n  Moderate: {detailed_info['severity_moderate']}\n  Severe: {detailed_info['severity_severe']}"))
        pdf.multi_cell(0, 10, sanitize_text(f"Progression (in days): {detailed_info['progression_days']}"))
        pdf.multi_cell(0, 10, sanitize_text("Treatment:\n" + "\n".join(f"- {t}" for t in detailed_info["treatment"])))
        pdf.multi_cell(0, 10, sanitize_text("Prevention:\n" + "\n".join(f"- {p}" for p in detailed_info["prevention"])))

    report_path = custom_filename if custom_filename.endswith('.pdf') else custom_filename + '.pdf'
    if not report_path.startswith("reports/"):
        os.makedirs("reports", exist_ok=True)
        report_path = os.path.join("reports", report_path)

    pdf.output(report_path)
    return report_path

# Convert PDF to downloadable base64 string
def create_download_link(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="plant_disease_report.pdf">📥 Download PDF Report</a>'
    return href

class_names = [
    "Apple Black Rot",
    "Apple Healthy",
    "Apple Rust",
    "Apple Scab",
    "Bell Pepper Healthy",
    "Bell Pepper Leaf Spot",
    "Blueberry Healthy",
    "Cherry Healthy",
    "Cherry Powdery Mildew",
    "Corn Gray Leaf Spot",
    "Corn Healthy",
    "Corn Leaf Blight",
    "Corn Rust",
    "Grape Black Rot",
    "Grape Esca",
    "Grape Healthy",
    "Grape Leaf Blight",
    "Orange Citrus Greening",
    "Peach Healthy",
    "Peach Leaf Spot",
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Raspberry Healthy",
    "Soybean Healthy",
    "Squash Powdery Mildew",
    "Strawberry Healthy",
    "Strawberry Leaf Scorch",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Mosaic Virus",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mite",
    "Tomato Target Spot",
    "Tomato Yellow Virus"
]

import streamlit as st
import os
import requests
from dotenv import load_dotenv

# === App UI ===
# Home Page
if menu_option == "Home":
    st.sidebar.success("You are on the Home page")
    st.title("Welcome to the Plant Disease Detector!")
    # Add Translate Dropdown (commented out)
    # selected_language = st.selectbox("Translate", list(languages.keys()), index=0, key="translate_dropdown")
    # translated_text = translate_text("Welcome to the Plant Disease Detector!", languages[selected_language])
    st.image("Image_for_PDA.jpg", caption="Healthy Leaves, Healthy Crops", use_container_width=True)
    st.markdown("""
    This AI-powered app helps you **identify plant diseases from leaf images** using deep learning.

    ---
    ### Overview:
    This project is a Plant Disease Detection system powered by deep learning/AI technologies.
    It leverages computer vision models to analyze plant leaf images and provide accurate classifications.
    The goal is to support early identification of diseases, helping reduce crop losses and improve yield.
    
    ---
    ### Why Plant Disease Detection?
    
    Globally, agriculture is a cornerstone of food security and livelihoods – roughly **26% of the world’s workforce** is employed in farming – yet plant diseases inflict massive losses. Humans get **80% of their food from plants**, but up to **40% of global crop production** is destroyed by pests and pathogens each year, costing on the order of **USD 220 billion annually**.

    In India, where agriculture remains a rural backbone, recent data show about **42.3% of the population** depends on farming (and roughly two-thirds of rural Indians derive their livelihood from agriculture). Crop diseases commonly cut Indian crop output by **10–30%**, with some analyses even citing **30%** annual productivity loss due to pest infestations. Such declines are especially harmful for millions of smallholder farmers, where even modest yield drops can undermine income and food access.

    Moreover, climate change and modern farming amplify these threats: warming allows pathogens (like fungi) to spread into new regions, and large-scale monocultures of genetically uniform crops provide ideal breeding grounds for outbreaks.

    In this context, early disease detection is vitally important. **AI and machine-learning tools** can rapidly scan plant images or sensor data to diagnose infections before they become widespread. By providing fast, precise, and automated diagnostics, these technologies cut labor and response time and enable targeted interventions—spot-treating only infected areas.

    In practice, AI-driven monitoring helps farmers act quickly, quarantining or treating diseased plants in time, thereby minimizing crop loss and strengthening food security. Given agriculture’s socio-economic importance (especially in India) and the growing strain of disease pressures, timely plant disease detection through AI/ML methods is critical to reduce losses, sustain yields, and support rural livelihoods.
    
    ---
    ### Features:
    - Capture or upload a leaf image
    - Get instant disease prediction using a ResNet18 model
    - Learn about symptoms, causes, treatments, and prevention
    - Lookup detailed info from Wikipedia or built-in database
    - Supports multiple plant types and diseases

    ---
    ### How to Use:
    1. Go to **Upload & Predict** in the sidebar
    2. Upload a leaf image or take a photo using your camera
    3. Click **Predict** to analyze the leaf
    4. Read the predicted disease and scroll down for detailed treatment info

    """, unsafe_allow_html=True)

    # st.markdown(f"**Translation in {selected_language}:** ") # {translated_text}
    # Load from .env if needed
    load_dotenv()

    # Load knowledge base
    #@st.cache_data
    #def load_knowledge_base():
        #with open("knowledge_base.txt", "r") as f:
            #return f.read()

    #knowledge_base = load_knowledge_base()

    # Chatbot using Hugging Face Inference API (no local download)
    #API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    #HF_TOKEN = os.getenv("HF_TOKEN")  # Add your token to .env as HF_TOKEN=<your_token>
    #headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    #def hf_inference(prompt):
       # response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        #try:
            # For text-generation models, response is a list of dicts
          #  return response.json()[0]['generated_text']
       # except Exception:
       #     return str(response.json())

    # Title + Chatbot Button
    #st.markdown("### Need Help?")
    #if st.button("I am PlantPal! How may I help you!"):
        #st.session_state.show_chat = True

    # Show Chat if Triggered
    #if st.session_state.get("show_chat", False):
       # st.markdown("#### Ask PlantPal")
        #user_query = st.text_input("Ask anything about plant diseases, symptoms, or the model:")

        #if user_query:
         #   with st.spinner("PlantPal is thinking..."):
           #     try:
          #          prompt = f"""You are PlantPal, an expert assistant on plant diseases and detection models.
#Only answer based on the knowledge base below. If the question is unrelated or the answer isn't available, say "I'm not sure based on my current knowledge."

### Knowledge Base:
#{knowledge_base}

### Question:
#{user_query}

### Answer:"""

                    #answer = hf_inference(prompt)
                    # Extract answer after "### Answer:" if present
                    #if "### Answer:" in answer:
                    #    answer = answer.split("### Answer:")[-1].strip()
                    #st.success(answer)

                #except Exception as e:
                    #st.error(f"Error: {str(e)}")

if menu_option == "Upload & Predict":
    st.sidebar.success("You are on the Upload & Predict page")
    st.title("Plant Disease Detection")
    st.markdown("Upload a **leaf image** or use **camera**, and the model will predict the disease.")

    # ----- Camera State -----
    if "show_camera" not in st.session_state:
        st.session_state["show_camera"] = False

    # Switch modes
    if st.button("Take a Photo"):
        st.session_state["show_camera"] = True
        # Clear input for new photo, but keep previous predictions/results
        st.session_state["uploaded_file"] = None
        st.session_state["input_image"] = None

    if st.session_state["show_camera"] and st.button("Upload"):
        st.session_state["show_camera"] = False
        st.session_state["camera_image"] = None
        st.session_state["input_image"] = None

    show_camera = st.session_state["show_camera"]

    # ----- Camera Mode -----
    if show_camera:
        camera_image = st.camera_input("Capture your leaf image")
        if camera_image is not None:
            st.session_state["camera_image"] = camera_image
            st.session_state["input_image"] = Image.open(camera_image).convert("RGB")
    elif st.session_state.get("camera_image") is not None:
        camera_image = st.session_state["camera_image"]
    else:
        camera_image = None

    # ----- Upload Mode -----
    if not show_camera:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.session_state["uploaded_file"] = uploaded_file
            st.session_state["input_image"] = Image.open(uploaded_file).convert("RGB")
    elif st.session_state.get("uploaded_file") is not None:
        uploaded_file = st.session_state["uploaded_file"]
    else:
        uploaded_file = None

    input_image = st.session_state.get("input_image")
    if input_image:
        st.image(input_image, caption="Input Image", use_container_width=True)

        # Predict logic
        if st.button("Predict"):
            with st.spinner("Analyzing the leaf..."):
                pred_class, confidence = predict_image_streamlit(model, input_image, test_transform, device)
                if pred_class is not None:
                    predicted_label = class_names[pred_class]
                else:
                    predicted_label = "Unknown"
                image_path = save_prediction_image(input_image)
                disease_info = get_disease_info(predicted_label)

                # GradCAM and explanation
                image_tensor = test_transform(input_image)
                overlay, cam, predicted_class_cam = generate_gradcam(model, image_tensor, device=device)
                overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
                explanation = explain_prediction_with_text(model, image_tensor, device=device)
                try:
                    with open("disease_info.json", "r") as f:
                        disease_data = json.load(f)
                    detailed_info = disease_data.get(predicted_label, None)
                except:
                    detailed_info = None

                # ----- Store ALL in session_state -----
                st.session_state["pred_class"] = pred_class
                st.session_state["confidence"] = confidence
                st.session_state["predicted_label"] = predicted_label
                st.session_state["image_path"] = image_path
                st.session_state["disease_name"] = predicted_label
                st.session_state["disease_info"] = disease_info
                st.session_state["explanation"] = explanation
                st.session_state["overlay_pil"] = overlay_pil
                st.session_state["last_detailed_info"] = detailed_info

    # ----- Display Results (if available in session_state) -----
    if st.session_state.get("predicted_label") is not None and st.session_state.get("confidence") is not None:
        st.success(f"Prediction: **{st.session_state['predicted_label']}** | Confidence: **{st.session_state['confidence']*100:.2f}%**")

        if st.session_state.get("overlay_pil") is not None:
            st.image(st.session_state["overlay_pil"], caption=f"Image Analysis: Grad-CAM for {st.session_state['predicted_label']}", use_container_width=True)
        if st.session_state.get("explanation"):
            st.markdown(f"**Explanation:** {st.session_state['explanation']}", unsafe_allow_html=True)

        # ----- LOG PREDICTION -----
        st.markdown("### Image ID")
        image_id = st.text_input("Enter an ID for this image:", value=st.session_state.get("image_id", ""))
        if st.button("Save Image ID"):
            if image_id:
                st.session_state["image_id"] = image_id
                with open("disease_log.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        image_id,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        st.session_state["predicted_label"],
                        f"{st.session_state['confidence']:.2f}%" if st.session_state["confidence"] is not None else "N/A",
                        st.session_state["image_path"]
                    ])
                st.success(f"Image ID '{image_id}' saved successfully!")
            else:
                st.error("Please enter a valid ID.")

        # ----- Agro Medicine Store -----
        if st.session_state.get("pred_class") is not None:
            st.subheader("Find Nearest Agro Medicine Store")
            if st.button("Search Agro Store"):
                st.markdown("Searching for nearest agro medicine store...")
                st.markdown("[Click here to find stores](https://www.google.com/maps/search/agro+medicine+store+near+me)")

        # ----- Disease Info -----
        disease_name = class_names[st.session_state.get("pred_class")] if st.session_state.get("pred_class") is not None else None
        if disease_name:
            st.subheader(f"Information about {disease_name}")
            disease_info = get_disease_info(disease_name)
            st.write(disease_info)

            st.subheader("Detailed Information")
            info = st.session_state.get("last_detailed_info")
            if info:
                st.write(f"**Plant**: {info['plant']}")
                st.write(f"**Disease Name**: {disease_name}")
                st.write("**Symptoms:**")
                for symptom in info["symptoms"]:
                    st.write(f"• {symptom}")

                st.write("**Causes:**")
                for cause in info["causes"]:
                    st.write(f"• {cause}")

                st.write(f"**Severity (Mild)**: {info['severity_mild']}")
                st.write(f"**Severity (Moderate)**: {info['severity_moderate']}")
                st.write(f"**Severity (Severe)**: {info['severity_severe']}")
                st.write(f"**Progression (in days)**: {info['progression_days']}")

                st.write("**Treatment:**")
                for step in info["treatment"]:
                    st.write(f"• {step}")

                if "medicines" in info and info["medicines"]:
                    st.write("**Medicines:**")
                    for medicine in info["medicines"]:
                        st.write(f"• {medicine}")

                st.write("**Prevention:**")
                for method in info["prevention"]:
                    st.write(f"• {method}")
            else:
                st.warning("No detailed information available for this disease.")

# === Reports Section ===
if menu_option == "Generate Report":
    st.sidebar.success("You are on the Reports page")
    st.title("PDF Report Generator")

    # Compatible with the session state variable names used in the prediction flow
    if (
        st.session_state.get("predicted_label") is not None and
        st.session_state.get("image_path") is not None
    ):
        st.write(f"**Prediction**: {st.session_state['predicted_label']}")
        st.image(st.session_state["image_path"], caption="Last Predicted Leaf", use_container_width=True)

        # Let user name the PDF
        default_filename = f"{st.session_state['predicted_label'].replace(' ', '_')}_report"
        pdf_filename = st.text_input("Enter a name for the PDF file (no extension):", value=default_filename)

        if st.button("Generate PDF Report"):
            final_pdf_name = pdf_filename.strip() + ".pdf"

            # Generate the PDF report
            pdf_path = generate_pdf_report(
                st.session_state["image_path"],
                st.session_state["predicted_label"],
                wiki_summary=st.session_state.get("disease_info"),
                detailed_info=st.session_state.get("last_detailed_info"),
                custom_filename=final_pdf_name,
            )

            # Read the PDF file and provide a download button
            with open(pdf_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_data,
                    file_name=final_pdf_name,
                    mime="application/pdf"
                )
    else:
        st.warning("No prediction available yet. Go to 'Upload & Predict' first.")

# === Help Section ===
# Help
if menu_option == "Help":
    st.sidebar.success("You are on the Help page")
    st.title("Help & Support")
    st.markdown("""
    ## How to Use:
    1. **Upload an Image**: Click on the "Choose an image..." button to upload a leaf image.
    2. **Use Camera**: Click on the "📸 Take a Photo" button to capture a leaf image using your device's camera.
    """)

    st.markdown("""
    ## If you encounter any issues or have questions, please contact us at:
    - **Email**: greenplantdoctor25@gmail.com
    """)

# === Write a Review Section ===
if menu_option == "Write a Review":
    st.sidebar.success("You are on the Write a Review page")
    st.title("Write a Review")
    st.markdown("""
    We value your feedback! Please share your experience using the Plant Disease Detector.
    """)

    review_text = st.text_area("Your Review", height=200)
    if st.button("Submit Review"):
        if review_text:
            # Save review to CSV
            with open("reviews.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), review_text])

            st.success("Thank you for your review!")
            
        else:
            st.error("Please enter a review before submitting.")
    
    MAX_REVIEWS = 100

    # After writing a new review, trim file
    def trim_reviews():
        try:
            with open("reviews.csv", "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > MAX_REVIEWS:
                lines = lines[-MAX_REVIEWS:]  # keep only last 100
            with open("reviews.csv", "w", encoding="utf-8") as f:
                f.writelines(lines)
        except FileNotFoundError:
            pass
    
    trim_reviews()

# === Contact Section ===
if menu_option == "Contact":
    st.sidebar.success("You are on the Contact page")
    st.title("Contact Us")
    st.markdown("""
    If you have any questions, suggestions, or need support, please reach out to us:
    - **Official Email**: greenplantdoctor25@gmail.com
    """)

# === Community Section ===
if menu_option == "Community":
    st.sidebar.success("You are on the Community page")
    st.title("Community Chat")

    name = st.text_input("Your Name")
    message = st.text_area("Message")
    if st.button("Send"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_msg = pd.DataFrame([[name, message, timestamp]], columns=["Name", "Message", "Time"])
        try:
            old_msgs = pd.read_csv("chat_log.csv")
            chat_log = pd.concat([old_msgs, new_msg], ignore_index=True)
        except:
            chat_log = new_msg
        chat_log.to_csv("chat_log.csv", index=False)

    # Display messages
    if st.checkbox("Show Messages"):
        try:
            msgs = pd.read_csv("chat_log.csv")
            for _, row in msgs.iterrows():
                st.markdown(f"**{row['Name']}** ({row['Time']}): {row['Message']}")
        except:
            st.info("No messages yet.")

# === Images Section ===
if menu_option == "Images":
    st.sidebar.success("You are on the Images page")
    st.title("Image Gallery")

    # Load images from a directory
    image_dir = "/Users/sandeep/VSCODE/Plant_Disease_Detection/predictions"  # Change this to your image directory
    if not os.path.exists(image_dir):
        st.error("Images not found. Try uploading images first.")
    else:
        images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            for img_file in images:
                img_path = os.path.join(image_dir, img_file)
                img = Image.open(img_path)
                st.image(img, caption=img_file, use_container_width=True)
        else:
            st.info("No images found.")

import streamlit as st
import json
import random

def generate_quiz_questions(disease_data, num_questions=5):
    # Only select diseases that have at least one symptom
    available_diseases = [
        (disease_name, info)
        for disease_name, info in disease_data.items()
        if info.get("symptoms") and len(info["symptoms"]) > 0
    ]

    # Generate up to num_questions random questions
    questions = []
    n = min(num_questions, len(available_diseases))
    for _ in range(n):
        disease_name, info = random.choice(available_diseases)
        # Choose up to 4 unique symptoms, always at least 1
        options = random.sample(info["symptoms"], min(4, len(info["symptoms"])))
        correct_answer = options[0]
        explanation = f"The symptoms of {disease_name} include: {', '.join(info['symptoms'])}."
        questions.append({
            "question": f"What are the symptoms of {disease_name}?",
            "options": options,
            "answer": correct_answer,
            "explanation": explanation
        })
    return questions

# === PlantPedia Section ===
if menu_option == "PlantPedia":
    st.sidebar.success("You are on the PlantPedia page")
    st.title("PlantPedia - Your Plant Encyclopedia")

    # Learn about plants and diseases
    try:
        # Load disease information from JSON
        with open("disease_info.json", "r") as f:
            disease_data = json.load(f)

        # 🔍 Search bar
        search_query = st.text_input("Search for a plant or disease", "").lower().strip()

        # Display available plants and diseases with a random image for each disease
        st.subheader("Available Plants and Diseases")

        # Filter diseases based on search query
        filtered_data = {
            disease: info for disease, info in disease_data.items()
            if search_query in disease.lower() or search_query in info["plant"].lower()
        } if search_query else disease_data

        if not filtered_data:
            st.warning("No results found. Try another search term!")
        else:
            for disease, info in filtered_data.items():
                # Display disease information
                st.markdown(f"### {disease}")
                # Display a random image for the disease
                show_random_image(disease)
                st.write(f"**Plant**: {info['plant']}")
                st.write(f"**Disease Name**: {disease}")
                st.write("**Symptoms:**")
                for symptom in info.get("symptoms", []):
                    st.write(f"• {symptom}")
                st.write("**Causes:**")
                for cause in info.get("causes", []):
                    st.write(f"• {cause}")
                st.write(f"**Severity (Mild)**: {info.get('severity_mild', 'N/A')}")
                st.write(f"**Severity (Moderate)**: {info.get('severity_moderate', 'N/A')}")
                st.write(f"**Severity (Severe)**: {info.get('severity_severe', 'N/A')}")
                st.write(f"**Progression (in days)**: {info.get('progression_days', 'N/A')}")
                st.write("**Treatment:**")
                for step in info.get("treatment", []):
                    st.write(f"• {step}")
                st.write("**Prevention:**")
                for method in info.get("prevention", []):
                    st.write(f"• {method}")
                st.markdown("---")

    except FileNotFoundError:
        st.error("Disease information file not found.")


    # Quiz Arena
    #st.subheader("Quiz Arena")
    #st.markdown("""Test your knowledge about plant diseases! Answer the following questions:""")

    #try:
        #with open("disease_info.json", "r") as f:
            #disease_data = json.load(f)

        # Generate and store questions in session state if not already present or if restarting
        #if "quiz_questions" not in st.session_state or st.session_state.get("restart_quiz", False):
            #st.session_state.quiz_questions = generate_quiz_questions(disease_data)
            # Reset answer tracking state
            #for idx in range(len(st.session_state.quiz_questions)):
                #st.session_state[f"answered_{idx}"] = False
          #  st.session_state.restart_quiz = False

       # questions = st.session_state.quiz_questions

       # if len(questions) == 0:
           # st.info("No diseases with symptoms found in the database. Add more data to enable the quiz!")
        #else:
            #for idx, q in enumerate(questions, start=1):
              #  st.subheader(f"Question {idx}: {q['question']}")
              #  user_answer = st.radio(
              #      "Choose an option:",
              #      q["options"],
              #      key=f"radio_{idx}"
              #  )
              #  if st.button(f"Submit Answer for Question {idx}", key=f"submit_{idx}"):
               #     st.session_state[f"answered_{idx}"] = True
               #     st.session_state[f"user_answer_{idx}"] = user_answer

                # Show result if answered
               # if st.session_state.get(f"answered_{idx}", False):
               #     if st.session_state.get(f"user_answer_{idx}") == q["answer"]:
               #         st.success("Correct!")
               #     else:
                #        st.error("Incorrect!")
                #    st.info(f"Explanation: {q['explanation']}")

        # Restart button to regenerate questions
       # if st.button("Restart Quiz"):
           # st.session_state.restart_quiz = True
           # st.rerun()

    #except FileNotFoundError:
       # st.error("Questions not found.")
        
# === PlantLog Section ===
if menu_option == "PlantLog":
    st.sidebar.success("You are on the PlantLog page")
    st.title("PlantLog - Your Plant Diary")

    # User-specific knowledge file
    user_name = st.text_input("Enter your name to personalize your PlantLog:")
    if user_name:
        knowledge_file = f"{user_name}_knowledge.txt"
        if not os.path.exists(knowledge_file):
            with open(knowledge_file, "w") as f:
                f.write("User-specific knowledge base for plant diseases.\n")

    # Plant Log Form
    with st.form("plant_log_form"):
        plant_name = st.text_input("Plant Name")
        disease_name = st.text_input("Disease Name")
        symptoms = st.text_area("Symptoms")
        treatment = st.text_area("Treatment")
        date = st.date_input("Date", datetime.now())
        submit_button = st.form_submit_button("Save Log")

        if submit_button:
            if plant_name and disease_name and symptoms and treatment and user_name:
                # Save to CSV
                log_entry = [date, plant_name, disease_name, symptoms, treatment]
                with open("plant_log.csv", "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(log_entry)

                # Append to user-specific knowledge file
                with open(knowledge_file, "a") as f:
                    f.write(f"\nDate: {date}\nPlant: {plant_name}\nDisease: {disease_name}\nSymptoms: {symptoms}\nTreatment: {treatment}\n")

                st.success("Log saved successfully!")
            else:
                st.error("Please fill in all fields and enter your name.")

    # Display logs
    if st.checkbox("Show Plant Logs"):
        try:
            logs_df = pd.read_csv("plant_log.csv")
            st.dataframe(logs_df)
        except FileNotFoundError:
            st.info("No logs found. Start logging your plants!")

    # Chatbot integration with reminders
    if user_name and st.checkbox("Enable Chatbot Reminders"):
        try:
            with open(knowledge_file, "r") as f:
                user_knowledge = f.read()
            
            with open("knowledge_base.json", "r") as f:
                knowledge_base = json.load(f)

            st.markdown("### Chatbot Interaction:")
            user_query = st.text_input("Ask PlantPal about your plants or diseases:")
            if user_query:
                with st.spinner("PlantPal is thinking..."):
                    try:
                        prompt = f"""You are PlantPal, an expert assistant on plant diseases and detection models.
    Use the user's personalized knowledge base below to provide reminders or answer questions.
    If the question is unrelated or the answer isn't available, say "I'm not sure based on my current knowledge."

    ### Knowledge Base:
    {knowledge_base}
    
    ### User Knowledge Base:
    {user_knowledge}

    ### Question:
    {user_query}

    ### Answer:"""

                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2
                        )
                        answer = response['choices'][0]['message']['content']
                        st.success(answer)

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        except FileNotFoundError:
            st.warning("No knowledge base found for this user. Start logging your plants!")

# === Stats Section ===
if menu_option == "Stats":
    st.sidebar.success("You are on the Stats page")
    st.title("Statistics and Insights")

    # Load and display stats from CSV
    try:
        stats_df = pd.read_csv("plant_log.csv")
        st.dataframe(stats_df)

        # Show basic stats
        st.subheader("Basic Statistics")
        st.write(f"Total Logs: {len(stats_df)}")
        st.write(f"Unique Plants: {stats_df['Plant Name'].nunique()}")
        st.write(f"Unique Diseases: {stats_df['Disease Name'].nunique()}")

        # Show most common diseases
        st.subheader("Most Common Diseases")
        disease_counts = stats_df['Disease Name'].value_counts().head(10)
        st.bar_chart(disease_counts)

    except FileNotFoundError:
        st.info("No logs found. Start logging your plants!")

    # Show a line graph of how diseases of specific plants have progressed over time based on confidence score of the predictions
    st.subheader("Disease Progression Over Time")
    try:
        if "last_prediction" in st.session_state and "last_image_path" in st.session_state and "confidence" in st.session_state:
            # Load the last prediction data
            last_prediction = st.session_state["last_prediction"]
            last_image_path = st.session_state["last_image_path"]
            confidence_score = st.session_state["confidence"]

            # Create a DataFrame for the last prediction
            last_data = {
                "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "Plant Name": ["Unknown"],
                "Disease Name": [last_prediction],
                "Confidence Score": [confidence_score],
                "Symptoms": ["Unknown"],
                "Treatment": ["Unknown"]
            }
            last_df = pd.DataFrame(last_data)

            # Append to existing stats DataFrame
            stats_df = pd.concat([stats_df, last_df], ignore_index=True)
            stats_df.to_csv("plant_log.csv", index=False)

            # Show the updated stats
            st.dataframe(stats_df)

            # Plot disease progression over time
            st.subheader("Disease Progression Over Time")
            if "Date" in stats_df.columns and "Confidence Score" in stats_df.columns:
                stats_df["Date"] = pd.to_datetime(stats_df["Date"])
                progression_data = stats_df.groupby("Date")["Confidence Score"].mean()
                st.line_chart(progression_data)

    except Exception as e:
        st.error(f"Error updating stats: {str(e)}")


    # Create a download button for the stats CSV and the line graph as pdf and jpg respectively
    st.subheader("Download Statistics")
    if st.button("Download Stats CSV"):
        try:
            stats_df.to_csv("plant_stats.csv", index=False)
            with open("plant_stats.csv", "rb") as f:
                st.download_button(
                    label="Download CSV",
                    data=f,
                    file_name="plant_stats.csv",
                    mime="text/csv"
                )
            st.success("CSV downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading CSV: {str(e)}")
    
    if menu_option == "Current Events":
        st.sidebar.success("You are on the Geo-Notifications page")
        st.title("Region-Specific Notifications")

        city, region, country = get_geo_location()
        if city and region and country:
            st.success(f"Your location: {city}, {region}, {country}")
            
            # Fetch current events based on user's location using a library

            def fetch_events_by_location(region):
                try:
                    import feedparser

                    feed = feedparser.parse("https://plantwiseplus.cabi.org/rss?region={region}")
                    for entry in feed.entries:
                        print(entry.title)
                        print(entry.link)

                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching events: {e}")
                    return []

            region_events = fetch_events_by_location(region)
            
            if region_events:
                st.subheader("Upcoming Events in Your Region:")
                for event in region_events:
                    st.write(f"- {event}")
            else:
                st.info("No events found for your region.")
        else:
            st.error("Unable to fetch your location.")

import pandas as pd
from fpdf import FPDF
import smtplib
import random
import matplotlib.pyplot as plt

if menu_option == "Trends":
    st.sidebar.success("You are on the Trends page")
    st.title("Disease Trends")

    log_file = "disease_log.csv"
    if not os.path.exists(log_file):
        st.error("No log file found. Start logging your plants!")
    else:
        # Load log
        df = pd.read_csv(log_file, header=None)
        df.columns = ["Plant", "Timestamp", "Disease", "Confidence", "ImagePath"]

        # Display unique plant names
        unique_plants = df["Plant"].unique()
        selected_plant = st.selectbox("Select a plant to view trends:", unique_plants)

        if selected_plant:
            # Filter by selected plant
            df_plant = df[df["Plant"] == selected_plant]
            if df_plant.empty:
                st.warning(f"No records found for plant '{selected_plant}'.")
            else:
                # Convert timestamp to datetime
                df_plant["Timestamp"] = pd.to_datetime(df_plant["Timestamp"])

                # Plotting
                st.subheader(f"Disease Confidence Over Time - {selected_plant}")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_plant["Timestamp"], df_plant["Confidence"], marker='o', linestyle='-')
                ax.set_title(f"Disease Confidence Over Time - {selected_plant}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Confidence Score")
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True)
                st.pyplot(fig)

                # Display clickable IDs
                st.subheader("View Individual Records")
                for _, row in df_plant.iterrows():
                    if st.button(f"View Plot for ID: {row['ImagePath']}"):
                        st.subheader(f"Details for ID: {row['ImagePath']}")
                        st.write(f"Plant: {row['Plant']}")
                        st.write(f"Disease: {row['Disease']}")
                        st.write(f"Confidence: {row['Confidence']}")
                        st.write(f"Timestamp: {row['Timestamp']}")
                        # Plot individual record
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot([row["Timestamp"]], [row["Confidence"]], marker='o', linestyle='-', color='red')
                        ax.set_title(f"Disease Confidence for ID: {row['ImagePath']}")
                        ax.set_xlabel("Time")
                        ax.set_ylabel("Confidence Score")
                        ax.grid(True)
                        st.pyplot(fig)

                        # Create a download button for the trends data as PDF

                        trends_pdf = f"{selected_plant}_trends.pdf"
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        pdf.cell(200, 10, txt=f"Disease Trends for {selected_plant}", ln=True, align='C')
                        pdf.ln(10)

                        for _, row in df_plant.iterrows():
                            pdf.cell(200, 10, txt=f"Timestamp: {row['Timestamp']}", ln=True)
                            pdf.cell(200, 10, txt=f"Disease: {row['Disease']}", ln=True)
                            pdf.cell(200, 10, txt=f"Confidence: {row['Confidence']}", ln=True)
                            pdf.cell(200, 10, txt=f"Image Path: {row['ImagePath']}", ln=True)
                            pdf.ln(5)

                        pdf.output(trends_pdf)

                        with open(trends_pdf, "rb") as f:
                            st.download_button(
                                label="Download Trends Data as PDF",
                                data=f,
                                file_name=trends_pdf,
                                mime="application/pdf"
                            )

# === Settings Section ===
if menu_option == "Settings":
    st.sidebar.success("You are on the Settings page")
    st.title("Settings")

    # Language Selection
    st.subheader("Language Settings")
    selected_language = st.selectbox("Select Language", list(languages.keys()), index=0, key="settings_language")
    st.session_state["selected_language"] = selected_language
    st.markdown(f"**Current Language:** {selected_language}")
    allow_notifications = st.checkbox("Enable Notifications", value=True, key="settings_notifications")
    if allow_notifications:
        st.success("Notifications are enabled.")
    else:
        st.warning("Notifications are disabled.")

    # Theme Selection
    st.subheader("Theme Settings")
    theme = st.selectbox("Select Theme", ["Light", "Dark"], index=0, key="settings_theme")
    if theme == "Dark":
        st.markdown("""
        <style>
        body {
            background-color: #121212;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        body {
            background-color: white;
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)

# === Upgrade to Pro Section ===
if menu_option == "Upgrade to Pro":
    st.sidebar.success("You are on the Upgrade to Pro page")
    st.title("Upgrade to Pro")

    st.markdown("""
    **Upgrade to Pro** for advanced features:
    - More classes
    - Unlimited predictions
    - Better accuracy
    - Priority support
    - Customized reports
    - Chatbot integration for personalized plant care advice
    """)

    if st.button("Avail a free trial for 7 days"):
        st.success("You have availed a free trial! Enjoy the Pro features for 7 days.")
        # Make a file called trial.txt which would have the users who have availed the free trial
        with open("trial.txt", "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {st.session_state.get('user_name', 'Anonymous')} availed a free trial\n")
    
    # When a particular user clicks on free trial, it should be logged in a file called trial.txt with the timestamp and user name and when 7 days is over, remove the user from trial.txt and disable the trial button for the user and save the user in trial_over.txt
    # Find the time for the user who has availed the free trial and if it is more than 7 days, remove the user from trial.txt and add the user to trial_over.txt
    if os.path.exists("trial.txt"):
        with open("trial.txt", "r") as f:
            trial_users = f.readlines()
        
        current_time = datetime.now()
        trial_over_users = []
        for line in trial_users:
            timestamp, user_name = line.strip().split(" - ")
            trial_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            if (current_time - trial_time).days > 7:
                trial_over_users.append(user_name)
        
        # Remove users who have completed their trial
        if trial_over_users:
            with open("trial.txt", "w") as f:
                for line in trial_users:
                    if not any(user in line for user in trial_over_users):
                        f.write(line)
            
            # Save to trial_over.txt
            with open("trial_over.txt", "a") as f:
                for user in trial_over_users:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {user} trial over\n")
            
            # Disable the trial button for these users
            st.warning("Your free trial has ended. Please upgrade to Pro to continue using the advanced features.")
            st.button("Avail a free trial for 7 days", disabled=True)

    # When a particular user clicks on upgrade to pro, it should be logged in a file called upgrade.txt with the timestamp and user name
    st.markdown("**Price:** $0.99/month or $9.99/year (20% off)")
    st.markdown("**Contact us at:** plantdoctor@gmail.com")

    if st.button("Upgrade Now"):
        # Show two options: Pay $0.99/month or $9.99/year
        st.markdown("### Choose your plan:")
        plan = st.radio("Select Plan", ["$0.99/month", "$9.99/year"], index=0)
        if plan == "$0.99/month":
            st.success("You have chosen the monthly plan. Please contact us at plantdoctor@gmail.com for more information.")
            with open("monthly_upgrade.txt", "a") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {st.session_state.get('user_name', 'Anonymous')} upgraded to Pro\n")
                # Remove the user from trial.txt if they are in it
                if os.path.exists("trial.txt"):
                    with open("trial.txt", "r") as trial_file:
                        trial_users = trial_file.readlines()
                    with open("trial.txt", "w") as trial_file:
                        for line in trial_users:
                            if st.session_state.get('user_name', 'Anonymous') not in line:
                                trial_file.write(line)

            # Also disable the upgrade and free trial buttons for the user
            st.button("Avail a free trial for 7 days", disabled=True)
            st.button("Upgrade Now", disabled=True)
            # Add a note that the user should be able to renew their subscription after a month
            st.markdown("**Note:** After a month, you will be able to renew your subscription. Please ensure you have a valid payment method.")
            # Add a note that the user should be deleted from monthly_upgrade.txt after a month
            st.markdown("**Note:** You will be removed from the monthly upgrade list after a month. Please ensure you renew your subscription to continue enjoying the Pro features.")
                
            # After a month, the user should be able to renew their subscription and the user should be deleted from monthly_upgrade.txt after a month
            # Delete the user from monthly_upgrade.txt after a month
            current_time = datetime.now()
            if os.path.exists("monthly_upgrade.txt"):
                with open("monthly_upgrade.txt", "r") as f:
                    monthly_users = f.readlines()
                
                # Remove users who have completed their month
                with open("monthly_upgrade.txt", "w") as f:
                    for line in monthly_users:
                        timestamp, user_name = line.strip().split(" - ")
                        upgrade_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                        if (current_time - upgrade_time).days < 30:
                            f.write(line)
                
                # Also enable the upgrade button for users who have completed their month if they are not in monthly_upgrade.txt
                if st.session_state.get('user_name', 'Anonymous') not in [line.split(" - ")[1] for line in monthly_users]:
                    st.button("Upgrade Now", disabled=False)

        elif plan == "$9.99/year":
            st.success("You have chosen the yearly plan. Please contact us at plantdoctor@gmail.com for more information.")
            with open("yearly_upgrade.txt", "a") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {st.session_state.get('user_name', 'Anonymous')} upgraded to Pro\n")
                # Remove the user from trial.txt if they are in it
                if os.path.exists("trial.txt"):
                    with open("trial.txt", "r") as trial_file:
                        trial_users = trial_file.readlines()
                    with open("trial.txt", "w") as trial_file:
                        for line in trial_users:
                            if st.session_state.get('user_name', 'Anonymous') not in line:
                                trial_file.write(line)

            # Also disable the upgrade and free trial buttons for the user
            st.button("Avail a free trial for 7 days", disabled=True)
            st.button("Upgrade Now", disabled=True)
            # Add a note that the user should be able to renew their subscription after a year
            st.markdown("**Note:** After a year, you will be able to renew your subscription. Please ensure you have a valid payment method.")
            # Add a note that the user should be deleted from yearly_upgrade.txt after a year
            st.markdown("**Note:** You will be removed from the yearly upgrade list after a year. Please ensure you renew your subscription to continue enjoying the Pro features.")

            # After a year, the user should be able to renew their subscription and the user should be deleted from yearly_upgrade.txt after a year
            # Delete the user from yearly_upgrade.txt after a year
            current_time = datetime.now()
            if os.path.exists("yearly_upgrade.txt"):
                with open("yearly_upgrade.txt", "r") as f:
                    yearly_users = f.readlines()
                
                # Remove users who have completed their year
                with open("yearly_upgrade.txt", "w") as f:
                    for line in yearly_users:
                        timestamp, user_name = line.strip().split(" - ")
                        upgrade_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                        if (current_time - upgrade_time).days < 365:
                            f.write(line)
                
                # Also enable the upgrade button for users who have completed their year if they are not in yearly_upgrade.txt
                if st.session_state.get('user_name', 'Anonymous') not in [line.split(" - ")[1] for line in yearly_users]:
                    st.button("Upgrade Now", disabled=False)

# === Play Section ===
if menu_option == "Predict Play":
    st.sidebar.success("You are on the Play page")
    st.title("Play")
    st.markdown("""
    Welcome to the Play section! Here you can:
    - **Play** interactive quizzes to test your knowledge.
    - **Learn** about plant diseases in a fun way.
    """)

    dataset_dir = os.path.join(os.path.dirname(__file__), "CombinedDataset")

    # Build mapping: class -> [list of image paths]
    class_to_images = {}
    for root, dirs, files in os.walk(dataset_dir):
        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            class_name = os.path.basename(root)  # ✅ only the last folder name
            full_paths = [os.path.join(root, f) for f in images]
            if class_name not in class_to_images:
                class_to_images[class_name] = []
            class_to_images[class_name].extend(full_paths)

    plant_disease_classes = list(class_to_images.keys())

    def new_question():
        """Helper to pick a new random question."""
        if not plant_disease_classes:
            st.error("No plant disease classes found.")
            return

        st.session_state.selected_class = random.choice(plant_disease_classes)
        st.session_state.selected_image = random.choice(
            class_to_images[st.session_state.selected_class]
        )

        # Generate options
        options = [st.session_state.selected_class]
        while len(options) < 4:
            random_class = random.choice(plant_disease_classes)
            if random_class not in options:
                options.append(random_class)
        random.shuffle(options)
        st.session_state.options = options

    # If no question yet, generate one
    if "selected_class" not in st.session_state:
        new_question()

    # Display question
    st.image(st.session_state.selected_image, caption="Can you guess the disease?")

    st.subheader("Options:")
    for idx, option in enumerate(st.session_state.options, start=1):
        st.write(f"{idx}. {option}")

    # User input
    user_guess = st.text_input("Your Guess:")

    if st.button("Submit Guess"):
        if user_guess.lower() == st.session_state.selected_class.lower():
            st.success("✅ Correct! Well done!")
        else:
            st.error(f"❌ Incorrect! The correct answer was: {st.session_state.selected_class}")

        # Load new question automatically after submission
        new_question()

    # Add a restart button to reset the game anytime
    if st.button("Restart Game"):
        new_question()
        st.info("🔄 Game restarted! Try again.")

# === Plant Saathi Section ===
if menu_option == "Plant Saathi":
    import json
    import streamlit as st
    from google import genai
    from datetime import datetime

    # -------------------------------
    # 1️⃣ Load Disease Data
    # -------------------------------
    with open("disease_info.json", "r") as f:
        disease_data = json.load(f)

    # -------------------------------
    # 2️⃣ Helper Function: Fetch Disease Info
    # -------------------------------
    def fetch_disease_info(query):
        query_lower = query.lower()
        for disease, info in disease_data.items():
            if disease.lower() in query_lower or info["plant"].lower() in query_lower:
                return info
        return None

    # -------------------------------
    # 3️⃣ Initialize Gemini Client
    # -------------------------------
    client = genai.Client()  # Make sure GEMINI_API_KEY is set

    # -------------------------------
    # 4️⃣ Streamlit UI Setup
    # -------------------------------
    st.set_page_config(page_title="Plant Doctor Chatbot", page_icon="🌱", layout="wide")
    st.title("🌱 Plant Doctor Chatbot MVP")
    st.markdown("Ask about plant diseases, symptoms, treatment, or prevention strategies!")

    # Initialize session state for chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar with Clear Chat
    with st.sidebar:
        if st.button("Clear Chat"):
            st.session_state.history = []
            st.experimental_rerun()

    # User input
    user_input = st.text_area("Type your question here:", height=80)

    if st.button("Send") and user_input:
        # Add user message to history
        st.session_state.history.append({
            "role": "user",
            "content": user_input,
            "time": datetime.now().strftime("%H:%M")
        })

        # Fetch disease info
        disease_info = fetch_disease_info(user_input)
        if disease_info:
            info_text = f"""
    Plant: {disease_info['plant']}
    Disease: {disease_info['disease']}
    Causes: {', '.join(disease_info['causes'])}
    Symptoms: {', '.join(disease_info['symptoms'])}
    Treatment: {', '.join(disease_info['treatment'])}
    Medicines: {', '.join(disease_info['medicines'])}
    Prevention: {', '.join(disease_info['prevention'])}
    Progression Days: {disease_info['progression_days']}
    """
        else:
            info_text = "No specific disease information found. Please describe symptoms or plant name."

        # Build context for multi-turn conversation (last 5 messages)
        context_text = ""
        for msg in st.session_state.history[-5:]:
            role = msg["role"]
            content = msg["content"]
            context_text += f"{role.capitalize()}: {content}\n"

        prompt = f"""
    You are a friendly plant doctor chatbot.
    Use the following disease info to answer questions:
    {info_text}

    Conversation context:
    {context_text}

    Answer clearly and politely. Include detailed explanations if needed.
    """

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            bot_message = response.text
        except Exception as e:
            bot_message = f"Error connecting to Gemini API: {e}"

        # Add bot message to history
        st.session_state.history.append({
            "role": "bot",
            "content": bot_message,
            "time": datetime.now().strftime("%H:%M")
        })

    # -------------------------------
    # 5️⃣ Display Chat in Clean Interface with Colors
    # -------------------------------
    for msg in st.session_state.history:
        timestamp = msg.get("time", "")
        if msg["role"] == "user":
            st.markdown(
                f"""
    <div style='background-color:#CFEFFF;color:#000;padding:10px;border-radius:10px;margin:5px 0px;width:70%;float:right;'>
    <b>You ({timestamp}):</b><br>{msg['content']}
    </div>
    <div style="clear:both;"></div>
    """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
    <div style='background-color:#E2FFE2;color:#000;padding:10px;border-radius:10px;margin:5px 0px;width:70%;float:left;'>
    <b>Plant Doctor ({timestamp}):</b><br>{msg['content']}
    </div>
    <div style="clear:both;"></div>
    """,
                unsafe_allow_html=True
            )
            # Optional: show images if present in JSON
            disease_name = msg['content'].split("**")[1] if "**" in msg['content'] else None
            if disease_name and disease_name in disease_data:
                for img_url in disease_data[disease_name].get("images", []):
                    st.image(img_url, width=300)