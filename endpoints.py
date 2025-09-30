import os
import json
import base64
import io
import time
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
from fpdf import FPDF
from flask import Flask, request, jsonify, send_file, Response, render_template_string, send_from_directory
from flask_cors import CORS
import requests

# Try to import ML dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    import cv2
    from torchvision.models import resnet18 as torch_resnet18
    ML_AVAILABLE = True
    print("✅ ML dependencies loaded successfully")
except ImportError as e:
    print(f"⚠️ ML dependencies not available: {e}")
    print("🔄 Running in demo mode")
    ML_AVAILABLE = False

# Class names for plant diseases
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

app = Flask(__name__, static_folder="assets", static_url_path="")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
model = None
test_transform = None
device = None
disease_info_data = None
predictions_cache = {}  # Store predictions temporarily
gradcam_cache = {}  # Store gradcam images temporarily

def create_resnet18_model():
    """Create ResNet18 model for plant disease classification"""
    if not ML_AVAILABLE:
        return None
    
    try:
        # Create ResNet18 model
        model = torch_resnet18(pretrained=False)
        # Modify the final layer for our number of classes
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        return model
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return None

def load_disease_info():
    """Load disease information from JSON file"""
    global disease_info_data
    try:
        disease_info_path = Path(__file__).parent / 'disease_info.json'
        if disease_info_path.exists():
            with open(disease_info_path, 'r') as f:
                disease_info_data = json.load(f)
            print("✅ Disease information loaded successfully")
        else:
            print("⚠️ disease_info.json not found, creating empty database")
            disease_info_data = {}
    except Exception as e:
        print(f"❌ Error loading disease info: {e}")
        disease_info_data = {}

def load_model():
    """Load the ResNet18 model if ML dependencies are available"""
    global model, test_transform, device
    
    if not ML_AVAILABLE:
        print("⚠️ ML dependencies not available, using demo mode")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Define transform (same as in Streamlit version)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        # Create model
        model = create_resnet18_model()
        if model is not None:
            # Try to load pre-trained weights if available
            model_paths = [
                Path(__file__).parent / "models" / "resnet18_plant_disease_detection.pth",
                Path(__file__).parent / "resnet18_plant_disease_detection.pth",
                "resnet18_plant_disease_detection.pth"
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if model_path.exists():
                    try:
                        state_dict = torch.load(model_path, map_location=device)
                        model.load_state_dict(state_dict, strict=True)
                        print(f"✅ Model weights loaded from {model_path}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"⚠️ Failed to load weights from {model_path}: {e}")
                        continue
            
            if not model_loaded:
                print("⚠️ No model weights found, using random initialization")
            
            model.eval().to(device)
        else:
            print("⚠️ Model creation failed, using demo mode")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

def predict_image_flask(image):
    """Predict disease from image using the loaded model (same logic as Streamlit)"""
    if model is None or not ML_AVAILABLE:
        # Demo prediction
        import random
        diseases = [
            'Tomato Late Blight',
            'Apple Scab', 
            'Potato Early Blight',
            'Corn Leaf Blight',
            'Grape Black Rot'
        ]
        prediction_idx = random.randint(0, len(class_names) - 1)
        prediction = class_names[prediction_idx]
        confidence = random.uniform(0.75, 0.95)
        return prediction_idx, prediction, confidence, True  # True indicates demo mode
    
    try:
        # Real ML prediction (same as Streamlit version)
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image_tensor = test_transform(image).unsqueeze(0).to(device)
        
        with torch.inference_mode():
            output = model(image_tensor)  # raw logits
            probs = F.softmax(output, dim=1)  # convert to probabilities
            predicted_idx = probs.argmax(dim=1).item()
            confidence = probs[0][predicted_idx].item()
        
        predicted_class = class_names[predicted_idx]
        return predicted_idx, predicted_class, confidence, False  # False indicates real ML mode
        
    except Exception as e:
        print(f"❌ Error in ML prediction: {e}")
        # Fallback to demo
        import random
        diseases = ['Tomato Late Blight', 'Apple Scab', 'Peach Leaf Spot']
        prediction_idx = random.randint(0, len(class_names) - 1)
        prediction = class_names[prediction_idx]
        confidence = random.uniform(0.75, 0.95)
        return prediction_idx, prediction, confidence, True

def generate_gradcam(image, target_class_idx=None):
    """Generate GradCAM heatmap for model explanation"""
    if model is None or not ML_AVAILABLE:
        # Demo GradCAM
        return generate_demo_gradcam(image), True
    
    try:
        # Real GradCAM implementation
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image_tensor = test_transform(image).unsqueeze(0).to(device)
        image_tensor.requires_grad_(True)
        
        gradients = []
        activations = []
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        # Find target layer (last conv layer of ResNet18)
        target_layer = model.layer4[-1].conv2
        
        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        # Forward pass
        output = model(image_tensor)
        
        if target_class_idx is None:
            target_class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        model.zero_grad()
        output[0, target_class_idx].backward()
        
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
        
        if not gradients or not activations:
            return generate_demo_gradcam(image), True
        
        # Generate GradCAM
        grads = gradients[0].cpu().data.numpy()[0]
        acts = activations[0].cpu().data.numpy()[0]
        
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        cam = cv2.resize(cam, (224, 224))
        
        # Create overlay
        orig_img = np.array(image.resize((224, 224)))
        orig_img = orig_img.astype(np.float32) / 255.0
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        overlay = heatmap * 0.4 + orig_img * 0.6
        overlay = overlay / np.max(overlay)
        
        return overlay, False
        
    except Exception as e:
        print(f"❌ Error generating real GradCAM: {e}")
        return generate_demo_gradcam(image), True

def generate_demo_gradcam(image):
    """Generate demo GradCAM for when ML is not available"""
    img_array = np.array(image.resize((224, 224)))
    
    # Create a simple circular heatmap pattern
    center_x, center_y = 112, 112
    y, x = np.ogrid[:224, :224]
    mask = (x - center_x)**2 + (y - center_y)**2 <= 80**2
    
    heatmap = np.zeros((224, 224, 3))
    heatmap[mask] = [1, 0.5, 0]  # Orange-red color
    
    # Blend with original image
    overlay = (img_array.astype(np.float32) / 255.0) * 0.6 + heatmap * 0.4
    
    return overlay

def get_disease_info(prediction):
    """Get disease information from the JSON database"""
    if disease_info_data and prediction in disease_info_data:
        return disease_info_data[prediction]
    return None

def format_disease_info_string(disease_info):
    """Format disease info into a readable string for the frontend"""
    if not disease_info:
        return "No specific information available for this condition."
    
    if isinstance(disease_info, dict):
        # Build comprehensive description from all available fields
        info_parts = []
        
        # Add basic info
        if disease_info.get('plant'):
            info_parts.append(f"Plant: {disease_info['plant']}")
        
        if disease_info.get('disease'):
            info_parts.append(f"Disease: {disease_info['disease']}")
        
        # Add symptoms
        if disease_info.get('symptoms') and len(disease_info['symptoms']) > 0:
            symptoms_str = ", ".join(disease_info['symptoms'][:3])
            info_parts.append(f"Symptoms: {symptoms_str}")
        
        # Add causes
        if disease_info.get('causes') and len(disease_info['causes']) > 0:
            causes_str = ", ".join(disease_info['causes'][:2])
            info_parts.append(f"Causes: {causes_str}")
        
        # Add severity info if available
        if disease_info.get('severity_moderate'):
            info_parts.append(f"Impact: {disease_info['severity_moderate']}")
        
        # Add progression info
        if disease_info.get('progression_days'):
            info_parts.append(f"Progression: {disease_info['progression_days']} days")
        
        return " | ".join(info_parts) if info_parts else "Disease information available"
    else:
        return str(disease_info)

def clean_text_for_pdf(text):
    """Clean text to remove unicode characters that can't be encoded in latin-1"""
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Replace unicode characters with latin-1 compatible alternatives
    replacements = {
        '\u2022': '-',  # bullet point
        '\u2013': '-',  # en dash
        '\u2014': '--', # em dash
        '\u2018': "'",  # left single quotation mark
        '\u2019': "'",  # right single quotation mark
        '\u201C': '"',  # left double quotation mark
        '\u201D': '"',  # right double quotation mark
        '\u2026': '...', # horizontal ellipsis
        '\u00B0': ' degrees',  # degree symbol
        '\u00A0': ' ',  # non-breaking space
    }
    
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    
    # Remove any remaining non-latin-1 characters
    try:
        text.encode('latin-1')
        return text
    except UnicodeEncodeError:
        # If there are still problematic characters, encode and decode to remove them
        return text.encode('ascii', 'ignore').decode('ascii')

def generate_pdf_report(image, prediction, confidence, disease_info, is_demo=False):
    """Generate PDF report with prediction results using the updated disease_info structure"""
    try:
        from fpdf import FPDF
    except ImportError:
        print("❌ FPDF not available - install with: pip install fpdf2")
        return None
    
    try:
        # Create PDF with explicit format
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        title = 'Plant Disease Detection Report'
        if is_demo:
            title += ' (Demo Mode)'
        pdf.cell(0, 10, clean_text_for_pdf(title), 0, 1, 'C')
        pdf.ln(10)
        
        # Basic info
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, clean_text_for_pdf(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'), 0, 1)
        pdf.cell(0, 10, clean_text_for_pdf(f'Prediction: {prediction}'), 0, 1)
        pdf.cell(0, 10, clean_text_for_pdf(f'Confidence: {confidence:.2%}'), 0, 1)
        
        if is_demo:
            pdf.ln(5)
            pdf.set_font('Arial', 'I', 10)
            pdf.multi_cell(0, 8, clean_text_for_pdf('Note: This report was generated in demo mode with simulated predictions.'))
            pdf.ln(3)
        
        pdf.ln(5)
        
        # Disease information using updated structure
        if disease_info:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, clean_text_for_pdf('Disease Information:'), 0, 1)
            pdf.set_font('Arial', '', 10)
            
            if disease_info.get('plant'):
                pdf.multi_cell(0, 8, clean_text_for_pdf(f"Plant: {disease_info['plant']}"))
            
            if disease_info.get('disease'):
                pdf.multi_cell(0, 8, clean_text_for_pdf(f"Disease: {disease_info['disease']}"))
            
            # Symptoms
            if disease_info.get('symptoms') and len(disease_info['symptoms']) > 0:
                pdf.multi_cell(0, 8, clean_text_for_pdf("Symptoms:"))
                for symptom in disease_info['symptoms'][:5]:  # Limit to first 5
                    if symptom:
                        pdf.multi_cell(0, 6, clean_text_for_pdf(f"- {str(symptom)}"))
                pdf.ln(3)
            
            # Causes
            if disease_info.get('causes') and len(disease_info['causes']) > 0:
                pdf.multi_cell(0, 8, clean_text_for_pdf("Causes:"))
                for cause in disease_info['causes'][:3]:  # Limit to first 3
                    if cause:
                        pdf.multi_cell(0, 6, clean_text_for_pdf(f"- {str(cause)}"))
                pdf.ln(3)
            
            # Severity Information
            severities = [
                ('Mild', disease_info.get('severity_mild')),
                ('Moderate', disease_info.get('severity_moderate')), 
                ('Severe', disease_info.get('severity_severe'))
            ]
            
            has_severity = any(severity[1] for severity in severities)
            if has_severity:
                pdf.multi_cell(0, 8, clean_text_for_pdf("Severity Levels:"))
                for level, description in severities:
                    if description:
                        pdf.multi_cell(0, 6, clean_text_for_pdf(f"- {level}: {str(description)}"))
                pdf.ln(3)
            
            # Treatment
            if disease_info.get('treatment') and len(disease_info['treatment']) > 0:
                pdf.multi_cell(0, 8, clean_text_for_pdf("Treatment:"))
                for treatment in disease_info['treatment'][:5]:  # Limit to first 5
                    if treatment:
                        pdf.multi_cell(0, 6, clean_text_for_pdf(f"- {str(treatment)}"))
                pdf.ln(3)
            
            # Medicines
            if disease_info.get('medicines') and len(disease_info['medicines']) > 0:
                pdf.multi_cell(0, 8, clean_text_for_pdf("Recommended medicines:"))
                for medicine in disease_info['medicines'][:5]:  # Limit to first 5
                    if medicine:
                        pdf.multi_cell(0, 6, clean_text_for_pdf(f"- {str(medicine)}"))
                pdf.ln(3)
            
            # Prevention
            if disease_info.get('prevention') and len(disease_info['prevention']) > 0:
                pdf.multi_cell(0, 8, clean_text_for_pdf("Prevention:"))
                for prevention in disease_info['prevention'][:5]:  # Limit to first 5
                    if prevention:
                        pdf.multi_cell(0, 6, clean_text_for_pdf(f"- {str(prevention)}"))
                pdf.ln(3)
            
            # Progression
            if disease_info.get('progression_days'):
                pdf.multi_cell(0, 8, clean_text_for_pdf(f"Disease Progression: {disease_info['progression_days']} days"))
        
        # Generate PDF output - Create temporary file approach
        try:
            import tempfile
            import os
            
            # Create a temporary file
            temp_dir = tempfile.gettempdir()
            temp_filename = f"plant_report_{int(time.time() * 1000)}.pdf"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            # Output PDF to temporary file
            pdf.output(temp_path)
            
            # Read the file back as bytes
            with open(temp_path, 'rb') as f:
                pdf_bytes = f.read()
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass  # Ignore cleanup errors
            
            print(f"✅ PDF generated successfully, size: {len(pdf_bytes)} bytes")
            return pdf_bytes
                
        except Exception as output_error:
            print(f"❌ Error in PDF output generation: {output_error}")
            import traceback
            print(traceback.format_exc())
            return None
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        import traceback
        print(traceback.format_exc())
        return None

# Flask routes

@app.route('/')
def index():
    """Serve the home page"""
    try:
        html_path = Path(__file__).parent / 'frontend' / 'index'
        if html_path.exists():
            with open(html_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # If no index.html found, return a simple fallback
            return """
            <!DOCTYPE html>
            <html lang="en">
            <head>
            <meta charset="UTF-8">
            <title>Plant Doctor AI</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                /* General styles */
                body {
                    margin: 0;
                    font-family: 'Segoe UI', 'Arial', sans-serif;
                    background: linear-gradient(135deg, #f0fdf4 0%, #d1fae5 100%);
                    color: #222;
                }
                .min-h-screen { min-height: 100vh; }
                .container {
                    max-width: 1100px;
                    margin: 0 auto;
                    padding: 0 1rem;
                }
                .flex { display: flex; }
                .flex-between { display: flex; align-items: center; justify-content: space-between; }
                .items-center { align-items: center; }
                .gap-2 { gap: 0.5rem; }
                .gap-3 { gap: 0.75rem; }
                .text-center { text-align: center; }
                .mb-12 { margin-bottom: 3rem; }
                .mb-10 { margin-bottom: 2.5rem; }
                .mt-12 { margin-top: 3rem; }
                .mb-4 { margin-bottom: 1rem; }
                .mr-3 { margin-right: 0.75rem; }
                .center { display: flex; justify-content: center; }
                .animate-pulse {
                    animation: pulse 2s infinite;
                }
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: .7; }
                }
                .animate-bounce {
                    animation: bounce 1.5s infinite;
                }
                @keyframes bounce {
                    0%, 100% { transform: translateY(0);}
                    50% { transform: translateY(-8px);}
                }
                .animate-spin {
                    animation: spin 8s linear infinite;
                }
                @keyframes spin {
                    100% { transform: rotate(360deg);}
                }
                
                /* Header */
                .header {
                    background: #fff;
                    box-shadow: 0 2px 8px -2px rgba(16, 185, 129, 0.08);
                    border-bottom: 1px solid #bbf7d0;
                }
                .title {
                    font-size: 2rem;
                    font-weight: bold;
                    color: #065f46;
                }
                .nav {
                    display: flex;
                    align-items: center;
                    gap: 1.5rem;
                }
                .nav a {
                    color: #047857;
                    font-weight: 500;
                    text-decoration: none;
                    transition: color 0.2s;
                }
                .nav a:hover {
                    color: #065f46;
                }
                .nav-actions {
                    display: flex;
                    gap: 0.75rem;
                    margin-left: 1.5rem;
                }
                .btn {
                    padding: 0.5rem 1rem;
                    border-radius: 0.75rem;
                    font-weight: 500;
                    transition: all .2s;
                    box-shadow: none;
                    border: none;
                    cursor: pointer;
                }
                .btn-green {
                    background: #16a34a;
                    color: #fff;
                }
                .btn-green:hover {
                    background: #166534;
                    box-shadow: 0 2px 8px -2px #16a34a44;
                }
                .btn-outline-green {
                    background: #fff;
                    border: 2px solid #16a34a;
                    color: #16a34a;
                }
                .btn-outline-green:hover {
                    background: #f0fdf4;
                    box-shadow: 0 2px 8px -2px #16a34a22;
                }
                
                /* Hero */
                .hero-title {
                    font-size: 2.5rem;
                    font-weight: bold;
                    color: #065f46;
                    margin-bottom: 1.5rem;
                }
                .hero-image {
                    position: relative;
                    width: 320px;
                    height: 240px;
                    background: #fff;
                    border-radius: 1.25rem;
                    box-shadow: 0 8px 32px -8px #16a34a22;
                    overflow: hidden;
                    transition: box-shadow .3s, transform .3s;
                }
                .hero-image img {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                }
                .hero-image:hover {
                    box-shadow: 0 16px 40px -10px #16a34a44;
                    transform: scale(1.05);
                }
                .img-gradient {
                    position: absolute;
                    inset: 0;
                    background: linear-gradient(to top, #16a34a33 0%, transparent 100%);
                    pointer-events: none;
                }
                .icon-top-right {
                    position: absolute;
                    top: 1rem; right: 1rem;
                }
                .img-bottom-left {
                    position: absolute;
                    left: 1rem; bottom: 1rem;
                }
                .tag {
                    font-size: 0.95rem;
                    font-weight: 500;
                    color: #fff;
                    text-shadow: 0 2px 6px #1116;
                }
                .desc {
                    font-size: 1.25rem;
                    color: #047857;
                    max-width: 700px;
                    margin: 0 auto;
                    line-height: 1.6;
                }
                
                /* Cards */
                .card {
                    background: #fff;
                    border-radius: 1.25rem;
                    box-shadow: 0 4px 24px -8px #16a34a22;
                    padding: 2rem;
                    transition: box-shadow .3s, transform .3s;
                }
                .card:hover {
                    box-shadow: 0 8px 32px -8px #16a34a22;
                    transform: translateY(-4px);
                }
                .section-title {
                    font-size: 1.75rem;
                    font-weight: bold;
                    color: #065f46;
                    margin-bottom: 1.5rem;
                    display: flex;
                    align-items: center;
                }
                .card-text {
                    color: #374151;
                    font-size: 1.1rem;
                    line-height: 1.7;
                }
                .card-text strong {
                    color: #047857;
                }
                
                /* Stats Grid */
                .grid-stats {
                    display: grid;
                    grid-template-columns: repeat(1, 1fr);
                    gap: 1.5rem;
                }
                @media (min-width: 768px) {
                    .grid-stats { grid-template-columns: repeat(2, 1fr); }
                }
                @media (min-width: 1024px) {
                    .grid-stats { grid-template-columns: repeat(4, 1fr); }
                }
                .stat-card {
                    background: #f0f9ff;
                    border-radius: 1rem;
                    padding: 1.2rem;
                    text-align: center;
                    transition: box-shadow .3s, transform .3s;
                }
                .stat-card:hover {
                    box-shadow: 0 4px 24px -8px #0369a1;
                    transform: scale(1.05);
                }
                .stat-blue { background: linear-gradient(135deg, #eff6ff 0%, #bae6fd 100%);}
                .stat-orange { background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%);}
                .stat-red { background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);}
                .stat-purple { background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%);}
                .stat-value {
                    font-size: 2rem;
                    font-weight: bold;
                    margin-bottom: 0.2rem;
                }
                .text-blue { color: #0369a1;}
                .text-orange { color: #ea580c;}
                .text-red { color: #db2777;}
                .text-purple { color: #7c3aed;}
                
                /* Features Grid */
                .grid-features {
                    display: grid;
                    grid-template-columns: repeat(1, 1fr);
                    gap: 1.5rem;
                }
                @media (min-width: 768px) {
                    .grid-features { grid-template-columns: repeat(2, 1fr);}
                }
                @media (min-width: 1024px) {
                    .grid-features { grid-template-columns: repeat(3, 1fr);}
                }
                .feature-card {
                    display: flex;
                    align-items: flex-start;
                    gap: 1rem;
                    padding: 1rem;
                    border-radius: 1rem;
                    transition: background .3s, box-shadow .3s, transform .3s;
                }
                .feature-card:hover {
                    background: #ecfdf5;
                    box-shadow: 0 4px 24px -8px #04785722;
                    transform: scale(1.05);
                }
                .bg-green-50 { background: #f0fdf4;}
                .bg-blue-50 { background: #eff6ff;}
                .bg-purple-50 { background: #f5f3ff;}
                .bg-orange-50 { background: #fff7ed;}
                .bg-teal-50 { background: #f0fdfa;}
                .bg-indigo-50 { background: #eef2ff;}
                .feature-title {
                    font-weight: 600;
                    margin-bottom: 0.3rem;
                    color: #065f46;
                }
                .feature-desc {
                    color: #047857;
                    font-size: 1rem;
                }
                .text-teal { color: #0d9488;}
                .text-indigo { color: #4338ca;}
                .text-orange { color: #ea580c;}
                .text-blue { color: #0369a1;}
                .text-purple { color: #7c3aed;}
                .text-green { color: #16a34a;}
                .text-white { color: #fff;}
                
                /* How to Use Grid */
                .gradient-card {
                    background: linear-gradient(90deg, #059669 0%, #2dd4bf 100%);
                }
                .grid-how {
                    display: grid;
                    grid-template-columns: repeat(1, 1fr);
                    gap: 1.5rem;
                }
                @media (min-width: 768px) {
                    .grid-how { grid-template-columns: repeat(2, 1fr);}
                }
                @media (min-width: 1024px) {
                    .grid-how { grid-template-columns: repeat(4, 1fr);}
                }
                .how-card {
                    text-align: center;
                }
                .how-circle {
                    background: rgba(255,255,255,0.18);
                    border-radius: 50%;
                    width: 64px; height: 64px;
                    display: flex; align-items: center; justify-content: center;
                    margin: 0 auto 1rem auto;
                    font-size: 2rem;
                    font-weight: bold;
                    color: #fff;
                }
                .how-title {
                    font-weight: 600;
                    margin-bottom: 0.3rem;
                }
                .how-card p {
                    color: #d1fae5;
                    font-size: 1rem;
                }
                
                /* CTA */
                .btn-cta {
                    background: #16a34a;
                    color: #fff;
                    font-weight: bold;
                    padding: 1.2rem 2.5rem;
                    border-radius: 1rem;
                    font-size: 1.2rem;
                    box-shadow: 0 4px 24px -8px #16a34a44;
                    border: none;
                    transition: background .3s, transform .3s;
                    cursor: pointer;
                }
                .btn-cta:hover {
                    background: #166534;
                    transform: scale(1.1);
                    animation: none;
                }
                
                /* Footer */
                .footer {
                    background: #065f46;
                    color: #fff;
                    margin-top: 5rem;
                    padding: 2rem 0;
                }
                .footer-title {
                    font-size: 1.1rem;
                    font-weight: 600;
                    color: #fff;
                }
                .footer-desc {
                    color: #bbf7d0;
                    font-size: 1rem;
                }
                .icon-lg { width: 32px; height: 32px;}
                .icon-md { width: 24px; height: 24px;}
                .icon-sm { width: 20px; height: 20px;}
                /* Utility */
                .flex { display: flex; }
                .justify-center { justify-content: center; }
                .items-center { align-items: center; }
                .gap-2 { gap: 0.5rem; }
                .gap-3 { gap: 0.75rem; }
                .profile-circle {
                    width: 40px;
                    height: 40px;
                    background: #16a34a;
                    color: #fff;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    font-size: 1.2rem;
                    cursor: pointer;
                    border: 2px solid #fff;
                    box-shadow: 0 2px 8px -2px #16a34a44;
                    transition: background .2s;
                }
                .profile-circle:hover {
                    background: #166534;
                }
            </style>
            </head>
            <body class="min-h-screen bg-gradient">

            <!-- Header -->
            <header class="header">
                <div class="container flex-between">
                <div class="flex items-center gap-2">
                    <!-- Leaf Icon SVG -->
                    <svg class="icon-lg text-green" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path d="M2 22s9-2 15-8 5-12 5-12-9 2-15 8-5 12-5 12z"/>
                    </svg>
                    <h1 class="title">Plant Doctor AI</h1>
                </div>
                <nav class="nav">
                    <a href="/">Home</a>
                    <a href="/predict">Upload & Predict</a>
                    <a href="/gallery">Disease Gallery</a>
                    <a href="/chatbot">Plant Saathi</a>
                    <a href="/about">About</a>
                    <div class="nav-actions" id="nav-actions">
                    <!-- Login/Signup or Profile will be injected here -->
                    </div>
                </nav>
                </div>
            </header>

            <!-- Main Content -->
            <main class="container">
                <!-- Hero Section -->
                <section class="text-center mb-12">
                <h2 class="hero-title animate-pulse">Welcome to Plant Doctor!</h2>
                <div class="center mb-8">
                    <div class="hero-image">
                    <img src="Image_for_PDA.jpg" alt="Plant Disease Analysis" />
                    <div class="img-gradient"></div>
                    <div class="icon-top-right">
                        <!-- Camera Icon SVG -->
                        <svg class="icon-md text-white animate-bounce" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h3l2-3h6l2 3h3a2 2 0 0 1 2 2z"/>
                        <circle cx="12" cy="13" r="4"/>
                        </svg>
                    </div>
                    <div class="img-bottom-left">
                        <p class="tag">AI-Powered Analysis</p>
                    </div>
                    </div>
                </div>
                <p class="desc">
                    This AI-powered app helps you <strong>identify plant diseases from leaf images</strong> using deep learning.
                </p>
                </section>

                <!-- Overview Section -->
                <section class="card mb-10">
                <h3 class="section-title flex items-center">
                    <!-- Shield Icon SVG -->
                    <svg class="icon-lg text-green animate-pulse mr-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    </svg>
                    Overview
                </h3>
                <p class="card-text">
                    This app is a Plant Disease Detection system powered by deep learning/AI technologies. It leverages computer vision models to analyze plant leaf images and provide accurate classifications. The goal is to support early identification of diseases, helping reduce crop losses and improve yield.
                </p>
                </section>

                <!-- Why Plant Disease Detection Section -->
                <section class="card mb-10">
                <h3 class="section-title flex items-center">
                    <!-- Globe Icon SVG -->
                    <svg class="icon-lg text-green animate-spin mr-3" style="animation-duration:8s" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M2 12h20M12 2a15.3 15.3 0 0 1 0 20M12 2a15.3 15.3 0 0 0 0 20"/>
                    </svg>
                    Why Plant Disease Detection?
                </h3>
                <div class="grid-stats mb-8">
                    <div class="stat-card stat-blue">
                    <!-- Users Icon SVG -->
                    <svg class="icon-lg text-blue animate-bounce mb-2" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path d="M17 21v-2a4 4 0 0 0-3-3.87M9 21v-2a4 4 0 0 1 3-3.87M7 8a4 4 0 1 1 10 0M12 8v4"/>
                    </svg>
                    <div class="stat-value text-blue">26%</div>
                    <div class="stat-label text-blue">Global workforce in farming</div>
                    </div>
                    <div class="stat-card stat-orange">
                    <!-- TrendingUp Icon SVG -->
                    <svg class="icon-lg text-orange animate-bounce mb-2" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <polyline points="17 1 17 7 23 7"/>
                        <polyline points="1 17 7 17 7 23"/>
                        <line x1="17" y1="7" x2="7" y2="17"/>
                    </svg>
                    <div class="stat-value text-orange">80%</div>
                    <div class="stat-label text-orange">Human food from plants</div>
                    </div>
                    <div class="stat-card stat-red">
                    <!-- Crop Icon SVG -->
                    <svg class="icon-lg text-red animate-bounce mb-2" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path d="M12 2L2 7h20L12 2z"/> <!-- Example: simple triangle leaf -->
                    </svg>
                    <div class="stat-value text-red">40%</div>
                    <div class="stat-label text-red">Crop loss to diseases</div>
                    </div>
                    
                    <div class="stat-card stat-purple">
                    <!-- Money Icon SVG -->
                    <svg class="icon-lg text-purple animate-bounce mb-2" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path d="M12 1v22M5 6h14M5 18h14"/> <!-- Example: simplified money -->
                    </svg>
                    <div class="stat-value text-purple">$220B</div>
                    <div class="stat-label text-purple">Annual cost of losses</div>
                    </div>
                    
                </div>
                <div class="card-text">
                    <p>
                    Globally, agriculture is a cornerstone of food security and livelihoods – roughly <strong>26% of the world's workforce</strong> is employed in farming – yet plant diseases inflict massive losses. Humans get <strong>80% of their food from plants</strong>, but up to <strong>40% of global crop production</strong> is destroyed by pests and pathogens each year, costing on the order of <strong>USD 220 billion annually</strong>.
                    </p>
                    <p>
                    In India, where agriculture remains a rural backbone, recent data show about <strong>42.3% of the population</strong> depends on farming (and roughly two-thirds of rural Indians derive their livelihood from agriculture). Crop diseases commonly cut Indian crop output by <strong>10–30%</strong>, with some analyses even citing <strong>30%</strong> annual productivity loss due to pest infestations. Such declines are especially harmful for millions of smallholder farmers, where even modest yield drops can undermine income and food access.
                    </p>
                    <p>
                    Moreover, climate change and modern farming amplify these threats: warming allows pathogens (like fungi) to spread into new regions, and large-scale monocultures of genetically uniform crops provide ideal breeding grounds for outbreaks.
                    </p>
                    <p>
                    In this context, early disease detection is vitally important. <strong>AI and machine-learning tools</strong> can rapidly scan plant images or sensor data to diagnose infections before they become widespread. By providing fast, precise, and automated diagnostics, these technologies cut labor and response time and enable targeted interventions—spot-treating only infected areas.
                    </p>
                    <p>
                    In practice, AI-driven monitoring helps farmers act quickly, quarantining or treating diseased plants in time, thereby minimizing crop loss and strengthening food security. Given agriculture's socio-economic importance (especially in India) and the growing strain of disease pressures, timely plant disease detection through AI/ML methods is critical to reduce losses, sustain yields, and support rural livelihoods.
                    </p>
                </div>
                </section>

                <!-- Features Section -->
                <section class="card mb-10">
                <h3 class="section-title text-center">Features</h3>
                <div class="grid-features">
                    <div class="feature-card bg-green-50">
                    <!-- Camera Icon SVG -->
                    <svg class="icon-lg text-green animate-pulse" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h3l2-3h6l2 3h3a2 2 0 0 1 2 2z"/>
                        <circle cx="12" cy="13" r="4"/>
                    </svg>
                    <div>
                        <h4 class="feature-title">Image Capture & Upload</h4>
                        <p class="feature-desc">Capture or upload a leaf image directly from your device</p>
                    </div>
                    </div>
                    <div class="feature-card bg-blue-50">
                    <!-- TrendingUp Icon SVG -->
                    <svg class="icon-lg text-blue animate-bounce" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <polyline points="17 1 17 7 23 7"/>
                        <polyline points="1 17 7 17 7 23"/>
                        <line x1="17" y1="7" x2="7" y2="17"/>
                    </svg>
                    <div>
                        <h4 class="feature-title text-blue">Instant Prediction</h4>
                        <p class="feature-desc text-blue">Get instant disease prediction using a ResNet18 model</p>
                    </div>
                    </div>
                    <div class="feature-card bg-purple-50">
                    <!-- Shield Icon SVG -->
                    <svg class="icon-lg text-purple animate-pulse" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    </svg>
                    <div>
                        <h4 class="feature-title text-purple">Detailed Information</h4>
                        <p class="feature-desc text-purple">Learn about symptoms, causes, treatments, and prevention</p>
                    </div>
                    </div>
                    <div class="feature-card bg-orange-50">
                    <!-- Search Icon SVG -->
                    <svg class="icon-lg text-orange animate-spin" style="animation-duration:4s" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <circle cx="11" cy="11" r="8"/>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                    </svg>
                    <div>
                        <h4 class="feature-title text-orange">Wikipedia Integration</h4>
                        <p class="feature-desc text-orange">Lookup detailed info from Wikipedia or built-in database</p>
                    </div>
                    </div>
                    <div class="feature-card bg-teal-50">
                    <!-- Leaf Icon SVG -->
                    <svg class="icon-lg text-teal animate-bounce" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path d="M2 22s9-2 15-8 5-12 5-12-9 2-15 8-5 12-5 12z"/>
                    </svg>
                    <div>
                        <h4 class="feature-title text-teal">Multiple Plant Types</h4>
                        <p class="feature-desc text-teal">Supports multiple plant types and diseases</p>
                    </div>
                    </div>
                    <div class="feature-card bg-indigo-50">
                    <!-- Upload Icon SVG -->
                    <svg class="icon-lg text-indigo animate-pulse" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path d="M16 16v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-4"/>
                        <polyline points="8 7 12 3 16 7"/>
                        <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                    <div>
                        <h4 class="feature-title text-indigo">Easy to Use</h4>
                        <p class="feature-desc text-indigo">Simple interface designed for farmers and gardeners</p>
                    </div>
                    </div>
                </div>
                </section>

                <!-- How to Use Section -->
                <section class="card gradient-card text-white">
                <h3 class="section-title text-center">How to Use</h3>
                <div class="grid-how">
                    <div class="how-card">
                    <div class="how-circle">1</div>
                    <h4 class="how-title">Navigate</h4>
                    <p>Go to <strong>Upload & Predict</strong> in the navigation bar</p>
                    </div>
                    <div class="how-card">
                    <div class="how-circle">2</div>
                    <h4 class="how-title">Upload Image</h4>
                    <p>Upload a leaf image or take a photo using your camera</p>
                    </div>
                    <div class="how-card">
                    <div class="how-circle">3</div>
                    <h4 class="how-title">Analyze</h4>
                    <p>Click <strong>Predict</strong> to analyze the leaf</p>
                    </div>
                    <div class="how-card">
                    <div class="how-circle">4</div>
                    <h4 class="how-title">Get Results</h4>
                    <p>Read the predicted disease and scroll down for detailed treatment info</p>
                    </div>
                </div>
                </section>

                <!-- CTA Section -->
                <div class="text-center mt-12">
                <button class="btn-cta animate-pulse" onclick="window.location.href='/predict'">
                    Start Diagnosing Plants &rarr;
                </button>

                </div>
            </main>

            <!-- Footer -->
            <footer class="footer">
                <div class="container text-center">
                <div class="flex items-center justify-center gap-2 mb-4">
                    <!-- Leaf Icon SVG -->
                    <svg class="icon-md text-white" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path d="M2 22s9-2 15-8 5-12 5-12-9 2-15 8-5 12-5 12z"/>
                    </svg>
                    <span class="footer-title">Plant Doctor AI</span>
                </div>
                <p class="footer-desc">Powered by AI • Helping farmers worldwide</p>
                </div>
            </footer>
            <!-- Firebase SDKs -->
            <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-app-compat.js"></script>
            <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-auth-compat.js"></script>
            <script>
                const firebaseConfig = {
                apiKey: "AIzaSyB-y52ekJkjgXYXhRNvvir0r9gU8CObpkM",
                authDomain: "plant-doctor-63da7.firebaseapp.com",
                projectId: "plant-doctor-63da7",
                storageBucket: "plant-doctor-63da7.appspot.com",
                messagingSenderId: "94281353808",
                appId: "1:94281353808:web:5fa49e3d6e494868be5f55",
                measurementId: "G-E5P4MZ68HW"
            };

            if (!firebase.apps.length) {
                firebase.initializeApp(firebaseConfig);
            }
            const auth = firebase.auth();

            function renderProfile(user) {
                const navActions = document.getElementById("nav-actions");
                if (!navActions) return;

                // Show profile circle with first letter
                let displayName = user.displayName || user.email || "U";
                let firstLetter = displayName.charAt(0).toUpperCase();

                navActions.innerHTML = `
                <div class="profile-circle" title="${displayName}" onclick="logout()">
                    ${firstLetter}
                </div>
                `;
            }

            function renderLoginSignup() {
                const navActions = document.getElementById("nav-actions");
                if (!navActions) return;
                navActions.innerHTML = `
                <a href="/signin" class="btn btn-green">Log In</a>
                <a href="/signup" class="btn btn-outline-green">Sign Up</a>
                `;
            }

            // Logout function
            function logout() {
                firebase.auth().signOut().then(() => {
                window.location.reload();
                });
            }

            // Listen for auth state changes
            auth.onAuthStateChanged(user => {
                if (user) {
                renderProfile(user);
                } else {
                renderLoginSignup();
                }
            });
            </script>
            </body>
            </html>
            """
    except Exception as e:
        return f"Error loading home page: {str(e)}"

@app.route('/predict')
def predict_page():
    """Serve the prediction page"""
    try:
        html_path = Path(__file__).parent / 'predict.html'
        if html_path.exists():
            with open(html_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return """
            <!DOCTYPE html>
            <html lang="en">
            <head>
            <meta charset="UTF-8">
            <title>Upload & Predict - Plant Doctor AI</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                /* General styles */
                body {
                margin: 0;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                background: linear-gradient(135deg, #f0fdf4 0%, #d1fae5 100%);
                color: #222;
                min-height: 100vh;
                }
                .container {
                max-width: 1100px;
                margin: 0 auto;
                padding: 0 1rem;
                }
                .flex { display: flex; }
                .flex-between { display: flex; align-items: center; justify-content: space-between; }
                .items-center { align-items: center; }
                .gap-2 { gap: 0.5rem; }
                .text-center { text-align: center; }
                .mb-12 { margin-bottom: 3rem; }
                .mb-10 { margin-bottom: 2.5rem; }
                .mb-8 { margin-bottom: 2rem; }
                .mb-4 { margin-bottom: 1rem; }
                .mr-3 { margin-right: 0.75rem; }
                
                /* Header */
                .header {
                background: #fff;
                box-shadow: 0 2px 8px -2px rgba(16, 185, 129, 0.08);
                border-bottom: 1px solid #bbf7d0;
                }
                .title {
                font-size: 2rem;
                font-weight: bold;
                color: #065f46;
                }
                .nav {
                display: flex;
                align-items: center;
                gap: 1.5rem;
                }
                .nav a {
                color: #047857;
                font-weight: 500;
                text-decoration: none;
                transition: color 0.2s;
                }
                .nav a:hover, .nav a.active {
                color: #065f46;
                }
                .nav-actions {
                display: flex;
                gap: 0.75rem;
                margin-left: 1.5rem;
                }
                .btn {
                padding: 0.5rem 1rem;
                border-radius: 0.75rem;
                font-weight: 500;
                transition: all .2s;
                box-shadow: none;
                border: none;
                cursor: pointer;
                }
                .btn-green {
                background: #16a34a;
                color: #fff;
                }
                .btn-green:hover {
                background: #166534;
                box-shadow: 0 2px 8px -2px #16a34a44;
                }
                .btn-outline-green {
                background: #fff;
                border: 2px solid #16a34a;
                color: #16a34a;
                }
                .btn-outline-green:hover {
                background: #f0fdf4;
                box-shadow: 0 2px 8px -2px #16a34a22;
                }
                
                /* Main content */
                .hero-title {
                font-size: 2.5rem;
                font-weight: bold;
                color: #065f46;
                margin-bottom: 1.5rem;
                }
                .hero-subtitle {
                font-size: 1.25rem;
                color: #047857;
                max-width: 700px;
                margin: 0 auto 2rem auto;
                line-height: 1.6;
                }
                
                /* Upload section */
                .upload-area {
                background: #fff;
                border-radius: 1.25rem;
                box-shadow: 0 4px 24px -8px #16a34a22;
                padding: 2rem;
                margin: 2rem 0;
                border: 2px dashed #bbf7d0;
                transition: all .3s;
                cursor: pointer;
                text-align: center;
                }
                .upload-area:hover, .upload-area.dragover {
                border-color: #16a34a;
                box-shadow: 0 8px 32px -8px #16a34a22;
                transform: translateY(-2px);
                }
                .upload-icon {
                width: 64px;
                height: 64px;
                background: #16a34a;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 1rem;
                color: #fff;
                }
                .upload-text {
                font-size: 1.2rem;
                font-weight: 600;
                color: #065f46;
                margin-bottom: 0.5rem;
                }
                .upload-subtext {
                color: #047857;
                font-size: 1rem;
                }
                
                /* Image preview */
                .image-preview {
                background: #fff;
                border-radius: 1.25rem;
                box-shadow: 0 4px 24px -8px #16a34a22;
                padding: 2rem;
                margin: 2rem 0;
                text-align: center;
                display: none;
                }
                .preview-image {
                max-width: 100%;
                max-height: 400px;
                border-radius: 1rem;
                box-shadow: 0 4px 16px -4px rgba(0,0,0,0.1);
                }
                
                /* Results section */
                .results-section {
                background: #fff;
                border-radius: 1.25rem;
                box-shadow: 0 4px 24px -8px #16a34a22;
                padding: 2rem;
                margin: 2rem 0;
                display: none;
                }
                .result-title {
                font-size: 1.5rem;
                font-weight: bold;
                color: #065f46;
                margin-bottom: 1rem;
                }
                .confidence-bar {
                background: #f0fdf4;
                border-radius: 1rem;
                overflow: hidden;
                margin: 1rem 0;
                }
                .confidence-fill {
                background: linear-gradient(90deg, #16a34a, #22c55e);
                height: 20px;
                border-radius: 1rem;
                transition: width 1s ease-out;
                }
                
                /* Enhanced Disease Information Cards */
                .disease-info {
                background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
                border-radius: 1.25rem;
                padding: 2rem;
                margin: 1rem 0;
                border-left: 6px solid #16a34a;
                box-shadow: 0 8px 32px -8px rgba(16, 163, 74, 0.15);
                animation: fadeInUp 0.6s ease-out;
                }
                
                @keyframes fadeInUp {
                from {
                opacity: 0;
                transform: translateY(30px);
                }
                to {
                opacity: 1;
                transform: translateY(0);
                }
                }
                
                .info-cards-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
                margin-top: 2rem;
                }
                
                .info-card {
                background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                border-radius: 1rem;
                padding: 1.5rem;
                border: 1px solid #e2e8f0;
                box-shadow: 0 4px 16px -4px rgba(0, 0, 0, 0.1);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
                animation: slideInCard 0.5s ease-out forwards;
                opacity: 0;
                transform: translateY(20px);
                }
                
                .info-card:nth-child(1) { animation-delay: 0.1s; }
                .info-card:nth-child(2) { animation-delay: 0.2s; }
                .info-card:nth-child(3) { animation-delay: 0.3s; }
                .info-card:nth-child(4) { animation-delay: 0.4s; }
                .info-card:nth-child(5) { animation-delay: 0.5s; }
                .info-card:nth-child(6) { animation-delay: 0.6s; }
                
                @keyframes slideInCard {
                to {
                opacity: 1;
                transform: translateY(0);
                }
                }
                
                .info-card:hover {
                transform: translateY(-8px) scale(1.02);
                box-shadow: 0 12px 40px -8px rgba(16, 163, 74, 0.25);
                border-color: #16a34a;
                }
                
                .info-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #16a34a, #22c55e, #34d399);
                border-radius: 1rem 1rem 0 0;
                }
                
                .card-header {
                display: flex;
                align-items: center;
                gap: 1rem;
                margin-bottom: 1rem;
                }
                
                .card-icon {
                width: 48px;
                height: 48px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #fff;
                font-size: 1.5rem;
                flex-shrink: 0;
                }
                
                .symptoms-icon { background: linear-gradient(135deg, #ef4444, #f87171); }
                .causes-icon { background: linear-gradient(135deg, #f59e0b, #fbbf24); }
                .treatment-icon { background: linear-gradient(135deg, #16a34a, #22c55e); }
                .medicines-icon { background: linear-gradient(135deg, #3b82f6, #60a5fa); }
                .prevention-icon { background: linear-gradient(135deg, #8b5cf6, #a78bfa); }
                .severity-icon { background: linear-gradient(135deg, #dc2626, #ef4444); }
                
                .card-title {
                font-size: 1.2rem;
                font-weight: 600;
                color: #1f2937;
                margin: 0;
                }
                
                .card-content {
                color: #4b5563;
                line-height: 1.6;
                }
                
                .card-list {
                list-style: none;
                padding: 0;
                margin: 0;
                }
                
                .card-list li {
                padding: 0.5rem 0;
                border-bottom: 1px solid #f1f5f9;
                display: flex;
                align-items: flex-start;
                gap: 0.75rem;
                transition: all 0.2s;
                }
                
                .card-list li:last-child {
                border-bottom: none;
                }
                
                .card-list li:hover {
                background: #f8fafc;
                margin: 0 -1rem;
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                }
                
                .list-bullet {
                width: 6px;
                height: 6px;
                background: #16a34a;
                border-radius: 50%;
                margin-top: 0.6rem;
                flex-shrink: 0;
                }
                
                /* Severity Cards */
                .severity-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
                }
                
                .severity-card {
                padding: 1rem;
                border-radius: 0.75rem;
                border-left: 4px solid;
                transition: all 0.3s;
                animation: pulseIn 0.6s ease-out;
                }
                
                @keyframes pulseIn {
                0% {
                opacity: 0;
                transform: scale(0.95);
                }
                100% {
                opacity: 1;
                transform: scale(1);
                }
                }
                
                .severity-mild {
                background: linear-gradient(135deg, #fef3c7, #fef9e7);
                border-left-color: #f59e0b;
                color: #92400e;
                }
                
                .severity-moderate {
                background: linear-gradient(135deg, #fed7aa, #fef2e2);
                border-left-color: #ea580c;
                color: #9a3412;
                }
                
                .severity-severe {
                background: linear-gradient(135deg, #fecaca, #fef2f2);
                border-left-color: #dc2626;
                color: #991b1b;
                }
                
                .severity-card h5 {
                font-weight: 600;
                margin: 0 0 0.5rem 0;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                }
                
                .severity-card p {
                margin: 0;
                font-size: 0.85rem;
                line-height: 1.4;
                }
                
                /* Medicine Carousel */
                .medicine-carousel {
                margin-top: 2rem;
                padding: 1.5rem;
                background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
                border-radius: 1rem;
                border: 1px solid #bae6fd;
                }
                
                .carousel-container {
                position: relative;
                display: flex;
                align-items: center;
                gap: 1rem;
                }
                
                .carousel-items {
                display: flex;
                gap: 1rem;
                overflow-x: auto;
                scroll-behavior: smooth;
                padding: 1rem 0;
                flex: 1;
                }
                
                .carousel-items::-webkit-scrollbar {
                height: 6px;
                }
                
                .carousel-items::-webkit-scrollbar-track {
                background: #f1f5f9;
                border-radius: 3px;
                }
                
                .carousel-items::-webkit-scrollbar-thumb {
                background: #cbd5e1;
                border-radius: 3px;
                }
                
                .medicine-card {
                min-width: 200px;
                padding: 1rem;
                background: #fff;
                border-radius: 0.75rem;
                border: 1px solid #e2e8f0;
                box-shadow: 0 2px 8px -2px rgba(0, 0, 0, 0.1);
                transition: all 0.3s;
                animation: slideInMedicine 0.4s ease-out;
                }
                
                @keyframes slideInMedicine {
                from {
                opacity: 0;
                transform: translateX(20px);
                }
                to {
                opacity: 1;
                transform: translateX(0);
                }
                }
                
                .medicine-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 8px 24px -4px rgba(59, 130, 246, 0.3);
                border-color: #3b82f6;
                }
                
                .carousel-btn {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                border: 2px solid #16a34a;
                background: #fff;
                color: #16a34a;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.2rem;
                transition: all 0.2s;
                }
                
                .carousel-btn:hover {
                background: #16a34a;
                color: #fff;
                transform: scale(1.1);
                }
                
                /* Action buttons */
                .action-buttons {
                display: flex;
                gap: 1rem;
                justify-content: center;
                flex-wrap: wrap;
                margin: 2rem 0;
                }
                
                .btn-action {
                background: linear-gradient(135deg, #16a34a, #22c55e);
                color: #fff;
                font-weight: 500;
                padding: 0.75rem 1.5rem;
                border-radius: 0.75rem;
                border: none;
                cursor: pointer;
                transition: all .3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
                }
                
                .btn-action::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                transition: left 0.5s;
                }
                
                .btn-action:hover::before {
                left: 100%;
                }
                
                .btn-action:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 24px -4px rgba(16, 163, 74, 0.4);
                }
                
                .btn-action:disabled {
                background: #9ca3af;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
                }
                
                .btn-action:disabled::before {
                display: none;
                }
                
                .btn-secondary {
                background: #fff;
                color: #16a34a;
                border: 2px solid #16a34a;
                }
                
                .btn-secondary:hover {
                background: #f0fdf4;
                }
                
                /* Loading spinner */
                .loading {
                display: none;
                text-align: center;
                padding: 2rem;
                }
                
                .spinner {
                border: 4px solid #f0fdf4;
                border-top: 4px solid #16a34a;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 1rem;
                }
                
                @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
                }
                
                /* Footer */
                .footer {
                background: #065f46;
                color: #fff;
                margin-top: 5rem;
                padding: 2rem 0;
                }
                .footer-title {
                font-size: 1.1rem;
                font-weight: 600;
                color: #fff;
                }
                .footer-desc {
                color: #bbf7d0;
                font-size: 1rem;
                }
                .icon-lg { width: 32px; height: 32px;}
                .icon-md { width: 24px; height: 24px;}
                
                /* Responsive */
                @media (max-width: 768px) {
                .action-buttons {
                flex-direction: column;
                align-items: center;
                }
                .btn-action {
                width: 100%;
                max-width: 300px;
                }
                .info-cards-grid {
                grid-template-columns: 1fr;
                }
                .severity-grid {
                grid-template-columns: 1fr;
                }
                }
                
                /* Utility */
                .hidden { display: none; }
                .profile-circle {
                width: 40px;
                height: 40px;
                background: #16a34a;
                color: #fff;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 1.2rem;
                cursor: pointer;
                border: 2px solid #fff;
                box-shadow: 0 2px 8px -2px #16a34a44;
                transition: background .2s;
                }
                .profile-circle:hover {
                background: #166534;
                }
                
                /* Progress indication */
                .progress-dots {
                display: flex;
                justify-content: center;
                gap: 0.5rem;
                margin: 2rem 0;
                }
                
                .progress-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #cbd5e1;
                transition: all 0.3s;
                }
                
                .progress-dot.active {
                background: #16a34a;
                transform: scale(1.2);
                }
                
                /* Enhanced animations */
                @keyframes bounce {
                0%, 20%, 53%, 80%, 100% {
                transform: translate3d(0, 0, 0);
                }
                40%, 43% {
                transform: translate3d(0, -8px, 0);
                }
                70% {
                transform: translate3d(0, -4px, 0);
                }
                90% {
                transform: translate3d(0, -2px, 0);
                }
                }
                
                .bounce-animation {
                animation: bounce 1s ease-in-out;
                }
                
                /* Floating elements */
                .floating {
                animation: float 3s ease-in-out infinite;
                }
                
                @keyframes float {
                0% {
                transform: translateY(0px);
                }
                50% {
                transform: translateY(-6px);
                }
                100% {
                transform: translateY(0px);
                }
                }
            body {
            margin: 0;
            font-family: 'Segoe UI', 'Arial', sans-serif;
            background: linear-gradient(135deg, #f0fdf4 0%, #d1fae5 100%);
            color: #222;
            min-height: 100vh;
            }
            </style>
            <!-- Firebase SDKs for profile.js -->
            <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-app-compat.js"></script>
            <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-auth-compat.js"></script>
            </head>
            <body>
            <!-- Header -->
            <header class="header">
            <div class="container flex-between">
            <div class="flex items-center gap-2">
              <!-- Leaf Icon SVG -->
              <svg class="icon-lg text-green" fill="none" stroke="green" stroke-width="2" viewBox="0 0 24 24">
              <path d="M2 22s9-2 15-8 5-12 5-12-9 2-15 8-5 12-5 12z"/>
              </svg>
              <h1 class="title">Plant Doctor AI</h1>
            </div>
            <nav class="nav">
                <a href="/">Home</a>
                <a href="/predict" class="active">Upload & Predict</a>
                <a href="/gallery">Disease Gallery</a>
                <a href="/chatbot">Plant Saathi</a>
                <a href="/about">About</a>
            <div class="nav-actions" id="nav-actions">
            <!-- Login/Signup or Profile will be injected here by profile.js -->
            </div>
            </nav>
            </div>
            </header>
            
            <!-- Main Content -->
            <main class="container">
            <!-- Hero Section -->
            <section class="text-center mb-12">
            <h2 class="hero-title">Plant Disease Detection</h2>
            <p class="hero-subtitle">
            Upload a <strong>leaf image</strong> and our AI will analyze it to detect potential diseases and provide treatment recommendations.
            </p>
            </section>
            
            <!-- Upload Section -->
            <section class="upload-area" id="uploadArea">
            <div class="upload-icon">
            <svg width="32" height="32" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h3l2-3h6l2 3h3a2 2 0 0 1 2 2z"/>
            <circle cx="12" cy="13" r="4"/>
            </svg>
            </div>
            <div class="upload-text">Drop your plant image here or click to browse</div>
            <div class="upload-subtext">Supported formats: JPG, PNG, GIF (Max size: 10MB)</div>
            <input type="file" id="imageInput" accept="image/*" style="display: none;">
            </section>
            
            <!-- Camera Option -->
            <div class="text-center mb-8">
            <button class="btn-action" id="cameraBtn">
            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" style="margin-right: 0.5rem;">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h3l2-3h6l2 3h3a2 2 0 0 1 2 2z"/>
            <circle cx="12" cy="13" r="4"/>
            </svg>
            Take Photo with Camera
            </button>
            <script>
            document.addEventListener('DOMContentLoaded', function() {
                document.getElementById('cameraBtn').addEventListener('click', openCamera);
            });

            </script>
            </div>
            
            <!-- Image Preview -->
            <section class="image-preview" id="imagePreview">
            <img id="previewImg" class="preview-image" alt="Preview">
            <div class="action-buttons" style="margin-top: 1rem;">
            <button class="btn-action" id="predictBtn">
            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" style="margin-right: 0.5rem;">
            <path d="M9 11l3 3L22 4"/>
            </svg>
            Analyze Plant
            </button>
            <button class="btn-action btn-secondary" id="newImageBtn">Choose Different Image</button>
            </div>
            </section>
            
            <!-- Loading Section -->
            <section class="loading" id="loadingSection">
            <div class="spinner"></div>
            <p style="color: #047857; font-weight: 500;">Analyzing your plant image...</p>
            <p style="color: #6b7280; font-size: 0.9rem;">This may take a few seconds</p>
            <div class="progress-dots">
            <div class="progress-dot active"></div>
            <div class="progress-dot"></div>
            <div class="progress-dot"></div>
            </div>
            </section>
            
            <!-- Results Section -->
            <section class="results-section" id="resultsSection">
            <h3 class="result-title" style="text-align: center;">Diagnosis🕵🏻‍♂️</h3>
            
            <!-- Prediction Result -->
            <div id="predictionResult" style="text-align: center; margin: 2rem 0;">
            <div style="font-size: 1.5rem; font-weight: bold; color: #065f46; margin-bottom: 1rem;" id="diseaseTitle"></div>
            <div style="font-size: 1.1rem; color: #047857; margin-bottom: 1rem;">Confidence: <span id="confidenceText"></span></div>
            <div class="confidence-bar">
            <div class="confidence-fill" id="confidenceFill" style="width: 0%"></div>
            </div>
            </div>
            
            <!-- Enhanced Disease Information with Beautiful Cards -->
            <div class="disease-info" id="diseaseInfo">
            <h4 style="color: #065f46; margin-bottom: 2rem; text-align: center; font-size: 1.3rem;">Disease Information & Treatment Guide</h4>
            
            <div class="info-cards-grid" id="infoCardsGrid">
            <!-- Cards will be dynamically generated here -->
            </div>
            
            <!-- Medicine Carousel -->
            <div id="medicineSection" class="medicine-carousel" style="display: none;">
            <h4 style="color: #3b82f6; margin-bottom: 1rem; text-align: center;">Recommended Medicines</h4>
            <div class="carousel-container">
            <button class="carousel-btn" id="medicinePrevBtn">‹</button>
            <div id="medicineItems" class="carousel-items"></div>
            <button class="carousel-btn" id="medicineNextBtn">›</button>
            </div>
            </div>
            </div>
            
            <!-- Action Buttons -->
            <div class="action-buttons">
            <button class="btn-action" id="generateReportBtn">
            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" style="margin-right: 0.5rem;">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14,2 14,8 20,8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
            </svg>
            Generate PDF Report
            </button>
            <button class="btn-action btn-secondary" id="newAnalysisBtn">Analyze Another Image</button>
            </div>
            </section>
            
            <!-- GradCAM Visualization -->
            <section class="results-section hidden" id="gradcamSection">
            <h3 class="result-title">Analysis Heatmap (GradCAM)</h3>
            <div style="text-align: center;">
            <p style="color: #6b7280; margin-bottom: 1rem;">This heatmap shows which areas of the leaf our AI focused on when making the diagnosis:</p>
            <img id="gradcamImage" style="max-width: 100%; border-radius: 1rem; box-shadow: 0 4px 16px -4px rgba(0,0,0,0.1);">
            </div>
            </section>
            </main>
            
            <!-- Footer -->
            <footer class="footer">
            <div class="container text-center">
            <div class="flex items-center justify-center gap-2 mb-4">
            <!-- Leaf Icon SVG -->
            <svg class="icon-md" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" style="color: #fff;">
            <path d="M2 22s9-2 15-8 5-12 5-12-9 2-15 8-5 12-5 12z"/>
            </svg>
            <span class="footer-title">Plant Doctor AI</span>
            </div>
            <p class="footer-desc">Powered by AI • Helping farmers worldwide</p>
            </div>
            </footer>
            
            <!-- Profile.js Logic -->
            <script>
            const firebaseConfig = {
                apiKey: "AIzaSyB-y52ekJkjgXYXhRNvvir0r9gU8CObpkM",
                authDomain: "plant-doctor-63da7.firebaseapp.com",
                projectId: "plant-doctor-63da7",
                storageBucket: "plant-doctor-63da7.appspot.com",
                messagingSenderId: "94281353808",
                appId: "1:94281353808:web:5fa49e3d6e494868be5f55",
                measurementId: "G-E5P4MZ68HW"
            };
            if (!firebase.apps.length) {
                firebase.initializeApp(firebaseConfig);
            }
            const auth = firebase.auth();
            
            function renderProfile(user) {
                const navActions = document.getElementById("nav-actions");
                if (!navActions) return;
            
                // Show profile circle with first letter
                let displayName = user.displayName || user.email || "U";
                let firstLetter = displayName.charAt(0).toUpperCase();
            
                navActions.innerHTML = `
                  <div class="profile-circle" title="${displayName}" onclick="logout()">
                    ${firstLetter}
                  </div>
                `;
            }
            
            function renderLoginSignup() {
                const navActions = document.getElementById("nav-actions");
                if (!navActions) return;
                navActions.innerHTML = `
                  <a href="/signin" class="btn btn-green">Log In</a>
                  <a href="/signup" class="btn btn-outline-green">Sign Up</a>
                `;
            }
            
            // Logout function
            function logout() {
                firebase.auth().signOut().then(() => {
                  window.location.reload();
                });
            }
            
            // Listen for auth state changes
            auth.onAuthStateChanged(user => {
                if (user) {
                  renderProfile(user);
                } else {
                  renderLoginSignup();
                }
            });
            </script>
            
            <!-- Main JS for prediction page -->
            <script>
            // Global variables
            let currentImage = null;
            let currentPredictionId = null;
            let currentDiseaseInfo = null;
            
            // DOM elements
            const uploadArea = document.getElementById('uploadArea');
            const imageInput = document.getElementById('imageInput');
            const imagePreview = document.getElementById('imagePreview');
            const previewImg = document.getElementById('previewImg');
            const loadingSection = document.getElementById('loadingSection');
            const resultsSection = document.getElementById('resultsSection');
            const gradcamSection = document.getElementById('gradcamSection');
            const infoCardsGrid = document.getElementById('infoCardsGrid');
            
            // Event listeners
            uploadArea.addEventListener('click', () => imageInput.click());
            uploadArea.addEventListener('dragover', handleDragOver);
            uploadArea.addEventListener('drop', handleDrop);
            imageInput.addEventListener('change', handleImageSelect);
            
            document.getElementById('predictBtn').addEventListener('click', analyzeImage);
            document.getElementById('newImageBtn').addEventListener('click', resetUpload);
            document.getElementById('newAnalysisBtn').addEventListener('click', resetUpload);
            document.getElementById('generateReportBtn').addEventListener('click', generateReport);
            document.getElementById('cameraBtn').addEventListener('click', openCamera);
            
            // Progress dots animation
            function animateProgressDots() {
            const dots = document.querySelectorAll('.progress-dot');
            let currentDot = 0;
            const interval = setInterval(() => {
            if (loadingSection.style.display === 'none') {
            clearInterval(interval);
            dots.forEach(dot => dot.classList.remove('active'));
            return;
            }
            dots.forEach(dot => dot.classList.remove('active'));
            dots[currentDot].classList.add('active');
            currentDot = (currentDot + 1) % dots.length;
            }, 500);
            }
            document.addEventListener('DOMContentLoaded', animateProgressDots);
            
            function handleDragOver(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
            }
            
            function handleDrop(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
            handleFile(files[0]);
            }
            }
            
            function handleImageSelect(e) {
            const file = e.target.files[0];
            if (file) {
            handleFile(file);
            }
            }
            
            function handleFile(file) {
            if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file.');
            return;
            }
            if (file.size > 10 * 1024 * 1024) { // 10MB limit
            alert('File size must be less than 10MB.');
            return;
            }
            currentImage = file;
            const reader = new FileReader();
            reader.onload = function(e) {
            previewImg.src = e.target.result;
            uploadArea.style.display = 'none';
            imagePreview.style.display = 'block';
            resultsSection.style.display = 'none';
            gradcamSection.classList.add('hidden');
            };
            reader.readAsDataURL(file);
            }
            
            // Replace the existing openCamera function in your HTML file with this improved version:

            async function openCamera() {
                const cameraBtn = document.getElementById('cameraBtn');
                const originalBtnText = cameraBtn.innerHTML;

                // Check for browser support
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    alert('Camera is not supported in this browser or requires HTTPS connection.');
                    return;
                }

                // Check HTTPS (required outside localhost)
                if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
                    alert('Camera access requires HTTPS connection for security reasons.');
                    return;
                }

                cameraBtn.innerHTML = 'Requesting Camera Access...';
                cameraBtn.disabled = true;

                let stream;
                try {
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: {
                            facingMode: 'environment', // back camera if available
                            width: { ideal: 1280 },
                            height: { ideal: 720 }
                        }
                    });
                } catch (error) {
                    console.error('Camera access error:', error);
                    cameraBtn.innerHTML = originalBtnText;
                    cameraBtn.disabled = false;

                    let errorMessage = 'Camera access failed: ';
                    if (error.name === 'NotAllowedError') {
                        errorMessage += 'Camera permission denied. Please allow camera access and try again.';
                    } else if (error.name === 'NotFoundError') {
                        errorMessage += 'No camera found on this device.';
                    } else if (error.name === 'NotSupportedError') {
                        errorMessage += 'Camera is not supported in this browser.';
                    } else if (error.name === 'NotReadableError') {
                        errorMessage += 'Camera is already in use by another application.';
                    } else {
                        errorMessage += error.message || 'Unknown error occurred.';
                    }
                    alert(errorMessage);
                    return;
                }

                // Reset button
                cameraBtn.innerHTML = originalBtnText;
                cameraBtn.disabled = false;

                // Create video element
                const video = document.createElement('video');
                video.srcObject = stream;
                video.autoplay = true;
                video.playsInline = true; // iOS compatibility

                // Create overlay
                const cameraDiv = document.createElement('div');
                cameraDiv.style.cssText = `
                    position: fixed;
                    top: 0; left: 0; width: 100%; height: 100%;
                    background: rgba(0,0,0,0.9); z-index: 1000;
                    display: flex; flex-direction: column;
                    align-items: center; justify-content: center;
                    padding: 20px; box-sizing: border-box;
                `;

                video.style.cssText = `
                    max-width: 90%; max-height: 60%;
                    border-radius: 1rem; object-fit: cover;
                `;

                const buttonContainer = document.createElement('div');
                buttonContainer.style.cssText = `
                    display: flex; gap: 1rem; margin-top: 1rem;
                    flex-wrap: wrap; justify-content: center;
                `;

                const captureBtn = document.createElement('button');
                captureBtn.textContent = '📸 Capture Photo';
                captureBtn.className = 'btn-action';
                captureBtn.style.cssText = 'padding: 1rem 2rem; font-size: 1.1rem;';

                const closeBtn = document.createElement('button');
                closeBtn.textContent = '❌ Close Camera';
                closeBtn.className = 'btn-action btn-secondary';
                closeBtn.style.cssText = 'padding: 1rem 2rem; font-size: 1.1rem;';

                buttonContainer.appendChild(captureBtn);
                buttonContainer.appendChild(closeBtn);
                cameraDiv.appendChild(video);
                cameraDiv.appendChild(buttonContainer);
                document.body.appendChild(cameraDiv);

                // Video loaded metadata
                video.addEventListener('loadedmetadata', () => console.log('Camera video loaded'));

                // Capture photo
                captureBtn.addEventListener('click', () => {
                    try {
                        const canvas = document.createElement('canvas');
                        canvas.width = video.videoWidth || 640;
                        canvas.height = video.videoHeight || 480;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                        canvas.toBlob((blob) => {
                            if (blob) {
                                const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
                                handleFile(file); // your existing handler
                                stream.getTracks().forEach(track => track.stop());
                                document.body.removeChild(cameraDiv);
                                console.log('Photo captured successfully');
                            } else {
                                alert('Failed to capture photo. Please try again.');
                            }
                        }, 'image/jpeg', 0.8);
                    } catch (err) {
                        console.error('Capture error:', err);
                        alert('Failed to capture photo: ' + err.message);
                    }
                });

                // Close camera
                closeBtn.addEventListener('click', () => {
                    stream.getTracks().forEach(track => track.stop());
                    document.body.removeChild(cameraDiv);
                    console.log('Camera closed by user');
                });

                // Escape key closes camera
                const escapeHandler = (e) => {
                    if (e.key === 'Escape' && document.body.contains(cameraDiv)) {
                        stream.getTracks().forEach(track => track.stop());
                        document.body.removeChild(cameraDiv);
                        document.removeEventListener('keydown', escapeHandler);
                    }
                };
                document.addEventListener('keydown', escapeHandler);
            }

            
            function analyzeImage() {
            if (!currentImage) return;
            imagePreview.style.display = 'none';
            loadingSection.style.display = 'block';
            resultsSection.style.display = 'none';
            animateProgressDots();
            const formData = new FormData();
            formData.append('image', currentImage);
            fetch('/api/predict', {
            method: 'POST',
            body: formData
            })
            .then(response => response.json())
            .then(data => {
            loadingSection.style.display = 'none';
            if (data.success) {
            currentPredictionId = data.prediction_id;
            displayResults(data);
            } else {
            alert('Error: ' + data.error);
            resetUpload();
            }
            })
            .catch(error => {
            loadingSection.style.display = 'none';
            alert('Network error. Please try again.');
            console.error('Error:', error);
            resetUpload();
            });
            }
            
            function displayResults(data) {
            document.getElementById('diseaseTitle').textContent = data.prediction;
            document.getElementById('confidenceText').textContent = data.confidence_percent + '%';
            document.getElementById('confidenceFill').style.width = data.confidence_percent + '%';
            fetchDetailedDiseaseInfo(data.prediction);
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth' });
            document.getElementById('diseaseTitle').classList.add('bounce-animation');
            }
            
            function fetchDetailedDiseaseInfo(diseaseName) {
            fetch(`/disease_info/${encodeURIComponent(diseaseName)}`)
            .then(response => response.json())
            .then(data => {
            if (data.success && data.disease_info) {
            currentDiseaseInfo = data.disease_info;
            createBeautifulInfoCards(data.disease_info);
            } else {
            document.getElementById('diseaseDescription').textContent =
            'Detailed information for this disease is being processed...';
            }
            })
            .catch(error => {
            console.error('Error fetching disease info:', error);
            document.getElementById('diseaseDescription').textContent =
            'Unable to fetch detailed information at this time.';
            });
            }
            
            function createBeautifulInfoCards(diseaseInfo) {
            const cardsGrid = document.getElementById('infoCardsGrid');
            cardsGrid.innerHTML = '';
            if (diseaseInfo.symptoms && diseaseInfo.symptoms.length > 0) {
            const symptomsCard = createInfoCard(
            '🩺',
            'symptoms-icon',
            'Symptoms',
            diseaseInfo.symptoms,
            'list'
            );
            cardsGrid.appendChild(symptomsCard);
            }
            if (diseaseInfo.causes && diseaseInfo.causes.length > 0) {
            const causesCard = createInfoCard(
            '🔍',
            'causes-icon',
            'Causes',
            diseaseInfo.causes,
            'list'
            );
            cardsGrid.appendChild(causesCard);
            }
            if (diseaseInfo.treatment && diseaseInfo.treatment.length > 0) {
            const treatmentCard = createInfoCard(
            '🌿',
            'treatment-icon',
            'Treatment',
            diseaseInfo.treatment,
            'list'
            );
            cardsGrid.appendChild(treatmentCard);
            }
            if (diseaseInfo.prevention && diseaseInfo.prevention.length > 0) {
            const preventionCard = createInfoCard(
            '🛡️',
            'prevention-icon',
            'Prevention',
            diseaseInfo.prevention,
            'list'
            );
            cardsGrid.appendChild(preventionCard);
            }
            if (diseaseInfo.severity_mild || diseaseInfo.severity_moderate || diseaseInfo.severity_severe) {
            const severityCard = createSeverityCard(diseaseInfo);
            cardsGrid.appendChild(severityCard);
            }
            if (diseaseInfo.medicines && diseaseInfo.medicines.length > 0) {
            const medicinesCard = createInfoCard(
            '💊',
            'medicines-icon',
            'Medicines',
            diseaseInfo.medicines,
            'list'
            );
            cardsGrid.appendChild(medicinesCard);
            createMedicineCarousel(diseaseInfo.medicines);
            }
            if (diseaseInfo.progression_days) {
            const progressCard = createProgressCard(diseaseInfo);
            cardsGrid.appendChild(progressCard);
            }
            }
            
            function createInfoCard(icon, iconClass, title, content, type = 'text') {
            const card = document.createElement('div');
            card.className = 'info-card';
            const cardHTML = `
            <div class="card-header">
            <div class="card-icon ${iconClass}">
            ${icon}
            </div>
            <h4 class="card-title">${title}</h4>
            </div>
            <div class="card-content">
            ${type === 'list' && Array.isArray(content) ?
            `<ul class="card-list">
            ${content.map(item => `
            <li>
            <div class="list-bullet"></div>
            <span>${item}</span>
            </li>
            `).join('')}
            </ul>` :
            `<p>${Array.isArray(content) ? content.join(', ') : content}</p>`
            }
            </div>
            `;
            card.innerHTML = cardHTML;
            return card;
            }
            
            function createSeverityCard(diseaseInfo) {
            const card = document.createElement('div');
            card.className = 'info-card';
            const severityHTML = `
            <div class="card-header">
            <div class="card-icon severity-icon">⚡</div>
            <h4 class="card-title">Severity Levels</h4>
            </div>
            <div class="card-content">
            <div class="severity-grid">
            ${diseaseInfo.severity_mild ? `
            <div class="severity-card severity-mild">
            <h5>Mild</h5>
            <p>${diseaseInfo.severity_mild}</p>
            </div>
            ` : ''}
            ${diseaseInfo.severity_moderate ? `
            <div class="severity-card severity-moderate">
            <h5>Moderate</h5>
            <p>${diseaseInfo.severity_moderate}</p>
            </div>
            ` : ''}
            ${diseaseInfo.severity_severe ? `
            <div class="severity-card severity-severe">
            <h5>Severe</h5>
            <p>${diseaseInfo.severity_severe}</p>
            </div>
            ` : ''}
            </div>
            </div>
            `;
            card.innerHTML = severityHTML;
            return card;
            }
            
            function createProgressCard(diseaseInfo) {
            const card = document.createElement('div');
            card.className = 'info-card';
            const progressHTML = `
            <div class="card-header">
            <div class="card-icon" style="background: linear-gradient(135deg, #06b6d4, #67e8f9);">⏱️</div>
            <h4 class="card-title">Disease Progression</h4>
            </div>
            <div class="card-content">
            <p><strong>Typical progression time:</strong> ${diseaseInfo.progression_days} days</p>
            <div style="margin-top: 1rem; padding: 1rem; background: #f0f9ff; border-radius: 0.5rem; border-left: 4px solid #06b6d4;">
            <p style="margin: 0; color: #0c4a6e; font-size: 0.9rem;">
            Early detection and treatment can significantly reduce the progression time and severity of symptoms.
            </p>
            </div>
            </div>
            `;
            card.innerHTML = progressHTML;
            return card;
            }
            
            function createMedicineCarousel(medicines) {
            const medicineSection = document.getElementById('medicineSection');
            const medicineItems = document.getElementById('medicineItems');
            const prevBtn = document.getElementById('medicinePrevBtn');
            const nextBtn = document.getElementById('medicineNextBtn');
            medicineItems.innerHTML = '';
            medicines.forEach((medicine, index) => {
            const medicineCard = document.createElement('div');
            medicineCard.className = 'medicine-card';
            medicineCard.style.animationDelay = `${index * 0.1}s`;
            medicineCard.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <div style="width: 8px; height: 8px; background: #3b82f6; border-radius: 50%;"></div>
            <strong style="color: #1e40af;">${medicine}</strong>
            </div>
            <p style="margin: 0; font-size: 0.85rem; color: #64748b;">
            Recommended treatment option
            </p>
            `;
            medicineItems.appendChild(medicineCard);
            });
            medicineSection.style.display = 'block';
            prevBtn.addEventListener('click', () => {
            medicineItems.scrollBy({ left: -200, behavior: 'smooth' });
            });
            nextBtn.addEventListener('click', () => {
            medicineItems.scrollBy({ left: 200, behavior: 'smooth' });
            });
            }
            
            function generateReport() {
            if (!currentPredictionId) {
            alert('No prediction available to generate report.');
            return;
            }
            console.log('Generating report for prediction ID:', currentPredictionId);
            const reportBtn = document.getElementById('generateReportBtn');
            const originalText = reportBtn.innerHTML;
            reportBtn.innerHTML = 'Generating Report...';
            reportBtn.disabled = true;
            fetch('/api/generate_report', {
            method: 'POST',
            headers: {
            'Content-Type': 'application/json',
            },
            body: JSON.stringify({
            prediction_id: currentPredictionId
            })
            })
            .then(response => {
            console.log('Generate report response status:', response.status);
            return response.json();
            })
            .then(data => {
            console.log('Generate report response data:', data);
            reportBtn.innerHTML = originalText;
            reportBtn.disabled = false;
            if (data.success && data.report_id) {
            console.log('Report generated successfully, downloading...');
            const downloadUrl = `/api/download_report/${data.report_id}`;
            const tempLink = document.createElement('a');
            tempLink.href = downloadUrl;
            tempLink.download = `plant_disease_report_${data.report_id}.pdf`;
            document.body.appendChild(tempLink);
            tempLink.click();
            document.body.removeChild(tempLink);
            } else {
            console.error('Report generation failed:', data.error);
            alert('Error generating report: ' + (data.error || 'Unknown error'));
            }
            })
            .catch(error => {
            console.error('Network error generating report:', error);
            reportBtn.innerHTML = originalText;
            reportBtn.disabled = false;
            alert('Network error. Please try again.');
            });
            }
            
            function showGradcam() {
            if (!currentPredictionId) return;
            fetch(`/api/gradcam/${currentPredictionId}`)
            .then(response => response.json())
            .then(data => {
            if (data.success) {
            document.getElementById('gradcamImage').src = 'data:image/png;base64,' + data.gradcam_image;
            gradcamSection.classList.remove('hidden');
            gradcamSection.scrollIntoView({ behavior: 'smooth' });
            } else {
            alert('Error generating heatmap: ' + data.error);
            }
            })
            .catch(error => {
            alert('Network error. Please try again.');
            console.error('Error:', error);
            });
            }
            
            function resetUpload() {
            currentImage = null;
            currentPredictionId = null;
            currentDiseaseInfo = null;
            uploadArea.style.display = 'block';
            imagePreview.style.display = 'none';
            loadingSection.style.display = 'none';
            resultsSection.style.display = 'none';
            gradcamSection.classList.add('hidden');
            imageInput.value = '';
            infoCardsGrid.innerHTML = '';
            document.getElementById('medicineSection').style.display = 'none';
            }
            </script>
            </body>
            </html>
            """
    except Exception as e:
        return f"Error loading predict page: {str(e)}"

@app.route('/about')
def about_page():
    """Serve the about page"""
    try:
        html_path = Path(__file__).parent / 'about'
        if html_path.exists():
            with open(html_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return """
            <!DOCTYPE html>
            <html lang="en">
            <head>
            <meta charset="UTF-8">
            <title>About - Plant Doctor AI</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                /* General styles */
                body {
                    margin: 0;
                    font-family: 'Segoe UI', 'Arial', sans-serif;
                    background: linear-gradient(135deg, #f0fdf4 0%, #d1fae5 100%);
                    color: #222;
                }
                .min-h-screen { min-height: 100vh; }
                .container {
                    max-width: 1100px;
                    margin: 0 auto;
                    padding: 0 1rem;
                }
                .flex { display: flex; }
                .flex-between { display: flex; align-items: center; justify-content: space-between; }
                .items-center { align-items: center; }
                .gap-2 { gap: 0.5rem; }
                .gap-3 { gap: 0.75rem; }
                .text-center { text-align: center; }
                .mb-12 { margin-bottom: 3rem; }
                .mb-10 { margin-bottom: 2.5rem; }
                .mb-8 { margin-bottom: 2rem; }
                .mt-12 { margin-top: 3rem; }
                .mb-4 { margin-bottom: 1rem; }
                .mr-3 { margin-right: 0.75rem; }
                .center { display: flex; justify-content: center; }
                .animate-pulse {
                    animation: pulse 2s infinite;
                }
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: .7; }
                }
                .animate-bounce {
                    animation: bounce 1.5s infinite;
                }
                @keyframes bounce {
                    0%, 100% { transform: translateY(0);}
                    50% { transform: translateY(-8px);}
                }
                .animate-spin {
                    animation: spin 8s linear infinite;
                }
                @keyframes spin {
                    100% { transform: rotate(360deg);}
                }
                
                /* Header */
                .header {
                    background: #fff;
                    box-shadow: 0 2px 8px -2px rgba(16, 185, 129, 0.08);
                    border-bottom: 1px solid #bbf7d0;
                }
                .title {
                    font-size: 2rem;
                    font-weight: bold;
                    color: #065f46;
                }
                .nav {
                    display: flex;
                    align-items: center;
                    gap: 1.5rem;
                }
                .nav a {
                    color: #047857;
                    font-weight: 500;
                    text-decoration: none;
                    transition: color 0.2s;
                }
                .nav a:hover, .nav a.active {
                    color: #065f46;
                }
                .nav-actions {
                    display: flex;
                    gap: 0.75rem;
                    margin-left: 1.5rem;
                }
                .btn {
                    padding: 0.5rem 1rem;
                    border-radius: 0.75rem;
                    font-weight: 500;
                    transition: all .2s;
                    box-shadow: none;
                    border: none;
                    cursor: pointer;
                }
                .btn-green {
                    background: #16a34a;
                    color: #fff;
                }
                .btn-green:hover {
                    background: #166534;
                    box-shadow: 0 2px 8px -2px #16a34a44;
                }
                .btn-outline-green {
                    background: #fff;
                    border: 2px solid #16a34a;
                    color: #16a34a;
                }
                .btn-outline-green:hover {
                    background: #f0fdf4;
                    box-shadow: 0 2px 8px -2px #16a34a22;
                }
                
                /* Hero */
                .hero-title {
                    font-size: 2.5rem;
                    font-weight: bold;
                    color: #065f46;
                    margin-bottom: 1.5rem;
                }
                .hero-subtitle {
                    font-size: 1.25rem;
                    color: #047857;
                    max-width: 700px;
                    margin: 0 auto 2rem auto;
                    line-height: 1.6;
                }
                
                /* Cards */
                .card {
                    background: #fff;
                    border-radius: 1.25rem;
                    box-shadow: 0 4px 24px -8px #16a34a22;
                    padding: 2rem;
                    transition: box-shadow .3s, transform .3s;
                }
                .card:hover {
                    box-shadow: 0 8px 32px -8px #16a34a22;
                    transform: translateY(-4px);
                }
                .section-title {
                    font-size: 1.75rem;
                    font-weight: bold;
                    color: #065f46;
                    margin-bottom: 1.5rem;
                    display: flex;
                    align-items: center;
                }
                .card-text {
                    color: #374151;
                    font-size: 1.1rem;
                    line-height: 1.7;
                }
                .card-text strong {
                    color: #047857;
                }
                
                /* Stats Grid */
                .grid-stats {
                    display: grid;
                    grid-template-columns: repeat(1, 1fr);
                    gap: 1.5rem;
                }
                @media (min-width: 768px) {
                    .grid-stats { grid-template-columns: repeat(2, 1fr); }
                }
                @media (min-width: 1024px) {
                    .grid-stats { grid-template-columns: repeat(3, 1fr); }
                }
                .stat-card {
                    background: #f0f9ff;
                    border-radius: 1rem;
                    padding: 1.2rem;
                    text-align: center;
                    transition: box-shadow .3s, transform .3s;
                }
                .stat-card:hover {
                    box-shadow: 0 4px 24px -8px #0369a1;
                    transform: scale(1.05);
                }
                .stat-blue { background: linear-gradient(135deg, #eff6ff 0%, #bae6fd 100%);}
                .stat-orange { background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%);}
                .stat-red { background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);}
                .stat-purple { background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%);}
                .stat-green { background: linear-gradient(135deg, #f0fdf4 0%, #bbf7d0 100%);}
                .stat-value {
                    font-size: 2rem;
                    font-weight: bold;
                    margin-bottom: 0.2rem;
                }
                .text-blue { color: #0369a1;}
                .text-orange { color: #ea580c;}
                .text-red { color: #db2777;}
                .text-purple { color: #7c3aed;}
                .text-green { color: #16a34a;}
                .text-white { color: #fff;}
                
                /* Team Grid */
                .grid-team {
                    display: grid;
                    grid-template-columns: repeat(1, 1fr);
                    gap: 1.5rem;
                }
                @media (min-width: 768px) {
                    .grid-team { grid-template-columns: repeat(2, 1fr);}
                }
                @media (min-width: 1024px) {
                    .grid-team { grid-template-columns: repeat(3, 1fr);}
                }
                .team-card {
                    background: #fff;
                    border-radius: 1rem;
                    padding: 1.5rem;
                    text-align: center;
                    border: 2px solid #f0fdf4;
                    transition: all .3s;
                }
                .team-card:hover {
                    border-color: #16a34a;
                    box-shadow: 0 8px 32px -8px #16a34a22;
                    transform: translateY(-4px);
                }
                .team-avatar {
                    width: 80px;
                    height: 80px;
                    background: #16a34a;
                    border-radius: 50%;
                    margin: 0 auto 1rem auto;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 2rem;
                    color: #fff;
                    font-weight: bold;
                }
                .team-name {
                    font-size: 1.2rem;
                    font-weight: 600;
                    color: #065f46;
                    margin-bottom: 0.25rem;
                }
                .team-role {
                    color: #047857;
                    font-size: 1rem;
                }
                
                /* Technology Grid */
                .grid-tech {
                    display: grid;
                    grid-template-columns: repeat(1, 1fr);
                    gap: 1rem;
                }
                @media (min-width: 768px) {
                    .grid-tech { grid-template-columns: repeat(2, 1fr);}
                }
                @media (min-width: 1024px) {
                    .grid-tech { grid-template-columns: repeat(4, 1fr);}
                }
                .tech-card {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    background: #f8fafc;
                    padding: 1rem;
                    border-radius: 0.75rem;
                    border: 1px solid #e2e8f0;
                    transition: all .3s;
                }
                .tech-card:hover {
                    background: #f0fdf4;
                    border-color: #16a34a;
                    transform: scale(1.02);
                }
                .tech-icon {
                    width: 40px;
                    height: 40px;
                    background: #16a34a;
                    border-radius: 0.5rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #fff;
                    font-weight: bold;
                    font-size: 0.9rem;
                }
                .tech-name {
                    font-weight: 600;
                    color: #065f46;
                }
                
                /* Footer */
                .footer {
                    background: #065f46;
                    color: #fff;
                    margin-top: 5rem;
                    padding: 2rem 0;
                }
                .footer-title {
                    font-size: 1.1rem;
                    font-weight: 600;
                    color: #fff;
                }
                .footer-desc {
                    color: #bbf7d0;
                    font-size: 1rem;
                }
                .icon-lg { width: 32px; height: 32px;}
                .icon-md { width: 24px; height: 24px;}
                .icon-sm { width: 20px; height: 20px;}
                
                /* Utility */
                .flex { display: flex; }
                .justify-center { justify-content: center; }
                .items-center { align-items: center; }
                .gap-2 { gap: 0.5rem; }
                .gap-3 { gap: 0.75rem; }
                .profile-circle {
                    width: 40px;
                    height: 40px;
                    background: #16a34a;
                    color: #fff;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    font-size: 1.2rem;
                    cursor: pointer;
                    border: 2px solid #fff;
                    box-shadow: 0 2px 8px -2px #16a34a44;
                    transition: background .2s;
                }
                .profile-circle:hover {
                    background: #166534;
                }
                
                /* Contact Section */
                .contact-grid {
                    display: grid;
                    grid-template-columns: repeat(1, 1fr);
                    gap: 1.5rem;
                }
                @media (min-width: 768px) {
                    .contact-grid { grid-template-columns: repeat(2, 1fr);}
                }
                .contact-card {
                    background: #fff;
                    border-radius: 1rem;
                    padding: 1.5rem;
                    border: 1px solid #e5e7eb;
                    transition: all .3s;
                }
                .contact-card:hover {
                    border-color: #16a34a;
                    box-shadow: 0 4px 20px -8px #16a34a22;
                }
                .contact-icon {
                    width: 48px;
                    height: 48px;
                    background: #16a34a;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-bottom: 1rem;
                }
                .contact-title {
                    font-size: 1.2rem;
                    font-weight: 600;
                    color: #065f46;
                    margin-bottom: 0.5rem;
                }
                .contact-desc {
                    color: #374151;
                }
            </style>
            <!-- Firebase SDKs for profile.js -->
            <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-app-compat.js"></script>
            <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-auth-compat.js"></script>
            </head>
            <body class="min-h-screen">

            <!-- Header -->
            <header class="header">
                <div class="container flex-between">
                <div class="flex items-center gap-2">
                    <svg class="icon-lg text-green" fill="none" stroke="green" stroke-width="2" viewBox="0 0 24 24">
                    <path d="M2 22s9-2 15-8 5-12 5-12-9 2-15 8-5 12-5 12z"/>
                    </svg>
                    <h1 class="title">Plant Doctor AI</h1>
                </div>
                <nav class="nav">
                    <a href="/">Home</a>
                    <a href="/predict">Upload & Predict</a>
                    <a href="/gallery">Disease Gallery</a>
                    <a href="/chatbot">Plant Saathi</a>
                    <a href="/about" class="active">About</a>
                    <div class="nav-actions" id="nav-actions">
                    <!-- Login/Signup or Profile will be injected here by profile.js -->
                    </div>
                </nav>
                </div>
            </header>

            <!-- Main Content -->
            <main class="container">
                <!-- Hero Section -->
                <section class="text-center mb-12">
                <h2 class="hero-title animate-pulse">About Plant Doctor AI</h2>
                <p class="hero-subtitle">
                    Empowering farmers and gardeners worldwide with <strong>cutting-edge AI technology</strong> to combat plant diseases and ensure food security for future generations.
                </p>
                </section>

                <!-- Mission Section -->
                <section class="card mb-10">
                <h3 class="section-title flex items-center">
                    <!-- Target Icon SVG -->
                    <svg class="icon-lg text-green animate-pulse mr-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <circle cx="12" cy="12" r="10"/>
                    <circle cx="12" cy="12" r="6"/>
                    <circle cx="12" cy="12" r="2"/>
                    </svg>
                    Our Mission
                </h3>
                <div class="card-text">
                    <p>
                    Our mission is to <strong>democratize plant disease detection</strong> by making advanced AI technology accessible to farmers, gardeners, and agricultural professionals worldwide. We believe that early and accurate disease identification is crucial for maintaining healthy crops and ensuring global food security.
                    </p>
                    <p>
                    Through innovative computer vision and deep learning techniques, we aim to reduce crop losses, minimize pesticide usage, and support sustainable agricultural practices that benefit both farmers and the environment.
                    </p>
                </div>
                </section>

                <!-- Technology Stack Section -->
                <section class="card mb-10">
                <h3 class="section-title flex items-center">
                    <!-- Code Icon SVG -->
                    <svg class="icon-lg text-green animate-bounce mr-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <polyline points="16,18 22,12 16,6"/>
                    <polyline points="8,6 2,12 8,18"/>
                    </svg>
                    Technology Stack
                </h3>
                <div class="grid-tech mb-8">
                    <div class="tech-card">
                    <div class="tech-icon">Py</div>
                    <div class="tech-name">Python</div>
                    </div>
                    <div class="tech-card">
                    <div class="tech-icon">PT</div>
                    <div class="tech-name">PyTorch</div>
                    </div>
                    <div class="tech-card">
                    <div class="tech-icon">R18</div>
                    <div class="tech-name">ResNet18</div>
                    </div>
                    <div class="tech-card">
                    <div class="tech-icon">CV</div>
                    <div class="tech-name">OpenCV</div>
                    </div>
                    <div class="tech-card">
                    <div class="tech-icon">FL</div>
                    <div class="tech-name">Flask</div>
                    </div>
                    <div class="tech-card">
                    <div class="tech-icon">JS</div>
                    <div class="tech-name">JavaScript</div>
                    </div>
                    <div class="tech-card">
                    <div class="tech-icon">H5</div>
                    <div class="tech-name">HTML5</div>
                    </div>
                    <div class="tech-card">
                    <div class="tech-icon">C3</div>
                    <div class="tech-name">CSS3</div>
                    </div>
                </div>
                <div class="card-text">
                    <p>
                    Our platform is built using <strong>state-of-the-art deep learning technologies</strong>. The core AI model utilizes a <strong>ResNet18 architecture</strong>, a proven convolutional neural network that excels at image classification tasks. This model has been specifically trained on thousands of plant leaf images to recognize various diseases with high accuracy.
                    </p>
                    <p>
                    The backend is powered by <strong>Python and Flask</strong>, providing a robust and scalable server infrastructure. Image processing is handled through <strong>OpenCV</strong> and <strong>PIL</strong>, ensuring optimal image preparation for AI analysis. The frontend delivers an intuitive user experience through modern <strong>HTML5, CSS3, and JavaScript</strong> technologies.
                    </p>
                </div>
                </section>

                <!-- Performance Stats Section -->
                <section class="card mb-10">
                <h3 class="section-title flex items-center">
                    <!-- Bar Chart Icon SVG -->
                    <svg class="icon-lg text-green animate-pulse mr-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <line x1="12" y1="20" x2="12" y2="10"/>
                    <line x1="18" y1="20" x2="18" y2="4"/>
                    <line x1="6" y1="20" x2="6" y2="16"/>
                    </svg>
                    Performance & Accuracy
                </h3>
                <div class="grid-stats mb-8">
                    <div class="stat-card stat-green">
                    <div class="stat-value text-green">99%</div>
                    <div class="stat-label text-green">Model Accuracy</div>
                    </div>
                    <div class="stat-card stat-blue">
                    <div class="stat-value text-blue">38</div>
                    <div class="stat-label text-blue">Disease Classes</div>
                    </div>
                </div>
                <div class="card-text">
                    <p>
                    Our AI model achieves an impressive <strong>99% accuracy rate</strong> across all supported plant diseases. The system can identify <strong>38 different disease classes</strong> spanning <strong>15 major plant species</strong> including tomatoes, potatoes, corn, apples, grapes, and many others.
                    </p>
                    <p>
                    The model processes images in <strong>real-time</strong>, typically providing results within 2-3 seconds. It has been trained on over <strong>60,000 high-quality plant images</strong> and continuously improves through additional data collection and model refinement.
                    </p>
                </div>
                </section>

                <!-- Team Section -->
                <section class="card mb-10">
                <h3 class="section-title flex items-center">
                    <!-- Users Icon SVG -->
                    <svg class="icon-lg text-green animate-bounce mr-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
                    <circle cx="9" cy="7" r="4"/>
                    <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/>
                    </svg>
                    Founder
                </h3>
                <div class="grid-team mb-8">
                    <div class="team-card">
                    <div class="team-avatar">AI</div>
                    <div class="team-name">Debottam Ghosh</div>
                    <div class="team-role">Machine Learning Enthusiast</div>
                    </div>
                </div>
                <div class="card-text">
                    <p>
                    Plant Doctor AI is developed by Debottam Ghosh, a skilled machine learning enthusiast, combining expertise in <strong>artificial intelligence, agriculture, and software development</strong>. He specializes in computer vision and deep learning, while also researching about plant diseases and agricultural practices.
                    </p>
                    <p>
                    He focuses on creating user-friendly interfaces and reliable backend systems that make advanced AI technology accessible to users regardless of their technical background.
                    </p>
                </div>
                </section>

                <!-- How It Works Section -->
                <section class="card mb-10">
                <h3 class="section-title flex items-center">
                    <!-- Settings Icon SVG -->
                    <svg class="icon-lg text-green animate-spin mr-3" style="animation-duration: 6s" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <circle cx="12" cy="12" r="3"/>
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1 1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                    </svg>
                    How It Works
                </h3>
                <div class="card-text">
                    <p>
                    Our AI system follows a sophisticated <strong>multi-step analysis process</strong>:
                    </p>
                    <p>
                    <strong>1. Image Preprocessing:</strong> Uploaded images are automatically resized, normalized, and enhanced to optimize them for AI analysis. This ensures consistent input quality regardless of camera type or lighting conditions.
                    </p>
                    <p>
                    <strong>2. Feature Extraction:</strong> The ResNet18 model extracts thousands of visual features from the leaf image, including color patterns, texture details, spot characteristics, and leaf structure anomalies.
                    </p>
                    <p>
                    <strong>3. Classification:</strong> These features are processed through our trained neural network, which compares them against learned patterns from our extensive disease database.
                    </p>
                    <p>
                    <strong>4. Results & Recommendations:</strong> The system provides the most likely disease diagnosis with confidence scores, along with detailed information about symptoms, causes, treatment options, and prevention strategies.
                    </p>
                </div>
                </section>

                <!-- Impact Section -->
                <section class="card mb-10">
                <h3 class="section-title flex items-center">
                    <!-- Award Icon SVG -->
                    <svg class="icon-lg text-green animate-pulse mr-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <circle cx="12" cy="8" r="7"/>
                    <polyline points="8.21,13.89 7,23 12,20 17,23 15.79,13.88"/>
                    </svg>
                    Our Impact
                </h3>
                <div class="card-text">
                    <p>
                    Plant Doctor AI represents a significant step forward in <strong>precision agriculture</strong> and sustainable farming practices. By enabling early disease detection, we help farmers:
                    </p>
                    <p>
                    <strong>• Reduce Crop Losses:</strong> Early intervention can prevent diseases from spreading, saving entire harvests.
                    <br>
                    <strong>• Optimize Pesticide Usage:</strong> Targeted treatment recommendations reduce unnecessary chemical applications.
                    <br>
                    <strong>• Improve Yield Quality:</strong> Healthier plants produce better quality crops with higher market value.
                    <br>
                    <strong>• Save Time & Resources:</strong> Quick AI diagnosis eliminates the need for time-consuming expert consultations.
                    <br>
                    <strong>• Support Sustainable Practices:</strong> Precision agriculture techniques promote environmental conservation.
                    </p>
                    <p>
                    Our vision extends beyond individual farmers to support <strong>global food security initiatives</strong> and contribute to the United Nations' Sustainable Development Goals related to zero hunger and sustainable agriculture.
                    </p>
                </div>
                </section>

                <!-- Contact Section -->
                <section class="card">
                <h3 class="section-title flex items-center">
                    <!-- Mail Icon SVG -->
                    <svg class="icon-lg text-green animate-bounce mr-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
                    <polyline points="22,6 12,13 2,6"/>
                    </svg>
                    Get In Touch
                </h3>
                <div class="contact-grid">
                    <div class="contact-card">
                    <div class="contact-icon">
                        <!-- Help Circle Icon SVG -->
                        <svg class="icon-md text-white" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                        <point cx="12" cy="17" r=".5"/>
                        </svg>
                    </div>
                    <div class="contact-title">Support</div>
                    <div class="contact-desc">Need help using Plant Doctor AI? Our support team is here to assist you with any questions or technical issues. Email us at <a href="mailto: greenplantdoctor25@gmail.com" class="text-green">greenplantdoctor25@gmail.com</a> for help.</div>
                    </div>
                    <div class="contact-card">
                    <div class="contact-icon">
                        <!-- Lightbulb Icon SVG -->
                        <svg class="icon-md text-white" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path d="M9 21h6"/>
                        <path d="M10 21v-2a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v2"/>
                        <path d="M7 17v-5a6 6 0 0 1 12 0v5"/>
                        </svg>
                    </div>
                    <div class="contact-title">Feedback</div>
                    <div class="contact-desc">Have suggestions for improvement? We value your feedback and use it to make Plant Doctor AI even better. Email us at <a href="mailto: greenplantdoctor25@gmail.com" class="text-green">greenplantdoctor25@gmail.com</a> to write feedback.</div>
                    </div>
                </div>
                </section>
            </main>

            <!-- Footer -->
            <footer class="footer">
                <div class="container text-center">
                <div class="flex items-center justify-center gap-2 mb-4">
                    <!-- Leaf Icon SVG -->
                    <svg class="icon-md text-white" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path d="M2 22s9-2 15-8 5-12 5-12-9 2-15 8-5 12-5 12z"/>
                    </svg>
                    <span class="footer-title">Plant Doctor AI</span>
                </div>
                <p class="footer-desc">Powered by AI • Helping farmers worldwide</p>
                </div>
            </footer>

            <!-- Profile.js Logic -->
            <script>
                const firebaseConfig = {
                apiKey: "AIzaSyB-y52ekJkjgXYXhRNvvir0r9gU8CObpkM",
                authDomain: "plant-doctor-63da7.firebaseapp.com",
                projectId: "plant-doctor-63da7",
                storageBucket: "plant-doctor-63da7.appspot.com",
                messagingSenderId: "94281353808",
                appId: "1:94281353808:web:5fa49e3d6e494868be5f55",
                measurementId: "G-E5P4MZ68HW"
                };
                if (!firebase.apps.length) {
                firebase.initializeApp(firebaseConfig);
                }
                const auth = firebase.auth();

                function renderProfile(user) {
                const navActions = document.getElementById("nav-actions");
                if (!navActions) return;

                // Show profile circle with first letter
                let displayName = user.displayName || user.email || "U";
                let firstLetter = displayName.charAt(0).toUpperCase();

                navActions.innerHTML = `
                    <div class="profile-circle" title="${displayName}" onclick="logout()">
                    ${firstLetter}
                    </div>
                `;
                }

                function renderLoginSignup() {
                const navActions = document.getElementById("nav-actions");
                if (!navActions) return;
                navActions.innerHTML = `
                    <a href="/signin" class="btn btn-green">Log In</a>
                    <a href="/signup" class="btn btn-outline-green">Sign Up</a>
                `;
                }

                // Logout function
                function logout() {
                firebase.auth().signOut().then(() => {
                    window.location.reload();
                });
                }

                // Listen for auth state changes
                auth.onAuthStateChanged(user => {
                if (user) {
                    renderProfile(user);
                } else {
                    renderLoginSignup();
                }
                });
            </script>
            </body>
            </html>

            """
    except Exception as e:
        return f"Error loading about page: {str(e)}"

@app.route('/gallery')
def gallery_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Disease Gallery - Plant Doctor AI</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            /* General styles */
            body {
                margin: 0;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                background: linear-gradient(135deg, #f0fdf4 0%, #d1fae5 100%);
                color: #222;
            }
            .min-h-screen { min-height: 100vh; }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 0 1rem;
            }
            .flex { display: flex; }
            .flex-between { display: flex; align-items: center; justify-content: space-between; }
            .items-center { align-items: center; }
            .justify-center { justify-content: center; }
            .gap-2 { gap: 0.5rem; }
            .gap-3 { gap: 0.75rem; }
            .text-center { text-align: center; }
            .mb-12 { margin-bottom: 3rem; }
            .mb-10 { margin-bottom: 2.5rem; }
            .mb-8 { margin-bottom: 2rem; }
            .mb-4 { margin-bottom: 1rem; }
            .mr-3 { margin-right: 0.75rem; }
            
            /* Header */
            .header {
                background: #fff;
                box-shadow: 0 2px 8px -2px rgba(16, 185, 129, 0.08);
                border-bottom: 1px solid #bbf7d0;
                padding: 1rem 0;
            }
            .title {
                font-size: 2rem;
                font-weight: bold;
                color: #065f46;
            }
            .nav {
                display: flex;
                align-items: center;
                gap: 1.5rem;
            }
            .nav a {
                color: #047857;
                font-weight: 500;
                text-decoration: none;
                transition: color 0.2s;
            }
            .nav a:hover {
                color: #065f46;
            }
            .nav-actions {
                display: flex;
                gap: 0.75rem;
                margin-left: 1.5rem;
            }
            .btn {
                padding: 0.5rem 1rem;
                border-radius: 0.75rem;
                font-weight: 500;
                transition: all .2s;
                box-shadow: none;
                border: none;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
            }
            .btn-green {
                background: #16a34a;
                color: #fff;
            }
            .btn-green:hover {
                background: #166534;
                box-shadow: 0 2px 8px -2px #16a34a44;
            }
            .btn-outline-green {
                background: #fff;
                border: 2px solid #16a34a;
                color: #16a34a;
            }
            .btn-outline-green:hover {
                background: #f0fdf4;
                box-shadow: 0 2px 8px -2px #16a34a22;
            }
            .profile-circle {
                width: 40px;
                height: 40px;
                background: #16a34a;
                color: #fff;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 1.2rem;
                cursor: pointer;
                border: 2px solid #fff;
                box-shadow: 0 2px 8px -2px #16a34a44;
                transition: background .2s;
            }
            .profile-circle:hover {
                background: #166534;
            }
            
            /* Hero Section */
            .hero-section {
                padding: 3rem 0 2rem;
            }
            .hero-title {
                font-size: 2.5rem;
                font-weight: bold;
                color: #065f46;
                margin-bottom: 1rem;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: .7; }
            }
            .hero-subtitle {
                font-size: 1.25rem;
                color: #047857;
                max-width: 700px;
                margin: 0 auto 2rem;
                line-height: 1.6;
            }
            
            /* Search Section */
            .search-container {
                max-width: 600px;
                margin: 0 auto 2rem;
                position: relative;
            }
            .search-input {
                width: 100%;
                padding: 1rem 3rem 1rem 1.5rem;
                border: 2px solid #bbf7d0;
                border-radius: 1rem;
                font-size: 1.1rem;
                outline: none;
                transition: all 0.3s;
                background: #fff;
                box-shadow: 0 4px 24px -8px rgba(22, 163, 74, 0.15);
            }
            .search-input:focus {
                border-color: #16a34a;
                box-shadow: 0 4px 24px -8px rgba(22, 163, 74, 0.25);
            }
            .search-icon {
                position: absolute;
                right: 1.5rem;
                top: 50%;
                transform: translateY(-50%);
                color: #047857;
                pointer-events: none;
                width: 24px;
                height: 24px;
            }
            
            /* Gallery Grid */
            .gallery-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                gap: 2rem;
                padding: 2rem 0;
            }
            
            /* Disease Card */
            .disease-card {
                background: #fff;
                border-radius: 1.25rem;
                box-shadow: 0 4px 24px -8px rgba(22, 163, 74, 0.15);
                overflow: hidden;
                transition: all 0.3s;
                cursor: pointer;
            }
            .disease-card:hover {
                transform: translateY(-8px);
                box-shadow: 0 12px 40px -8px rgba(22, 163, 74, 0.25);
            }
            .card-image-container {
                position: relative;
                width: 100%;
                height: 220px;
                overflow: hidden;
                background: linear-gradient(135deg, #f0fdf4 0%, #d1fae5 100%);
            }
            .card-image {
                width: 100%;
                height: 100%;
                object-fit: cover;
                transition: transform 0.3s;
            }
            .disease-card:hover .card-image {
                transform: scale(1.1);
            }
            .card-badge {
                position: absolute;
                top: 1rem;
                right: 1rem;
                background: rgba(22, 163, 74, 0.9);
                color: #fff;
                padding: 0.4rem 0.8rem;
                border-radius: 1rem;
                font-size: 0.85rem;
                font-weight: 600;
                box-shadow: 0 2px 8px -2px rgba(0, 0, 0, 0.3);
            }
            .card-badge.healthy {
                background: rgba(34, 197, 94, 0.9);
            }
            .img-gradient {
                position: absolute;
                inset: 0;
                background: linear-gradient(to top, rgba(22, 163, 74, 0.2) 0%, transparent 100%);
                pointer-events: none;
            }
            .card-content {
                padding: 1.5rem;
            }
            .card-title {
                font-size: 1.4rem;
                font-weight: bold;
                color: #065f46;
                margin-bottom: 0.5rem;
            }
            .card-plant {
                color: #047857;
                font-size: 0.95rem;
                margin-bottom: 1rem;
                font-weight: 500;
                display: flex;
                align-items: center;
                gap: 0.3rem;
            }
            .card-info {
                display: flex;
                gap: 0.75rem;
                margin-bottom: 1rem;
                flex-wrap: wrap;
            }
            .info-badge {
                display: flex;
                align-items: center;
                gap: 0.3rem;
                font-size: 0.85rem;
                color: #059669;
                background: #f0fdf4;
                padding: 0.3rem 0.6rem;
                border-radius: 0.5rem;
            }
            .card-symptoms {
                color: #374151;
                font-size: 0.9rem;
                line-height: 1.5;
                margin-bottom: 1rem;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
                overflow: hidden;
            }
            .card-footer {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding-top: 1rem;
                border-top: 1px solid #f0fdf4;
            }
            .view-details-btn {
                background: #16a34a;
                color: #fff;
                padding: 0.5rem 1rem;
                border-radius: 0.75rem;
                border: none;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s;
                font-size: 0.9rem;
            }
            .view-details-btn:hover {
                background: #166534;
                transform: translateY(-2px);
                box-shadow: 0 4px 16px -4px rgba(22, 163, 74, 0.4);
            }
            
            /* Modal */
            .modal {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.7);
                z-index: 1000;
                overflow-y: auto;
            }
            .modal.active {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 2rem;
            }
            .modal-content {
                background: #fff;
                border-radius: 1.5rem;
                max-width: 800px;
                width: 100%;
                max-height: 90vh;
                overflow-y: auto;
                position: relative;
                box-shadow: 0 20px 60px -10px rgba(0, 0, 0, 0.5);
            }
            .modal-header {
                padding: 2rem;
                border-bottom: 1px solid #e5e7eb;
                position: sticky;
                top: 0;
                background: #fff;
                z-index: 10;
                border-radius: 1.5rem 1.5rem 0 0;
            }
            .modal-close {
                position: absolute;
                top: 1.5rem;
                right: 1.5rem;
                background: #f3f4f6;
                border: none;
                width: 36px;
                height: 36px;
                border-radius: 50%;
                cursor: pointer;
                font-size: 1.5rem;
                color: #374151;
                transition: all 0.3s;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .modal-close:hover {
                background: #e5e7eb;
                transform: scale(1.1);
            }
            .modal-body {
                padding: 2rem;
            }
            .modal-image {
                width: 100%;
                height: 300px;
                object-fit: cover;
                border-radius: 1rem;
                margin-bottom: 1.5rem;
            }
            .modal-title {
                font-size: 2rem;
                font-weight: bold;
                color: #065f46;
                margin-bottom: 0.5rem;
            }
            .modal-section {
                margin-bottom: 1.5rem;
            }
            .section-title {
                font-size: 1.2rem;
                font-weight: bold;
                color: #047857;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .section-content {
                color: #374151;
                line-height: 1.7;
            }
            .section-list {
                list-style: none;
                padding: 0;
            }
            .section-list li {
                padding: 0.5rem 0;
                padding-left: 1.5rem;
                position: relative;
                color: #374151;
            }
            .section-list li:before {
                content: "✓";
                position: absolute;
                left: 0;
                color: #16a34a;
                font-weight: bold;
            }
            
            /* No Results */
            .no-results {
                text-align: center;
                padding: 4rem 2rem;
                display: none;
            }
            .no-results.show {
                display: block;
            }
            .no-results-icon {
                font-size: 4rem;
                margin-bottom: 1rem;
            }
            .no-results-text {
                font-size: 1.5rem;
                color: #047857;
                font-weight: 600;
            }
            
            /* Loading */
            .loading {
                text-align: center;
                padding: 4rem 2rem;
            }
            .spinner {
                width: 50px;
                height: 50px;
                border: 4px solid #bbf7d0;
                border-top: 4px solid #16a34a;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 1rem;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            /* Footer */
            .footer {
                background: #065f46;
                color: #fff;
                margin-top: 5rem;
                padding: 2rem 0;
                text-align: center;
            }
            .footer-title {
                font-size: 1.1rem;
                font-weight: 600;
                color: #fff;
                margin-bottom: 0.5rem;
            }
            .footer-desc {
                color: #bbf7d0;
                font-size: 1rem;
            }
            
            /* Icons */
            .icon-sm { width: 16px; height: 16px; }
            .icon-md { width: 24px; height: 24px; }
            .icon-lg { width: 32px; height: 32px; }
            
            /* Responsive */
            @media (max-width: 768px) {
                .gallery-grid {
                    grid-template-columns: 1fr;
                }
                .hero-title {
                    font-size: 2rem;
                }
                .modal-content {
                    margin: 1rem;
                }
                .nav {
                    flex-wrap: wrap;
                    gap: 1rem;
                }
            }
        </style>
    </head>
    <body class="min-h-screen">
        <!-- Header -->
        <header class="header">
            <div class="container flex-between">
                <div class="flex items-center gap-2">
                    <svg class="icon-lg" style="color: #16a34a;" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path d="M2 22s9-2 15-8 5-12 5-12-9 2-15 8-5 12-5 12z"/>
                    </svg>
                    <h1 class="title">Plant Doctor AI</h1>
                </div>
                <nav class="nav">
                    <a href="/">Home</a>
                    <a href="/predict">Upload & Predict</a>
                    <a href="/gallery">Disease Gallery</a>
                    <a href="/chatbot">Plant Saathi</a>
                    <a href="/about">About</a>
                    <div class="nav-actions" id="nav-actions">
                        <!-- Login/Signup or Profile will be injected here -->
                    </div>
                </nav>
            </div>
        </header>

        <!-- Main Content -->
        <main class="container">
            <!-- Hero Section -->
            <section class="hero-section text-center">
                <h2 class="hero-title">Disease Gallery</h2>
                <p class="hero-subtitle">
                    Browse through our comprehensive collection of plant diseases. Search by plant type or disease name to learn about symptoms, causes, and treatments.
                </p>
            </section>

            <!-- Search Section -->
            <section class="text-center mb-8">
                <div class="search-container">
                    <input 
                        type="text" 
                        id="searchInput" 
                        class="search-input" 
                        placeholder="Search by plant or disease name..."
                        autocomplete="off"
                    >
                    <svg class="search-icon" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <circle cx="11" cy="11" r="8"/>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                    </svg>
                </div>
            </section>

            <!-- Loading -->
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="color: #047857; font-weight: 500;">Loading disease gallery...</p>
            </div>

            <!-- Gallery Grid -->
            <div class="gallery-grid" id="galleryGrid"></div>

            <!-- No Results -->
            <div class="no-results" id="noResults">
                <div class="no-results-icon">🔍</div>
                <p class="no-results-text">No diseases found matching your search</p>
                <p style="color: #059669; margin-top: 1rem;">Try adjusting your search terms</p>
            </div>
        </main>

        <!-- Modal -->
        <div class="modal" id="diseaseModal">
            <div class="modal-content">
                <div class="modal-header">
                    <button class="modal-close" onclick="closeModal()">×</button>
                </div>
                <div class="modal-body" id="modalBody"></div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="footer">
            <div class="container">
                <div class="flex items-center justify-center gap-2 mb-4">
                    <svg class="icon-md" style="color: #fff;" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                        <path d="M2 22s9-2 15-8 5-12 5-12-9 2-15 8-5 12-5 12z"/>
                    </svg>
                    <span class="footer-title">Plant Doctor AI</span>
                </div>
                <p class="footer-desc">Powered by AI • Helping farmers worldwide</p>
            </div>
        </footer>

        <!-- Firebase SDKs -->
        <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-app-compat.js"></script>
        <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-auth-compat.js"></script>
        
        <script>
            // Firebase Configuration
            const firebaseConfig = {
                apiKey: "AIzaSyB-y52ekJkjgXYXhRNvvir0r9gU8CObpkM",
                authDomain: "plant-doctor-63da7.firebaseapp.com",
                projectId: "plant-doctor-63da7",
                storageBucket: "plant-doctor-63da7.appspot.com",
                messagingSenderId: "94281353808",
                appId: "1:94281353808:web:5fa49e3d6e494868be5f55",
                measurementId: "G-E5P4MZ68HW"
            };

            if (!firebase.apps.length) {
                firebase.initializeApp(firebaseConfig);
            }
            const auth = firebase.auth();

            function renderProfile(user) {
                const navActions = document.getElementById("nav-actions");
                if (!navActions) return;

                let displayName = user.displayName || user.email || "U";
                let firstLetter = displayName.charAt(0).toUpperCase();

                navActions.innerHTML = `
                    <div class="profile-circle" title="${displayName}" onclick="logout()">
                        ${firstLetter}
                    </div>
                `;
            }

            function renderLoginSignup() {
                const navActions = document.getElementById("nav-actions");
                if (!navActions) return;
                navActions.innerHTML = `
                    <a href="/signin" class="btn btn-green">Log In</a>
                    <a href="/signup" class="btn btn-outline-green">Sign Up</a>
                `;
            }

            function logout() {
                firebase.auth().signOut().then(() => {
                    window.location.reload();
                });
            }

            auth.onAuthStateChanged(user => {
                if (user) {
                    renderProfile(user);
                } else {
                    renderLoginSignup();
                }
            });

            // Disease data embedded from JSON
            const diseaseData = {
        "Peach Leaf Curl": {
            "plant": "Peach",
            "disease": "Leaf Curl",
            "causes": [
                "Fungal infection",
                "Poor air circulation",
                "High humidity"
            ],
            "symptoms": [
                "Curling of leaves",
                "Red or yellow discoloration",
                "Leaf drop"
            ],
            "severity_mild": "Leaves may curl slightly but the plant remains healthy overall.",
            "severity_moderate": "Leaves show significant curling and discoloration, affecting plant health.",
            "severity_severe": "Leaves are severely curled and discolored, leading to potential death of the plant if untreated.",
            "treatment": [
                "Prune affected leaves",
                "Improve air circulation",
                "Apply fungicide",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 14,
            "prevention": [
                "Regularly inspect plants for early signs of disease",
                "Ensure good air circulation around plants",
                "Avoid overhead watering"
            ]
        },
        "Tomato Late Blight": {
            "plant": "Tomato",
            "disease": "Late Blight",
            "causes": [
                "Fungal infection (Phytophthora infestans)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark, water-soaked spots on leaves",
                "White fungal growth on undersides of leaves",
                "Wilting and browning of leaves"
            ],
            "severity_mild": "Few leaves show dark spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant wilting and discoloration.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected plants",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Corn Leaf Blight": {
            "plant": "Corn",
            "disease": "Leaf Blight",
            "causes": [
                "Fungal infection",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Brown spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show brown spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 12,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Apple Rust": {
            "plant": "Apple",
            "disease": "Rust",
            "causes": [
                "Fungal infection",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Orange or yellow spots on leaves",
                "Rusty appearance on undersides of leaves",
                "Leaf drop"
            ],
            "severity_mild": "Few leaves show orange spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant discoloration and leaf drop.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Apple Black Rot": {
            "plant": "Apple",
            "disease": "Black Rot",
            "causes": [
                "Fungal infection (Botryosphaeria obtusa)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark, sunken lesions on fruit",
                "Black streaks on leaves",
                "Leaf drop"
            ],
            "severity_mild": "Few fruits show dark lesions but the plant remains healthy.",
            "severity_moderate": "Multiple fruits affected with significant lesions and leaf drop.",
            "severity_severe": "Widespread fruit damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected fruits",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 14,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Tomato Mosaic Virus": {
            "plant": "Tomato",
            "disease": "Mosaic Virus",
            "causes": [
                "Viral infection",
                "Insect vectors (aphids, whiteflies)",
                "Contaminated tools or seeds"
            ],
            "symptoms": [
                "Mosaic pattern on leaves",
                "Stunted growth",
                "Yellowing of leaves"
            ],
            "severity_mild": "Few leaves show mosaic patterns but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant stunting and yellowing.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected plants",
                "Control insect vectors",
                "Sanitize tools and equipment"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Use virus-resistant varieties",
                "Control insect populations",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Peach Healthy": {
            "plant": "Peach",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Potato Healthy": {
            "plant": "Potato",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Tomato Yellow Virus": {
            "plant": "Tomato",
            "disease": "Yellow Virus",
            "causes": [
                "Viral infection",
                "Insect vectors (aphids, whiteflies)",
                "Contaminated tools or seeds"
            ],
            "symptoms": [
                "Yellowing of leaves",
                "Stunted growth",
                "Leaf curling"
            ],
            "severity_mild": "Few leaves show yellowing but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant stunting and curling.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected plants",
                "Control insect vectors",
                "Sanitize tools and equipment"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Use virus-resistant varieties",
                "Control insect populations",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Squash Powdery Mildew": {
            "plant": "Squash",
            "disease": "Powdery Mildew",
            "causes": [
                "Fungal infection",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "White, powdery spots on leaves",
                "Leaf curling and distortion",
                "Yellowing of leaves"
            ],
            "severity_mild": "Few leaves show white spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant curling and yellowing.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Sulfur fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Tomato Early Blight": {
            "plant": "Tomato",
            "disease": "Early Blight",
            "causes": [
                "Fungal infection (Alternaria solani)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark, concentric rings on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show dark rings but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 12,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Orange Citrus Greening": {
            "plant": "Orange",
            "disease": "Citrus Greening",
            "causes": [
                "Bacterial infection (Candidatus Liberibacter asiaticus)",
                "Insect vectors (Asian citrus psyllid)"
            ],
            "symptoms": [
                "Yellowing of leaves",
                "Stunted growth",
                "Bitter, misshapen fruit"
            ],
            "severity_mild": "Few leaves show yellowing but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant stunting and fruit deformities.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected plants",
                "Control insect vectors",
                "Sanitize tools and equipment"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 14,
            "prevention": [
                "Use disease-resistant varieties",
                "Control insect populations",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Bell Pepper Leaf Spot": {
            "plant": "Bell Pepper",
            "disease": "Leaf Spot",
            "causes": [
                "Fungal infection",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Brown or black spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Strawberry Healthy": {
            "plant": "Strawberry",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Apple Scab": {
            "plant": "Apple",
            "disease": "Scab",
            "causes": [
                "Fungal infection (Venturia inaequalis)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Olive-green to black lesions on leaves",
                "Deformed fruit",
                "Leaf drop"
            ],
            "severity_mild": "Few leaves show lesions but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant lesions and leaf drop.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 12,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Tomato Leaf Mold": {
            "plant": "Tomato",
            "disease": "Leaf Mold",
            "causes": [
                "Fungal infection (Fulvia fulva)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Yellowing of leaves",
                "Grayish mold on undersides of leaves",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show yellowing but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant wilting and discoloration.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Tomato Target Spot": {
            "plant": "Tomato",
            "disease": "Target Spot",
            "causes": [
                "Fungal infection (Corynespora cassiicola)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark, circular spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show dark spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 12,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Soybean Healthy": {
            "plant": "Soybean",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Grape Esca": {
            "plant": "Grape",
            "disease": "Esca",
            "causes": [
                "Fungal infection (Phaeomoniella chlamydospora)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Leaf yellowing and wilting",
                "Dark streaks on wood",
                "Poor fruit development"
            ],
            "severity_mild": "Few leaves show yellowing but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant wilting and poor fruit development.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected plants",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 14,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Tomato Bacterial Spot": {
            "plant": "Tomato",
            "disease": "Bacterial Spot",
            "causes": [
                "Bacterial infection (Xanthomonas campestris pv. vesicatoria)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Water-soaked spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show water-soaked spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply bactericide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper bactericide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Strawberry Leaf Scorch": {
            "plant": "Strawberry",
            "disease": "Leaf Scorch",
            "causes": [
                "Fungal infection",
                "High temperatures",
                "Poor air circulation"
            ],
            "symptoms": [
                "Brown, scorched edges on leaves",
                "Wilting of leaves",
                "Reduced fruit yield"
            ],
            "severity_mild": "Few leaves show scorched edges but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant wilting and reduced yield.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy affected leaves",
                "Improve air circulation",
                "Maintain proper watering practices"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Ensure proper spacing between plants",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Grape Leaf Blight": {
            "plant": "Grape",
            "disease": "Leaf Blight",
            "causes": [
                "Fungal infection (Phomopsis viticola)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Brown spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show brown spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 12,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Potato Late Blight": {
            "plant": "Potato",
            "disease": "Late Blight",
            "causes": [
                "Fungal infection (Phytophthora infestans)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark, water-soaked spots on leaves",
                "White fungal growth on undersides of leaves",
                "Wilting and browning of leaves"
            ],
            "severity_mild": "Few leaves show dark spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant wilting and discoloration.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected plants",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Corn Healthy": {
            "plant": "Corn",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Cherry Healthy": {
            "plant": "Cherry",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Bell Pepper Healthy": {
            "plant": "Bell Pepper",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Corn Rust": {
            "plant": "Corn",
            "disease": "Rust",
            "causes": [
                "Fungal infection (Puccinia sorghi)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Orange or brown pustules on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show pustules but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Blueberry Healthy": {
            "plant": "Blueberry",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Apple Healthy": {
            "plant": "Apple",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Potato Early Blight": {
            "plant": "Potato",
            "disease": "Early Blight",
            "causes": [
                "Fungal infection (Alternaria solani)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark, concentric rings on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show dark rings but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 12,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Cherry Powdery Mildew": {
            "plant": "Cherry",
            "disease": "Powdery Mildew",
            "causes": [
                "Fungal infection",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "White, powdery spots on leaves",
                "Leaf curling and distortion",
                "Yellowing of leaves"
            ],
            "severity_mild": "Few leaves show white spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant curling and yellowing.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Sulfur fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Corn Gray Leaf Spot": {
            "plant": "Corn",
            "disease": "Gray Leaf Spot",
            "causes": [
                "Fungal infection (Cercospora zeae-maydis)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Grayish-brown spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show gray spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 12,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Grape Black Rot": {
            "plant": "Grape",
            "disease": "Black Rot",
            "causes": [
                "Fungal infection (Guignardia bidwellii)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark, sunken lesions on fruit",
                "Black streaks on leaves",
                "Leaf drop"
            ],
            "severity_mild": "Few fruits show dark lesions but the plant remains healthy.",
            "severity_moderate": "Multiple fruits affected with significant lesions and leaf drop.",
            "severity_severe": "Widespread fruit damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected fruits",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 14,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Raspberry Healthy": {
            "plant": "Raspberry",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Grape Healthy": {
            "plant": "Grape",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Tomato Spider Mite": {
            "plant": "Tomato",
            "disease": "Spider Mite",
            "causes": [
                "Insect infestation (Tetranychus urticae)",
                "High temperatures",
                "Low humidity"
            ],
            "symptoms": [
                "Fine webbing on leaves",
                "Yellowing and stippling of leaves",
                "Leaf drop"
            ],
            "severity_mild": "Few leaves show webbing but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and stippling.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infested leaves",
                "Apply miticide",
                "Increase humidity around plants"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Regularly inspect plants for early signs of infestation",
                "Maintain proper humidity levels",
                "Avoid overcrowding plants"
            ]
        },
        "Tomato Healthy": {
            "plant": "Tomato",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Tomato Septoria Leaf Spot": {
            "plant": "Tomato",
            "disease": "Septoria Leaf Spot",
            "causes": [
                "Fungal infection (Septoria lycopersici)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Small, dark spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show dark spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Banana Bunchy Top Virus": {
            "plant": "Banana",
            "disease": "Bunchy Top Virus",
            "causes": [
                "Viral infection",
                "Insect vectors (aphids)",
                "Contaminated tools or seeds"
            ],
            "symptoms": [
                "Stunted growth",
                "Bunchy appearance of leaves",
                "Yellowing of leaf edges"
            ],
            "severity_mild": "Few leaves show stunted growth but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant stunting and yellowing.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected plants",
                "Control insect vectors",
                "Sanitize tools and equipment"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Use virus-resistant varieties",
                "Control insect populations",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Banana Sigatoka": {
            "plant": "Banana",
            "disease": "Sigatoka",
            "causes": [
                "Fungal infection (Mycosphaerella fijiensis)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark streaks on leaves",
                "Yellowing of leaf edges",
                "Leaf drop"
            ],
            "severity_mild": "Few leaves show dark streaks but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and leaf drop.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Guava Rust": {
            "plant": "Guava",
            "disease": "Rust",
            "causes": [
                "Fungal infection (Puccinia psidii)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Rusty, orange spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show rusty spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Guava Wilt": {
            "plant": "Guava",
            "disease": "Wilt",
            "causes": [
                "Fungal infection (Fusarium oxysporum)",
                "High temperatures",
                "Poor drainage"
            ],
            "symptoms": [
                "Wilting of leaves",
                "Yellowing of leaf edges",
                "Stunted growth"
            ],
            "severity_mild": "Few leaves show wilting but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and stunting.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected plants",
                "Improve soil drainage",
                "Avoid overwatering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 14,
            "prevention": [
                "Plant resistant varieties",
                "Ensure proper soil drainage",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Mango Anthracnose": {
            "plant": "Mango",
            "disease": "Anthracnose",
            "causes": [
                "Fungal infection (Colletotrichum gloeosporioides)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark, sunken lesions on fruit",
                "Leaf spots",
                "Wilting of leaves"
            ],
            "severity_mild": "Few fruits show dark lesions but the plant remains healthy.",
            "severity_moderate": "Multiple fruits affected with significant lesions and leaf spots.",
            "severity_severe": "Widespread fruit damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected fruits",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 14,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Mango Powdery Mildew": {
            "plant": "Mango",
            "disease": "Powdery Mildew",
            "causes": [
                "Fungal infection",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "White, powdery spots on leaves",
                "Leaf curling and distortion",
                "Yellowing of leaves"
            ],
            "severity_mild": "Few leaves show white spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant curling and yellowing.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Sulfur fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Papaya Ring Spot Virus": {
            "plant": "Papaya",
            "disease": "Ring Spot Virus",
            "causes": [
                "Viral infection",
                "Insect vectors (aphids)",
                "Contaminated tools or seeds"
            ],
            "symptoms": [
                "Yellow rings on leaves",
                "Stunted growth",
                "Reduced fruit yield"
            ],
            "severity_mild": "Few leaves show yellow rings but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant stunting and reduced yield.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected plants",
                "Control insect vectors",
                "Sanitize tools and equipment"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Use virus-resistant varieties",
                "Control insect populations",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Peach Leaf Spot": {
            "plant": "Peach",
            "disease": "Leaf Spot",
            "causes": [
                "Fungal infection (Cladosporium carpophilum)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Brown or black spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Pomegranate Bacterial Blight": {
            "plant": "Pomegranate",
            "disease": "Bacterial Blight",
            "causes": [
                "Bacterial infection (Xanthomonas axonopodis)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Water-soaked spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show water-soaked spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply bactericide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper bactericide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Pomegranate Leaf Spot": {
            "plant": "Pomegranate",
            "disease": "Leaf Spot",
            "causes": [
                "Fungal infection (Alternaria spp.)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark, circular spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show dark spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 12,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Rice Bacterial Leaf Blight": {
            "plant": "Rice",
            "disease": "Bacterial Leaf Blight",
            "causes": [
                "Bacterial infection (Xanthomonas oryzae pv. oryzae)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Water-soaked lesions on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show water-soaked lesions but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply bactericide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper bactericide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Rice Brown Spot": {
            "plant": "Rice",
            "disease": "Brown Spot",
            "causes": [
                "Fungal infection (Bipolaris oryzae)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Brown, oval spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show brown spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 12,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Sugarcane Red Rot": {
            "plant": "Sugarcane",
            "disease": "Red Rot",
            "causes": [
                "Fungal infection (Colletotrichum falcatum)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Red or brown lesions on stalks",
                "Wilting of leaves",
                "Reduced sugar content"
            ],
            "severity_mild": "Few stalks show red lesions but the plant remains healthy.",
            "severity_moderate": "Multiple stalks affected with significant wilting and reduced sugar content.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected plants",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 14,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Wheat Leaf Rust": {
            "plant": "Wheat",
            "disease": "Leaf Rust",
            "causes": [
                "Fungal infection (Puccinia triticina)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Orange or brown pustules on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show pustules but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Wheat Stem Rust": {
            "plant": "Wheat",
            "disease": "Stem Rust",
            "causes": [
                "Fungal infection (Puccinia graminis)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark brown or black pustules on stems",
                "Yellowing of leaves",
                "Wilting of leaves"
            ],
            "severity_mild": "Few stems show pustules but the plant remains healthy.",
            "severity_moderate": "Multiple stems affected with significant yellowing and wilting.",
            "severity_severe": "Widespread stem damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected plants",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 12,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Tea Blister Blight": {
            "plant": "Tea",
            "disease": "Blister Blight",
            "causes": [
                "Fungal infection (Exobasidium vexans)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Blister-like lesions on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show blisters but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Cabbage Looper": {
            "plant": "Cabbage",
            "disease": "Looper",
            "causes": [
                "Insect infestation (Trichoplusia ni)",
                "High temperatures",
                "Poor air circulation"
            ],
            "symptoms": [
                "Chewed holes in leaves",
                "Caterpillar-like larvae on leaves",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show chewed holes but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant chewing and wilting.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infested leaves",
                "Apply insecticide",
                "Increase air circulation",
                "Regularly inspect plants for early signs of infestation"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Regularly inspect plants for early signs of infestation",
                "Maintain proper humidity levels",
                "Avoid overcrowding plants"
            ]
        },
        "Apple Powdery Mildew": {
            "plant": "Apple",
            "disease": "Powdery Mildew",
            "causes": [
                "Fungal infection",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "White, powdery spots on leaves",
                "Leaf curling and distortion",
                "Yellowing of leaves"
            ],
            "severity_mild": "Few leaves show white spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant curling and yellowing.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Sulfur fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Cashew Anthracnose": {
            "plant": "Cashew",
            "disease": "Anthracnose",
            "causes": [
                "Fungal infection (Colletotrichum gloeosporioides)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark, sunken lesions on leaves and fruit",
                "Leaf drop",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show dark lesions but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant lesions and leaf drop.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves and fruit",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 14,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Cashew Gumosis": {
            "plant": "Cashew",
            "disease": "Gummosis",
            "causes": [
                "Fungal infection (Phytophthora spp.)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Gum oozing from bark",
                "Wilting of leaves",
                "Stunted growth"
            ],
            "severity_mild": "Few branches show gum oozing but the plant remains healthy.",
            "severity_moderate": "Multiple branches affected with significant wilting and stunting.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected branches",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Cashew Healthy": {
            "plant": "Cashew",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Cashew Leaf Miner": {
            "plant": "Cashew",
            "disease": "Leaf Miner",
            "causes": [
                "Insect infestation (Leucoptera coffeella)",
                "High temperatures",
                "Poor air circulation"
            ],
            "symptoms": [
                "Tunnels in leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show tunnels but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infested leaves",
                "Apply insecticide",
                "Increase air circulation",
                "Regularly inspect plants for early signs of infestation"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Regularly inspect plants for early signs of infestation",
                "Maintain proper humidity levels",
                "Avoid overcrowding plants"
            ]
        },
        "Cashew Red Rust": {
            "plant": "Cashew",
            "disease": "Red Rust",
            "causes": [
                "Fungal infection (Puccinia psidii)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Rusty, orange spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show rusty spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Cassava Bacterial Blight": {
            "plant": "Cassava",
            "disease": "Bacterial Blight",
            "causes": [
                "Bacterial infection (Xanthomonas axonopodis)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Water-soaked spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show water-soaked spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply bactericide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper bactericide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Cassava Brown Spot": {
            "plant": "Cassava",
            "disease": "Brown Spot",
            "causes": [
                "Fungal infection (Alternaria spp.)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark, circular spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show dark spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 12,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Cassava Green Mite": {
            "plant": "Cassava",
            "disease": "Green Mite",
            "causes": [
                "Insect infestation (Mononychellus tanajoa)",
                "High temperatures",
                "Low humidity"
            ],
            "symptoms": [
                "Yellowing and stippling of leaves",
                "Webbing on leaves",
                "Leaf drop"
            ],
            "severity_mild": "Few leaves show yellowing but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant stippling and webbing.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infested leaves",
                "Apply miticide",
                "Increase humidity around plants"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Regularly inspect plants for early signs of infestation",
                "Maintain proper humidity levels",
                "Avoid overcrowding plants"
            ]
        },
        "Cassava Healthy": {
            "plant": "Cassava",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Cassava Mosaic": {
            "plant": "Cassava",
            "disease": "Mosaic",
            "causes": [
                "Viral infection (Cassava mosaic virus)",
                "Insect vectors (whiteflies)",
                "Contaminated tools or seeds"
            ],
            "symptoms": [
                "Mosaic patterns on leaves",
                "Stunted growth",
                "Reduced yield"
            ],
            "severity_mild": "Few leaves show mosaic patterns but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant stunting and reduced yield.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected plants",
                "Control insect vectors",
                "Sanitize tools and equipment"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Use virus-resistant varieties",
                "Control insect populations",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Cercospora Leaf Spot": {
            "plant": "Cassava",
            "disease": "Cercospora Leaf Spot",
            "causes": [
                "Fungal infection (Cercospora spp.)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark, circular spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show dark spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 12,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Citrus Rust Mite": {
            "plant": "Citrus",
            "disease": "Rust Mite",
            "causes": [
                "Insect infestation (Phyllocoptruta oleivora)",
                "High temperatures",
                "Low humidity"
            ],
            "symptoms": [
                "Bronzing of leaves",
                "Stippling on fruit",
                "Reduced fruit quality"
            ],
            "severity_mild": "Few leaves show bronzing but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant stippling and reduced fruit quality.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infested leaves",
                "Apply miticide",
                "Increase humidity around plants"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Regularly inspect plants for early signs of infestation",
                "Maintain proper humidity levels",
                "Avoid overcrowding plants"
            ]
        },
        "Coffee Rust": {
            "plant": "Coffee",
            "disease": "Rust",
            "causes": [
                "Fungal infection (Hemileia vastatrix)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Yellow-orange spots on leaves",
                "Leaf drop",
                "Reduced yield"
            ],
            "severity_mild": "Few leaves show yellow-orange spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant leaf drop and reduced yield.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 14,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Garlic Healthy": {
            "plant": "Garlic",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Ginger Healthy": {
            "plant": "Ginger",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Grape Leaf Rust Mite": {
            "plant": "Grape",
            "disease": "Leaf Rust Mite",
            "causes": [
                "Insect infestation (Calepitrimerus vitis)",
                "High temperatures",
                "Low humidity"
            ],
            "symptoms": [
                "Bronzing of leaves",
                "Stippling on fruit",
                "Reduced fruit quality"
            ],
            "severity_mild": "Few leaves show bronzing but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant stippling and reduced fruit quality.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infested leaves",
                "Apply miticide",
                "Increase humidity around plants"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Regularly inspect plants for early signs of infestation",
                "Maintain proper humidity levels",
                "Avoid overcrowding plants"
            ]
        },
        "Lemon Canker": {
            "plant": "Lemon",
            "disease": "Canker",
            "causes": [
                "Bacterial infection (Xanthomonas citri)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Water-soaked lesions on leaves and fruit",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show water-soaked lesions but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves and fruit",
                "Apply bactericide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper bactericide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Maize Fall Armyworm": {
            "plant": "Maize",
            "disease": "Fall Armyworm",
            "causes": [
                "Insect infestation (Spodoptera frugiperda)",
                "High temperatures",
                "Poor air circulation"
            ],
            "symptoms": [
                "Chewed holes in leaves",
                "Caterpillar-like larvae on leaves",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show chewed holes but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant chewing and wilting.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infested leaves",
                "Apply insecticide",
                "Increase air circulation",
                "Regularly inspect plants for early signs of infestation"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Regularly inspect plants for early signs of infestation",
                "Maintain proper humidity levels",
                "Avoid overcrowding plants"
            ]
        },
        "Maize Grasshopper": {
            "plant": "Maize",
            "disease": "Grasshopper",
            "causes": [
                "Insect infestation (Zonocerus variegatus)",
                "High temperatures",
                "Poor air circulation"
            ],
            "symptoms": [
                "Chewed holes in leaves",
                "Wilting of leaves",
                "Reduced yield"
            ],
            "severity_mild": "Few leaves show chewed holes but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant wilting and reduced yield.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infested leaves",
                "Apply insecticide",
                "Increase air circulation",
                "Regularly inspect plants for early signs of infestation"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Regularly inspect plants for early signs of infestation",
                "Maintain proper humidity levels",
                "Avoid overcrowding plants"
            ]
        },
        "Maize Healthy": {
            "plant": "Maize",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Maize Leaf Beetle": {
            "plant": "Maize",
            "disease": "Leaf Beetle",
            "causes": [
                "Insect infestation (Diabrotica spp.)",
                "High temperatures",
                "Poor air circulation"
            ],
            "symptoms": [
                "Chewed holes in leaves",
                "Wilting of leaves",
                "Reduced yield"
            ],
            "severity_mild": "Few leaves show chewed holes but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant wilting and reduced yield.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infested leaves",
                "Apply insecticide",
                "Increase air circulation",
                "Regularly inspect plants for early signs of infestation"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Regularly inspect plants for early signs of infestation",
                "Maintain proper humidity levels",
                "Avoid overcrowding plants"
            ]
        },
        "Maize Leaf Blight": {
            "plant": "Maize",
            "disease": "Leaf Blight",
            "causes": [
                "Fungal infection (Exserohilum turcicum)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Water-soaked lesions on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show water-soaked lesions but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Maize Leaf Spot": {
            "plant": "Maize",
            "disease": "Leaf Spot",
            "causes": [
                "Fungal infection (Cercospora zeae-maydis)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark, circular spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show dark spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 12,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Maize Streak Virus": {
            "plant": "Maize",
            "disease": "Streak Virus",
            "causes": [
                "Viral infection (Maize streak virus)",
                "Insect vectors (leafhoppers)",
                "Contaminated tools or seeds"
            ],
            "symptoms": [
                "Streaks on leaves",
                "Stunted growth",
                "Reduced yield"
            ],
            "severity_mild": "Few leaves show streaks but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant stunting and reduced yield.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected plants",
                "Control insect vectors",
                "Sanitize tools and equipment"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Use virus-resistant varieties",
                "Control insect populations",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Myrtle Rust": {
            "plant": "Myrtle",
            "disease": "Rust",
            "causes": [
                "Fungal infection (Puccinia psidii)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Rusty, orange spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show rusty spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Nitrogen Deficiency": {
            "plant": "General",
            "disease": "Nitrogen Deficiency",
            "causes": [
                "Insufficient nitrogen in soil",
                "Poor soil fertility",
                "Lack of organic matter"
            ],
            "symptoms": [
                "Yellowing of older leaves",
                "Stunted growth",
                "Reduced yield"
            ],
            "severity_mild": "Few leaves show yellowing but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant stunting and reduced yield.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Apply nitrogen-rich fertilizer",
                "Incorporate organic matter into soil",
                "Regularly test soil for nutrient levels"
            ],
            "medicines": [
                "Urea fertilizer",
                "Compost"
            ],
            "progression_days": 14,
            "prevention": [
                "Regularly test soil for nutrient levels",
                "Maintain proper soil fertility",
                "Incorporate organic matter into soil"
            ]
        },
        "Onion Healthy": {
            "plant": "Onion",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Peach Scab": {
            "plant": "Peach",
            "disease": "Scab",
            "causes": [
                "Fungal infection (Cladosporium carpophilum)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark, sunken lesions on fruit",
                "Leaf drop",
                "Wilting of leaves"
            ],
            "severity_mild": "Few fruits show dark lesions but the plant remains healthy.",
            "severity_moderate": "Multiple fruits affected with significant leaf drop and wilting.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected fruit",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 14,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Pear Scab": {
            "plant": "Pear",
            "disease": "Scab",
            "causes": [
                "Fungal infection (Venturia pirina)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark, sunken lesions on leaves and fruit",
                "Leaf drop",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves and fruits show dark lesions but the plant remains healthy.",
            "severity_moderate": "Multiple leaves and fruits affected with significant leaf drop and wilting.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves and fruit",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 14,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Potassium Deficiency": {
            "plant": "General",
            "disease": "Potassium Deficiency",
            "causes": [
                "Insufficient potassium in soil",
                "Poor soil fertility",
                "Lack of organic matter"
            ],
            "symptoms": [
                "Yellowing of leaf edges",
                "Wilting of leaves",
                "Reduced yield"
            ],
            "severity_mild": "Few leaves show yellowing but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant wilting and reduced yield.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Apply potassium-rich fertilizer",
                "Incorporate organic matter into soil",
                "Regularly test soil for nutrient levels"
            ],
            "medicines": [
                "Potassium sulfate fertilizer",
                "Compost"
            ],
            "progression_days": 14,
            "prevention": [
                "Regularly test soil for nutrient levels",
                "Maintain proper soil fertility",
                "Incorporate organic matter into soil"
            ]
        },
        "Potato Hollow Heart": {
            "plant": "Potato",
            "disease": "Hollow Heart",
            "causes": [
                "Rapid growth due to excess nitrogen",
                "Inconsistent watering",
                "Poor soil aeration"
            ],
            "symptoms": [
                "Hollow cavities in tubers",
                "Reduced yield",
                "Poor quality of tubers"
            ],
            "severity_mild": "Few tubers show small cavities but the plant remains healthy.",
            "severity_moderate": "Multiple tubers affected with significant cavities and reduced yield.",
            "severity_severe": "Widespread damage leading to poor quality of tubers if untreated.",
            "treatment": [
                "Adjust nitrogen levels in soil",
                "Ensure consistent watering",
                "Improve soil aeration"
            ],
            "medicines": [],
            "progression_days": 14,
            "prevention": [
                "Regularly test soil for nutrient levels",
                "Maintain proper soil fertility",
                "Avoid excessive nitrogen fertilization"
            ]
        },
        "Rice Leaf Smut": {
            "plant": "Rice",
            "disease": "Leaf Smut",
            "causes": [
                "Fungal infection (Entyloma oryzae)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "White, powdery spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show white spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Sogatella Rice": {
            "plant": "Rice",
            "disease": "Sogatella",
            "causes": [
                "Insect infestation (Sogatella furcifera)",
                "High temperatures",
                "Poor air circulation"
            ],
            "symptoms": [
                "Yellowing and wilting of leaves",
                "Stippling on leaves",
                "Reduced yield"
            ],
            "severity_mild": "Few leaves show yellowing but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant stippling and reduced yield.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infested leaves",
                "Apply insecticide",
                "Increase air circulation",
                "Regularly inspect plants for early signs of infestation"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Regularly inspect plants for early signs of infestation",
                "Maintain proper humidity levels",
                "Avoid overcrowding plants"
            ]
        },
        "Tea Algal Leaf": {
            "plant": "Tea",
            "disease": "Algal Leaf",
            "causes": [
                "Algal infection (Cephaleuros virescens)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Green, velvety patches on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show green patches but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Tea Anthracnose": {
            "plant": "Tea",
            "disease": "Anthracnose",
            "causes": [
                "Fungal infection (Colletotrichum camelliae)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Dark, sunken lesions on leaves",
                "Leaf drop",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show dark lesions but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant leaf drop and wilting.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 14,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Tea Bird Eye Spot": {
            "plant": "Tea",
            "disease": "Bird Eye Spot",
            "causes": [
                "Fungal infection (Cercospora theae)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Small, circular spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show small spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 12,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Tea Brown Blight": {
            "plant": "Tea",
            "disease": "Brown Blight",
            "causes": [
                "Fungal infection (Pestalotiopsis spp.)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Brown, sunken lesions on leaves",
                "Leaf drop",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show brown lesions but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant leaf drop and wilting.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Maintain proper humidity levels"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 14,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Tea Healthy": {
            "plant": "Tea",
            "disease": "Healthy",
            "causes": [],
            "symptoms": [],
            "severity_mild": "",
            "severity_moderate": "",
            "severity_severe": "",
            "treatment": [],
            "medicines": [],
            "progression_days": 0,
            "prevention": []
        },
        "Tea Red Leaf Spot": {
            "plant": "Tea",
            "disease": "Red Leaf Spot",
            "causes": [
                "Fungal infection (Diplocarpon earliana)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Red, circular spots on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show red spots but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 12,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Tomato Canker": {
            "plant": "Tomato",
            "disease": "Canker",
            "causes": [
                "Bacterial infection (Clavibacter michiganensis)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Water-soaked lesions on stems and leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few stems show water-soaked lesions but the plant remains healthy.",
            "severity_moderate": "Multiple stems affected with significant yellowing and wilting.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected stems and leaves",
                "Apply bactericide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper bactericide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Tomato Leaf Blight": {
            "plant": "Tomato",
            "disease": "Leaf Blight",
            "causes": [
                "Fungal infection (Alternaria solani)",
                "High humidity",
                "Poor air circulation"
            ],
            "symptoms": [
                "Water-soaked lesions on leaves",
                "Yellowing of leaf edges",
                "Wilting of leaves"
            ],
            "severity_mild": "Few leaves show water-soaked lesions but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and wilting.",
            "severity_severe": "Widespread leaf damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected leaves",
                "Apply fungicide",
                "Improve air circulation",
                "Avoid overhead watering"
            ],
            "medicines": [
                "Copper fungicide",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Plant resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Tomato Leaf Curl": {
            "plant": "Tomato",
            "disease": "Leaf Curl",
            "causes": [
                "Viral infection (Tomato yellow leaf curl virus)",
                "Insect vectors (whiteflies)",
                "Contaminated tools or seeds"
            ],
            "symptoms": [
                "Curling of leaves",
                "Stunted growth",
                "Reduced yield"
            ],
            "severity_mild": "Few leaves show curling but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant stunting and reduced yield.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected plants",
                "Control insect vectors",
                "Sanitize tools and equipment"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Use virus-resistant varieties",
                "Control insect populations",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Tomato Two-Spotted Spider Mite": {
            "plant": "Tomato",
            "disease": "Two-Spotted Spider Mite",
            "causes": [
                "Insect infestation (Tetranychus urticae)",
                "High temperatures",
                "Low humidity"
            ],
            "symptoms": [
                "Fine webbing on leaves",
                "Yellowing and stippling of leaves",
                "Reduced yield"
            ],
            "severity_mild": "Few leaves show fine webbing but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and stippling.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infested leaves",
                "Apply miticide",
                "Increase humidity around plants"
            ],
            "medicines": [
                "Insecticidal soap",
                "Neem oil"
            ],
            "progression_days": 7,
            "prevention": [
                "Regularly inspect plants for early signs of infestation",
                "Maintain proper humidity levels",
                "Avoid overcrowding plants"
            ]
        },
        "Tomato Verticulium Wilt": {
            "plant": "Tomato",
            "disease": "Verticillium Wilt",
            "causes": [
                "Fungal infection (Verticillium dahliae)",
                "Poor soil drainage",
                "Contaminated tools or seeds"
            ],
            "symptoms": [
                "Wilting of leaves",
                "Yellowing of leaf edges",
                "Stunted growth"
            ],
            "severity_mild": "Few leaves show wilting but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and stunting.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Remove and destroy infected plants",
                "Improve soil drainage",
                "Sanitize tools and equipment"
            ],
            "medicines": [
                "Fungicidal drench",
                "Neem oil"
            ],
            "progression_days": 10,
            "prevention": [
                "Use resistant varieties",
                "Rotate crops annually",
                "Regularly inspect plants for early signs of disease"
            ]
        },
        "Waterlogging in Plant": {
            "plant": "General",
            "disease": "Waterlogging",
            "causes": [
                "Excessive rainfall",
                "Poor soil drainage",
                "Overwatering"
            ],
            "symptoms": [
                "Wilting of leaves",
                "Yellowing of leaf edges",
                "Root rot"
            ],
            "severity_mild": "Few leaves show wilting but the plant remains healthy.",
            "severity_moderate": "Multiple leaves affected with significant yellowing and root issues.",
            "severity_severe": "Widespread damage leading to potential death of the plant if untreated.",
            "treatment": [
                "Improve soil drainage",
                "Reduce watering frequency",
                "Remove affected plants"
            ],
            "medicines": [],
            "progression_days": 14,
            "prevention": [
                "Ensure proper soil drainage",
                "Avoid overwatering",
                "Regularly inspect plants for early signs of waterlogging"
            ]
        }
    }

            let allDiseases = [];
            let filteredDiseases = [];

            function init() {
                allDiseases = Object.keys(diseaseData).map(key => ({
                    key: key,
                    ...diseaseData[key]
                }));
                filteredDiseases = [...allDiseases];
                
                renderGallery();
                
                document.getElementById('loading').style.display = 'none';
            }

            document.getElementById('searchInput').addEventListener('input', (e) => {
                applySearch(e.target.value);
            });

            function applySearch(searchTerm) {
                const term = searchTerm.toLowerCase().trim();
                
                if (!term) {
                    filteredDiseases = [...allDiseases];
                    renderGallery();
                    return;
                }
                
                filteredDiseases = allDiseases.filter(d => {
                    return d.plant?.toLowerCase().includes(term) ||
                        d.disease?.toLowerCase().includes(term) ||
                        d.key.toLowerCase().includes(term);
                });
                
                renderGallery();
            }

            async function renderGallery() {
                const grid = document.getElementById('galleryGrid');
                const noResults = document.getElementById('noResults');
                
                if (filteredDiseases.length === 0) {
                    grid.innerHTML = '';
                    noResults.classList.add('show');
                    return;
                }
                
                noResults.classList.remove('show');
                
                const cardPromises = filteredDiseases.map(async (disease) => {
                    const isHealthy = disease.disease === 'Healthy';
                    const imagePath = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='300'%3E%3Crect fill='%23f0fdf4' width='800' height='300'/%3E%3Ctext fill='%23047857' font-family='Arial' font-size='20' x='50%25' y='50%25' text-anchor='middle'%3E${disease.plant} - ${disease.disease}%3C/text%3E%3C/svg%3E`;
                    
                    return `
                        <div class="disease-card" onclick='openModal(${JSON.stringify(disease.key)})'>
                            <div class="card-image-container">
                                <img src="${imagePath}" alt="${disease.key}" class="card-image">
                                <div class="img-gradient"></div>
                                <div class="card-badge ${isHealthy ? 'healthy' : ''}">${isHealthy ? '✓ Healthy' : '⚠ Disease'}</div>
                            </div>
                            <div class="card-content">
                                <h3 class="card-title">${disease.disease}</h3>
                                <p class="card-plant">
                                    <svg class="icon-sm" style="color: #047857;" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                        <path d="M2 22s9-2 15-8 5-12 5-12-9 2-15 8-5 12-5 12z"/>
                                    </svg>
                                    ${disease.plant}
                                </p>
                                ${!isHealthy ? `
                                    <div class="card-info">
                                        ${disease.progression_days > 0 ? `
                                            <span class="info-badge">
                                                <svg class="icon-sm" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                                    <circle cx="12" cy="12" r="10"/>
                                                    <polyline points="12 6 12 12 16 14"/>
                                                </svg>
                                                ${disease.progression_days} days
                                            </span>
                                        ` : ''}
                                        ${disease.symptoms?.length > 0 ? `
                                            <span class="info-badge">
                                                <svg class="icon-sm" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                                                </svg>
                                                ${disease.symptoms.length} symptoms
                                            </span>
                                        ` : ''}
                                    </div>
                                    ${disease.symptoms?.length > 0 ? `
                                        <p class="card-symptoms">${disease.symptoms.join(', ')}</p>
                                    ` : ''}
                                ` : '<p class="card-symptoms">Plant is healthy with no disease detected.</p>'}
                                <div class="card-footer">
                                    <button class="view-details-btn">View Details →</button>
                                </div>
                            </div>
                        </div>
                    `;
                });

                const cards = await Promise.all(cardPromises);
                grid.innerHTML = cards.join('');
            }

            async function openModal(diseaseKey) {
                const disease = diseaseData[diseaseKey];
                const modal = document.getElementById('diseaseModal');
                const modalBody = document.getElementById('modalBody');
                const isHealthy = disease.disease === 'Healthy';
                const imagePath = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='300'%3E%3Crect fill='%23f0fdf4' width='800' height='300'/%3E%3Ctext fill='%23047857' font-family='Arial' font-size='20' x='50%25' y='50%25' text-anchor='middle'%3E${disease.plant} - ${disease.disease}%3C/text%3E%3C/svg%3E`;

                modalBody.innerHTML = `
                    <img src="${imagePath}" alt="${diseaseKey}" class="modal-image">
                    <h2 class="modal-title">${disease.plant} - ${disease.disease}</h2>
                    
                    ${!isHealthy ? `
                        ${disease.causes?.length > 0 ? `
                            <div class="modal-section">
                                <h3 class="section-title">
                                    <svg class="icon-sm" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                        <circle cx="12" cy="12" r="10"/>
                                        <line x1="12" y1="16" x2="12" y2="12"/>
                                        <line x1="12" y1="8" x2="12.01" y2="8"/>
                                    </svg>
                                    Causes
                                </h3>
                                <ul class="section-list">
                                    ${disease.causes.map(c => `<li>${c}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                        
                        ${disease.symptoms?.length > 0 ? `
                            <div class="modal-section">
                                <h3 class="section-title">
                                    <svg class="icon-sm" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                                    </svg>
                                    Symptoms
                                </h3>
                                <ul class="section-list">
                                    ${disease.symptoms.map(s => `<li>${s}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                        
                        ${disease.severity_mild || disease.severity_moderate || disease.severity_severe ? `
                            <div class="modal-section">
                                <h3 class="section-title">
                                    <svg class="icon-sm" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                                    </svg>
                                    Severity Levels
                                </h3>
                                ${disease.severity_mild ? `<p class="section-content"><strong>Mild:</strong> ${disease.severity_mild}</p>` : ''}
                                ${disease.severity_moderate ? `<p class="section-content" style="margin-top: 0.5rem;"><strong>Moderate:</strong> ${disease.severity_moderate}</p>` : ''}
                                ${disease.severity_severe ? `<p class="section-content" style="margin-top: 0.5rem;"><strong>Severe:</strong> ${disease.severity_severe}</p>` : ''}
                            </div>
                        ` : ''}
                        
                        ${disease.treatment?.length > 0 ? `
                            <div class="modal-section">
                                <h3 class="section-title">
                                    <svg class="icon-sm" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                                        <polyline points="22 4 12 14.01 9 11.01"/>
                                    </svg>
                                    Treatment
                                </h3>
                                <ul class="section-list">
                                    ${disease.treatment.map(t => `<li>${t}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                        
                        ${disease.medicines?.length > 0 ? `
                            <div class="modal-section">
                                <h3 class="section-title">
                                    <svg class="icon-sm" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                                        <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                                    </svg>
                                    Recommended Medicines
                                </h3>
                                <ul class="section-list">
                                    ${disease.medicines.map(m => `<li>${m}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                        
                        ${disease.prevention?.length > 0 ? `
                            <div class="modal-section">
                                <h3 class="section-title">
                                    <svg class="icon-sm" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                                    </svg>
                                    Prevention
                                </h3>
                                <ul class="section-list">
                                    ${disease.prevention.map(p => `<li>${p}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                    ` : `
                        <div class="modal-section">
                            <p class="section-content">This plant is healthy with no disease detected. Continue regular care and monitoring to maintain plant health.</p>
                        </div>
                    `}
                `;
                
                modal.classList.add('active');
                document.body.style.overflow = 'hidden';
            }

            function closeModal() {
                const modal = document.getElementById('diseaseModal');
                modal.classList.remove('active');
                document.body.style.overflow = 'auto';
            }

            document.getElementById('diseaseModal').addEventListener('click', (e) => {
                if (e.target.id === 'diseaseModal') {
                    closeModal();
                }
            });

            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    closeModal();
                }
            });

            init();
        </script>
    </body>
    </html>
    """

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image file selected'})
        
        # Read and process image
        image = Image.open(file.stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Make prediction (same as Streamlit version)
        prediction_idx, prediction, confidence, is_demo = predict_image_flask(image)
        
        if prediction is None:
            return jsonify({'success': False, 'error': 'Failed to make prediction'})
        
        # Generate prediction ID
        prediction_id = f"pred_{int(time.time() * 1000)}"
        
        # Get disease information
        disease_info = get_disease_info(prediction)
        
        # Format disease info string for frontend display
        disease_info_str = format_disease_info_string(disease_info)
        
        # Cache the prediction data for later use
        predictions_cache[prediction_id] = {
            'prediction_idx': prediction_idx,
            'prediction': prediction,
            'confidence': confidence,
            'disease_info': disease_info,
            'is_demo': is_demo,
            'image': image.copy(),
            'timestamp': datetime.now()
        }
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'confidence_percent': round(confidence * 100, 1),
            'disease_info': disease_info_str,
            'prediction_id': prediction_id,
            'is_demo': is_demo,
            'ml_available': ML_AVAILABLE
        })
        
    except Exception as e:
        print(f"❌ Error in prediction endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/gradcam/<prediction_id>', methods=['GET'])
def gradcam_by_id(prediction_id):
    """Generate GradCAM heatmap by prediction ID"""
    try:
        if prediction_id not in predictions_cache:
            return jsonify({'success': False, 'error': 'Prediction not found'})
        
        cached_data = predictions_cache[prediction_id]
        image = cached_data['image']
        target_class_idx = cached_data.get('prediction_idx')
        
        # Generate GradCAM
        gradcam_result, is_demo = generate_gradcam(image, target_class_idx)
        
        if gradcam_result is None:
            return jsonify({'success': False, 'error': 'Failed to generate GradCAM'})
        
        # Convert to base64 for response
        gradcam_uint8 = (gradcam_result * 255).astype(np.uint8)
        gradcam_img = Image.fromarray(gradcam_uint8)
        
        # Save to bytes
        img_buffer = io.BytesIO()
        gradcam_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Cache the gradcam for potential report generation
        gradcam_cache[prediction_id] = img_base64
        
        return jsonify({
            'success': True,
            'gradcam_image': img_base64,
            'is_demo': is_demo,
            'ml_available': ML_AVAILABLE
        })
        
    except Exception as e:
        print(f"❌ Error in GradCAM endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/generate_report', methods=['POST'])
def generate_report_api():
    """Generate PDF report for the prediction"""
    try:
        data = request.get_json()
        prediction_id = data.get('prediction_id')
        
        if not prediction_id:
            return jsonify({'success': False, 'error': 'No prediction ID provided'})
        
        if prediction_id not in predictions_cache:
            return jsonify({'success': False, 'error': 'Prediction not found'})
        
        cached_data = predictions_cache[prediction_id]
        
        print(f"🔄 Generating PDF report for prediction: {cached_data['prediction']}")
        
        # Generate PDF
        pdf_bytes = generate_pdf_report(
            cached_data['image'],
            cached_data['prediction'],
            cached_data['confidence'],
            cached_data['disease_info'],
            cached_data['is_demo']
        )
        
        if pdf_bytes is None:
            return jsonify({'success': False, 'error': 'Failed to generate PDF'})
        
        print(f"✅ PDF generated successfully, size: {len(pdf_bytes)} bytes")
        
        # Generate report ID and cache the PDF
        report_id = f"report_{int(time.time() * 1000)}"
        
        # Store PDF temporarily in memory
        if not hasattr(app, 'reports_cache'):
            app.reports_cache = {}
        app.reports_cache[report_id] = pdf_bytes
        
        return jsonify({
            'success': True,
            'report_id': report_id
        })
        
    except Exception as e:
        print(f"❌ Error in report generation endpoint: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/download_report/<report_id>')
def download_report(report_id):
    """Download generated report"""
    try:
        if not hasattr(app, 'reports_cache') or report_id not in app.reports_cache:
            return "Report not found", 404
        
        pdf_bytes = app.reports_cache[report_id]
        print(f"📥 Serving PDF report, size: {len(pdf_bytes)} bytes")
        
        # Validate PDF format
        if not pdf_bytes or len(pdf_bytes) < 100:
            print("❌ PDF bytes are too small or empty")
            return "Invalid PDF data", 500
            
        # Check PDF header
        if not pdf_bytes.startswith(b'%PDF'):
            print("❌ PDF does not have valid header")
            return "Invalid PDF format", 500
        
        # Create Flask response with proper headers
        from flask import Response
        
        response = Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="plant_disease_report_{report_id}.pdf"',
                'Content-Length': str(len(pdf_bytes)),
                'Content-Type': 'application/pdf'
            }
        )
        
        # Clean up after response is created
        try:
            del app.reports_cache[report_id]
        except:
            pass
            
        return response
        
    except Exception as e:
        print(f"❌ Error in download report endpoint: {e}")
        import traceback
        print(traceback.format_exc())
        return f"Error downloading report: {str(e)}", 500

@app.route('/disease_info/<disease_name>')
def get_disease_info_endpoint(disease_name):
    """Get disease information by name"""
    try:
        info = get_disease_info(disease_name)
        if info:
            return jsonify({'success': True, 'disease_info': info})
        else:
            return jsonify({'success': False, 'error': 'Disease information not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'disease_info_loaded': disease_info_data is not None,
        'ml_available': ML_AVAILABLE,
        'mode': 'production' if ML_AVAILABLE and model is not None else 'demo',
        'timestamp': datetime.now().isoformat(),
        'disease_count': len(disease_info_data) if disease_info_data else 0,
        'device': str(device) if device else 'N/A',
        'class_count': len(class_names)
    })

# Initialize the application
def initialize_app():
    """Initialize model and data"""
    mode = "Production" if ML_AVAILABLE else "Demo"
    print(f"🌱 Initializing Plant Doctor AI ({mode} Mode)...")
    load_disease_info()
    load_model()
    print(f"✅ Plant Doctor AI initialized successfully in {mode} mode!")
    if disease_info_data:
        print(f"📚 Loaded information for {len(disease_info_data)} diseases")
    print(f"🎯 Supporting {len(class_names)} plant disease classes")

@app.route('/signup')
def signup():
    """Serve the signup page"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Up</title>
    <style>
        body {
        font-family: 'Segoe UI', 'Arial', sans-serif;
        min-height: 100vh;
        margin: 0;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        display: flex;
        justify-content: center;
        align-items: center;
        }
        .container {
        background: #fff;
        padding: 48px 36px;
        border-radius: 24px;
        box-shadow: 0 8px 32px -8px #00b89433;
        max-width: 350px;
        width: 100%;
        text-align: center;
        transition: box-shadow 0.3s;
        position: relative;
        overflow: hidden;
        }
        .container::before {
        content: '';
        position: absolute;
        top: -60px; left: -60px;
        width: 120px; height: 120px;
        background: radial-gradient(circle at 30% 30%, #00b89488 0%, transparent 70%);
        z-index: 0;
        }
        .container::after {
        content: '';
        position: absolute;
        bottom: -60px; right: -60px;
        width: 120px; height: 120px;
        background: radial-gradient(circle at 70% 70%, #fed6e388 0%, transparent 70%);
        z-index: 0;
        }
        h2 {
        margin-bottom: 18px;
        color: #00b894;
        font-weight: 700;
        letter-spacing: 1px;
        font-size: 2rem;
        position: relative;
        z-index: 1;
        }
        input {
        display: block;
        width: 90%;
        padding: 12px;
        margin: 14px auto 0 auto;
        border-radius: 8px;
        border: 1px solid #dedede;
        font-size: 1rem;
        outline: none;
        transition: border-color 0.2s;
        background: #f7fafc;
        position: relative;
        z-index: 1;
        }
        input:focus {
        border-color: #00b894;
        background: #eafaf7;
        }
        button {
        padding: 12px 0;
        border: none;
        border-radius: 8px;
        background: linear-gradient(90deg, #00b894, #43e97b);
        color: #fff;
        cursor: pointer;
        font-size: 1.08rem;
        width: 90%;
        margin-top: 24px;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 16px -4px #00b89444;
        transition: background 0.2s, box-shadow 0.2s, transform 0.1s;
        position: relative;
        z-index: 1;
        }
        button:hover, button:focus {
        background: linear-gradient(90deg, #019875, #2fe3b6);
        box-shadow: 0 8px 32px -4px #00b89455;
        transform: translateY(-2px) scale(1.02);
        }
        #signup-error-msg {
        color: #dc3545;
        margin-top: 12px;
        font-size: 1rem;
        min-height: 24px;
        transition: color 0.2s;
        position: relative;
        z-index: 1;
        }
        .container p {
        margin-top: 20px;
        font-size: 0.98rem;
        color: #636e72;
        position: relative;
        z-index: 1;
        }
        .container a {
        color: #00b894;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s;
        }
        .container a:hover {
        color: #019875;
        text-decoration: underline;
        }
        @media (max-width: 480px) {
        .container {
            padding: 32px 12px;
            border-radius: 12px;
            max-width: 98vw;
        }
        h2 {
            font-size: 1.5rem;
        }
        input, button {
            width: 98%;
            font-size: 1rem;
        }
        }
    </style>
    <!-- Firebase SDKs -->
    <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-app-compat.js"></script>
    <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-auth-compat.js"></script>
    </head>
    <body>
    <div class="container">
        <h2>Create Your Account</h2>
        <input type="email" id="signup-email" placeholder="Email">
        <input type="password" id="signup-password" placeholder="Password">
        <input type="password" id="signup-password-confirm" placeholder="Confirm Password">
        <button onclick="signUp()">Sign Up</button>
        <p>Already have an account? <a href="/signin">Sign In</a></p>
        <p id="signup-error-msg"></p>
    </div>

    <script>
        // Firebase configuration
        const firebaseConfig = {
        apiKey: "AIzaSyB-y52ekJkjgXYXhRNvvir0r9gU8CObpkM",
        authDomain: "plant-doctor-63da7.firebaseapp.com",
        projectId: "plant-doctor-63da7",
        storageBucket: "plant-doctor-63da7.firebasestorage.app",
        messagingSenderId: "94281353808",
        appId: "1:94281353808:web:5fa49e3d6e494868be5f55",
        measurementId: "G-E5P4MZ68HW"
        };
        firebase.initializeApp(firebaseConfig);
        const auth = firebase.auth();

        function signUp() {
        const email = document.getElementById("signup-email").value.trim();
        const password = document.getElementById("signup-password").value;
        const passwordConfirm = document.getElementById("signup-password-confirm").value;
        const errorMsg = document.getElementById("signup-error-msg");

        errorMsg.textContent = "";

        if (!email) {
            errorMsg.textContent = "Please enter your email.";
            return;
        }
        if (!password) {
            errorMsg.textContent = "Please enter a password.";
            return;
        }
        if (password.length < 6) {
            errorMsg.textContent = "Password must be at least 6 characters.";
            return;
        }
        if (password !== passwordConfirm) {
            errorMsg.textContent = "Passwords do not match!";
            return;
        }

        auth.createUserWithEmailAndPassword(email, password)
            .then((userCredential) => {
            alert("Account created successfully!");
            window.location.href = "/signin";
            })
            .catch((error) => {
            errorMsg.textContent = error.message;
            });
        }
    </script>
    </body>
    </html>
    """

@app.route('/signin')
def signin():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign In</title>
    <style>
        body {
        font-family: 'Segoe UI', 'Arial', sans-serif;
        min-height: 100vh;
        margin: 0;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        display: flex;
        justify-content: center;
        align-items: center;
        }
        .container {
        background: #fff;
        padding: 48px 36px;
        border-radius: 24px;
        box-shadow: 0 8px 32px -8px #00b89433;
        max-width: 350px;
        width: 100%;
        text-align: center;
        transition: box-shadow 0.3s;
        position: relative;
        overflow: hidden;
        }
        .container::before {
        content: '';
        position: absolute;
        top: -60px; left: -60px;
        width: 120px; height: 120px;
        background: radial-gradient(circle at 30% 30%, #00b89488 0%, transparent 70%);
        z-index: 0;
        }
        .container::after {
        content: '';
        position: absolute;
        bottom: -60px; right: -60px;
        width: 120px; height: 120px;
        background: radial-gradient(circle at 70% 70%, #fed6e388 0%, transparent 70%);
        z-index: 0;
        }
        h2 {
        margin-bottom: 18px;
        color: #00b894;
        font-weight: 700;
        letter-spacing: 1px;
        font-size: 2rem;
        position: relative;
        z-index: 1;
        }
        input {
        display: block;
        width: 90%;
        padding: 12px;
        margin: 14px auto 0 auto;
        border-radius: 8px;
        border: 1px solid #dedede;
        font-size: 1rem;
        outline: none;
        transition: border-color 0.2s;
        background: #f7fafc;
        position: relative;
        z-index: 1;
        }
        input:focus {
        border-color: #00b894;
        background: #eafaf7;
        }
        button {
        padding: 12px 0;
        border: none;
        border-radius: 8px;
        background: linear-gradient(90deg, #00b894, #43e97b);
        color: #fff;
        cursor: pointer;
        font-size: 1.08rem;
        width: 90%;
        margin-top: 24px;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 16px -4px #00b89444;
        transition: background 0.2s, box-shadow 0.2s, transform 0.1s;
        position: relative;
        z-index: 1;
        }
        button:hover, button:focus {
        background: linear-gradient(90deg, #019875, #2fe3b6);
        box-shadow: 0 8px 32px -4px #00b89455;
        transform: translateY(-2px) scale(1.02);
        }
        #error-msg {
        color: #dc3545;
        margin-top: 12px;
        font-size: 1rem;
        min-height: 24px;
        transition: color 0.2s;
        position: relative;
        z-index: 1;
        }
        .container p {
        margin-top: 20px;
        font-size: 0.98rem;
        color: #636e72;
        position: relative;
        z-index: 1;
        }
        .container a {
        color: #00b894;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s;
        }
        .container a:hover {
        color: #019875;
        text-decoration: underline;
        }
        @media (max-width: 480px) {
        .container {
            padding: 32px 12px;
            border-radius: 12px;
            max-width: 98vw;
        }
        h2 {
            font-size: 1.5rem;
        }
        input, button {
            width: 98%;
            font-size: 1rem;
        }
        }
    </style>
    <!-- Firebase SDKs -->
    <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-app-compat.js"></script>
    <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-auth-compat.js"></script>
    </head>
    <body>
    <div class="container">
        <h2>Sign In</h2>
        <input type="email" id="email" placeholder="Email">
        <input type="password" id="password" placeholder="Password">
        <button onclick="signIn()">Sign In</button>
        <p>Don't have an account? <a href="/signup">Sign Up</a></p>
        <p id="error-msg"></p>
    </div>

    <script>
        const firebaseConfig = {
        apiKey: "AIzaSyB-y52ekJkjgXYXhRNvvir0r9gU8CObpkM",
        authDomain: "plant-doctor-63da7.firebaseapp.com",
        projectId: "plant-doctor-63da7",
        storageBucket: "plant-doctor-63da7.appspot.com",
        messagingSenderId: "94281353808",
        appId: "1:94281353808:web:5fa49e3d6e494868be5f55",
        measurementId: "G-E5P4MZ68HW"
        };

        if (!firebase.apps.length) {
        firebase.initializeApp(firebaseConfig);
        }
        const auth = firebase.auth();

        function signIn() {
        const email = document.getElementById("email").value.trim();
        const password = document.getElementById("password").value;
        const errorMsg = document.getElementById("error-msg");

        errorMsg.textContent = ""; // Clear any previous error messages

        if (!email) {
            errorMsg.textContent = "Please enter your email.";
            return;
        }
        if (!password) {
            errorMsg.textContent = "Please enter your password.";
            return;
        }

        auth.signInWithEmailAndPassword(email, password)
            .then((userCredential) => {
            window.location.href = "/";
            })
            .catch((error) => {
            errorMsg.textContent = error.message;
            console.error("Sign-in error:", error);
            });
        }

        auth.onAuthStateChanged((user) => {
        if (user) {
            console.log("User logged in:", user.email);
        } else {
            console.log("No user logged in");
        }
        });
    </script>
    </body>
    </html>
    """

CORS(app)

# Get Gemini API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

@app.route('/chatbot')
def chatbot():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Plant Saathi - Plant Doctor AI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            margin: 0;
            font-family: 'Segoe UI', 'Arial', sans-serif;
            background: linear-gradient(135deg, #f0fdf4 0%, #d1fae5 100%);
            color: #222;
        }
        .min-h-screen { min-height: 100vh; }
        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        .flex { display: flex; }
        .flex-between { display: flex; align-items: center; justify-content: space-between; }
        .items-center { align-items: center; }
        .gap-2 { gap: 0.5rem; }
        .text-center { text-align: center; }
        .animate-pulse { animation: pulse 2s infinite; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: .7; }
        }
        .header {
            background: #fff;
            box-shadow: 0 2px 8px -2px rgba(16, 185, 129, 0.08);
            border-bottom: 1px solid #bbf7d0;
        }
        .icon-lg { width: 32px; height: 32px; }
        .title {
            font-size: 2rem;
            font-weight: bold;
            color: #065f46;
        }
        .nav {
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }
        .nav a {
            color: #047857;
            font-weight: 500;
            text-decoration: none;
            transition: color 0.2s;
        }
        .nav a:hover, .nav a.active { color: #065f46; }
        .nav-actions {
            display: flex;
            gap: 0.75rem;
            margin-left: 1.5rem;
        }
        .btn {
            padding: 0.5rem 1rem;
            border-radius: 0.75rem;
            font-weight: 500;
            transition: all .2s;
            box-shadow: none;
            border: none;
            cursor: pointer;
        }
        .btn-green {
            background: #16a34a;
            color: #fff;
        }
        .btn-green:hover { background: #166534; }
        .btn-outline-green {
            background: #fff;
            border: 2px solid #16a34a;
            color: #16a34a;
        }
        .btn-outline-green:hover { background: #f0fdf4; }
        .profile-circle {
            width: 40px;
            height: 40px;
            background: #16a34a;
            color: #fff;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.2rem;
            cursor: pointer;
            border: 2px solid #fff;
            box-shadow: 0 2px 8px -2px #16a34a44;
            transition: background .2s;
        }
        .profile-circle:hover { background: #166534; }
        .chatbot-section {
            max-width: 700px;
            margin: 0 auto 3rem auto;
            background: #fff;
            border-radius: 1.25rem;
            box-shadow: 0 4px 32px -8px #16a34a22;
            padding: 2rem;
            margin-top: 3rem;
        }
        .chat-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .chat-header h2 {
            color: #065f46;
            font-size: 2rem;
            font-weight: 700;
        }
        .chat-header p {
            color: #047857;
            font-size: 1.2rem;
            margin-top: 0.5rem;
        }
        .chat-area {
            width: 100%;
            min-height: 340px;
            max-height: 420px;
            overflow-y: auto;
            margin-bottom: 1rem;
            padding-right: 6px;
        }
        .chat-message {
            margin: 8px 0;
            clear: both;
            max-width: 70%;
        }
        .chat-message.user {
            background: #CFEFFF;
            color: #000;
            padding: 10px;
            border-radius: 10px;
            float: right;
            box-shadow: 0 1px 8px -2px #16a34a22;
            margin-right: 0;
            margin-left: auto;
        }
        .chat-message.bot {
            background: #E2FFE2;
            color: #000;
            padding: 10px;
            border-radius: 10px;
            float: left;
            box-shadow: 0 1px 8px -2px #16a34a22;
            margin-left: 0;
            margin-right: auto;
            line-height: 1.5;
        }
        .chat-message.bot h1,
        .chat-message.bot h2, 
        .chat-message.bot h3 {
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .chat-message.bot h1:first-child,
        .chat-message.bot h2:first-child,
        .chat-message.bot h3:first-child {
            margin-top: 0;
        }
        .chat-message.bot ul,
        .chat-message.bot ol {
            margin: 0.5rem 0;
            padding-left: 1.5rem;
        }
        .chat-message.bot p {
            margin: 0.5rem 0;
        }
        .chat-timestamp {
            font-size: 0.85em;
            color: #888;
            margin-bottom: 4px;
            display: block;
        }
        .chat-clear { clear: both; }
        .chat-form {
            display: flex;
            gap: 1rem;
            align-items: flex-end;
            margin-top: 1rem;
        }
        .chat-input {
            flex: 1;
            padding: 0.75rem;
            border: 1px solid #bbf7d0;
            border-radius: 0.75rem;
            font-size: 1.05rem;
            background: #f0fdf4;
            color: #222;
            transition: border .2s;
            min-height: 50px;
            resize: vertical;
        }
        .chat-input:focus {
            border: 2px solid #16a34a;
            outline: none;
            background: #e6fcf0;
        }
        .chat-send-btn {
            padding: 0.75rem 1.5rem;
            background: linear-gradient(135deg, #16a34a, #22c55e);
            color: #fff;
            font-weight: 500;
            border-radius: 0.75rem;
            border: none;
            cursor: pointer;
            font-size: 1.05rem;
            transition: background .2s;
            box-shadow: 0 2px 8px -2px #16a34a22;
        }
        .chat-send-btn:hover:not(:disabled) { background: #166534; }
        .chat-send-btn:disabled {
            opacity: 0.7;
            cursor: not-allowed;
        }
        .chat-clear-btn {
            padding: 0.75rem 1rem;
            background: #fff;
            color: #16a34a;
            border: 2px solid #16a34a;
            border-radius: 0.75rem;
            font-size: 1rem;
            cursor: pointer;
            margin-left: 1rem;
            transition: background .2s;
        }
        .chat-clear-btn:hover { background: #f0fdf4; }
        .chat-loader {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f0fdf4;
            border-top: 3px solid #16a34a;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .status-indicator {
            text-align: center;
            margin-bottom: 16px;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 0.9em;
            font-weight: 500;
        }
        .status-connected {
            background: #dcfce7;
            color: #15803d;
            border: 1px solid #bbf7d0;
        }
        .status-error {
            background: #fecaca;
            color: #dc2626;
            border: 1px solid #fca5a5;
        }
        .welcome-message {
            background: #f0fdf4;
            border-left: 4px solid #16a34a;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 0 8px 8px 0;
        }
        .welcome-message h3 {
            margin: 0 0 0.5rem 0;
            color: #065f46;
        }
        .welcome-message p {
            margin: 0;
            color: #047857;
            font-size: 0.95em;
        }
        @media (max-width: 800px) {
            .chatbot-section {
                max-width: 99vw;
                padding: 1rem;
            }
            .chat-form {
                flex-direction: column;
                gap: 0.5rem;
            }
            .chat-clear-btn {
                margin-left: 0;
                width: 100%;
            }
        }
        @media (max-width: 600px) {
            .chatbot-section { padding: 0.5rem; }
            .chat-header h2 { font-size: 1.5rem; }
            .chat-header p { font-size: 1rem; }
        }
    </style>
    <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-app-compat.js"></script>
    <script src="https://www.gstatic.com/firebasejs/9.22.2/firebase-auth-compat.js"></script>
</head>
<body class="min-h-screen">
    <header class="header">
        <div class="container flex-between">
            <div class="flex items-center gap-2">
                <svg class="icon-lg text-green" fill="none" stroke="green" stroke-width="2" viewBox="0 0 24 24">
                    <path d="M2 22s9-2 15-8 5-12 5-12-9 2-15 8-5 12-5 12z"/>
                </svg>
                <h1 class="title">Plant Doctor AI</h1>
            </div>
            <nav class="nav">
                <a href="/">Home</a>
                <a href="/predict">Upload & Predict</a>
                <a href="/gallery">Disease Gallery</a>
                <a href="/chatbot" class="active">Plant Saathi</a>
                <a href="/about">About</a>
                <div class="nav-actions" id="nav-actions"></div>
            </nav>
        </div>
    </header>
    
    <main class="container">
        <section class="chatbot-section">
            <div class="chat-header">
                <h2 class="animate-pulse">Plant Saathi</h2>
                <p>Ask about plant diseases, symptoms, treatment, or prevention strategies!</p>
            </div>
            
            <div id="connection-status" class="status-indicator status-connected">Ready to chat!</div>
            
            <div class="welcome-message">
                <h3>Welcome to Plant Saathi!</h3>
                <p>I'm here to help you with plant diseases, pest problems, treatments, and agricultural advice. Feel free to ask me anything about plant health!</p>
            </div>
            
            <div id="chat-area" class="chat-area"></div>
            <form id="chat-form" class="chat-form" autocomplete="off">
                <textarea id="chat-input" class="chat-input" rows="2" placeholder="Ask me about plant diseases, symptoms, or treatments..." required></textarea>
                <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                    <button id="send-btn" type="submit" class="chat-send-btn">Send</button>
                    <button id="clear-btn" type="button" class="chat-clear-btn">Clear Chat</button>
                </div>
            </form>
        </section>
    </main>
    
    <script>
        const firebaseConfig = {
            apiKey: "AIzaSyB-y52ekJkjgXYXhRNvvir0r9gU8CObpkM",
            authDomain: "plant-doctor-63da7.firebaseapp.com",
            projectId: "plant-doctor-63da7",
            storageBucket: "plant-doctor-63da7.appspot.com",
            messagingSenderId: "94281353808",
            appId: "1:94281353808:web:5fa49e3d6e494868be5f55",
            measurementId: "G-E5P4MZ68HW"
        };
        
        if (!firebase.apps.length) {
            firebase.initializeApp(firebaseConfig);
        }
        const auth = firebase.auth();
        
        function renderProfile(user) {
            const navActions = document.getElementById("nav-actions");
            if (!navActions) return;
            let displayName = user.displayName || user.email || "U";
            let firstLetter = displayName.charAt(0).toUpperCase();
            navActions.innerHTML = '<div class="profile-circle" title="' + displayName + '" onclick="logout()">' + firstLetter + '</div>';
        }
        
        function renderLoginSignup() {
            const navActions = document.getElementById("nav-actions");
            if (!navActions) return;
            navActions.innerHTML = '<a href="/signin" class="btn btn-green">Log In</a><a href="/signup" class="btn btn-outline-green">Sign Up</a>';
        }
        
        function logout() {
            firebase.auth().signOut().then(() => {
                window.location.reload();
            });
        }
        
        auth.onAuthStateChanged(user => {
            if (user) {
                renderProfile(user);
            } else {
                renderLoginSignup();
            }
        });
    </script>
    
    <script>
        const chatArea = document.getElementById('chat-area');
        const chatForm = document.getElementById('chat-form');
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        const clearBtn = document.getElementById('clear-btn');
        const connectionStatus = document.getElementById('connection-status');
        
        const BACKEND_URL = window.location.origin;
        
        let history = [];
        try {
            const savedHistory = localStorage.getItem('pd_chat_history');
            if (savedHistory) {
                history = JSON.parse(savedHistory);
            }
        } catch (e) {
            console.warn('Could not load chat history:', e);
            history = [];
        }
        
        function renderChatHistory() {
            chatArea.innerHTML = '';
            history.forEach(msg => {
                const msgDiv = document.createElement('div');
                msgDiv.className = 'chat-message ' + (msg.role === "user" ? "user" : "bot");
                msgDiv.innerHTML = '<span class="chat-timestamp">' + 
                    (msg.role === "user" ? "You" : "Plant Doctor") + 
                    ' (' + (msg.time || "") + '):</span>' +
                    (msg.role === "bot" ? formatBotMessage(msg.content) : escapeHTML(msg.content));
                chatArea.appendChild(msgDiv);
                const clearDiv = document.createElement('div');
                clearDiv.className = "chat-clear";
                chatArea.appendChild(clearDiv);
            });
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        function escapeHTML(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function formatBotMessage(text) {
            let formatted = escapeHTML(text);
            
            formatted = formatted.replace(/^\\|(.+)\\|$/gm, function(match) {
                return 'TABLE_ROW_MARKER' + match + 'TABLE_ROW_MARKER';
            });
            
            const tableRegex = /TABLE_ROW_MARKER\\|(.+?)\\|TABLE_ROW_MARKER(?:\\n(?:TABLE_ROW_MARKER\\|.*?\\|TABLE_ROW_MARKER|\\|.*?\\||[\\-\\|: ]+))*(?=\\n[^|]|$)/g;
            
            formatted = formatted.replace(tableRegex, function(tableMatch) {
                const rows = tableMatch.split('\\n').filter(row => row.includes('|'));
                let tableHTML = '<table style="border-collapse: collapse; margin: 1rem 0; width: 100%; font-size: 0.9em;">';
                let isHeader = true;
                
                rows.forEach((row) => {
                    row = row.replace(/TABLE_ROW_MARKER/g, '');
                    
                    if (row.match(/^\\s*\\|[\\s\\-\\|:]*\\|\\s*$/)) {
                        isHeader = false;
                        return;
                    }
                    
                    const cells = row.split('|').slice(1, -1).map(cell => cell.trim());
                    
                    if (cells.length > 0) {
                        const cellTag = isHeader ? 'th' : 'td';
                        const cellStyle = isHeader 
                            ? 'background: #f0fdf4; border: 1px solid #bbf7d0; padding: 8px 12px; font-weight: 600; color: #065f46;'
                            : 'border: 1px solid #bbf7d0; padding: 8px 12px; background: #fff;';
                        
                        tableHTML += '<tr>';
                        cells.forEach(cell => {
                            tableHTML += '<' + cellTag + ' style="' + cellStyle + '">' + cell + '</' + cellTag + '>';
                        });
                        tableHTML += '</tr>';
                        
                        if (isHeader) isHeader = false;
                    }
                });
                
                tableHTML += '</table>';
                return tableHTML;
            });
            
            formatted = formatted.replace(/^#### (.*$)/gm, '<h4 style="color: #065f46; margin: 1rem 0 0.5rem 0; font-size: 1.1em; font-weight: 600;">$1</h4>');
            formatted = formatted.replace(/^### (.*$)/gm, '<h3 style="color: #065f46; margin: 1rem 0 0.5rem 0; font-size: 1.2em; font-weight: 600;">$1</h3>');
            formatted = formatted.replace(/^## (.*$)/gm, '<h2 style="color: #065f46; margin: 1.2rem 0 0.6rem 0; font-size: 1.4em; font-weight: 700;">$1</h2>');
            formatted = formatted.replace(/^# (.*$)/gm, '<h1 style="color: #065f46; margin: 1.5rem 0 0.8rem 0; font-size: 1.6em; font-weight: 700;">$1</h1>');
            
            formatted = formatted.replace(/```([\\s\\S]*?)```/g, '<pre style="background: #f3f4f6; border: 1px solid #d1d5db; border-radius: 6px; padding: 12px; margin: 1rem 0; overflow-x: auto; font-family: monospace; font-size: 0.9em;"><code>$1</code></pre>');
            
            formatted = formatted.replace(/`([^`\\n]+)`/g, '<code style="background: #f3f4f6; border: 1px solid #d1d5db; padding: 2px 6px; border-radius: 3px; font-family: monospace; font-size: 0.9em;">$1</code>');
            
            formatted = formatted.replace(/\\*\\*([^\\*\\n]+)\\*\\*/g, '<strong>$1</strong>');
            formatted = formatted.replace(/__([^_\\n]+)__/g, '<strong>$1</strong>');
            
            formatted = formatted.replace(/(?<!\\*)\\*([^\\*\\n]+)\\*(?!\\*)/g, '<em>$1</em>');
            formatted = formatted.replace(/(?<!_)_([^_\\n]+)_(?!_)/g, '<em>$1</em>');
            
            formatted = formatted.replace(/~~([^~\\n]+)~~/g, '<del>$1</del>');
            
            formatted = formatted.replace(/^&gt; (.+)$/gm, '<blockquote style="border-left: 4px solid #16a34a; margin: 1rem 0; padding: 0.5rem 1rem; background: #f0fdf4; font-style: italic;">$1</blockquote>');
            
            formatted = formatted.replace(/^(---+|\\*{3,})$/gm, '<hr style="border: none; border-top: 2px solid #e5e7eb; margin: 1.5rem 0;">');
            
            formatted = formatted.replace(/^[\\*\\-\\+] (.+)$/gm, '<li style="margin: 0.25rem 0;">• $1</li>');
            
            formatted = formatted.replace(/^\\d+\\. (.+)$/gm, '<li style="margin: 0.25rem 0;">$1</li>');
            
            formatted = formatted.replace(/(<li[^>]*>.*?<\\/li>(?:\\s*<li[^>]*>.*?<\\/li>)*)/g, '<ul style="margin: 0.5rem 0; padding-left: 1.5rem; list-style: none;">$1</ul>');
            
            formatted = formatted.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" style="color: #16a34a; text-decoration: underline;" target="_blank" rel="noopener noreferrer">$1</a>');
            
            formatted = formatted.replace(/\\n/g, '<br>');
            
            return formatted;
        }
        
        renderChatHistory();
        
        chatForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const question = chatInput.value.trim();
            if (!question) return;
            
            const now = new Date();
            const time = now.getHours().toString().padStart(2,'0') + ":" + now.getMinutes().toString().padStart(2,'0');
            
            history.push({role:"user", content:question, time});
            renderChatHistory();
            chatInput.value = "";
            sendBtn.disabled = true;
            sendBtn.innerHTML = '<span class="chat-loader"></span>Thinking...';
        
            let botMsg = {role:"bot", content:"Sorry, I couldn't process your request.", time};
            
            try {
                const response = await fetch(BACKEND_URL + '/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    mode: 'cors',
                    body: JSON.stringify({
                        message: question,
                        history: history.slice(0, -1)
                    })
                });
            
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || 'HTTP ' + response.status + ': ' + (data.details || 'Unknown error'));
                }
                
                if (data.response) {
                    botMsg.content = data.response;
                    connectionStatus.innerHTML = 'Connected and ready!';
                    connectionStatus.className = 'status-indicator status-connected';
                } else {
                    botMsg.content = "Sorry, I didn't receive a proper response from the AI.";
                }
                
            } catch (error) {
                console.error('Error calling Flask API:', error);
                
                if (error.message.includes('Failed to fetch') || error.message.includes('fetch')) {
                    botMsg.content = 'Cannot connect to Flask backend. Please make sure you\\'re running: python app.py\\n\\nError: ' + error.message;
                    connectionStatus.innerHTML = 'Backend Connection Failed';
                } else if (error.message.includes('404')) {
                    botMsg.content = 'API endpoint not found. Make sure Flask backend is running with correct routes.';
                    connectionStatus.innerHTML = 'API Route Error';
                } else {
                    botMsg.content = 'Error: ' + error.message;
                    connectionStatus.innerHTML = 'API Error - Check Console';
                }
                connectionStatus.className = 'status-indicator status-error';
            }
            
            botMsg.time = new Date().getHours().toString().padStart(2,'0') + ":" + new Date().getMinutes().toString().padStart(2,'0');
            history.push(botMsg);
            
            try {
                localStorage.setItem('pd_chat_history', JSON.stringify(history));
            } catch (e) {
                console.warn('Could not save chat history:', e);
            }
            
            sendBtn.disabled = false;
            sendBtn.innerHTML = "Send";
            renderChatHistory();
        });
        
        clearBtn.addEventListener('click', function() {
            history = [];
            try {
                localStorage.setItem('pd_chat_history', JSON.stringify(history));
            } catch (e) {
                console.warn('Could not clear chat history:', e);
            }
            renderChatHistory();
        });
        
        chatInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                chatForm.dispatchEvent(new Event('submit'));
            }
        });
        
        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    </script>
</body>
</html>
"""

@app.route('/api/chat', methods=['POST'])
def chat():
    try:        
        if not GEMINI_API_KEY:
            return jsonify({
                'error': 'GEMINI_API_KEY environment variable not set'
            }), 500
        
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_message = data['message']
        chat_history = data.get('history', [])
        
        contents = []
        
        contents.append({
            "role": "user",
            "parts": [{
                "text": "You are a helpful plant doctor chatbot specializing in plant diseases and agriculture. Answer questions about plant diseases, symptoms, causes, treatments, medicines, and prevention clearly and professionally. Use structured formatting with headings when appropriate."
            }]
        })
        
        contents.append({
            "role": "model", 
            "parts": [{
                "text": "I understand. I'm here to help with plant diseases, symptoms, treatments, and agricultural advice. Please ask me your question about plants!"
            }]
        })
        
        recent_history = chat_history[-8:] if len(chat_history) > 8 else chat_history
        for msg in recent_history:
            role = "user" if msg['role'] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg['content']}]
            })
        
        contents.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })
        
        request_body = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': GEMINI_API_KEY
        }
        
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            json=request_body,
            timeout=30
        )
        
        if response.status_code != 200:
            error_text = response.text
            try:
                error_json = response.json()
                error_message = error_json.get('error', {}).get('message', error_text)
            except:
                error_message = error_text
            
            return jsonify({
                'error': f'Gemini API Error (Status {response.status_code}): {error_message}',
                'status_code': response.status_code,
                'details': error_text
            }), 500
        
        response_data = response.json()
        
        if (response_data.get('candidates') and 
            len(response_data['candidates']) > 0 and 
            response_data['candidates'][0].get('content') and 
            response_data['candidates'][0]['content'].get('parts') and
            len(response_data['candidates'][0]['content']['parts']) > 0):
            
            generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
            
            return jsonify({
                'response': generated_text,
                'status': 'success'
            })
        else:
            return jsonify({
                'error': 'No content generated by Gemini API',
                'details': str(response_data)
            }), 500
            
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request to Gemini API timed out'}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Network error: {str(e)}'}), 500
    except json.JSONDecodeError as e:
        return jsonify({'error': f'JSON decode error: {str(e)}'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
    
print("Starting Flask app...")
initialize_app()
print(f"GEMINI_API_KEY set: {'Yes' if GEMINI_API_KEY else 'No'}")
app.run(debug=False, host='0.0.0.0', port=5000)
