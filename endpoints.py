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

app = Flask(__name__, static_folder="static", static_url_path="")
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
                    <a href="predict">Upload & Predict</a>
                    <a href="/chatbot">Plant Saathi</a>
                    <a href="about">About</a>
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
            <html>
            <head><title>Plant Doctor AI - Predict</title></head>
            <body>
            <h1>Plant Doctor AI</h1>
            <p>Upload an image to predict plant diseases</p>
            <p>HTML file not found. Please ensure predict.html is in the same directory.</p>
            <a href="/">← Back to Home</a>
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

CORS(app)  # Enable CORS for all routes

# Gemini API configuration
GEMINI_API_KEY = "AIzaSyCnAHW3IzqHD1yb3MkE-V4kELMs2gse8AI"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

@app.route('/chatbot')
def chatbot():
    # Serve the chatbot.html file directly
    try:
        return send_from_directory('.', 'chatbot.html')
    except FileNotFoundError:
        return """
        <h1>Error: chatbot.html not found</h1>
        <p>Make sure chatbot.html is in the same directory as app.py</p>
        <p>Current directory files:</p>
        <ul>
        """ + "".join([f"<li>{f}</li>" for f in os.listdir('.')]) + """
        </ul>
        """, 404

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_message = data['message']
        chat_history = data.get('history', [])
        
        print(f"📩 Received message: {user_message}")
        print(f"📚 Chat history length: {len(chat_history)}")
        
        # Prepare the contents array for Gemini API
        contents = []
        
        # Add system prompt
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
        
        # Add recent chat history (last 4 exchanges)
        recent_history = chat_history[-8:] if len(chat_history) > 8 else chat_history
        for msg in recent_history:
            role = "user" if msg['role'] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg['content']}]
            })
        
        # Add current user message
        contents.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })
        
        # Prepare the request body for Gemini API
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
        
        print(f"🚀 Making request to Gemini API...")
        
        # Make request to Gemini API with correct headers
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': GEMINI_API_KEY
        }
        
        print(f"🔗 Request URL: {GEMINI_API_URL}")
        
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            json=request_body,
            timeout=30
        )
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code != 200:
            error_text = response.text
            print(f"❌ Gemini API Error: {response.status_code}")
            print(f"❌ Response text: {error_text}")
            
            # Try to parse error response
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
        print(f"✅ Gemini response received successfully")
        
        # Extract the generated text
        if (response_data.get('candidates') and 
            len(response_data['candidates']) > 0 and 
            response_data['candidates'][0].get('content') and 
            response_data['candidates'][0]['content'].get('parts') and
            len(response_data['candidates'][0]['content']['parts']) > 0):
            
            generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
            print(f"📝 Generated response length: {len(generated_text)} characters")
            
            return jsonify({
                'response': generated_text,
                'status': 'success'
            })
        else:
            print(f"❌ Unexpected response format from Gemini")
            print(f"Response data: {json.dumps(response_data, indent=2)}")
            return jsonify({
                'error': 'No content generated by Gemini API',
                'details': str(response_data)
            }), 500
            
    except requests.exceptions.Timeout:
        print("⏰ Request timeout")
        return jsonify({'error': 'Request to Gemini API timed out'}), 504
    except requests.exceptions.RequestException as e:
        print(f"🌐 Network error: {str(e)}")
        return jsonify({'error': f'Network error: {str(e)}'}), 500
    except json.JSONDecodeError as e:
        print(f"📄 JSON decode error: {str(e)}")
        return jsonify({'error': f'JSON decode error: {str(e)}'}), 500
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/test-gemini')
def test_gemini():
    """Test endpoint to check Gemini API connectivity"""
    try:
        print("🧪 Testing Gemini API...")
        
        test_request = {
            "contents": [{
                "role": "user",
                "parts": [{"text": "Hello! Can you help with plant diseases?"}]
            }],
            "generationConfig": {
                "maxOutputTokens": 100
            }
        }
        
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': GEMINI_API_KEY
        }
        
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            json=test_request,
            timeout=15
        )
        
        result = {
            'status_code': response.status_code,
            'api_key_set': bool(GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE"),
            'api_url': GEMINI_API_URL,
            'success': response.status_code == 200
        }
        
        if response.status_code == 200:
            response_data = response.json()
            if response_data.get('candidates'):
                result['test_response'] = response_data['candidates'][0]['content']['parts'][0]['text'][:100] + "..."
            result['message'] = "✅ Gemini API is working correctly!"
        else:
            result['error'] = response.text
            result['message'] = f"❌ Gemini API test failed with status {response.status_code}"
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'api_key_set': bool(GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE"),
            'api_url': GEMINI_API_URL,
            'success': False,
            'message': f"❌ Test failed: {str(e)}"
        })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    initialize_app()
    # Configure Flask to run on all hosts (required for cloud deployment)
    app.run(host='0.0.0.0', port=5000, debug=True)
