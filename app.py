from flask import Flask, request, jsonify, session, make_response, url_for
import torch
from PIL import Image
from torchvision import transforms
import os
import torch.nn as nn
import torchvision.models as models
from groq import Groq
from dotenv import load_dotenv
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask_caching import Cache
from hashlib import md5
from flask_session import Session
from redis import Redis
from bs4 import BeautifulSoup
from google import genai

# Load environment variables from the .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = os.getenv('FLASK_SECRET_KEY')
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Initialize the cache
cache = Cache(config={'CACHE_TYPE': 'simple'})  # Simple in-memory cache
cache.init_app(app)

# Configure session to use Redis
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False  # Set to True if you want sessions to persist between server restarts
app.config['SESSION_USE_SIGNER'] = True  # Sign the session to prevent tampering
app.config['SESSION_KEY_PREFIX'] = 'your_app_session:'  # Prefix for Redis keys
app.config['SESSION_REDIS'] = Redis(host='localhost', port=6379, db=0)

from flask_cors import CORS

CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://localhost:3000"}})

# Initialize the session extension
Session(app)

def is_health_related(prompt):
    health_keywords = [
        'health', 'medicine', 'medical', 'doctor', 'disease', 'symptom',
        'treatment', 'well-being', 'ophthalmology', 'eye', 'vision', 'glaucoma',
        'diabetes', 'cataract', 'hypertension', 'myopia', 'AMD', 'amd', 'macular degeneration',
        'infection', 'inflammation', 'surgery', 'surgical', 'diagnosis', 'prescription',
        'therapy', 'clinic', 'hospital', 'pharmacy', 'medication', 'drug', 'illness',
        'condition', 'chronic', 'acute', 'recovery', 'rehabilitation', 'disorder',
        'consultation', 'appointment', 'specialist', 'physician', 'nurse', 'allergy',
        'immune system', 'immunity', 'pathology', 'anatomy', 'biopsy', 'imaging',
        'x-ray', 'MRI', 'CT scan', 'ultrasound', 'blood pressure', 'heart rate',
        'pulse', 'respiration', 'nutrition', 'diet', 'exercise', 'physical therapy',
        'mental health', 'psychiatry', 'psychology', 'counseling', 'wellness',
        'screening', 'preventive care', 'vaccination', 'immunization', 'epidemic',
        'pandemic', 'virus', 'bacteria', 'microorganism', 'pathogen', 'infection',
        'fever', 'pain', 'headache', 'migraine', 'cancer', 'tumor', 'oncology',
        'cardiology', 'neurology', 'dermatology', 'orthopedics', 'pediatrics',
        'gerontology', 'geriatrics', 'gastroenterology', 'urology', 'nephrology',
        'endocrinology', 'hematology', 'gynecology', 'obstetrics', 'radiology',
        'pulmonology', 'rheumatology', 'infectious disease', 'sore throat', 'cough',
        'flu', 'cold', 'COVID', 'virus', 'antibiotic', 'antiviral', 'antifungal',
        'blood test', 'cholesterol', 'glucose', 'insulin', 'metabolism', 'gene',
        'genetic', 'biological', 'biomedical', 'cell', 'tissue', 'organ', 'body',
        'muscle', 'bone', 'joint', 'arthritis', 'tendon', 'ligament', 'fracture',
        'sprain', 'strain', 'wound', 'laceration', 'bruise', 'burn', 'rash',
        'eczema', 'psoriasis', 'acne', 'dermatitis', 'skin', 'hair', 'nail',
        'hormone', 'thyroid', 'adrenal', 'pituitary', 'corticosteroid',
        'anti-inflammatory', 'analgesic', 'painkiller', 'sedative', 'anesthetic',
        'antiseptic', 'disinfectant', 'sterilization', 'hygiene', 'sanitation',
        'clinical trial', 'study', 'research', 'vaccine', 'booster', 'injection',
        'IV', 'intravenous', 'syringe', 'needle', 'scalpel', 'stethoscope',
        'blood pressure cuff', 'thermometer', 'sphygmomanometer', 'otoscope',
        'ophthalmoscope', 'reflex hammer', 'medical chart', 'record', 'report',
        'laboratory', 'specimen', 'sample', 'swab', 'culture', 'petri dish',
        'microscope', 'slide', 'pathogen', 'symptomatic', 'asymptomatic', 'outbreak',
        'quarantine', 'isolation', 'healthcare', 'health provider', 'clinical',
        'telemedicine', 'telehealth', 'consultation', 'triage', 'first aid', 'CPR',
        'emergency', 'urgent care', 'ambulance', 'paramedic', 'ER', 'ICU', 'sutures',
        'stitches', 'bandage', 'cast', 'splint', 'crutches', 'wheelchair', 'prosthesis',
        'orthotic', 'hearing aid', 'eyeglasses', 'contact lenses', 'vision correction',
        'LASIK', 'refractive surgery', 'eye exam', 'fundus', 'retina', 'cornea',
        'pupil', 'iris', 'lens', 'optic nerve', 'sclera', 'tear duct', 'vitreous humor', 'kids', 'adults',
        'elderly', 'infants', 'toddlers', 'ocular', 'diagnosis', 'prognosis', 'care', 'children', 'child', 'patient'
        # Add any more relevant keywords as necessary
    ]
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in health_keywords)

@app.route('/')
def index():
    # Example of setting a value in the session
    session['key'] = 'value'
    return 'Session set!'

# Define your image classification models
class DiseaseSpecificModel(nn.Module):
    def __init__(self, num_classes=8):
        super(DiseaseSpecificModel, self).__init__()
        # Use a pre-trained ResNet50 model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Modify the output layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

# Load and configure the dual-image classification model (processes single images)
dual_image_model_path = 'newDualModel.pth'  # Path to the new dual-image model
dual_image_model = DiseaseSpecificModel(num_classes=8)
dual_image_model.load_state_dict(torch.load(dual_image_model_path, map_location=torch.device('cpu')))
dual_image_model.eval()

# Load and configure the single-image classification model
single_image_model_path = 'single_Model.pth'  # Path to the single-image model
single_image_model = DiseaseSpecificModel(num_classes=8)
single_image_model.load_state_dict(torch.load(single_image_model_path, map_location=torch.device('cpu')))
single_image_model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define class-to-disease mapping and treatment mapping
class_to_disease = {
    0: 'Normal',
    1: 'Diabetes',
    2: 'Glaucoma',
    3: 'Cataract',
    4: 'Age-related Macular Degeneration (AMD)',
    5: 'Hypertension',
    6: 'Myopia',
    7: 'Other Diseases/Abnormalities'
}

disease_treatment = {
    'Normal': 'Your eyes appear to be normal. Maintain regular eye check-ups and a healthy lifestyle.',
    'Diabetes': 'Manage blood sugar levels, regular eye exams, possible laser treatment or surgery.',
    'Glaucoma': 'Medications, laser treatment, or surgery to lower intraocular pressure.',
    'Cataract': 'Surgery to replace the cloudy lens with an artificial lens.',
    'Age-related Macular Degeneration (AMD)': 'Anti-VEGF injections, laser therapy, lifestyle changes (diet, smoking cessation).',
    'Hypertension': 'Blood pressure control, lifestyle changes, medications.',
    'Myopia': 'Corrective lenses, orthokeratology, or refractive surgery for severe cases.',
    'Other Diseases/Abnormalities': 'Consult with an ophthalmologist for specific treatment options.'
}

# Map diseases to diagram image filenames
disease_diagram = {
    'Diabetes': 'diabetes.jpg',
    'Glaucoma': 'glaucoma.jpg',
    'Cataract': 'cataract.jpg',
    'Age-related Macular Degeneration (AMD)': 'amd.png',
    'Hypertension': 'hypertension.jpg',
    'Myopia': 'myopia.jpg',
    'Other Diseases/Abnormalities': 'Other_Diseases.jpg'
}

# Initialize Groq client
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Updated function to get the LLM response using Groq
def get_llm_response(left_diseases=None, right_diseases=None, chat_history=None, is_image_upload=False):
    # Simplified system message
    system_message = (
        "You are a helpful and knowledgeable medical assistant specialized in ophthalmology. "
        "Provide factual answers from a medical perspective. "
        "Ensure the information is accurate and easy to understand. "
        "Present the information in plain text without any HTML or markup. "
        "You must not answer queries unrelated to health, medicine, or overall well-being."
    )

    # Prepare messages for chat completion
    messages = [{"role": "system", "content": system_message}]

    # Include chat history, limiting to the last 5 messages
    if chat_history:
        messages.extend(chat_history[-5:])

    # Generate the prompt based on the diagnosis
    if is_image_upload:
        # Determine which eyes have diagnoses
        prompt_parts = []
        eye_descriptions = []

        if left_diseases is not None:
            left_diseases_str = ', '.join(left_diseases) if left_diseases else 'No detectable diseases'
            prompt_parts.append(f"- Left Eye: {left_diseases_str}")
            if left_diseases == ['Normal']:
                eye_descriptions.append("left eye")

        if right_diseases is not None:
            right_diseases_str = ', '.join(right_diseases) if right_diseases else 'No detectable diseases'
            prompt_parts.append(f"- Right Eye: {right_diseases_str}")
            if right_diseases == ['Normal']:
                eye_descriptions.append("right eye")

        # Check if all provided eyes are normal
        if (
            (left_diseases == ['Normal'] if left_diseases is not None else True) and
            (right_diseases == ['Normal'] if right_diseases is not None else True)
        ):
            if eye_descriptions:
                eye_str = " and ".join(eye_descriptions)
                prompt = (
                    f"The patient's {'eyes are' if len(eye_descriptions) > 1 else 'eye is'} normal with no detectable signs of ocular diseases "
                    f"in the {eye_str}. "
                    "Provide guidelines for maintaining optimal eye health, including lifestyle recommendations."
                )
            else:
                # No eyes provided or no diagnoses, default message
                prompt = (
                    "No detectable diseases were found in the uploaded eye image. "
                    "Provide guidelines for maintaining optimal eye health, including lifestyle recommendations."
                )
        else:
            prompt = (
                "A patient is diagnosed with the following condition(s):\n"
                f"{chr(10).join(prompt_parts)}\n"
                "Provide a detailed medical report that includes recommended treatment options, lifestyle changes, possible outcomes, and future prognosis for each condition."
            )

        messages.append({"role": "user", "content": prompt})
    elif chat_history:
        # If not an image upload, rely on the existing chat history
        pass
    else:
        return "No input provided."

    print(f"Generated Messages: {messages}")  # For debugging

    # Generate text using the Groq API
    try:
        response = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192"
        )
        response_text = response.choices[0].message.content
        print(f"LLM Response: {response_text}")  # For debugging

        # Remove HTML tags from the response using BeautifulSoup
        soup = BeautifulSoup(response_text, 'html.parser')
        clean_response_text = soup.get_text()
        print(f"Cleaned LLM Response: {clean_response_text}")  # For debugging

    except Exception as e:
        print(f"Error during text generation: {e}")
        return "Error during text generation."

    return clean_response_text

# Custom function to generate a cache key based on the request body (prompt)
def make_cache_key():
    if request.json and 'prompt' in request.json:
        prompt = request.json['prompt'].strip().lower()  # Normalize prompt
        return md5(prompt.encode('utf-8')).hexdigest()
    return None

@app.route('/api/chat', methods=['POST'])
@cache.cached(timeout=300, key_prefix=make_cache_key)  # Cache using a custom key based on the prompt
def handle_chat():
    if request.json and 'prompt' in request.json:
        user_prompt = request.json['prompt']

        # Check if the prompt is health-related
        if not is_health_related(user_prompt):
            return jsonify({"response": "I can only assist with medical and health-related inquiries."}), 200

        session['chat_history'] = session.get('chat_history', [])
        session['chat_history'].append({'role': 'user', 'content': user_prompt})

        # Get the LLM response with chat history
        llm_response = get_llm_response([], [], chat_history=session['chat_history'])

        # Append the LLM response to the chat history with role 'assistant'
        session['chat_history'].append({'role': 'assistant', 'content': llm_response})

        return jsonify({"response": llm_response, "chat_history": session['chat_history']})
    
    return jsonify({"error": "Invalid input"}), 400

# API endpoint for handling file upload
@app.route('/api/upload', methods=['POST'])
def handle_upload():
    # Extract the user message if available
    message = request.form.get('message', '')

    # Check if the message is health-related if present
    if message and not is_health_related(message):
        return jsonify({"response": "I can only assist with medical and health-related inquiries."}), 200

    # Get the list of uploaded files
    uploaded_files = request.files.getlist('file')
    eye_labels = request.form.getlist('eye_labels')

    # Check if the number of images matches the number of eye labels
    if len(uploaded_files) != len(eye_labels):
        return jsonify({"error": "The number of images does not match the number of eye labels."}), 400

    # Validate eye labels
    valid_eye_labels = ['left', 'right']
    for label in eye_labels:
        if label.lower() not in valid_eye_labels:
            return jsonify({"error": f"Invalid eye label '{label}'. Accepted values are 'left' or 'right'."}), 400

    if len(uploaded_files) == 0:
        return jsonify({"error": "Please upload at least one eye image."}), 400

    if len(uploaded_files) > 2:
        return jsonify({"error": "Please upload no more than two images (left eye and right eye)."}), 400

    # Process uploaded files and eye labels
    images = {}
    for file, eye_label in zip(uploaded_files, eye_labels):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Open and preprocess the image
        try:
            image = Image.open(filepath).convert('RGB')
            image = transform(image)
            images[eye_label.lower()] = image
        except Exception as e:
            return jsonify({"error": f"Error processing image '{filename}': {e}"}), 400

    # Initialize dictionaries to hold predictions
    predicted_diseases = {}
    treatment_texts = {}
    diagram_urls = {}

    # Threshold for predictions
    threshold = 0.5
    disease_indices = list(range(1, len(class_to_disease)))  # indices 1-7

    # Predict diseases for each eye
    for eye_label in images:
        image_tensor = images[eye_label].unsqueeze(0)  # Add batch dimension

        # Predict diseases using the single-image model
        with torch.no_grad():
            outputs = single_image_model(image_tensor)
            preds = torch.sigmoid(outputs).squeeze(0)  # Shape: (num_classes,)

        # Apply threshold to predictions
        pred_disease_indices = (preds[disease_indices] > threshold).nonzero(as_tuple=True)[0].tolist()
        pred_disease_indices = [i+1 for i in pred_disease_indices]  # Adjust indices back to original

        normal_pred = preds[0] > threshold

        if pred_disease_indices:
            diseases = [class_to_disease[idx] for idx in pred_disease_indices]
            treatments = [disease_treatment.get(disease, 'No treatment available') for disease in diseases]
        elif normal_pred:
            diseases = ['Normal']
            treatments = [disease_treatment.get('Normal', 'Maintain regular eye check-ups and a healthy lifestyle.')]
        else:
            diseases = []
            treatments = ['No specific treatment required. Maintain regular eye check-ups and a healthy lifestyle.']

        predicted_diseases[eye_label] = diseases
        treatment_texts[eye_label] = treatments

        # Get diagram URLs for diagnosed diseases
        diagrams = []
        for disease in diseases:
            diagram_filename = disease_diagram.get(disease)
            if diagram_filename:
                diagram_url = url_for('static', filename=f'diagrams/{diagram_filename}', _external=True)
                diagrams.append(diagram_url)
        diagram_urls[eye_label] = diagrams

    # Prepare the diagnosis text
    diagnosis_text = ""
    for eye_label in ['left', 'right']:
        if eye_label in predicted_diseases:
            diseases = predicted_diseases[eye_label]
            diagnosis = ', '.join(diseases) if diseases else 'No detectable diseases'
            diagnosis_text += f"{eye_label.capitalize()} Eye: {diagnosis}\n"

    # Update chat history
    session['chat_history'] = session.get('chat_history', [])
    user_message = f"Uploaded images diagnosed with:\n{diagnosis_text}"
    session['chat_history'].append({'role': 'user', 'content': user_message})

    # Get the LLM response with chat history
    # Prepare left and right diseases, using None if the eye was not uploaded
    left_diseases = predicted_diseases.get('left', None)  # Will be None if 'left' is not in predicted_diseases
    right_diseases = predicted_diseases.get('right', None)  # Will be None if 'right' is not in predicted_diseases

    # Get the LLM response
    llm_response = get_llm_response(
    left_diseases=left_diseases,
    right_diseases=right_diseases,
    chat_history=session['chat_history'],
    is_image_upload=True
    )

    # Append the LLM response to the chat history
    session['chat_history'].append({'role': 'assistant', 'content': llm_response})

    # Include diagram URLs in the response
    response_data = {
        "diagnosis": llm_response,
        "chat_history": session['chat_history']
    }
    if 'left' in predicted_diseases:
        response_data['left_eye'] = {
            "diagnosis": ', '.join(predicted_diseases['left']) if predicted_diseases['left'] else 'No detectable diseases',
            "diagrams": diagram_urls.get('left', [])
        }
    if 'right' in predicted_diseases:
        response_data['right_eye'] = {
            "diagnosis": ', '.join(predicted_diseases['right']) if predicted_diseases['right'] else 'No detectable diseases',
            "diagrams": diagram_urls.get('right', [])
        }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
