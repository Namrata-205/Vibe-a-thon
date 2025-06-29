from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import io
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

from twilio.rest import Client

import os
from dotenv import load_dotenv
load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
MY_PHONE_NUMBER = os.getenv("MY_PHONE_NUMBER")


twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = Flask(__name__)

# Define your class labels
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

solution_map = {
    'Pepper__bell___Bacterial_spot': 'Remove infected leaves and apply copper-based spray.',
    'Pepper__bell___healthy': 'No issues detected.',
    'Potato___Early_blight': 'Use fungicides like mancozeb and rotate crops.',
    'Potato___Late_blight': 'Apply metalaxyl-based fungicides and remove affected parts.',
    'Potato___healthy': 'Your crop looks healthy!',
    'Tomato_Bacterial_spot': 'Apply copper spray and avoid overhead watering.',
    'Tomato_Early_blight': 'Use chlorothalonil fungicide and remove affected leaves.',
    'Tomato_Late_blight': 'Remove infected plants and apply copper fungicide.',
    'Tomato_Leaf_Mold': 'Improve air circulation and apply fungicide if needed.',
    'Tomato_Septoria_leaf_spot': 'Prune lower leaves and use fungicide like chlorothalonil.',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Use neem oil or insecticidal soap.',
    'Tomato__Target_Spot': 'Use disease-free seeds and apply appropriate fungicide.',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Use resistant varieties and control whiteflies.',
    'Tomato__Tomato_mosaic_virus': 'Remove infected plants and disinfect tools.',
    'Tomato_healthy': 'Your tomato plant is healthy!'
}

# Define the model architecture (MUST match what you trained)
import torchvision.models as models

class PlantDiseaseModel(nn.Module):
    def __init__(self):
        super(PlantDiseaseModel, self).__init__()
        self.model = models.densenet121(pretrained=False)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, len(class_names))

    def forward(self, x):
        return self.model(x)


# Load the model
from torchvision import models

model = models.densenet121(weights=None)  # or weights="DEFAULT" for pretrained
model.classifier = nn.Sequential(
    nn.Dropout(),
    nn.Linear(model.classifier.in_features, len(class_names))
)

model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()



# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET'])
def home():
    return "Welcome to Field Doctor API. Use POST /predict to get predictions."


# Store location info (you can later connect this to a DB)
user_locations = {}

@app.route('/set_location', methods=['POST'])
def set_location():
    data = request.json
    user_id = data.get('user_id')  # Assume you identify users
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    if not all([user_id, latitude, longitude]):
        return jsonify({'error': 'Missing user_id or coordinates'}), 400

    user_locations[user_id] = {'lat': latitude, 'lon': longitude}
    return jsonify({'message': 'Location saved successfully'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    return jsonify({
        'disease': predicted_class,
        'solution': solution_map.get(predicted_class, 'No solution available')
    })
# app.py (add this above your routes)
import requests

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Mock user database
user_data = {
    "farmer123": {
        "phone": MY_PHONE_NUMBER,
        "language": "hi"  # ISO 639-1 language code
    }
}

# Simple severe weather detection logic
def is_severe_weather(weather_data):
    alerts = weather_data.get("alerts", [])
    if alerts:
        return True, alerts[0]["description"]
    return False, ""
@app.route('/weather_alert', methods=['POST'])
def check_weather_for_all_users():
    print("🔄 Running scheduled weather check...")
    for user_id, location in user_locations.items():
        lat = location['lat']
        lon = location['lon']

        if user_id not in user_data:
            continue

        try:
            url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&lang={user_data[user_id]['language']}"
            res = requests.get(url)
            weather_json = res.json()
            severe, alert_msg = is_severe_weather(weather_json)

            if severe:
                print(f"⚠️ ALERT for {user_id}: {alert_msg}")
                try:
                    message = twilio_client.messages.create(
                        from_=TWILIO_WHATSAPP_NUMBER,
                        body=f"⚠️ Weather Alert ({user_data[user_id]['language']}): {alert_msg}",
                        to=user_data[user_id]["phone"]
                    )
                    print(f"✅ Sent WhatsApp alert to {user_id}: {message.sid}")
                except Exception as e:
                    print(f"❌ Failed to send WhatsApp alert to {user_id}: {e}")
        except Exception as e:
            print(f"❌ Error checking weather for {user_id}: {e}")

    return jsonify({"message": "Weather check completed."})

@app.route('/test_twilio', methods=['GET'])
def test_twilio():
    try:
        message = twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body="✅ This is a test WhatsApp message from Field Doctor.",
            to=user_data["farmer123"]["phone"]
        )
        return jsonify({"status": "Sent", "sid": message.sid})
    except Exception as e:
        return jsonify({"status": "Failed", "error": str(e)})

# Schedule auto checks every hour
scheduler = BackgroundScheduler()
scheduler.add_job(func=check_weather_for_all_users, trigger="interval", hours=1)
scheduler.start()

# Ensure clean shutdown
atexit.register(lambda: scheduler.shutdown())


import google.generativeai as genai

# Initialize Gemini
genai.configure(api_key=os.getenv("OMNIDIMENSION_API_KEY"))
gemini = genai.GenerativeModel("gemini-pro")

@app.route('/chatbot', methods=['POST'])
def chatbot_reply():
    data = request.json
    user_message = data.get("message")

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        response = gemini.generate_content(user_message)
        reply = response.text.strip() if hasattr(response, "text") else "No response."

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": f"Chatbot failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
