from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import numpy as np
from joblib import load
import io
from pydantic import BaseModel
import random  # Import random for selecting a recommendation
import requests  # For making HTTP requests to ESP32

# Load your trained model
try:
    model = load("eggplant_model.joblib")
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

# Constants
IMAGE_SIZE = (64, 64)
PLANT_CATEGORIES = ["Healthy Leaf", "Leaf Spot Disease", "Insect Pest Disease", "Mosaic Virus Disease", "Small Leaf Disease", "White Mold Disease", "Wilt Disease"]

# Recommendations based on disease and soil moisture
RECOMMENDATIONS = {
    "Healthy Leaf": {
        "causes": "The plant is healthy and free from diseases or pests.",
        "solutions": [
            "Continue regular care and monitoring.",
            "Ensure the plant receives adequate sunlight and water.",
            "Apply a balanced fertilizer periodically."
        ],
        "low": [
            "Water the plant to maintain healthy growth.",
            "Ensure the plant is receiving adequate sunlight.",
            "Apply a balanced fertilizer to promote growth."
        ],
        "high": [
            "No action needed. The plant is healthy.",
            "Monitor for any signs of pests or diseases.",
            "Maintain regular pruning to encourage healthy growth."
        ]
    },
    "Leaf Spot Disease": {
        "causes": "Caused by fungal or bacterial infections, often due to high humidity or wet leaves.",
        "solutions": [
            "Remove affected leaves to prevent further infection.",
            "Apply fungicides to control the spread.",
            "Improve air circulation around the plant."
        ],
        "low": [
            "Water the plant and apply fungicides to control the spread.",
            "Remove affected leaves to prevent further infection.",
            "Improve air circulation around the plant."
        ],
        "high": [
            "Reduce watering and apply fungicides to control the spread.",
            "Avoid overhead watering to minimize leaf wetness.",
            "Use resistant plant varieties if available."
        ]
    },
    "Insect Pest Disease": {
        "causes": "Caused by pests such as aphids, whiteflies, or caterpillars feeding on the plant.",
        "solutions": [
            "Inspect leaves regularly and remove pests manually.",
            "Use insecticides or organic pest control solutions.",
            "Introduce natural predators like ladybugs to control pests."
        ],
        "low": [
            "Water the plant and use insecticides to control pests.",
            "Introduce natural predators like ladybugs to control pests.",
            "Inspect leaves regularly and remove pests manually."
        ],
        "high": [
            "Use insecticides to control pests.",
            "Apply neem oil or other organic pest control solutions.",
            "Ensure the plant is not stressed due to overwatering."
        ]
    },
    "Mosaic Virus Disease": {
        "causes": "Caused by viruses transmitted through insect vectors like aphids or contaminated tools.",
        "solutions": [
            "Remove infected plants to prevent the spread.",
            "Disinfect tools to avoid contamination.",
            "Control insect vectors like aphids that spread the virus."
        ],
        "low": [
            "Remove infected plants and ensure proper watering.",
            "Disinfect tools to prevent the spread of the virus.",
            "Plant resistant varieties to reduce susceptibility."
        ],
        "high": [
            "Remove infected plants and reduce watering.",
            "Avoid handling plants when they are wet.",
            "Control insect vectors like aphids that spread the virus."
        ]
    },
    "Small Leaf Disease": {
        "causes": "Caused by nutrient deficiencies, poor soil conditions, or root damage.",
        "solutions": [
            "Ensure proper nutrient supply and test soil pH.",
            "Apply micronutrient fertilizers to address deficiencies.",
            "Check for root damage and address it promptly."
        ],
        "low": [
            "Ensure proper nutrient supply and water the plant.",
            "Apply micronutrient fertilizers to address deficiencies.",
            "Check for root damage and address it promptly."
        ],
        "high": [
            "Ensure proper nutrient supply and avoid overwatering.",
            "Prune affected areas to encourage healthy growth.",
            "Test soil pH and adjust if necessary."
        ]
    },
    "White Mold Disease": {
        "causes": "Caused by fungal pathogens thriving in humid and poorly ventilated conditions.",
        "solutions": [
            "Improve air circulation and reduce humidity.",
            "Remove infected plant parts to prevent spread.",
            "Apply fungicides specifically for white mold."
        ],
        "low": [
            "Improve air circulation and water the plant moderately.",
            "Remove infected plant parts to prevent spread.",
            "Apply fungicides specifically for white mold."
        ],
        "high": [
            "Improve air circulation and reduce watering.",
            "Avoid planting in overly shaded areas.",
            "Use mulch to prevent soil-borne spores from splashing onto leaves."
        ]
    },
    "Wilt Disease": {
        "causes": "Caused by soil-borne fungi or bacteria that block water flow in the plant.",
        "solutions": [
            "Check for root rot and ensure proper drainage.",
            "Remove severely affected plants to prevent spread.",
            "Apply fungicides if fungal wilt is suspected."
        ],
        "low": [
            "Check for root rot and water the plant adequately.",
            "Ensure proper drainage to prevent waterlogging.",
            "Apply fungicides if fungal wilt is suspected."
        ],
        "high": [
            "Check for root rot and avoid overwatering.",
            "Remove severely affected plants to prevent spread.",
            "Sterilize soil before planting to reduce pathogen load."
        ]
    },
    "Not an Eggplant": {
        "causes": "The uploaded image does not match an eggplant leaf.",
        "solutions": [
            "Ensure the image is of an Eggplant leaf.",
            "Verify the plant species and provide accurate input.",
            "Consult an expert for proper identification."
        ],
        "low": [
            "Ensure the image is of an Eggplant leaf and check soil moisture.",
            "Verify the plant species and provide accurate input.",
            "Consult an expert for proper identification."
        ],
        "high": [
            "Ensure the image is of an Eggplant leaf.",
            "Double-check the uploaded image for clarity.",
            "Provide a clear and close-up image of the leaf."
        ]
    }
}

# Initialize app
app = FastAPI()

# Enable CORS for React Native (adjust IP if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://192.168.100.181:8000"],  # Replace with your actual mobile app IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the disease and provide recommendations based on soil moisture.
    """
    try:
        # Fetch soil moisture data from ESP32
        soil_moisture_data = await fetch_soil_moisture()
        soil_moisture = soil_moisture_data["moisture"]
        print(f"Fetched soil moisture: {soil_moisture}")

        # Read the uploaded file (image)
        contents = await file.read()

        # Try to open the image and handle invalid files
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Resize image to a standard size (64x64)
        image = image.resize(IMAGE_SIZE)

        # Convert the image to a NumPy array for prediction
        image_array = np.array(image).flatten().reshape(1, -1)

        # Make the prediction using the model
        probabilities = model.predict_proba(image_array)[0]
        confidence = max(probabilities)
        predicted_label = model.classes_[np.argmax(probabilities)]

        # Debugging logs
        print(f"Probabilities: {probabilities}")
        print(f"Confidence: {confidence}")
        print(f"Predicted Label: {predicted_label}")

        # Separate logic for "Not an Eggplant"
        if predicted_label == "Not an Eggplant":
            print("Prediction: Not an Eggplant")
            return {
                "prediction": "This is not an Eggplant",
                "confidence": confidence,
                "soil_moisture": soil_moisture,
                "recommendation": random.choice(RECOMMENDATIONS["Not an Eggplant"]["low" if soil_moisture < 40 else "high"])
            }

        # Handle low-confidence predictions for valid labels
        if confidence < 0.5:
            print("Low confidence for valid prediction.")
            return {
                "prediction": "This is not an Eggplant",
                "confidence": confidence,
                "soil_moisture": soil_moisture,
                "recommendation": "Make sure the eggplant is real and fully visible within the frame before capturing the image."
            }

        # Determine soil moisture condition
        moisture_condition = "low" if soil_moisture < 40 else "high"
        print(f"Moisture condition: {moisture_condition}")

        # Get recommendations based on prediction and soil moisture
        recommendation_list = RECOMMENDATIONS.get(predicted_label, {}).get(moisture_condition, [])
        recommendation = random.choice(recommendation_list) if recommendation_list else "No recommendation available."

        # Get causes and solutions for the predicted label
        causes = RECOMMENDATIONS.get(predicted_label, {}).get("causes", "No causes available.")
        solutions = RECOMMENDATIONS.get(predicted_label, {}).get("solutions", [])

        return {
            "prediction": predicted_label,
            "confidence": confidence,
            "soil_moisture": soil_moisture,
            "recommendation": recommendation,
            "causes": causes,
            "solutions": solutions
        }

    except Exception as e:
        # Handle unexpected errors gracefully
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "AgriShield API is running!"}

# Define the request payload structure
class SoilMoisture(BaseModel):
    moisture: int

# Endpoint to receive soil moisture data
@app.post("/soil-moisture")
async def receive_soil_moisture(soil_moisture: SoilMoisture):
    """
    Receive soil moisture data from an external source (e.g., ESP32).
    """
    print(f"Received soil moisture value: {soil_moisture.moisture}")
    # You can store or process the moisture value here
    return {"message": "Data received successfully", "moisture": soil_moisture.moisture}

# Endpoint to retrieve the latest soil moisture (optional)
@app.get("/soil-moisture")
async def get_soil_moisture():
    # Here, you would retrieve the latest value from your database or cache
    # For now, just return a dummy value
    return {"moisture": 55}  # Example moisture value

# ESP32 Configuration
ESP32_IP = "192.168.100.187"  # Replace with your ESP32's IP address
ESP32_SOIL_MOISTURE_ENDPOINT = f"http://{ESP32_IP}/soil-moisture"

@app.get("/fetch-soil-moisture")
async def fetch_soil_moisture():
    """
    Fetch soil moisture data from the ESP32.
    """
    try:
        # Make a GET request to the ESP32's soil moisture endpoint
        response = requests.get(ESP32_SOIL_MOISTURE_ENDPOINT, timeout=5)
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)

        # Parse the JSON response from the ESP32
        data = response.json()
        moisture = data.get("moisture")

        if moisture is None:
            raise HTTPException(status_code=500, detail="Invalid response from ESP32")

        return {"moisture": moisture}

    except requests.RequestException as e:
        # Handle connection errors or timeouts
        raise HTTPException(status_code=500, detail=f"Failed to fetch data from ESP32: {str(e)}")


@app.post("/soil-moisture")
async def receive_soil_moisture(soil_moisture: SoilMoisture):
    """
    Receive soil moisture data from an external source (e.g., ESP32).
    """
    print(f"Received soil moisture value: {soil_moisture.moisture}")
    # You can store or process the moisture value here
    return {"message": "Data received successfully", "moisture": soil_moisture.moisture}