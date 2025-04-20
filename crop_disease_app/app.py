import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load labels
with open("labels.txt", "r") as f:
    labels = f.read().splitlines()


# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def predict(image):
    image = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_idx = np.argmax(output_data)
    confidence = np.max(output_data)
    return class_names[predicted_idx], confidence

# Streamlit UI
st.title("🌿 Crop Disease Detection")
uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner('Analyzing...'):
        label, confidence = predict(image)
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: {confidence:.2f}")
