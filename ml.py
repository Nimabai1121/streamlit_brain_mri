import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

model_path = "/Users/nimatashi/Desktop/ML/streamlit code/my_model.h5"
model = load_model(model_path)
model.save("our_model.h5")


def preprocess_input_data(input_data):
    
    return input_data

def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def make_prediction(input_data):
    preprocessed_data = preprocess_input_data(input_data)
    prediction = model.predict(preprocessed_data)
    
    # Check if the predicted probability for class 1 is between 0.5 and 1
    if 0.5 <= prediction[0, 1] <= 1:
        predicted_class = "Cancer"
    else:
        predicted_class = "fine"
    
    return predicted_class, prediction

def main():
    st.title("Brain MRI predictions")


    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
       
        input_image = load_and_preprocess_image(uploaded_file)

        if st.button("Make Prediction"):
            
            predicted_class, prediction_probs = make_prediction(input_image)

    
            st.success(f"Predicted Class: {predicted_class}")
            st.success(f"Raw Probabilities: {prediction_probs}")
    else:
        st.warning("Please upload an image.")
        
        
    st.markdown("Copyright © N.T.Tamang")
    
    st.markdown("No machine learning model is perfect, and there can be errors in their predictions.")
    st.markdown("Factors liketraining data quality and inherent uncertainties in the problem can affect the accuracy.")

if __name__ == "__main__":
    main()