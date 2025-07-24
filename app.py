import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the CNN model
cnn = tf.keras.models.load_model('cnn_model_pneumonia_detection.keras') ### YOUR CODE HERE

# Create a title
### YOUR CODE HERE
st.title("Pneumonia or Not?")

# Function to process and classify the uploaded image
def process_image(image):
    # Convert the image to RGB model
    image = image.convert('RGB')

    # Resize the image
    image = image.resize((64, 64))

    # Convert the image to an array and normalize
    img_array = np.array(image).astype('float32')
    img_array = img_array / 255.0

    # Expand dimensions to match the expected input shape
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Display the image upload widget using st.file_uploader, set it to `uploaded_file`
### YOUR CODE HERE
uploaded_file = st.file_uploader('Upload Image Here')

# Perform prediction and display the result
if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)

    # Process the image using the `process_image()` function from above!
    processed_image = process_image(image)### YOUR CODE HERE

    # Get the prediction from your model and convert to a diagnosis
    prediction = cnn.predict(processed_image)[0] ### YOUR CODE HERE
    st.write(prediction)
    pred_diagnosis = "Pneumonia" if prediction == 1 else "Not Pneumonia" ### YOUR CODE HERE

    # Display the processed image
    ### YOUR CODE HERE
    st.image(processed_image)

    # Display the predicted diagnosis
    st.header("Prediction")
    st.subheader("Diagnosis")
    st.write(pred_diagnosis)
