
    import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')

# Streamlit app
st.title('Mental Health Classifier')

# Text input
input_text = st.text_area("Enter your text here:")

# Button to make prediction
if st.button("Classify"):
    if input_text.strip() == "":
        st.write("Please enter some text to classify.")
    else:
        # Preprocess and vectorize the input
        input_vectorized = vectorizer.transform([input_text])

        # Make prediction
        prediction = model.predict(input_vectorized)

        # Output the result
        st.write(f"The predicted mental health issue is: {prediction[0]}")

    