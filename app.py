import streamlit as st
import pickle

# Function to load model and vectorizer with error handling
def load_resources():
    try:
        model = pickle.load(open('spam.pkl', 'rb'))
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()

# Load the model and vectorizer
model, vectorizer = load_resources()

# Custom CSS for Main UI and elements
st.markdown(
    """
    <style>
    /* Body styling */
    body {
        background: linear-gradient(135deg, #00bcd4, #8e44ad);
        color: white;
        font-family: 'Arial', sans-serif;
    }

    /* Main content styling */
    .main {
        border-radius: 20px;
        padding: 30px;
        margin-top: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        background: rgba(0, 0, 0, 0.5);
    }

    h1 {
        font-size: 3rem;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Roboto', sans-serif;
        color: #f1c40f;
    }

    h3 {
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        color: #ecf0f1;
    }

    .stTextArea textarea {
        font-size: 1.2rem;
        border-radius: 10px;
        padding: 1rem;
        border: none;
        width: 100%;
        margin-bottom: 1.5rem;
    }

    /* Button styling */
    .stButton button {
        background-color: #e74c3c;
        color: white;
        font-size: 1.2rem;
        border-radius: 10px;
        padding: 1rem 2rem;
        border: none;
        transition: 0.3s ease;
        cursor: pointer;
    }

    .stButton button:hover {
        background-color: #c0392b;
    }

    /* Result styling */
    .result {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        text-align: center;
    }

    /* Footer styling */
    footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 1rem;
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main App UI
st.title("Email Spam Detector")
st.markdown(
    """
    ### Welcome to the Spam Detection App!  
    This application uses **Machine Learning** to classify emails or SMS messages as **Spam** or **Not Spam**.
    """
)

# Sample messages
st.markdown("### 📜 **Sample Messages**")
st.markdown("#### 1. **Spam Message**")
st.markdown("""
    - **Subject:** You've won a lottery!
    - **Message:** Congratulations! You have been selected to win a huge prize. Please click here to claim your money now!
    """)

st.markdown("#### 2. **Not Spam Message**")
st.markdown("""
    - **Subject:** Meeting reminder for tomorrow
    - **Message:** Hello, just a reminder about the team meeting scheduled for tomorrow at 10 AM. Please be on time.
    """)

st.markdown("---")

# Input box for user
user_input = st.text_area("Enter the email or SMS content", height=200, placeholder="Type your content here...", key="input")

# Placeholder for result or error message
result_placeholder = st.empty()

# Classification logic
if st.button("Classify", use_container_width=True):
    if user_input.strip():  # Ensure input is valid
        try:
            # Preprocess and classify
            vectorized_data = vectorizer.transform([user_input]).toarray()
            result = model.predict(vectorized_data)

            # Display result
            if result[0] == 0:
                result_placeholder.markdown('<div class="result" style="color: #27ae60;">✅ The message is <strong>NOT SPAM</strong>.</div>', unsafe_allow_html=True)
            else:
                result_placeholder.markdown('<div class="result" style="color: #e74c3c;">⚠️ The message is <strong>SPAM</strong>.</div>', unsafe_allow_html=True)
        except Exception as e:
            result_placeholder.error(f"An error occurred: {e}")
    else:
        result_placeholder.warning("⚠️ Please enter a valid message!")

# Footer
st.markdown(
    """
    <footer>
        Made by [SANEE KUMAR]
        <a href="https://github.com" target="https://github.com/SaneeKumar1">GitHub</a> | <a href="https://linkedin.com" target="https://www.linkedin.com/in/saneekumar1/">LinkedIn</a>
    </footer>
    """,
    unsafe_allow_html=True,
)
