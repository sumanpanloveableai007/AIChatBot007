import os
import streamlit as st
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load API keys from .env file
load_dotenv()

# Set up Streamlit Page Configuration
st.set_page_config(page_title="Gemini Vision Describer", page_icon="📸", layout="centered")
st.title("📸 Gemini 2.5 Flash Image Describer")
st.caption("Upload up to 3 images to get a concise, 50-word description for each.")

# Initialize the Gemini 2.5 Flash model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# File uploader restricted to a maximum of 3 images
uploaded_files = st.file_uploader(
    "Choose images (Max 3)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

# Enforce the 3-image limit visually
if len(uploaded_files) > 3:
    st.error("Please upload a maximum of 3 images at a time.")
    uploaded_files = uploaded_files[:3]

# Action button to trigger processing
if uploaded_files and st.button("Generate Descriptions"):
    
    # Process each uploaded file sequentially
    for index, file in enumerate(uploaded_files):
        st.write("---")
        
        # Create side-by-side columns: image on left (size 1), description on right (size 2)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Streamlit displays raw uploaded file handles cleanly without PIL initialization overhead
            st.image(file, caption=f"Image {index + 1}: {file.name}", use_container_width=True)
            
        with col2:
            with st.spinner(f"Analysing image {index + 1}..."):
                try:
                    # 1. Get raw bytes and convert them to a base64 encoded string
                    image_bytes = file.getvalue()
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # 2. Extract the file extension explicitly forcing it to string type
                    filename_str = str(file.name)
                    file_ext = filename_str.split('.')[-1].lower()
                    mime_type = "image/png" if file_ext == "png" else "image/jpeg"
                    
                    # 3. Construct the clean multimodal content block required by ChatGoogleGenerativeAI
                    raw_message = HumanMessage(
                        content=[
                            {
                                "type": "text", 
                                "text": "Describe the contents of this image. Keep your answer strictly under 50 words."
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:{mime_type};base64,{base64_image}"
                            }
                        ]
                    )
                    
                    # 4. Query the Gemini 2.5 model
                    response = llm.invoke([raw_message])
                    
                    # Display the generated description
                    st.success("**Description:**")
                    st.write(response.content)
                    
                    # Word count utility
                    word_count = len(response.content.split())
                    st.caption(f"*Word Count: {word_count} words*")
                    
                except Exception as e:
                    st.error(f"Error processing image {index + 1}: {str(e)}")


# python -m streamlit run vision_app.py
