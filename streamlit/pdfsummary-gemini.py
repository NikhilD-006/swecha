import streamlit as st
import google.generativeai as genai
import time

# --- Configuration ---
# Configure the Gemini API key from Streamlit's secrets
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("Your Google API Key is not set! Please add it to your Streamlit secrets.")
    st.stop()

# --- Gemini API Function ---

def get_gemini_response_for_pdf(pdf_content: bytes, prompt: str) -> str:
    """
    Sends the PDF content to the Gemini API and gets a text response.
    """
    # Use a model that supports PDF inputs, like gemini-1.5-flash
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    # The API directly accepts the PDF bytes
    pdf_file_part = {"mime_type": "application/pdf", "data": pdf_content}
    
    try:
        # The model can take a list of parts, including the prompt and the file
        response = model.generate_content([prompt, pdf_file_part])
        return response.text
    except Exception as e:
        return f"An error occurred while calling the Gemini API: {e}"

# --- Streamlit App UI ---

st.set_page_config(page_title="PDF Teacher (Gemini API)", page_icon="🧑‍🏫", layout="wide")
st.title("🧑‍🏫 PDF Teacher powered by Google Gemini")
st.markdown(
    "Upload a PDF, and Gemini will analyze the entire document to provide a page-by-page summary and explanation. This is much faster and simpler than processing each page as an image."
)

# Define the detailed prompt for the model
teacher_prompt = """
You are an expert teacher with a goal to make the content of the provided PDF document easy to understand for a beginner.
Your task is to analyze the document page by page and provide a detailed explanation for each.

Follow these instructions carefully:
1.  Go through the document sequentially, from the first page to the last.
2.  For each page, create a clear heading like "--- Page X ---".
3.  Under each page's heading, provide a clear and simple explanation of the text on that page.
4.  If a page contains diagrams, charts, or images, describe them in detail and explain how they relate to the text.
5.  Use analogies or real-world examples to make complex concepts more relatable.
6.  At the end of your entire analysis, provide a final, concise summary of the key takeaways from the whole document.

Your output should be well-structured, easy to read, and cover every piece of information in the PDF.
"""

uploaded_file = st.file_uploader("Choose a PDF file (up to 1000 pages)", type="pdf")

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    st.success(f"File uploaded: **{uploaded_file.name}**")

    if st.button("Start Analysis with Gemini"):
        with st.spinner("Gemini is reading and analyzing your PDF... This may take a few moments."):
            start_time = time.time()
            explanation = get_gemini_response_for_pdf(pdf_bytes, teacher_prompt)
            end_time = time.time()
            
            st.info(f"Analysis completed in {end_time - start_time:.2f} seconds.")
            st.markdown("---")
            st.subheader("Gemini's Teaching Output:")
            st.markdown(explanation)
            
            # Add a download button for the text output
            st.download_button(
                label="📥 Download Explanation",
                data=explanation,
                file_name=f"{uploaded_file.name}_explanation.txt",
                mime="text/plain"
            )
