import streamlit as st
import requests
from pdf2image import convert_from_bytes
from PIL import Image
import base64
import ollama
from io import BytesIO
import json

# --- Configuration ---
PDF_IMAGE_DPI = 200  # Lower DPI for faster conversion and smaller images
JPEG_QUALITY = 85    # Reduce JPEG quality for smaller file size

# --- Core Functions ---

def is_ollama_running(base_url="http://localhost:11434"):
    """Checks if the Ollama server is running and accessible."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False

def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a Base64 encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buffered.getvalue()).decode()

def gemma3_teacher_page_prompt_cpu() -> str:
    """
    A simplified prompt optimized for CPU execution to reduce generation time.
    """
    return (
        "You are an efficient assistant. Analyze the provided image of a document page. "
        "Provide a concise, bullet-point summary of the key takeaways. "
        "Focus only on the most critical information."
    )

def analyze_page_with_gemma3(image: Image.Image) -> str:
    """
    Sends a page image to the Gemma 3 model for analysis and returns the explanation.
    Uses a CPU-friendly prompt.
    """
    prompt = gemma3_teacher_page_prompt_cpu()
    img_b64 = image_to_base64(image)
    try:
        response = ollama.chat(
            model='gemma3:4b-it-qat',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [img_b64]
            }]
        )
        return response['message']['content']
    except Exception as e:
        st.error(f"An error occurred while communicating with Ollama: {e}")
        return "Could not get a response from the model. It may have timed out due to high CPU load."

# --- Streamlit App UI ---

st.set_page_config(page_title="🧑‍🏫 PDF Page Teacher (Gemma 3) - CPU Optimized", page_icon="🧑‍🏫", layout="wide")
st.title("🧑‍🏫 PDF Page Teacher (Gemma 3) - CPU Optimized")
st.markdown(
    "Upload a PDF. Each page will be converted to an image and analyzed sequentially by `gemma3:4b-it-qat` with a simplified prompt optimized for CPU usage."
)

# Check for Ollama connection before proceeding
if not is_ollama_running():
    st.error(
        "**Cannot connect to Ollama.**\n\n"
        "Please make sure the Ollama application is running and accessible at `http://localhost:11434`.\n\n"
        "If you are using the command line, run `ollama serve`."
    )
    st.stop()
else:
    st.success("Successfully connected to Ollama. Ready to process your PDF.")


uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    st.success(f"File uploaded: **{uploaded_file.name}**")

    try:
        with st.spinner("Converting PDF pages to images... This may take a moment for large files."):
            page_images = convert_from_bytes(pdf_bytes, dpi=PDF_IMAGE_DPI)
    except Exception as e:
        st.error(f"Failed to convert PDF to images. Please ensure you have Poppler installed. Error: {e}")
        st.stop()

    total_pages = len(page_images)
    st.info(f"Found {total_pages} page(s). Starting sequential analysis on CPU. This may take several minutes per page. Please be patient.")

    results = []

    for idx, img in enumerate(page_images):
        st.markdown("---")
        st.subheader(f"Page {idx + 1} of {total_pages}")

        cols = st.columns([1, 2])
        with cols[0]:
            st.image(img, caption=f"Image of Page {idx + 1}", use_column_width=True)

        with cols[1]:
            with st.spinner(f"Analyzing page {idx + 1} on CPU... Please be patient."):
                explanation = analyze_page_with_gemma3(img)
            
            st.markdown("**Gemma 3 Teaching Output:**")
            st.write(explanation)
            
            results.append({
                "page_number": idx + 1,
                "gemma3_teaching_output": explanation
            })

    st.markdown("---")
    st.success("All pages have been analyzed!")

    json_string = json.dumps(results, indent=2)
    
    st.download_button(
        label="📥 Download Explanations (JSON)",
        data=json_string,
        file_name=f"{uploaded_file.name}_page_teaching_cpu.json",
        mime="application/json"
    )
