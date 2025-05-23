import streamlit as st
import os
import json
import tempfile
import subprocess
import shutil
import re


from pydub import AudioSegment
import yt_dlp

import google.generativeai as genai
from google.api_core.client_options import ClientOptions

# Load environment variables (for local testing)


# --- Configuration ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set it in Streamlit secrets (as 'GEMINI_API_KEY') or as an environment variable.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini API. Check your API key: {e}")
    st.stop()

SAMPLE_RATE = 16000

if 'temp_file_paths' not in st.session_state:
    st.session_state.temp_file_paths = []
if 'temp_dirs_to_clean' not in st.session_state:
    st.session_state.temp_dirs_to_clean = []

def download_and_extract_audio_from_url(url):
    st.info(f"Attempting to download audio from: {url}")
    temp_dir = tempfile.mkdtemp()
    st.session_state.temp_dirs_to_clean.append(temp_dir)
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, 'yt_dlp_audio_%(id)s.%(ext)s'),
            'noplaylist': True, 'quiet': True, 'no_warnings': True, 'retries': 3,
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
            'postprocessor_args': ['-ac', '1', '-ar', str(SAMPLE_RATE)],
            'final_ext': 'wav'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            base_filename = os.path.splitext(ydl.prepare_filename(info_dict))[0]
            temp_downloaded_audio_path = base_filename + '.wav'
            if not os.path.exists(temp_downloaded_audio_path) or os.stat(temp_downloaded_audio_path).st_size == 0:
                st.error("Error: yt-dlp failed to download and convert audio to WAV, or the file is empty.")
                return None
            st.success(f"Audio downloaded and processed to: {os.path.basename(temp_downloaded_audio_path)}")
            st.session_state.temp_file_paths.append(temp_downloaded_audio_path)
            return temp_downloaded_audio_path
    except yt_dlp.utils.DownloadError as e:
        st.error(f"yt-dlp download error: {e}. This might be due to video privacy, geo-restrictions, or an invalid URL.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during audio download/processing: {e}")
        st.warning("Ensure `ffmpeg` is installed and available in your system's PATH for yt-dlp's audio extraction.")
        return None

def convert_to_wav(uploaded_file):
    temp_input_path = None
    temp_wav_path = None
    try:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
            temp_input.write(uploaded_file.read())
            temp_input_path = temp_input.name
        st.session_state.temp_file_paths.append(temp_input_path)
        
        st.info(f"Converting '{uploaded_file.name}' to WAV (mono, {SAMPLE_RATE}Hz)...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
            temp_wav_path = temp_output.name
        st.session_state.temp_file_paths.append(temp_wav_path)

        audio = AudioSegment.from_file(temp_input_path)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        audio.export(temp_wav_path, format="wav")
        
        st.success("File converted to WAV successfully.")
        return temp_wav_path
    except Exception as e:
        st.error(f"Error converting file to WAV: {e}")
        st.warning("Ensure `ffmpeg` is installed and available in your system's PATH.")
        return None

def analyze_with_gemini_direct_audio(audio_file_path):
    if not audio_file_path or not os.path.exists(audio_file_path):
        st.error("Audio file path is invalid for Gemini analysis.")
        return None

    try:
        # Use genai.upload_file() for models that require file IDs, e.g., gemini-1.5-flash-latest
        # For gemini-1.5-flash (older versions), genai.Part.from_data directly might be sufficient
        # If using gemini-2.0-flash as you have it, genai.upload_file is the correct way for larger audio.
        file = genai.upload_file(audio_file_path)
        model = genai.GenerativeModel('gemini-2.0-flash') # Changed to 1.5-flash as 2.0-flash might not be generally available or has different access patterns

        prompt=f"""
            Analyze the provided audio. Your response MUST be a JSON object with the following keys:
            "accent_prediction": string (e.g., "American English", "British English", "Indian English", "australian english" ..etc )
            "confidence": string (e.g., "75%", "0%", always a percentage string)
            "summary": string (a concise summary of the main content, 2-4 sentences)

            Example JSON structure:
            ```json
            {{
              "accent_prediction": "American English",
              "confidence": "85%",
              "summary": "This audio contains a brief discussion about climate change and its impacts."
            }}
            ```
            Do not include any other text or formatting outside of the JSON object.
            """
        contents=[prompt,file]

        st.info("Sending audio directly to Gemini API for analysis... (This may take a moment)")

        response = model.generate_content(contents)
        full_response_text = response.text.strip()

        if not full_response_text:
            st.warning("Gemini returned an empty response.")
            return None

        st.success("Gemini analysis complete.")
        return full_response_text

    except Exception as e:
        st.error(f"Error calling Gemini API with audio: {e}")
        st.warning("Ensure your Gemini API key is correct, the model is available, and the audio format/size is supported.")
        st.exception(e)
        return None

def parse_gemini_output(gemini_output):
    parsed_data = {
        "accent_prediction": "N/A",
        "confidence": "N/A",
        "summary": "N/A"
    }
    try:
        json_match = re.search(r"```json\n({.*?})\n```", gemini_output, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
        else:
            json_string = gemini_output

        data = json.loads(json_string)
        parsed_data["accent_prediction"] = data.get("accent_prediction", "N/A")
        parsed_data["confidence"] = data.get("confidence", "N/A")
        parsed_data["summary"] = data.get("summary", "N/A")
    except json.JSONDecodeError as e:
        st.error(f"Error parsing Gemini's JSON output: {e}")
        st.code(gemini_output)
        parsed_data["summary"] = f"JSON Parsing Error: {e}. Raw output: {gemini_output}"
    except Exception as e:
        st.error(f"An unexpected error occurred during output parsing: {e}")
        st.code(gemini_output)
        parsed_data["summary"] = f"Parsing Error: {e}. Raw output: {gemini_output}"
    return parsed_data

def cleanup_temp_resources():
    for f_path in st.session_state.temp_file_paths:
        if os.path.exists(f_path):
            try: os.remove(f_path)
            except Exception: pass
    st.session_state.temp_file_paths = []
    for d_path in st.session_state.temp_dirs_to_clean:
        if os.path.exists(d_path):
            try: shutil.rmtree(d_path)
            except Exception: pass
    st.session_state.temp_dirs_to_clean = []

def main():
    st.set_page_config(page_title="Gemini-Powered Audio Analyzer", layout="centered")

    st.title("🗣️ Gemini-Powered Audio Analyzer")
    st.markdown("""
        Upload an audio/video file or provide a mp4/Loom URL.
        The app will convert the audio and then send it directly to **Gemini 2.0 Flash**
        to infer the speaker's accent and provide a concise summary of the content.
        """)

    st.header("1. Enter Audio Source")
    audio_source_option = st.radio(
        "Choose audio source:",
        ("Upload Audio/Video File", "mp4/Loom URL"),
        index=0
    )

    # Initialize variables for file/URL input, but don't process yet
    uploaded_file = None
    video_url = ""

    if audio_source_option == "Upload Audio/Video File":
        uploaded_file = st.file_uploader(
            "Upload an audio or video file (e.g., .wav, .mp3, .mp4, .mov)",
            type=["wav", "mp3", "flac", "ogg", "m4a", "mp4", "mov", "avi", "webm"]
        )
    else: # YouTube/Loom URL
        video_url = st.text_input("Enter YouTube or Loom URL:")


    st.header("2. Analyze with Gemini")
    if st.button("Analyze Audio with Gemini", type="primary", use_container_width=True):
        processed_audio_path = None # Reset for this specific analysis click

        if audio_source_option == "Upload Audio/Video File":
            if uploaded_file is None:
                st.error("Please upload an audio/video file first before clicking Analyze.")
                return
            with st.spinner("Processing uploaded file..."):
                processed_audio_path = convert_to_wav(uploaded_file)
            
        else: # YouTube/Loom URL
            if not video_url:
                st.error("Please enter a mp4 or Loom URL first before clicking Analyze.")
                return
            with st.spinner("Downloading audio from URL... This might take a moment."):
                processed_audio_path = download_and_extract_audio_from_url(video_url)
        
        # Check if we successfully got a processed audio file before proceeding to analysis
        if processed_audio_path is None or not os.path.exists(processed_audio_path) or os.stat(processed_audio_path).st_size == 0:
            st.error("Failed to prepare audio for analysis. Please try a different file or URL.")
            return

        # Display the audio player ONLY after successful processing/download
        st.success(f"Audio ready for analysis: {os.path.basename(processed_audio_path)}")
        try:
            st.audio(processed_audio_path, format='audio/wav')
        except Exception as e:
            st.warning(f"Could not display audio player for the prepared audio: {e}")


        # --- Proceed with Gemini Analysis ---
        try:
            gemini_raw_output = analyze_with_gemini_direct_audio(processed_audio_path)
            
            if gemini_raw_output:
                st.subheader("Gemini Analysis Results:")
                
                parsed_results = parse_gemini_output(gemini_raw_output)

                st.markdown(f"**Predicted Accent:** `{parsed_results['accent_prediction']}`")
                st.markdown(f"**Confidence:** `{parsed_results['confidence']}`")
                st.markdown(f"**Summary:** {parsed_results['summary']}")

                with st.expander("Show Raw Gemini Output"):
                    st.code(gemini_raw_output)

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.exception(e)
        finally:
            cleanup_temp_resources()

def clean_on_rerun():
    cleanup_temp_resources()

if __name__ == "__main__":
    st.session_state.on_rerun = clean_on_rerun
    main()