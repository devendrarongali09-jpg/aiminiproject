import os
import os
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# 1. API Key directly configured
api_key = "your api key"

# 2. Page Configuration
st.set_page_config(
    page_title="Document QA Assistant",
    page_icon="📄",
    layout="wide"
)

# 3. Initialize Gemini Client
@st.cache_resource
def get_gemini_client():
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

client = get_gemini_client()

# 4. Helper Function: Extract Text from Uploaded File
def extract_text(uploaded_file) -> str:
    extracted_text = ""
    try:
        if uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    extracted_text += f"\n--- [Page {page_idx + 1}] ---\n" + page_text
        elif uploaded_file.name.endswith(".txt"):
            extracted_text = uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error parsing file: {e}")
    return extracted_text.strip()

# 5. Initialize Chat History & Document Context in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""

# 6. Sidebar: Document Upload & Metadata
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"],
        help="Upload lecture notes, research papers, or syllabus documents."
    )

    if uploaded_file is not None:
        if st.button("Process Document", type="primary") or not st.session_state.doc_text:
            with st.spinner("Extracting text..."):
                text = extract_text(uploaded_file)
                if text:
                    st.session_state.doc_text = text
                    st.session_state.messages = []
                    st.success(f"Loaded: {uploaded_file.name}")
                    st.caption(f"Estimated length: {len(text.split())} words")
                else:
                    st.warning("No selectable text found in the file.")

    if st.session_state.doc_text:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# 7. Main Interface
st.title("📄 Grounded Document QA")
st.caption("Ask questions strictly based on your uploaded document with verifiable quotes.")

if not api_key:
    st.error("API Key missing. Please set your API key in the script.")
    st.stop()

if not st.session_state.doc_text:
    st.info("👈 Please upload and process a document in the sidebar to begin.")
    st.stop()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if user_prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    system_prompt = (
        "You are an academic research assistant specializing in accurate document question-answering.\n"
        "Strict rules:\n"
        "1. Answer using ONLY information found in the PROVIDED DOCUMENT.\n"
        "2. If the document does not contain the answer, explicitly state: "
        "'The provided document does not contain sufficient information to answer this question.'\n"
        "3. Always ground your key claims by citing direct quotes or page numbers when available.\n"
        "4. Be concise, direct, and structured."
    )

    full_context_prompt = (
        f"### PROVIDED DOCUMENT:\n{st.session_state.doc_text}\n\n"
        f"### USER QUESTION:\n{user_prompt}"
    )

    # Generate Model Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing document..."):
            try:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=full_context_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.2,
                    ),
                )
                assistant_reply = response.text
                st.markdown(assistant_reply)
                st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
            except Exception as err:
                error_msg = f"API Error: {err}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})