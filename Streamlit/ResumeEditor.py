import base64
import streamlit as st
import streamlit.components.v1 as components
import time
import requests
import time

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import tempfile
import shutil
import chardet

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document as LangchainDocument
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import  UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PDFPlumberLoader

import json
from PIL import Image



#overwrites the size of image to 64x64

#add new image here
# avatar_path = os.path.join(os.getcwd(), "images", "alqimi.png")

# Resize and overwrite the image to make it 64x64
# if os.path.exists(avatar_path):
#     img = Image.open(avatar_path)
#     img = img.resize((64, 64))
#     img.save(avatar_path)  # This overwrites the image

#global variables
full_response = ""

#html style for the chat messages
st.markdown("""
    <style>
    .fixed-reset-btn {
        position: fixed;
        top: 16px;
        left: 16px;
        z-index: 9999;
        background-color: white;
        padding: 0.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    </style>
    <div class="fixed-reset-btn" id="reset-btn-anchor">
    """, unsafe_allow_html=True)



#helper functions

def stream_response_lines_and_buffer(response):
    for line in response.iter_lines(decode_unicode=True):
        if line:
            try:
                chunk = json.loads(line)
                content_piece = chunk.get("message", {}).get("content")
                if content_piece:
                    yield content_piece
            except json.JSONDecodeError as e:
                print("Failed to parse:", line, e)

# Now: stream and buffer at the same time
def stream_and_buffer_to_ui(response):
    global full_response
    full_response = ""  # Reset full_response for each new response
    for piece in stream_response_lines_and_buffer(response):
        full_response += piece
        yield piece

#function to extract text from the uploaded file, currently only supports pdf, docx, txt
def load_document(file):
    ext = os.path.splitext(file.name)[-1].lower()

    file.seek(0)  # Reset file pointer to the beginning

    # Save the uploaded file to a temporary location
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name

    if ext == ".pdf":
        loader = PDFPlumberLoader(tmp_path)
        documents = loader.load()  #  list[Document]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    
    os.remove(tmp_path)  # Clean up the temporary file
    print("document loaded")
    return documents


#used to clear the conversation whenever button is clicked
def reset_conversation():
    st.session_state.messages = []

    # Clear uploaded file and FAISS state
    if "uploaded_file" in st.session_state:
        st.session_state["uploader_key"] += 1
    if "faiss_loaded" in st.session_state:
        del st.session_state.faiss_loaded
    if "db" in st.session_state:
        del st.session_state.db


if __name__ == "__main__":
    st.markdown("Solverah-Resume Editor", unsafe_allow_html=True)



    #used for removing the file when someone clicks the reset button
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 1

    #reset button
    st.button('Reset Chat', on_click=reset_conversation)
    st.markdown("</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a resume", type=["pdf"], key=st.session_state["uploader_key"])

    # Save uploaded file to session_state
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file


    #initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful and precise assistant."
                    "Always respond in English. "
                    "You are an expert in resume editing and job application processes. "
                    "Use the Resume and Job Description provided to edit resumes to include keywords and phrases from the job description. "
                    "Provide accurate, factual, and well-reasoned answers. "
                    "If you are unsure about something or do not have enough information, say so clearly. "
                    "Do not make up facts, names, numbers, or citations. "
                    "Use clear and concise language. "
                    "Always prefer being correct over sounding confident. "
                    "If applicable, cite sources or explain how the answer was derived."
                )
            }
        ]


    #display chat history
    # for message in st.session_state.messages:
    #     avatar = "👤" if message["role"] == "user" else avatar_path
    #     with st.chat_message(message["role"], avatar=avatar):
    #         st.markdown(message["content"])
   

    # ---- Always define user_prompt ----
    user_prompt = None

    # Always show the chat box
    chat_input_value = st.chat_input("Paste job description here")

    #user input
    if user_prompt:
        with st.chat_message("user",avatar="👤"):
            if uploaded_file:         
                
                resume_text=load_document(uploaded_file)

                prompt = (
                    f"Here is the resume:\n\n{resume_text}\n\n"
                    f"Based on this job description here:\n\n{user_prompt}"
                    f"\n\nPlease edit the resume to include keywords and phrases from the job description."
                    f"\n\n Do not change the format of the resume, just edit the content to include the keywords and phrases from the job description."
                    f"\n\nDo not make up any information, skills, or experiences. "
                    f"\n\nOnly edit the content of the resume to include the keywords and phrases from the job description if stating this skill is not a blatant lie."
                    f"\n\nIf the resume is already well-tailored to the job description, just say that it is already well-tailored."
                    f"\n\nIf the resume does not show any relevant skills or experiences for the job description, say that the resume does not show any relevant skills or experiences for the job description."
                )
            #if no file is uploaded, just use the user prompt
            else:
                prompt = user_prompt
            st.session_state.messages.append({
                "role": "user",
                "content": user_prompt,      
                "llm_content": prompt,
                "avatar": "👤"         
            })
            st.markdown(user_prompt)


       
        #with st.chat_message("assistant", avatar=avatar_path):
        with st.chat_message("assistant"):
            response_area = st.empty()
            #adds processing... to response before it is output to show it is "thinking"
            placeholder = st.empty()

            placeholder.markdown("""
                <div class="typing-indicator">
                <span></span><span></span><span></span>
                </div>

                <style>
                .typing-indicator {
                display: flex;
                gap: 6px;
                padding: 6px 10px;
                background-color: #f0f0f0;
                border-radius: 20px;
                width: fit-content;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                margin-left: 10px;
                align-items: center;
                }

                .typing-indicator span {
                width: 6px;
                height: 6px;
                background-color: #999;
                border-radius: 50%;
                animation: bounce 1.4s infinite ease-in-out both;
                }

                .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
                .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
                .typing-indicator span:nth-child(3) { animation-delay: 0; }

                @keyframes bounce {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1); }
                }
                </style>

                """, unsafe_allow_html=True)


            messages = [
                {"role": m["role"], "content": m.get("llm_content", m["content"])}
                for m in st.session_state.messages
            ]

            #change this if using a new model
            print("message sent to model")
            start_time = time.time()
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "llama3",
                    "messages": messages,
                    "stream": True
                },
                stream=True  
            )

            end_time = time.time()
            print(f"Response time: {end_time - start_time:.2f} seconds")
            print("response received from model")
            
            print(response.status_code)


            placeholder.markdown("")
            
            response_area.empty()
            # Write to UI while buffering

            response_area.write_stream(stream_and_buffer_to_ui(response))
            st.markdown('<div id="scroll-to-bottom"></div>', unsafe_allow_html=True)
            components.html(
                """
                <script>
                    const scrollAnchor = window.parent.document.getElementById("scroll-to-bottom");
                    if (scrollAnchor) {
                        scrollAnchor.scrollIntoView({ behavior: "smooth" });
                    }
                </script>
                """,
                height=0
            )

            # Store in chat history
            #st.session_state.messages.append({"role": "assistant", "content": full_response, "avatar": avatar_path})
            st.session_state.messages.append({"role": "assistant", "content": full_response})

