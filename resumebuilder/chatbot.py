import streamlit as st
import time
import requests
import pandas as pd
import fitz
from docx import Document

#used for finding specifc words in user prompt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


#helper function

#outputs text char by char, makes it appear like the output is being typed out
def response_generator(response):
    for char in response:
        yield char
        time.sleep(0.02)  # Adjust speed here (0.02s = 50 chars/sec)

#used to clear the conversation whenever button is clicked
def reset_conversation():
    st.session_state.messages = []
    #need to add the clearing of uploaded file if button is clicked

#extracting text from files, currently only tested pdf and txt files, not sure if there are more types of files that can be added
def extract_text_from_file(file):
    name = file.name.lower()
    if name.endswith('.pdf'):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif name.endswith('.txt'):
        return file.read().decode("utf-8")
    elif name.endswith('.docx'):
        file.seek(0)
        document = Document(file)
        return "\n".join([para.text for para in document.paragraphs])
    elif name.endswith('.csv'):
        df = pd.read_csv(file)
        return df.to_string(index=False)
    elif name.endswith('.xlsx'):
        df = pd.read_excel(file)
        return df.to_string(index=False)
    else:
        #error message here
        return "unsupported file"
    
#prepare file input to be split into chunks based on max_words limit, used for large files, might need some refining and 500 is most definetly too small but just a placeholder for now while testing
def prepare_file_content(text, max_words=500):
    text.replace("\n\n", "\n").strip()
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]




st.title("Chatgpt Clone")

#reset button
st.button('Reset Chat', on_click=reset_conversation)

uploaded_files = st.file_uploader(
    "Upload up to 6 files", 
    type=["pdf", "txt", "docx", "csv", "xlsx"], 
    accept_multiple_files=True
)

#file display here, probably going to change to displaying as the actual file instead of a block of text
file_contents = []
if uploaded_files:
    for file in uploaded_files[:6]:
        content = extract_text_from_file(file)
        file_contents.append(content)
        st.text_area(f"File Content: {file.name}", content, height=200)


#initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#user input
if user_prompt := st.chat_input("How can I help you?"):
    with st.chat_message("user"):
        if file_contents:
            chunked_file_contents = []
            for content in file_contents:
                chunked_file_contents.extend(prepare_file_content(content))
            context = "\n\n".join(chunked_file_contents)
            st.session_state.messages.append({
                "role": "system",
                "content": f"The following documents are relevant context:\n\n{context}\n\n"
            })
        st.session_state.messages.append({
            "role": "user",
            "content": user_prompt
        })

        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        #adds processing... to response before it is output to show it is "thinking"
        placeholder = st.empty()
        placeholder.markdown("Processing...")
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        #change this if using a new model
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3",
                "messages": messages,
                "stream": False
            }
        )
        print(response.status_code)
        print(response.json())

        content = response.json()["message"]["content"]
        placeholder.markdown("")
        response_area =st.empty()
        response_area.write_stream(response_generator(content))
        st.session_state.messages.append({"role": "assistant", "content": content})