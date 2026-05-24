import time
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import chat_session
from typing import List, Dict
from transformers import AutoTokenizer

# Change the following variables as needed
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
IP1 = "127.0.0.1"
PORT1 = 8000

@st.cache_resource
def get_tokenizer():
    global MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


tokenizer = get_tokenizer()


@st.cache_data
def read_chunks(file_folder) -> Dict[str, str]:
    """
    Read all the txt files in the folder and return the filenames
    """
    filenames = os.listdir(file_folder)
    ret = {}
    for filename in filenames:
        if not filename.endswith("txt"):
            continue
        key = filename.removesuffix(".txt")
        with open(os.path.join(file_folder, filename), "r") as fin:
            value = fin.read()
        ret[key] = value

    return ret

chunks = read_chunks("data/")
selected_chunks = st.multiselect(
    "Select the chunks into the context",
    list(chunks.keys()),
    default = [],
    placeholder = "Select in the drop-down menu")
contexts = [chunks[key] for key in selected_chunks]

container = st.container(border=True)

with st.sidebar:
    system_prompt = st.text_area(
            "System prompt:",
            "You are a helpful assistant. I will now give you a document and "
            "please answer my question afterwards based on the content in document"
        )

    session = chat_session.ChatSession(IP1,PORT1)
    session.set_context([system_prompt] + contexts)

    num_tokens = tokenizer.encode(session.get_context())
    container.header(f"The context given to LLM: ({len(num_tokens)} tokens)", divider = "grey")
    container.text(session.get_context())

    messages = st.container(height=300)
    messages.markdown("*vLLM instance 1*")
    if prompt := st.chat_input("Type the question here", key=1):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write_stream(session.chat(prompt))
