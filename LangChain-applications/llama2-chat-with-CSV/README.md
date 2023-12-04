# Conversational AI with Language Model Fine-Tuning and Retrieval

This repository contains a Python script showcasing a Conversational AI system built using language model fine-tuning, text embeddings, and retrieval techniques. The script enables answering queries based on a given dataset and user input.

## Overview

The script demonstrates the integration of various components:

- **Data Loading**: Utilizes `CSVLoader` from `langchain` to load a dataset ('loan_default.csv'). Modify `file_path` and `encoding` parameters in the `CSVLoader` instantiation for your dataset.

- **Text Preprocessing**: Splits text into manageable chunks using `RecursiveCharacterTextSplitter` for efficient processing.

- **Embeddings Generation**: Hugging Face's `HuggingFaceEmbeddings` converts text chunks into embeddings using `sentence-transformers/all-MiniLM-L6-v2`. Adjust the `model_name` parameter for different embeddings.

- **Vector Storage**: Employs FAISS (`FAISS`) to save embeddings into a knowledge base, saved locally for retrieval.

- **Language Model Fine-Tuning (LLM)**: Uses `CTransformers` with a fine-tuned language model ('models/llama-2-7b-chat.ggml'), customizable with parameters like `max_new_tokens` and `temperature`.

- **Conversational Retrieval Chain**: Connects the LLM with the retriever (`ConversationalRetrievalChain`) to generate conversational responses based on the retrieved information.

## Usage

1. **Setup**: Install required dependencies and packages using `pip install requirements.txt`

2. **Data Preparation**: Replace 'loan_default.csv' with your dataset and adjust loader parameters.

3. **Run the Script**: Execute the script to perform text splitting, embedding generation, knowledge base creation, LLM setup, and conversational interaction.

4. **Interactive Chat**: Upon running, input queries as prompted to engage in conversation with the AI system. Type 'exit' to end the conversation.
