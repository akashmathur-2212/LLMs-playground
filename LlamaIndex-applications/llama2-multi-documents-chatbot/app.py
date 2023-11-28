import streamlit as st
from dotenv import load_dotenv
import torch
import sys
import os
from transformers import BitsAndBytesConfig

# llama_index
from langchain.embeddings import HuggingFaceInstructEmbeddings
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, ServiceContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.text_splitter import SentenceSplitter
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM

# chromadb
import chromadb

# gpu
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


######################### Data Connectors #########################
def load_text_and_get_chunks(path_to_pdfs):
    documents = SimpleDirectoryReader(path_to_pdfs).load_data()
    
    return documents

######################### Models #########################
def load_llm():
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # load the model with quantized features
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    )

    llm = HuggingFaceLLM(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        query_wrapper_prompt=PromptTemplate("<s> [INST] {query_str} [/INST] "),
        context_window=3900,
        model_kwargs={"token": hf_token, "quantization_config": quantization_config},
        tokenizer_kwargs={"token": hf_token},
        device_map="auto",
    )

    return llm

def load_embeddings():
        
    embed_model = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
        )
    
    return embed_model

######################### Service Context #########################
def setting_the_service_context(llm, embed_model):
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20) # using the default chunk_# values as they work just fine

    # set context window
    context_window = 4096
    # set number of output tokens
    num_output = 256

    service_context = ServiceContext.from_defaults(llm=llm,
                                                embed_model=embed_model,
                                                text_splitter=text_splitter,
                                                context_window=context_window,
                                                num_output=num_output,
                                                )
    
    return service_context

######################### Storage AND Indexing #########################
def setup_vector_database_and_create_vector_index(documents, service_context, collection_name):
    # # To make ephemeral client that is a short lasting client or an in-memory client
    # db = chromadb.EphemeralClient()

    # initialize client, setting path to save data
    db = chromadb.PersistentClient(path="./chroma_db")

    # create collection
    chroma_collection = db.get_or_create_collection(collection_name)

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # create your index
    vector_index = VectorStoreIndex.from_documents(documents=documents,
                                                   storage_context=storage_context,
                                                   service_context=service_context
                                                   )
    
    return vector_index

######################### Query Engine #########################

def setup_retriver_query_engine(index, top_k, similarity_cutoff, exclude_keywords):
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k
        )

    # configure node postprocessors
    s_processor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
    k_processor = KeywordNodePostprocessor(
        exclude_keywords=exclude_keywords
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[s_processor, k_processor],
    )

    return query_engine

######################### Chat Engine #########################

def chat_engine_response(index, prompt_input):
    chat_engine = index.as_chat_engine(chat_mode="condense_question")
    response = chat_engine.chat(prompt_input).response

    return response

 
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):

                # get documents
                documents = load_text_and_get_chunks(path_to_pdfs="./sample_data/pdfs")
                
                # get llm
                llm = load_llm()
                
                # get embeddings
                embed_model = load_embeddings()

                # create service context
                service_context = setting_the_service_context(llm=llm, embed_model=embed_model)
                
                # create vector store and index
                vector_index = setup_vector_database_and_create_vector_index(documents=documents,
                                                              service_context=service_context,
                                                              collection_name="bank-earnings-database")
                
                # # create query engine
                # st.session_state.conversation = setup_retriver_query_engine(index=vector_index,
                #                                                             top_k=4,
                #                                                             similarity_cutoff=0.8,
                #                                                             exclude_keywords=[])
                
                # create chat engine
                if "messages" not in st.session_state.keys():
                    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])

                if prompt := st.chat_input():
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.write(prompt)


                if st.session_state.messages[-1]["role"] != "assistant":
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking ... "):
                            response = chat_engine_response(index=vector_index, prompt_input=prompt)
                            st.write(response)
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)


                
if __name__ == '__main__':
    main()