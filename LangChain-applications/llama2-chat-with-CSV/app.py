from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys
import streamlit as st
from streamlit_chat import message
import tempfile

def main():
  st.title("☔ ☔Chat with CSV using Llama2 ☔ ☔")
  st.markdown("Built by ♻️ CSVQConnect ♻️ ", unsafe_allow_html=True)
  # Your background image URL goes here
  #background_image_url = 'https://www.bing.com/images/search?view=detailV2&ccid=lFAWXtbv&id=BB57AC3541361FF3844CAA706B667014CB515B92&thid=OIP.lFAWXtbvpchf66BryfJQ1QHaE8&mediaurl=https%3a%2f%2fimage.freepik.com%2ffree-photo%2ftwo-llamas-andean-highland-bolivia_107467-2006.jpg&exph=418&expw=626&q=llama2+image&simid=608011097331751957&FORM=IRPRST&ck=F66D65F1AFAAA4BBCF9986ADF8ED1643&selectedIndex=4'
  background_image_path = 'llama-image.jpg'
  # Set the background image and color

  uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

  if uploaded_file :
    #use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)

    llm = CTransformers(model='TheBloke/Llama-2-7B-Chat-GGML',
                    model_file='llama-2-7b-chat.ggmlv3.q8_0.bin',
                    max_new_tokens=512,
                    temperature=0.1,
                    gpu_layers=50)
    
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! What is your query about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):

            user_input = st.text_input("Query:", placeholder="Search answer from your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

if __name__ == '__main__':
	main()
