import streamlit as st
import os
import pickle
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_llm import HuggingFaceLLM
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from datetime import date
from transformers import AutoModelForCausalLM, AutoTokenizer

# Environment variables
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# os.environ['LANGCHAIN_API_KEY'] = os.environ.get('LANGCHAIN_API_KEY')


# Load model and tokenizer
# @st.cache_resource
# def load_model():
#     model_name = "allenai/OLMo-7B-Instruct"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     return model, tokenizer

# model, tokenizer = load_model()
def load_from_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)
def load_retriever(docstore_path,chroma_path,embeddings,child_splitter,parent_splitter):
    """Loads the vector store and document store, initializing the retriever."""
    db3 = Chroma(collection_name="full_documents", #collection_name shoud be the same as in the first time
                     embedding_function=embeddings,
                     persist_directory=chroma_path
    )
    store_dict = load_from_pickle(docstore_path)

    store = InMemoryStore()
    store.mset(list(store_dict.items()))

    retriever = ParentDocumentRetriever(
        vectorstore=db3,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 5}
    )
    return retriever
def inspect(state):
    if "context_sources" not in st.session_state:
        st.session_state.context_sources = []
    context = state['normal_context']
    st.session_state.context_sources =[doc.metadata['source']  for doc in context]
    st.session_state.context_content = [doc.page_content for doc in context]
    return state
def retrieve_normal_context(retriever, question):
    docs = retriever.invoke(question)
    return docs

# Your OLMOLLM class implementation here (adapted for the Hugging Face model)

@st.cache_resource
def get_chain(temperature):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    
    docstore_path = 'repos_github_opensmodel.pcl'
    chroma_path   = 'repos_github_opensmodel'
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                    chunk_overlap=500)

    # create the child documents - The small chunks
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300,
                                                    chunk_overlap=50)
    retriever = load_retriever(docstore_path,chroma_path,embeddings,child_splitter,parent_splitter)
    
    # Replace the local OLMOLLM with the Hugging Face model
    llm = HuggingFaceLLM(
            model_id="EleutherAI/gpt-neo-1.3B",  # or another suitable model
            temperature=temperature,
            max_tokens=256
        )    
    
    today = date.today()
    # Response prompt 
    response_prompt_template = """You are an assistant who helps Ocean Hack Week community to answer their questions. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.
    Keep track of chat history: {chat_history}
    Today's date: {date}
    ## Normal Context:
    {normal_context}
 
    # Original Question: {question}

    # Answer (embed links where relevant):
    
    """
    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)
    context_chain = RunnableLambda(lambda x: {
    "question": x["question"],
    "normal_context": retrieve_normal_context(retriever,x["question"]),
    # "step_back_context": retrieve_step_back_context(retriever,generate_queries_step_back.invoke({"question": x["question"]})),
    "chat_history": x["chat_history"],
    "date": today})
    chain = (
        context_chain
        | RunnableLambda(inspect)
        | response_prompt
        | llm
        | StrOutputParser()
    )
    return chain

def clear_chat_history():
    st.session_state.messages = []
    st.session_state.context_sources = []
    st.session_state.key = 0

st.set_page_config(page_title='OHW AI')

# Sidebar
with st.sidebar:
    st.title("OHW Assistant")
    temperature = st.slider("Temperature: ", 0.0, 1.0, 0.5, 0.1)
    chain = get_chain(temperature)
    st.button('Clear Chat History', on_click=clear_chat_history)

# Main app
if "messages" not in st.session_state:
    st.session_state.messages = []


for q, message in enumerate(st.session_state.messages):
    if (message["role"] == 'assistant'):
        with st.chat_message(message["role"]):
            tab1, tab2 = st.tabs(["Answer", "Sources"])
            with tab1:
                st.markdown(message["content"])
    
            with tab2:
                for i, source in enumerate(message["sources"]):
                    name = f'{source}'
                    with st.expander(name):
                        st.markdown(f'{message["context"][i]}')
            
    else:
        question = message["content"]
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if prompt := st.chat_input("How may I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        query=st.session_state.messages[-1]['content']
        tab1, tab2 = st.tabs(["Answer", "Sources"])
        with tab1:
            placeholder = st.empty()  # Create a placeholder in Streamlit
            full_answer = ""
            for chunk in chain.stream({"question": query, "chat_history":st.session_state.messages}):
                
                full_answer += chunk
                placeholder.markdown(full_answer,unsafe_allow_html=True)
            
        with tab2:
            for i, source in enumerate(st.session_state.context_sources):
                name = f'{source}'
                with st.expander(name):
                    st.markdown(f'{st.session_state.context_content[i]}')




    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    st.session_state.messages[-1]['sources'] = st.session_state.context_sources
    st.session_state.messages[-1]['context'] = st.session_state.context_content
