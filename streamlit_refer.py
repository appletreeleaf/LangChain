import streamlit as st
# count token
import tiktoken
# logging from web site by streamlit
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models.openai import ChatOpenAI

# document loaders (pdf, docs, ppt, url, etc...)
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.document_loaders.csv_loader import CSVLoader

# from langchain.d
# Text splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# VectorDB
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(
    page_title="DirChat",
    page_icon=":books:"
    )

    st.title("Private Data _:red[Private Assistant BOT]_ :books:")

    # 추후 session state를 변수처럼 활용하기 위해 먼저 선언해둠
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    # 사이드 바에 하위 기능을 포함하기 위함
    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    # Process 버튼 실행시 문서를 load-> split -> VectorDB 저장 -> chain 구동
    if process:
        if not openai_api_key:
            # api_key 미입력시 경고 문구 출력
            st.info("Please add your OpenAI API key to continue. If you don't have API key, please visit https://platform.openai.com/api-key")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key)

        st.session_state.processComplete = True
    # 초기 대화창 구성
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # 채팅 history
    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    ## 질문창 구성
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            # 앞서 정의한 retrival chain
            chain = st.session_state.conversation
            # 답변 생성하는 동안의 spinner 구성
            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    # result의 chat_history를 session_state에 저장하여
                    # LLM이 답변 생성시 history를 참조할 수 있도록 함
                    st.session_state.chat_history = result['chat_history']
                # 실제 답변
                response = result['answer']
                # 참조한 문서
                source_documents = result['source_documents']

                st.markdown(response)
                # expander: markdown내 접고 필 수 있는 기능
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)



# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# 토큰 수를 return
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()
        # elif '.csv' in doc.name:
        #     loader = CSVLoader(file_name, dele)
        #     documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    # 클라우드 내에서 일시적으로 존재하는 vectorDB
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True),
            # chat history를 저장할 buffer 선언
            # output_key = 'answer' 설정하여, 답변에 해당하는 부분만 history에 저장되도록 함
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()
