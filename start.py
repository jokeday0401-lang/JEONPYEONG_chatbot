## streamlit 관련 모듈 불러오기
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
import fitz  # PyMuPDF
import re


############################### 🔑 환경변수 (Streamlit Cloud Secrets 자동 로드) ##########################
import streamlit as st
import os

# Streamlit Cloud에서 Secrets에 등록된 값 자동 불러오기
# 👉 Streamlit Cloud에서 [Settings → Secrets] 메뉴에 다음처럼 등록
# OPENAI_API_KEY = sk-xxxxxxx

openai_api_key = st.secrets["OPENAI_API_KEY"]

if not openai_api_key:
    st.error("⚠️ Streamlit Secrets에 OPENAI_API_KEY가 설정되지 않았습니다.")
    st.stop()

# LangChain과 OpenAI 라이브러리가 참조할 수 있도록 환경변수에도 등록
os.environ["OPENAI_API_KEY"] = openai_api_key

############################### 1단계 : PDF 문서를 벡터DB에 저장하는 함수들 ##########################

def save_uploadedfile(uploadedfile: UploadedFile) -> str:
    temp_dir = "PDF_임시폴더"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.read())
    return file_path


def pdf_to_documents(pdf_path: str) -> List[Document]:
    loader = PyMuPDFLoader(pdf_path)
    doc = loader.load()
    for d in doc:
        d.metadata['file_path'] = pdf_path
    return doc


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(documents)


def save_to_vector_store(documents: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")


############################### 2단계 : RAG 기능 구현 ##########################

@st.cache_data
def process_question(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs={"k": 3})
    retrieve_docs: List[Document] = retriever.invoke(user_question)
    chain = get_rag_chain()
    response = chain.invoke({"question": user_question, "context": retrieve_docs})
    return response, retrieve_docs


def get_rag_chain() -> Runnable:
    template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘.
    - 질문에 대한 응답을 해줘.
    - 간결하게 5줄 이내로 해줘.
    - 곧바로 응답결과를 말해줘.

    컨텍스트 : {context}

    질문: {question}

    응답:"""
    prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-5", api_key=openai_api_key)
    return prompt | model | StrOutputParser()


############################### 3단계 : PDF 페이지 표시 ##########################

@st.cache_data(show_spinner=False)
def convert_pdf_to_images(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path)
    image_paths = []
    output_folder = "PDF_이미지"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
    return image_paths


def display_pdf_page(image_path: str, page_number: int):
    image_bytes = open(image_path, "rb").read()
    st.image(image_bytes, caption=f"Page {page_number}", output_format="PNG", width=600)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


############################### 메인 함수 ##########################

def main():
    st.set_page_config("전력계통영향평가 FAQ 챗봇", layout="wide")

    left, right = st.columns([1, 1])

    with left:
        st.header("⚡ 전평 ChatBot")

        pdf_doc = st.file_uploader("PDF 업로드", type="pdf")
        if st.button("PDF 업로드하기") and pdf_doc:
            with st.spinner("PDF 문서 처리 중..."):
                pdf_path = save_uploadedfile(pdf_doc)
                pdf_document = pdf_to_documents(pdf_path)
                smaller_documents = chunk_documents(pdf_document)
                save_to_vector_store(smaller_documents)

            with st.spinner("PDF 페이지 이미지 변환 중..."):
                st.session_state.images = convert_pdf_to_images(pdf_path)

        user_question = st.text_input("전력계통영향평가 관련 질문을 입력하세요", 
                                      placeholder="예: 20MW를 공급받으려 하는데 전력계통영향평가를 받아야 하나요?")

        if user_question:
            response, context = process_question(user_question)
            st.text(response)

            for i, doc in enumerate(context):
                with st.expander("관련 문서"):
                    st.text(doc.page_content)
                    file_path = doc.metadata.get("source", "")
                    page_number = doc.metadata.get("page", 0) + 1
                    file_path = file_path.replace("\\", "/")
                    if st.button(f"🔎 {os.path.basename(file_path)} - {page_number}페이지", key=f"{file_path}_{i}"):
                        st.session_state.page_number = str(page_number)

    with right:
        page_number = st.session_state.get("page_number")
        if page_number:
            page_number = int(page_number)
            images = sorted(os.listdir("PDF_이미지"), key=natural_sort_key)
            image_paths = [os.path.join("PDF_이미지", img) for img in images]
            display_pdf_page(image_paths[page_number - 1], page_number)


if __name__ == "__main__":
    main()

스트림릿 클라우드로 배포하고 있고, 너 말대로 스트림릿 클라우드에서 API 키 비밀로 등록할테니까, 그걸 자동으로 불러오는 코드를 써줘 
