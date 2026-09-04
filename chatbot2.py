import os
import streamlit as st
from dotenv import load_dotenv

# 기존 임포트 수정
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 환경변수 및 API Key 로드
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Streamlit Cloud Secrets 대응
if not api_key and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

if not api_key:
    st.error("GEMINI_API_KEY가 설정되지 않았습니다.")
    st.stop()

# 2. 페이지 및 UI 설정 (상단/하단 메인 메뉴 및 배너 숨기기)
st.set_page_config(page_title="PDF 기반 RAG 챗봇", layout="centered")

custom_css = """
    <style>
    header, #MainMenu, [data-testid="stHeader"], footer, .stAppFooter, [data-testid="stFooter"] {
        display: none !important;
        height: 0px !important;
    }
    div[class*="viewerBadge"], div[class*="profileOwnerEl"], [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
    }
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("📄 PDF 문서 기반 RAG 챗봇")
st.caption("PDF 문서를 업로드하고 궁금한 점을 질문해 보세요!")

# 3. 사이드바 - PDF 업로드 파트
with st.sidebar:
    st.header("📄 문서 업로드")
    uploaded_file = st.file_uploader("PDF 파일을 선택하세요", type=["pdf"])

# 4. PDF 문서 처리 및 벡터DB(FAISS) 생성 함수
@st.cache_resource(show_spinner="PDF 문서를 분석하고 벡터DB를 생성 중입니다...")
def process_pdf(uploaded_file_bytes, file_name):
    # 임시 파일로 저장하여 PyPDFLoader에서 읽을 수 있도록 함
    temp_path = f"./temp_{file_name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file_bytes)

    # PDF 로드
    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    # 텍스트 분할 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    # Gemini 임베딩 모델 준비
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004", 
        google_api_key=api_key,
        task_type="retrieval_document"
    )

    # FAISS 벡터 스토어 생성
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # 임시 파일 삭제
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return vectorstore

# 5. 세션 상태(대화 내역 및 벡터DB) 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "안녕하세요! PDF 문서를 왼쪽 사이드바에 업로드하신 후 질문해 주세요."}
    ]

# 대화 내용 출력
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 6. PDF 업로드 시 처리 흐름
vectorstore = None
if uploaded_file is not None:
    # 파일 바이트 추출 및 처리
    file_bytes = uploaded_file.getvalue()
    vectorstore = process_pdf(file_bytes, uploaded_file.name)
    st.sidebar.success("PDF 분석이 완료되었습니다!")

# 7. 사용자 질문 처리 (RAG 대화 로직)
if user_input := st.chat_input("문서 내용에 대해 질문하세요..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        if vectorstore is None:
            response_text = "먼저 왼쪽 사이드바에서 PDF 문서를 업로드해 주세요!"
            st.write(response_text)
        else:
            with st.spinner("문서에서 정보를 검색하여 답변을 작성 중입니다..."):
                # LLM 설정 (gemini-3.1-flash-lite)
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.1-flash-lite",
                    google_api_key=api_key,
                    temperature=0.3
                )

                # 프롬프트 설정
                system_prompt = (
                    "당신은 제시된 참고 문서를 바탕으로 사용자 질문에 정확하게 답변하는 AI입니다.\n"
                    "아래에 제공된 [참고 문서 내용]만을 기반으로 질문에 답변하세요.\n"
                    "만약 참고 문서 내용에 답변을 도출할 정보가 없다면, 솔직하게 '제시된 문서에서 해당 정보를 찾을 수 없습니다.'라고 답변하세요.\n\n"
                    "[참고 문서 내용]:\n{context}"
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                # 문서 결합 함수 (검색된 문서들을 하나의 텍스트로 합침)
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

                # 최신 LCEL 기반 RAG 체인 구성
                rag_chain = (
                    {"context": retriever | format_docs, "input": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                # 답변 생성
                response_text = rag_chain.invoke(user_input)
                st.write(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})