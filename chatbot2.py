import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# 1. 환경 변수 로드
load_dotenv()
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일이나 Streamlit Secrets를 확인해 주세요.")
    st.stop()

os.environ["GEMINI_API_KEY"] = api_key

# 2. 웹앱 기본 설정
st.set_page_config(page_title="스마트 PDF 문서 챗봇", page_icon="📄", layout="wide")
st.title("📄 스마트 PDF 문서 대화 챗봇")

# 3. 세션 상태 초기화 (대화 기록 및 LangChain Message 객체 저장)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! PDF 문서를 업로드해 주시면 문서 내용에 대해 대화 맥락을 기억하며 답변해 드립니다."}
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # LangChain용 (HumanMessage, AIMessage)

# 4. pdfplumber 파서를 사용한 PDF 분석 및 FAISS 벡터 DB 생성
@st.cache_resource(show_spinner="PDF 문서의 텍스트와 레이아웃을 분석하는 중입니다...")
def process_pdf(uploaded_file_bytes, file_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file_bytes)
        temp_path = tmp_file.name

    docs = []
    try:
        with pdfplumber.open(temp_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(layout=True)
                if text and text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={"source": file_name, "page": i + 1}
                    )
                    docs.append(doc)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if not docs:
        st.sidebar.error("PDF에서 텍스트를 추출하지 못했습니다.")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore, docs

# 5. 사이드바 - 파일 업로드, 대화 초기화 및 디버깅용 텍스트 확인
with st.sidebar:
    st.header("📄 문서 업로드")
    uploaded_file = st.file_uploader("분석할 PDF 파일을 선택하세요", type=["pdf"])
    
    # 대화 초기화 버튼
    if st.button("🔄 대화 기록 초기화", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요! PDF 문서를 업로드해 주시면 문서 내용에 대해 대화 맥락을 기억하며 답변해 드립니다."}
        ]
        st.session_state.chat_history = []
        st.rerun()

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        vectorstore, raw_docs = process_pdf(file_bytes, uploaded_file.name)
        st.success(f"'{uploaded_file.name}' 분석 완료!")
        
        st.divider()
        st.subheader("🔍 PDF 추출 원문 확인")
        page_num = st.number_input("확인할 페이지 번호", min_value=1, max_value=len(raw_docs), value=1, step=1)
        selected_doc = next((d for d in raw_docs if d.metadata["page"] == page_num), None)
        
        if selected_doc:
            st.text_area(
                f"{page_num}페이지 실제 읽어온 텍스트", 
                value=selected_doc.page_content, 
                height=300
            )

# 6. 이전 대화 기록 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. 질문 입력 및 처리 (대화 맥락 반영 RAG)
if user_input := st.chat_input("문서 내용에 대해 무엇이든 질문해 보세요..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if uploaded_file is None:
        with st.chat_message("assistant"):
            st.markdown("먼저 사이드바에서 PDF 파일을 업로드해 주세요.")
            st.session_state.messages.append({"role": "assistant", "content": "먼저 사이드바에서 PDF 파일을 업로드해 주세요."})
    else:
        with st.chat_message("assistant"):
            with st.spinner("문서 내용을 바탕으로 답변을 작성 중입니다..."):
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.1-flash-lite",
                    temperature=0.2
                )

                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

                # 이전 대화 맥락 반영 질문 재구성 프롬프트
                contextualize_q_system_prompt = (
                    "이전 대화 내용과 최신 사용자 질문이 주어졌을 때, "
                    "이전 대화 내용 없이도 이해할 수 있는 독립적인 질문으로 재구성하세요. "
                    "질문에 답변하지 말고, 필요하다면 질문을 재구성하기만 하고 그대로 반환하세요."
                )
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                
                history_aware_retriever = create_history_aware_retriever(
                    llm, retriever, contextualize_q_prompt
                )

                # 범용 문서 분석 및 답변 프롬프트
                system_prompt = (
                    "당신은 친절하고 정교한 문서 분석 전문 AI 보조입니다.\n"
                    "아래 제공된 [참고 문서 내용]만을 바탕으로 질문에 정확하고 명확하게 답변하세요.\n"
                    "정리하기 적합한 내용(비교, 항목별 분류, 수치 등)이 있다면 보기 쉽게 요약문이나 표(Table) 형태로 작성해 주세요.\n"
                    "참고 문서에서 답을 찾을 수 없다면 억지로 추측하지 말고 '제공된 문서에서 관련 정보를 찾을 수 없습니다'라고 솔직하게 답하세요.\n\n"
                    "[참고 문서 내용]:\n{context}"
                )
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])

                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                # 토큰 절약을 위한 대화 이력 슬라이싱 (최근 6개 메시지)
                recent_chat_history = st.session_state.chat_history[-6:]

                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": recent_chat_history
                })

                answer_text = response["answer"]
                st.markdown(answer_text)

                # 출처 표기 (Expander)
                if "context" in response and response["context"]:
                    with st.expander("🔍 AI가 참고한 PDF 페이지 및 원문 보기"):
                        for i, doc in enumerate(response["context"]):
                            page_num = doc.metadata.get("page", "?")
                            st.markdown(f"**[참고 {i+1}] (Page {page_num})**")
                            st.caption(doc.page_content[:300] + "...")

                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=answer_text))
                st.session_state.messages.append({"role": "assistant", "content": answer_text})