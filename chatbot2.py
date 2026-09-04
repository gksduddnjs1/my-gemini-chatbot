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
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# 1. 환경 변수 로드
load_dotenv()
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일이나 Streamlit Secrets를 확인해 주세요.")
    st.stop()

os.environ["GEMINI_API_KEY"] = api_key

# 2. 웹앱 기본 설정
st.set_page_config(page_title="자전거 홍보 PDF 챗봇", page_icon="🚲", layout="wide")
st.title("🚲 자전거 카탈로그/홍보 PDF Q&A 챗봇")

# 3. 세션 상태 초기화 (대화 기록 및 LangChain Message 객체 저장)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 자전거 홍보 PDF 문서를 업로드해 주시면 대화 맥락을 기억하며 질문에 답변해 드립니다."}
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # LangChain용 (HumanMessage, AIMessage)

# 4. pdfplumber 파서를 사용한 PDF 분석 및 FAISS 벡터 DB 생성
@st.cache_resource(show_spinner="자전거 PDF의 레이아웃과 표 구조를 분석하는 중입니다...")
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore, docs

# 5. 사이드바 - 파일 업로드 및 디버깅용 텍스트 확인
with st.sidebar:
    st.header("📄 문서 업로드")
    uploaded_file = st.file_uploader("자전거 홍보/카탈로그 PDF 파일을 선택하세요", type=["pdf"])
    
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
if user_input := st.chat_input("자전거 스펙, 부품, 가격 등에 대해 질문해 보세요..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if uploaded_file is None:
        with st.chat_message("assistant"):
            st.markdown("먼저 사이드바에서 자전거 홍보 PDF 파일을 업로드해 주세요.")
            st.session_state.messages.append({"role": "assistant", "content": "먼저 사이드바에서 자전거 홍보 PDF 파일을 업로드해 주세요."})
    else:
        with st.chat_message("assistant"):
            with st.spinner("이전 대화와 PDF 문서를 바탕으로 답변을 작성 중입니다..."):
                # LLM 모델 고정: gemini-3.1-flash-lite
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.1-flash-lite",
                    temperature=0.2
                )

                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

                # A. 대화 내역을 바탕으로 사용자의 재질문/대명사("이 자전거", "그건 얼마야?")를 독립적인 질문으로 재구성하는 프롬프트
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
                
                # 대화 이력 인식 검색기 생성
                history_aware_retriever = create_history_aware_retriever(
                    llm, retriever, contextualize_q_prompt
                )

                # B. 재구성된 질문과 검색된 문맥을 바탕으로 답변을 생성하는 프롬프트
                system_prompt = (
                    "당신은 자전거 전문 상담사 및 문서 분석 전문가입니다.\n"
                    "아래 제공된 [참고 문서 내용]만을 바탕으로 질문에 정확하게 답변하세요.\n"
                    "스펙표나 부품 정보가 포함되어 있다면 이해하기 쉽게 목록 형태로 정리해 주세요.\n"
                    "참고 문서에서 답을 찾을 수 없다면 솔직하게 모른다고 답변하세요.\n\n"
                    "[참고 문서 내용]:\n{context}"
                )
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])

                # 문서 결합 체인 및 전체 대화형 RAG 체인 구성
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                # 체인 실행 (대화 이력 전달)
                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })

                answer_text = response["answer"]
                st.markdown(answer_text)

                # 참고한 PDF 페이지 번호 보여주기 (Expander)
                if "context" in response and response["context"]:
                    with st.expander("🔍 AI가 참고한 PDF 페이지 및 원문 보기"):
                        for i, doc in enumerate(response["context"]):
                            page_num = doc.metadata.get("page", "?")
                            st.markdown(f"**[참고 {i+1}] (Page {page_num})**")
                            st.caption(doc.page_content[:300] + "...")

                # 대화 이력 업데이트 (LangChain Message 객체 및 UI 메시지)
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=answer_text))
                st.session_state.messages.append({"role": "assistant", "content": answer_text})