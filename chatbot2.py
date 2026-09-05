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

# ---------------------------------------------------------
# 1. 환경 변수 및 공통 유틸리티 설정
# ---------------------------------------------------------
load_dotenv()
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일이나 Streamlit Secrets를 확인해 주세요.")
    st.stop()

os.environ["GEMINI_API_KEY"] = api_key


def parse_llm_response(response_obj) -> str:
    """LLM 응답 객체에서 순수 텍스트만 안전하게 추출하는 방어적 유틸리티 함수"""
    if isinstance(response_obj, str):
        return response_obj
    
    # LangChain BaseMessage 형태
    if hasattr(response_obj, "content"):
        content = response_obj.content
        if isinstance(content, str):
            return content
        if isinstance(content, list) and len(content) > 0:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
            return str(first)
        return str(content)
    
    # 딕셔너리 또는 리스트 형태
    if isinstance(response_obj, list) and len(response_obj) > 0:
        first = response_obj[0]
        if isinstance(first, dict) and "text" in first:
            return first["text"]
        return str(first)
        
    return str(response_obj)


# ---------------------------------------------------------
# 2. 웹앱 기본 설정 및 세션 초기화
# ---------------------------------------------------------
st.set_page_config(page_title="스마트 AI & PDF 챗봇", page_icon="💬", layout="wide")
st.title("💬 스마트 AI & PDF 대화 챗봇")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 자유롭게 대화를 나누시거나, 필요할 때 사이드바에서 PDF 문서를 업로드해 주세요."}
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0


# ---------------------------------------------------------
# 3. PDF 분석 및 벡터 DB 생성 (캐싱)
# ---------------------------------------------------------
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
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": file_name, "page": i + 1}
                    ))
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


# ---------------------------------------------------------
# 4. 사이드바 구성 (파일 업로드, 요약, 초기화, 원문 확인)
# ---------------------------------------------------------
with st.sidebar:
    st.header("📄 문서 업로드 (선택)")
    
    uploaded_file = st.file_uploader(
        "분석할 PDF 파일을 선택하세요 (선택 사항)", 
        type=["pdf"],
        key=f"pdf_uploader_{st.session_state.file_uploader_key}"
    )
    
    if uploaded_file is not None:
        if st.button("🗑️ 업로드된 PDF 제거", use_container_width=True, type="secondary"):
            st.session_state.file_uploader_key += 1
            st.cache_resource.clear()
            st.success("PDF 문서가 제거되었습니다. 일반 대화 모드로 전환합니다.")
            st.rerun()

    st.divider()

    if st.button("🔄 대화 기록 초기화", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요! 자유롭게 대화를 나누시거나, 필요할 때 사이드바에서 PDF 문서를 업로드해 주세요."}
        ]
        st.session_state.chat_history = []
        st.rerun()

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        vectorstore, raw_docs = process_pdf(file_bytes, uploaded_file.name)
        st.success(f"'{uploaded_file.name}' 분석 완료!")

        if st.button("📝 문서 전체 요약하기", use_container_width=True, type="primary"):
            with st.spinner("문서 전체 내용을 바탕으로 핵심 요약을 작성하는 중입니다..."):
                full_text = "\n\n".join([f"[Page {d.metadata['page']}]\n{d.page_content}" for d in raw_docs])
                truncated_text = full_text[:12000]

                llm_summary = ChatGoogleGenerativeAI(
                    model="gemini-3.1-flash-lite",
                    temperature=0.2
                )
                
                summary_prompt = (
                    "당신은 뛰어난 문서 요약 전문가입니다. 아래 제공된 [문서 내용]을 바탕으로 "
                    "전체 문서의 핵심 주제, 주요 내용, 그리고 중요한 포인트들을 깔끔하고 가독성 높게 요약해 주세요.\n\n"
                    "작성 양식:\n"
                    "1. 📌 **개요 및 한 줄 요약**\n"
                    "2. 💡 **주요 핵심 내용 (불릿 포인트)**\n"
                    "3. 📊 **특이사항 또는 결론 (필요 시 표나 리스트 활용)**\n\n"
                    f"[문서 내용]:\n{truncated_text}"
                )
                
                summary_response = llm_summary.invoke(summary_prompt)
                summary_result = parse_llm_response(summary_response)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"📋 **[{uploaded_file.name}] 전체 문서 요약**\n\n{summary_result}"
                })
                st.session_state.chat_history.append(AIMessage(content=summary_result))
                st.rerun()

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


# ---------------------------------------------------------
# 5. 메인 대화 창 및 사용자 입력 처리
# ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("질문이나 대화를 입력해 보세요..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            temperature=0.3
        )
        recent_chat_history = st.session_state.chat_history[-6:]

        # 모드 A: PDF 미업로드 시 (일반 LLM 대화 모드)
        if uploaded_file is None:
            with st.spinner("생각 중입니다..."):
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "당신은 친절하고 유능한 AI 보조입니다. 사용자의 질문에 지식과 대화 맥락을 활용하여 정확하고 명확하게 답변해 주세요."),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ])
                chain = prompt | llm
                response = chain.invoke({
                    "input": user_input,
                    "chat_history": recent_chat_history
                })
                answer_text = parse_llm_response(response)
                st.markdown(answer_text)

        # 모드 B: PDF 업로드 시 (문서 기반 RAG 모드)
        else:
            with st.spinner("문서 내용을 바탕으로 답변을 작성 중입니다..."):
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

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

                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": recent_chat_history
                })

                answer_text = parse_llm_response(response.get("answer", response))
                st.markdown(answer_text)

                if "context" in response and response["context"]:
                    with st.expander("🔍 AI가 참고한 PDF 페이지 및 원문 보기"):
                        for i, doc in enumerate(response["context"]):
                            page_num = doc.metadata.get("page", "?")
                            st.markdown(f"**[참고 {i+1}] (Page {page_num})**")
                            st.caption(doc.page_content[:300] + "...")

        # 대화 이력 업데이트 (공통)
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer_text))
        st.session_state.messages.append({"role": "assistant", "content": answer_text})