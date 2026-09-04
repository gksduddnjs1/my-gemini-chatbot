import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

# 1. .env 환경변수 로드
load_dotenv()

# Streamlit 페이지 기본 설정
st.set_page_config(page_title="Gemini AI Chatbot", page_icon="🤖")
st.title("🤖 Gemini AI 챗봇")

# 2. API Key 확인
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error(".env 파일에서 GEMINI_API_KEY를 찾을 수 없습니다.")
    st.stop()

# 3. Streamlit 세션 상태 초기화 (대화 기록 전용)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. 이전 대화 기록 화면에 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. 사용자 입력 처리
if user_input := st.chat_input("메시지를 입력하세요..."):
    # (1) 화면에 사용자 메시지 표시 및 세션 저장
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # (2) Gemini API 호출 (요청 시점에 클라이언트 및 세션 구성)
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            try:
                # 클라이언트를 세션에 저장하지 않고 매 요청 시 생성 (연결 닫힘 방지)
                client = genai.Client(api_key=api_key)
                
                # 지금까지의 대화 기록을 Gemini 형식으로 재구성하여 전송
                # (또는 client.chats.create를 활용)
                chat = client.chats.create(
                    model="gemini-3.1-flash-lite",
                    history=[
                        {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                        for m in st.session_state.messages[:-1]  # 현재 입력 이전의 기록들
                    ]
                )
                
                response = chat.send_message(user_input)
                ai_response = response.text
                
                # 답변 출력
                st.markdown(ai_response)
                
                # (3) 세션에 AI 답변 저장
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")