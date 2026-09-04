import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

# 1. .env 환경변수 로드
load_dotenv()

import streamlit as st

# 페이지 기본 설정
st.set_page_config(page_title="Gemini Chatbot", layout="centered")

import streamlit as st

st.set_page_config(page_title="Gemini Chatbot", layout="centered")

# Streamlit 최신 버전 대응 - 모든 불필요한 UI 완벽 숨기기
custom_css = """
    <style>
    /* 1. 상단 헤더, 메인 메뉴, 깃허브 버튼 숨기기 */
    header, #MainMenu, [data-testid="stHeader"] {
        display: none !important;
        height: 0px !important;
    }

    /* 2. 하단 푸터 및 'Made with Streamlit' 배너 완전 차단 */
    footer, .stAppFooter, [data-testid="stFooter"], [data-testid="stStatusWidget"] {
        display: none !important;
        height: 0px !important;
    }

    /* 3. 우측 하단 시크릿/비로그인 유저용 프로필 및 Streamlit 홍보 배너 숨기기 */
    div[class*="viewerBadge"], 
    div[class*="profileOwnerEl"],
    [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* 4. 상단 빈 여백 줄여서 채팅창 밀착 */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Streamlit 페이지 기본 설정
st.set_page_config(page_title="홍보봇", page_icon="🤖")
st.title("🤖 홍보봇")

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