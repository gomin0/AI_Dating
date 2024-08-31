import streamlit as st
import boto3
from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import base64
import json
import os
import random

page_bg_img = '''
<style>
.stApp {
    background-color: #FFC0CB; /* 핑크색 배경 */
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Bedrock 클라이언트 설정
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Claude 3.5 파라미터 설정
model_kwargs = { 
    "max_tokens": 1000,
    "temperature": 0.01,
    "top_p": 0.01,
}

# Bedrock LLM 설정
llm = ChatBedrock(
    client=bedrock_runtime,
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_kwargs=model_kwargs,
    streaming=True
)

# OpenAI API 키 설정
def save_image(base64_image_data, prompt):
    # 이미지 저장 폴더 생성
    output_folder = "images"
    os.makedirs(output_folder, exist_ok=True)

    # 이미지 파일 경로 생성
    file_path = os.path.join(output_folder, f"{prompt}.png")
    # base64 인코딩된 데이터를 이미지 파일로 변환하여 저장
    with open(file_path, "wb") as file:
        file.write(base64.b64decode(base64_image_data))
    # 이미지 저장 경로를 로깅함
    return file_path


def edit_image(image_path, prompt):
    with open(image_path, "rb") as image_file:
        input_image = base64.b64encode(image_file.read()).decode("utf8")

    body = json.dumps(
        {
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {"text": prompt, "images": [input_image]},
        }
    )

    try:
        # 모델을 호출해 이미지 생성
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId="amazon.titan-image-generator-v1",
        )

        # base64 인코딩된 이미지 데이터 추출
        base64_image_data = json.loads(response["body"].read())["images"][0]
        return save_image(base64_image_data, prompt)
    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {e}")
        raise


def generate_image(prompt):
    # 이미지 생성을 위한 난수 시드를 생성
    seed = random.randint(0, 2147483647)
    # 이미지 생성 요청 데이터를 구성함
    body = json.dumps(
        {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": prompt},
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "standard",
                "cfgScale": 7.5,
                "height": 512,
                "width": 512,
                "seed": seed,
            },
        }
    )
    try:
        # 모델을 호출해 이미지 생성
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId="amazon.titan-image-generator-v1",
        )

        # base64 인코딩된 이미지 데이터 추출
        base64_image_data = json.loads(response["body"].read())["images"][0]
        return save_image(base64_image_data, prompt)
    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {e}")
        raise


# 페이지 상태 초기화
if "page" not in st.session_state:
    st.session_state.page = "input"  # 처음에는 입력 페이지로 시작
if "message_count" not in st.session_state:
    st.session_state.message_count = 0
if "character_image_url" not in st.session_state:
    st.session_state.character_image_url = None

if st.session_state.page == "input":
    st.markdown("# 💖 나의 이상형 💖")

    # 사용자로부터 캐릭터 설정을 받는 UI
    hair_style = st.selectbox("머리 스타일을 선택하세요:", ["긴 생머리", "단발 머리", "긴 웨이브 머리", "포마드 스타일", "가르마 스타일", "땋은 머리", "빡빡이"])
    hair_color = st.selectbox("머리 색깔을 선택하세요:", ["검정색", "갈색", "노란색", "빨간색"])
    skin_color = st.selectbox("피부 색깔을 선택하세요:", ["흑인", "백인", "황인"])
    gender = st.radio("성별을 선택하세요:", ["여성", "남성"])

    # 성격 선택 항목 (중복 선택 가능)
    personalities = st.multiselect(
        "이상형의 성격을 선택하세요(2개 이하):",
        ["소심한", "내향적인", "다정한", "외향적인", "도도한", "애교있는", "착한", "화끈한", "매력있는"]
    )

    # 페이지 상태 초기화
    if "name" not in st.session_state:
        st.session_state.name = ""
    
    # 사용자로부터 이름 입력 받기
    name = st.text_input("이상형의 이름을 입력하세요:")
    
    # 글자 수 제한 설정
    max_length = 10
    
    # 입력된 이름의 길이를 체크
    if len(name) > max_length:
        st.error(f"이름은 {max_length}자 이하로 입력해 주세요.")
    else:
        st.session_state.name = name

    # 선택 항목을 영어로 변환
    hair_style_en = {"긴 생머리": "long straight hair", "단발 머리": "short hair", "긴 웨이브 머리": "long wavy hair", "포마드 스타일": "pompadour style",
        "가르마 스타일": "side parting style", "땋은 머리": "ponytail", "빡빡이": "bald"}[hair_style]
    hair_color_en = {"검정색": "black", "갈색": "brown", "노란색": "blonde", "빨간색": "red"}[hair_color]
    skin_color_en = {"백인": "white", "흑인": "black", "황인": "asian"}[skin_color]
    gender_en = {"여성": "female", "남성": "male"}[gender]
    personalities_en = {
        "소심한": "shy",
        "내향적인": "introverted",
        "다정한": "kind",
        "외향적인": "extroverted",
        "도도한": "aloof",
        "애교있는": "cute",
        "착한": "good-hearted",
        "화끈한": "bold",
        "매력있는": "charming"
    }

    # 성격을 영어로 변환하여 'and'로 연결
    selected_personalities_en = " and ".join([personalities_en[p] for p in personalities])

    # 전체 외모 설명을 하나의 영어 문장으로 생성
    if gender == "여성":
        appearance_description = f"Create a portrait of a beautiful young female anime character with {hair_style_en} hair and {hair_color_en} color and {skin_color_en} skin. She should have large, expressive eyes, a warm smile, and a classic anime style. her personality of {selected_personalities_en}"
    else:
        appearance_description = f"Create a portrait of a handsome young male anime character with {hair_style_en} hair and {hair_color_en} color and {skin_color_en} skin. He should have a strong, confident appearance, and a classic anime style. his personality of {selected_personalities_en}"

    if st.button("이상형 생성"):
        if selected_personalities_en and name:
            # 변수를 저장하고 이미지 생성 페이지로 이동
            st.session_state.personality = selected_personalities_en
            st.session_state.appearance = appearance_description
            st.session_state.name = name

            # 외모 이미지 생성
            image_url = generate_image(st.session_state.appearance)
            if image_url:
                st.session_state.character_image_url = image_url
                st.success("이상형이 생성되었습니다!")
                st.button("이동하기")
                st.session_state.page = "chat"  # 채팅 페이지로 이동
        else:
            st.error("모든 입력 필드를 작성해주세요.")

elif st.session_state.page == "chat":
    st.title(f"{st.session_state.name}와(과) 대화하기")

    # 채팅 메시지 히스토리 설정
    message_history = StreamlitChatMessageHistory(key="chat_messages")

    # 프롬프트 템플릿 설정
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"Assume the character has a personality of {st.session_state.personality}. \
            Create dialogue and actions for this character in a flirty interaction with the protagonist. \
            Respond as if you are completely embodying this personality.\
            your name is {st.session_state.name}"),
            MessagesPlaceholder(variable_name="message_history"),
            ("human", "{query}"),
        ]
    )

    # 대화 체인 설정
    chain_with_history = RunnableWithMessageHistory(
        prompt | llm,
        lambda session_id: message_history,  # 항상 이전 대화를 리턴
        input_messages_key="query",
        history_messages_key="message_history",
    )

    # 레이아웃을 좌우로 나누기
    col1, col2 = st.columns([1, 1])  # 비율을 1:2로 설정

    with col1:
        # 캐릭터 이미지가 있다면 배경으로 설정
        if st.session_state.character_image_url:
            st.image(st.session_state.character_image_url, caption="대화 상대의 이미지", use_column_width=True)

    with col2:
        # 채팅 인터페이스
        if query := st.chat_input(f"{st.session_state.name}와(과) 대화하기"):
            # 사용자 입력 메시지 표시
            st.chat_message("human").write(query)
    
            # chain이 호출되면 새 메시지가 자동으로 StreamlitChatMessageHistory에 저장됨
            config = {"configurable": {"session_id": "any"}}
            response_stream = chain_with_history.stream({"query": query}, config=config)
            st.chat_message("ai").write_stream(response_stream)

        # "새로운 상대 찾기" 버튼을 항상 표시
        if st.button("새로운 상대 찾기"):
            # 대화 초기화 및 페이지 변경
            st.session_state.message_count = 0  # 대화 횟수 초기화
            st.session_state.character_image_url = None  # 캐릭터 이미지 초기화
            st.session_state.page = "input"  # 입력 페이지로 이동
            
            # 상태 변경 후 페이지를 새로 고침
            st.rerun()
