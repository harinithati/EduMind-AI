import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from google import genai

st.set_page_config(
    page_title="EduMind AI | Learning Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
@st.cache_resource
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


client = get_gemini_client()
# ---------- PROFESSIONAL DESIGN ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F5F7FB;
}

[data-testid="stSidebar"] {
    background: #101828;
}

[data-testid="stSidebar"] * {
    color: #F2F4F7;
}

.hero {
    padding: 32px;
    border-radius: 20px;
    background: linear-gradient(120deg, #4338CA, #6366F1, #818CF8);
    color: white;
    margin-bottom: 26px;
}

.hero h1 {
    font-size: 34px;
    font-weight: 800;
    color: white;
}

.hero p {
    font-size: 16px;
    color: #EEF2FF;
}

.metric-card {
    background: white;
    border: 1px solid #E4E7EC;
    border-radius: 16px;
    padding: 22px;
    min-height: 140px;
}

.metric-label {
    color: #667085;
    font-size: 14px;
}

.metric-value {
    color: #101828;
    font-size: 30px;
    font-weight: 800;
    margin-top: 12px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #101828;
    margin: 24px 0 14px 0;
}

div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 42px;
}
</style>
""", unsafe_allow_html=True)


# ---------- SESSION DATA ----------
if "quiz_history" not in st.session_state:
    st.session_state.quiz_history = []

if "study_material_name" not in st.session_state:
    st.session_state.study_material_name = None


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("# 🎓 EduMind AI")
    st.caption("LEARNING INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "AI Learning Tutor",
            "Adaptive Quiz",
            "Skill Gap Analysis",
            "Learning Progress",
            "Study Materials"
        ]
    )

    st.divider()
    st.caption("SDG 4 • QUALITY EDUCATION")
    st.caption("Personalized learning for everyone")


# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <h1>Learn Smarter with EduMind AI</h1>
    <p>
        Your personalized AI learning companion.
        Discover skill gaps, practise concepts, and track your progress.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------- OVERVIEW ----------
if page == "Overview":

    st.markdown("## Your Learning Workspace")
    st.write(
        "Build knowledge, strengthen weak concepts, "
        "and follow a personalized learning journey."
    )

    history = st.session_state.quiz_history

    attempts = len(history)
    average = (
        sum(item["score"] for item in history) / attempts
        if attempts else 0
    )
    best = max(
        (item["score"] for item in history),
        default=0
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">'
            f'📝 Quiz Attempts</div><div class="metric-value">'
            f'{attempts}</div></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">'
            f'📊 Average Score</div><div class="metric-value">'
            f'{average:.1f}%</div></div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">'
            f'🏆 Best Score</div><div class="metric-value">'
            f'{best:.1f}%</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">Learning Modules</div>',
                unsafe_allow_html=True)

    a, b, c = st.columns(3)

    with a:
        st.info("🤖 **AI Tutor**\n\nGet explanations for difficult concepts.")

    with b:
        st.info("📝 **Adaptive Quiz**\n\nPractise concepts and test your understanding.")

    with c:
        st.info("🎯 **Skill Analysis**\n\nIdentify areas that need more practice.")

    st.caption(
        "Metrics update as you complete quizzes. "
        "This prototype stores session data temporarily."
    )


# ---------- AI TUTOR ----------
elif page == "AI Learning Tutor":

    st.markdown("## 🤖 AI Learning Tutor")

    question = st.text_area(
        "What would you like to learn?",
        placeholder="Explain Moore and Mealy machines with examples."
    )

    level = st.selectbox(
        "Explanation level",
        ["Beginner", "Intermediate", "Advanced"]
    )

    if st.button("Generate Explanation", use_container_width=True):

        if not question.strip():
            st.warning("Please enter a question.")

        elif client is None:
            st.error(
                "Gemini API key is missing. "
                "Add GEMINI_API_KEY in Streamlit Secrets."
            )

        else:
            with st.spinner("Generating your explanation..."):

                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=(
                            f"Act as an educational tutor. "
                            f"Explain at a {level} level. "
                            f"Use simple language, examples, and a short summary.\n\n"
                            f"Student question: {question}"
                        )
                    )

                    st.markdown("### 📘 AI Explanation")
                    st.write(response.text)

                except Exception as e:
                    st.error(
                        "The AI request failed. Check the API key, "
                        "model availability, quota, and connection."
                    )
                    st.caption(str(e))
            )


# ---------- ADAPTIVE QUIZ ----------
elif page == "Adaptive Quiz":

    st.markdown("## 📝 Adaptive Knowledge Assessment")

    topic = st.text_input(
        "Topic",
        "Moore and Mealy Machines"
    )

    difficulty = st.selectbox(
        "Difficulty",
        ["Easy", "Medium", "Hard", "Advanced"]
    )

    number = st.slider(
        "Number of questions",
        min_value=3,
        max_value=10,
        value=5
    )

    st.info(
        "The finished quiz module will generate questions from "
        "your uploaded study material and evaluate your answers."
    )

    st.button(
        "Generate Assessment",
        use_container_width=True,
        disabled=True
    )


# ---------- SKILL GAP ----------
elif page == "Skill Gap Analysis":

    st.markdown("## 🎯 Skill Gap Analysis")

    st.write(
        "Enter your assessment score to receive a preliminary "
        "recommendation. Full topic-level analysis requires "
        "recorded question-level results."
    )

    score = st.slider("Assessment score", 0, 100, 50)

    if score < 40:
        st.error("Priority: Revisit the fundamentals.")
        st.write("Recommended: Review definitions and solve easy questions.")

    elif score < 70:
        st.warning("Priority: Strengthen conceptual understanding.")
        st.write("Recommended: Revise weak concepts and practise medium questions.")

    elif score < 90:
        st.success("Good progress!")
        st.write("Recommended: Practise applications and problem-solving.")

    else:
        st.success("Excellent assessment score!")
        st.write("Recommended: Attempt advanced problems and consolidate learning.")


# ---------- PROGRESS ----------
elif page == "Learning Progress":

    st.markdown("## 📈 Learning Progress")

    history = st.session_state.quiz_history

    if history:
        df = pd.DataFrame(history)

        fig = px.line(
            df,
            x="attempt",
            y="score",
            markers=True,
            title="Assessment Score Trend",
            labels={
                "attempt": "Quiz Attempt",
                "score": "Score (%)"
            }
        )

        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df, use_container_width=True)

    else:
        st.info(
            "Your progress chart will appear here after quiz results "
            "are recorded."
        )


# ---------- STUDY MATERIALS ----------
elif page == "Study Materials":

    st.markdown("## 📚 Study Material Library")

    uploaded_file = st.file_uploader(
        "Upload a study material PDF",
        type=["pdf"]
    )

    if uploaded_file:
        st.session_state.study_material_name = uploaded_file.name

        st.success(f"Uploaded: {uploaded_file.name}")

        st.info(
            "PDF extraction, semantic search, and grounded AI answers "
            "will be connected in the next implementation step."
        )

    if st.session_state.study_material_name:
        st.write(
            "Current session file:",
            st.session_state.study_material_name
        )


st.divider()

st.caption(
    f"EduMind AI • Learning Intelligence Platform • "
    f"{datetime.now().year} • SDG 4"
)

