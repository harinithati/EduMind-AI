import streamlit as st

st.set_page_config(
    page_title="EduMind AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 EduMind AI")
st.subheader("Adaptive Learning & Skill-Gap Detection Agent")

st.markdown("""
Welcome to EduMind AI — your personalized learning assistant!

### Features
- 📚 Study material assistance
- 🤖 AI Tutor
- 📝 Adaptive quizzes
- 🎯 Skill-gap detection
- 📊 Student progress tracking

Designed to support **SDG 4: Quality Education**.
""")

st.divider()

page = st.sidebar.radio(
    "Choose a Feature",
    [
        "Home",
        "AI Tutor",
        "Adaptive Quiz",
        "Skill-Gap Analysis",
        "Progress Dashboard"
    ]
)

if page == "Home":
    st.success("Welcome! Select a feature from the sidebar.")

elif page == "AI Tutor":
    st.header("🤖 AI Tutor")
    question = st.text_area("Ask a question about your study material")

    if st.button("Ask AI"):
        if question.strip():
            st.info("AI Tutor integration will be added next.")
        else:
            st.warning("Please enter a question.")

elif page == "Adaptive Quiz":
    st.header("📝 Adaptive Quiz")
    topic = st.text_input(
        "Enter a topic",
        "Moore and Mealy Machines"
    )

    difficulty = st.selectbox(
        "Select Difficulty",
        ["Easy", "Medium", "Hard", "Advanced"]
    )

    if st.button("Generate Quiz"):
        st.info(
            f"Quiz interface ready for {topic} at {difficulty} difficulty."
        )

elif page == "Skill-Gap Analysis":
    st.header("🎯 Skill-Gap Analysis")

    score = st.slider("Enter your quiz score", 0, 100, 50)

    if score < 40:
        st.warning("Revise fundamental concepts.")
    elif score < 70:
        st.info("Practise more questions to strengthen understanding.")
    elif score < 90:
        st.success("Good progress! Practise application-based questions.")
    else:
        st.balloons()
        st.success("Excellent! Try advanced questions.")

elif page == "Progress Dashboard":
    st.header("📊 Progress Dashboard")
    st.info("Student progress tracking will be integrated next.")

st.divider()
st.caption("EduMind AI | SDG 4: Quality Education")
