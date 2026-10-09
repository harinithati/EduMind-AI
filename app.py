import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from google import genai
from pypdf import PdfReader
import io

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
def extract_pdf_text(uploaded_file):
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            pages.append(
                f"Page {page_number}:\n{page_text}"
            )

    return "\n\n".join(pages)
def split_text_into_chunks(text, chunk_size=1200, overlap=200):
    """
    Divide extracted PDF text into overlapping chunks.
    Overlap helps preserve context between adjacent chunks.
    """
    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

import numpy as np
import faiss


def create_semantic_index(text_chunks):
    """Create a FAISS index for the supplied PDF text chunks."""

    if client is None:
        raise ValueError("Gemini API key is missing.")

    if not text_chunks:
        raise ValueError("No text chunks were provided.")

    embeddings = []

    for chunk in text_chunks:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=chunk,
            config={"output_dimensionality": 768}
        )

        embeddings.append(response.embeddings[0].values)

    embedding_matrix = np.asarray(embeddings, dtype="float32")

    # Normalize vectors so inner product represents cosine similarity.
    faiss.normalize_L2(embedding_matrix)

    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    return index, embedding_matrix


def search_pdf(question, text_chunks, index, top_k=3):
    """Retrieve the most relevant PDF passages for a question."""

    if client is None:
        raise ValueError("Gemini API key is missing.")

    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=question,
        config={"output_dimensionality": 768}
    )

    query_vector = np.asarray(
        [response.embeddings[0].values],
        dtype="float32"
    )

    faiss.normalize_L2(query_vector)

    k = min(top_k, len(text_chunks))
    scores, indices = index.search(query_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                "text": text_chunks[idx],
                "similarity": float(score)
            })

    return results
def answer_from_retrieved_chunks(question, retrieved_chunks):

    if client is None:
        return "Gemini API key is missing. Check Streamlit Secrets."

    if not retrieved_chunks:
        return "No relevant information was found in the uploaded PDF."

    context = "\n\n".join(
        [
            f"Passage {i + 1}:\n{item['text']}"
            for i, item in enumerate(retrieved_chunks)
        ]
    )

    prompt = f"""
You are EduMind AI, an educational assistant.

Answer the student's question using the retrieved study material below.

Instructions:
1. Use the retrieved passages as your primary source.
2. Explain concepts clearly in student-friendly language.
3. Do not invent information that is not supported by the passages.
4. If the passages do not contain the answer, say so.
5. Include a short example when supported by the material.

STUDY MATERIAL:
{context}

STUDENT QUESTION:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text or "No answer was generated."

def answer_from_pdf(question, pdf_text):

    if not pdf_text or not pdf_text.strip():
        return "Please upload and process a PDF first."

    if client is None:
        return "Gemini API key is missing. Check Streamlit Secrets."

    # Limit context to avoid sending an excessively large prompt.
    context = pdf_text[:30000]

    prompt = f"""
You are EduMind AI, an educational assistant.

Answer the student's question using the provided study material.

Rules:
1. Use the study material as your primary source.
2. Do not invent facts that are not supported by the material.
3. If the answer is not found in the material, say so clearly.
4. Explain the answer in simple language.
5. Include the relevant page number if it is available in the context.

STUDY MATERIAL:
{context}

STUDENT QUESTION:
{question}

Provide a clear, structured answer.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text or "No answer was generated."
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

    st.markdown("## 🤖 EduMind AI Tutor")
    st.write("Ask questions and explore concepts interactively.")

    for message in st.session_state.tutor_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask your learning question...")

    if question:

        st.session_state.tutor_messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        if client is None:
            answer = "Gemini API key is missing. Check Streamlit Secrets."

        else:
            conversation = "\n".join(
                f"{m['role']}: {m['content']}"
                for m in st.session_state.tutor_messages[-10:]
            )

            prompt = f"""
You are EduMind AI, a patient educational tutor.

Explain concepts clearly, using examples where helpful.
Adapt your explanation to the student's question.
Use simple language unless advanced detail is requested.

Conversation:
{conversation}

Respond to the latest student question.
"""

            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                answer = response.text or "No response was generated."

            except Exception:
                answer = (
                    "I couldn't generate an answer. "
                    "Please check your Gemini API configuration and try again."
                )

        st.session_state.tutor_messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):
            st.markdown(answer)

    if st.button("Clear Conversation"):
        st.session_state.tutor_messages = []
        st.rerun()


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
if "tutor_messages" not in st.session_state:
    st.session_state.tutor_messages = []

if "tutor_history" not in st.session_state:
    st.session_state.tutor_history = []

# ---------- STUDY MATERIALS ----------
elif page == "Study Materials":

    st.markdown("## 📚 AI Study Material Assistant")
    st.write("Upload a PDF and ask questions grounded in its content.")

    uploaded_file = st.file_uploader(
        "Upload your study material",
        type=["pdf"],
        key="rag_pdf_upload"
    )

    if uploaded_file:
        if st.button("Process PDF", type="primary"):

            with st.spinner("Extracting text and building the search index..."):
                try:
                    pdf_text = extract_pdf_text(uploaded_file)

                    if not pdf_text.strip():
                        st.error("No readable text found in this PDF.")
                    else:
                        chunks, index = process_pdf_for_rag(pdf_text)

                        st.session_state["rag_pdf_text"] = pdf_text
                        st.session_state["rag_chunks"] = chunks
                        st.session_state["rag_index"] = index
                        st.session_state["rag_pdf_name"] = uploaded_file.name

                        st.success("Study material processed successfully!")
                        st.write("**File:**", uploaded_file.name)
                        st.write("**Text chunks indexed:**", len(chunks))

                except Exception as e:
                    st.error("Unable to process the PDF.")
                    st.exception(e)

    if st.session_state.get("rag_index") is not None:

        st.divider()
        st.subheader("💬 Ask Your Study Material")

        question = st.text_area(
            "Enter your question",
            placeholder="Explain the difference between Moore and Mealy machines.",
            key="rag_question"
        )

        if st.button("Ask EduMind AI", type="primary"):

            if not question.strip():
                st.warning("Please enter a question.")

            else:
                with st.spinner("Searching your PDF and generating an answer..."):
                    try:
                        retrieved_chunks = search_pdf(
                            question,
                            st.session_state["rag_chunks"],
                            st.session_state["rag_index"],
                            top_k=3
                        )

                        answer = answer_from_retrieved_chunks(
                            question,
                            retrieved_chunks
                        )

                        st.markdown("### 📘 Answer")
                        st.write(answer)

                        with st.expander("View retrieved passages"):
                            for i, item in enumerate(retrieved_chunks, start=1):
                                st.markdown(
                                    f"**Passage {i}** "
                                    f"(similarity: {item['similarity']:.3f})"
                                )
                                st.write(item["text"])

                    except Exception as e:
                        st.error("Unable to answer this question.")
                        st.exception(e)


    st.divider()

    st.subheader("💬 Ask Questions About Your PDF")

    question = st.text_area(
        "Enter your question",
        placeholder="Example: What is the difference between Moore and Mealy machines?"
    )

    if st.button("Ask EduMind AI"):

        pdf_text = st.session_state.get("pdf_text", "")

        if not pdf_text:
            st.warning("Upload and process a PDF first.")

        elif not question.strip():
            st.warning("Please enter a question.")

        else:
            with st.spinner("Finding an answer in your study material..."):

                try:
                    answer = answer_from_pdf(question, pdf_text)

                    st.markdown("### 📘 Answer")
                    st.write(answer)

                except Exception as e:
                    st.error("Unable to generate an answer.")
                    st.caption(str(e))

