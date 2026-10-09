import json
import os
import random
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types

# ============================================================
# EduMind AI — Adaptive Learning & Skill-Gap Detection Agent
# ============================================================

st.set_page_config(
    page_title="EduMind AI | Adaptive Learning",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "EduMind AI"
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_PDF_CHARS = 120_000
CHUNK_SIZE = 1_500
CHUNK_OVERLAP = 250

# ------------------------- Styling ---------------------------

st.markdown(
    """
    <style>
      .stApp { background: #f5f7fc; }
      [data-testid="stSidebar"] { background: #15172b; }
      [data-testid="stSidebar"] * { color: #f7f7ff; }
      .hero {
        padding: 1.6rem 1.8rem; border-radius: 22px; color: white;
        background: linear-gradient(120deg, #5b4bdb, #8a5cf6 60%, #bd6af0);
        margin-bottom: 1.2rem;
      }
      .hero h1 { color: white; margin: 0; font-size: 2.2rem; }
      .hero p { color: #f0eaff; margin: .5rem 0 0 0; font-size: 1rem; }
      .small-note { color: #65708a; font-size: .9rem; }
      div[data-testid="stMetric"] {
        background: white; padding: 1rem; border-radius: 16px;
        border: 1px solid #e8eaf4;
      }
      div.stButton > button { border-radius: 10px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- Session state -----------------------

DEFAULT_STATE = {
    "pdf_name": None,
    "pdf_text": "",
    "pdf_chunks": [],
    "quiz": [],
    "quiz_topic": "Theory of Automata and Formal Languages",
    "quiz_difficulty": "Medium",
    "quiz_submitted": False,
    "quiz_answers": {},
    "quiz_results": None,
    "quiz_score": None,
    "quiz_history": [],
    "tutor_history": [],
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, (list, dict)) else value

# ------------------------ Gemini API ------------------------

def get_api_key():
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""
    return secret_key or os.getenv("GEMINI_API_KEY", "")


@st.cache_resource
def make_client(api_key):
    return genai.Client(api_key=api_key)


def get_client():
    key = get_api_key()
    if not key:
        return None
    try:
        return make_client(key)
    except Exception:
        return None


def generate_text(prompt, model=DEFAULT_MODEL, temperature=0.3):
    client = get_client()
    if client is None:
        raise RuntimeError(
            "Gemini API key not configured. Add GEMINI_API_KEY in Streamlit "
            "Settings → Secrets, or set it as an environment variable."
        )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="text/plain",
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("The AI returned an empty response. Please try again.")
    return text.strip()


def extract_json(text):
    """Parse JSON from a model response, including fenced JSON."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise ValueError("AI response was not valid JSON. Please try again.")


# ------------------------- PDF / RAG -------------------------

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    page_text = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            page_text.append(f"[Page {page_num}]\n{text.strip()}")
    full_text = "\n\n".join(page_text)
    if not full_text.strip():
        raise ValueError(
            "No selectable text was found. This may be a scanned PDF; OCR is not "
            "included in this version."
        )
    return full_text[:MAX_PDF_CHARS]


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def retrieve_relevant_chunks(question, chunks, top_k=4):
    """Lightweight lexical retrieval; avoids a separate embedding service/index."""
    terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9_+-]+", question)
        if len(term) > 2
    }
    if not terms:
        return chunks[:top_k]
    scored = []
    for index, chunk in enumerate(chunks):
        lower = chunk.lower()
        score = sum(lower.count(term) for term in terms)
        # Small preference for chunks with a denser match.
        score = score / max(len(chunk.split()) ** 0.35, 1)
        if score > 0:
            scored.append((score, index, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return chunks[:min(top_k, len(chunks))]
    return [item[2] for item in scored[:top_k]]


def answer_from_pdf(question):
    chunks = retrieve_relevant_chunks(
        question,
        st.session_state["pdf_chunks"],
        top_k=4,
    )
    if not chunks:
        raise ValueError("No readable PDF content is available. Upload and process a PDF first.")
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""
You are EduMind AI, a careful study assistant.
Answer the student's question using ONLY the supplied study-material context.
If the context does not contain enough information, clearly say that the uploaded
document does not provide enough information. Do not invent citations or facts.
Explain in beginner-friendly language. Use examples only when supported or clearly
labelled as a general illustration. Keep the answer structured.

STUDY MATERIAL CONTEXT:
{context}

STUDENT QUESTION:
{question}

Answer:
"""
    return generate_text(prompt, temperature=0.2), chunks


# ------------------------ Quiz logic -------------------------

def normalize_question(item):
    if not isinstance(item, dict):
        return None
    question = str(item.get("question", "")).strip()
    options = item.get("options", [])
    answer = str(item.get("answer", "")).strip().upper()
    explanation = str(item.get("explanation", "")).strip()
    topic = str(item.get("topic", "General concepts")).strip()
    if not question or not isinstance(options, list) or len(options) != 4:
        return None
    if answer not in {"A", "B", "C", "D"}:
        return None
    return {
        "question": question,
        "options": [str(option) for option in options],
        "answer": answer,
        "explanation": explanation or "Review the relevant concept in your study material.",
        "topic": topic,
    }


def generate_quiz(topic, difficulty, number_of_questions, use_pdf):
    if use_pdf and st.session_state["pdf_chunks"]:
        chunks = retrieve_relevant_chunks(
            f"{topic} {difficulty} key concepts definitions examples",
            st.session_state["pdf_chunks"],
            top_k=5,
        )
        source_context = "\n\n".join(chunks)
        source_instruction = (
            "Base the quiz on the study-material context below. Do not test facts "
            "that are unsupported by the context.\n\n" + source_context
        )
    else:
        source_instruction = (
            "Use standard, well-established educational knowledge about the topic."
        )

    prompt = f"""
Create exactly {number_of_questions} multiple-choice questions for a student.
Topic: {topic}
Difficulty: {difficulty}
{source_instruction}

Return ONLY valid JSON in this exact format:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
      "answer": "A",
      "explanation": "Short explanation of why the answer is correct",
      "topic": "Specific subtopic"
    }}
  ]
}}
Rules:
- Every question must have exactly four options.
- "answer" must be A, B, C, or D and match the option at that position.
- Make questions distinct and appropriate for the selected difficulty.
- Explanations should be clear and concise.
"""
    response = generate_text(prompt, temperature=0.4)
    data = extract_json(response)
    raw_questions = data.get("questions", []) if isinstance(data, dict) else data
    questions = [normalize_question(item) for item in raw_questions]
    questions = [question for question in questions if question]
    if len(questions) < 1:
        raise ValueError("The AI did not return valid quiz questions. Please try again.")
    return questions[:number_of_questions]


def evaluate_quiz(questions, answers):
    rows = []
    correct_count = 0
    for index, question in enumerate(questions):
        chosen = answers.get(str(index), "")
        correct = question["answer"]
        is_correct = chosen == correct
        correct_count += int(is_correct)
        chosen_text = (
            question["options"][ord(chosen) - ord("A")]
            if chosen in {"A", "B", "C", "D"}
            else "Not answered"
        )
        correct_text = question["options"][ord(correct) - ord("A")]
        rows.append({
            "Question": question["question"],
            "Topic": question.get("topic", "General concepts"),
            "Your Answer": f"{chosen}. {chosen_text}" if chosen else "Not answered",
            "Correct Answer": f"{correct}. {correct_text}",
            "Result": "Correct" if is_correct else "Needs revision",
            "Explanation": question.get("explanation", ""),
        })
    score = round(100 * correct_count / len(questions)) if questions else 0
    return score, pd.DataFrame(rows)


def build_learning_path(score, difficulty):
    if score < 40:
        level = "Foundation"
        next_difficulty = "Easy"
        actions = [
            "Review definitions and core concepts before attempting another quiz.",
            "Create short notes for each concept you missed.",
            "Try a short Easy quiz and explain each answer in your own words.",
        ]
    elif score < 70:
        level = "Developing"
        next_difficulty = "Medium"
        actions = [
            "Revise the concepts linked to incorrect answers.",
            "Solve worked examples and compare similar concepts.",
            "Attempt another Medium quiz after revision.",
        ]
    elif score < 90:
        level = "Proficient"
        next_difficulty = "Hard"
        actions = [
            "Practise application and analysis questions.",
            "Explain why each incorrect option is wrong.",
            "Attempt a Hard quiz to strengthen transfer of knowledge.",
        ]
    else:
        level = "Advanced"
        next_difficulty = "Hard"
        actions = [
            "Move to advanced problems and real-world applications.",
            "Teach the topic to someone else or write a concise summary.",
            "Revisit this topic later with a mixed revision quiz.",
        ]
    return {
        "learning_level": level,
        "recommended_difficulty": next_difficulty,
        "actions": actions,
    }


# ---------------------- Navigation / Header ------------------

with st.sidebar:
    st.markdown("## 🎓 EduMind AI")
    st.caption("Adaptive learning • Skill-gap detection")
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "AI Learning Tutor",
            "Adaptive Quiz",
            "Skill Gap Analysis",
            "Learning Progress",
            "Study Materials",
        ],
        key="main_navigation",
    )
    st.divider()
    st.markdown("**Project focus**")
    st.caption("SDG 4 — Quality Education")
    st.caption("Personalized learning through AI-assisted practice.")
    if get_api_key():
        st.success("Gemini API key detected")
    else:
        st.warning("Gemini API key not configured")


def render_hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


# --------------------------- Overview ------------------------

if page == "Overview":
    render_hero(
        "Your Personalized Learning Path 🎯",
        "Learn at your pace, practise with AI, and discover which concepts need revision.",
    )
    history = st.session_state["quiz_history"]
    attempts = len(history)
    average = round(sum(item["score"] for item in history) / attempts, 1) if attempts else 0
    latest = history[-1]["score"] if history else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quiz Attempts", attempts)
    c2.metric("Average Score", f"{average}%")
    c3.metric("Latest Score", f"{latest}%" if latest is not None else "—")
    c4.metric("Study Material", "Loaded" if st.session_state["pdf_text"] else "Not loaded")
    st.subheader("✨ What you can do")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("### 💬 AI Tutor")
        st.write("Ask questions and get explanations from your uploaded PDF.")
    with cols[1]:
        st.markdown("### 🧠 Adaptive Quiz")
        st.write("Generate topic-based quizzes at different difficulty levels.")
    with cols[2]:
        st.markdown("### 🎯 Skill Gaps")
        st.write("Review missed questions and focus on topics that need attention.")
    if history:
        st.subheader("Recent quiz performance")
        recent = pd.DataFrame(history[-10:])
        st.line_chart(recent.set_index("date")["score"])
    else:
        st.info("Start by opening Adaptive Quiz or uploading a PDF under Study Materials.")

# ----------------------- AI Learning Tutor -------------------

elif page == "AI Learning Tutor":
    render_hero(
        "AI Learning Tutor 💬",
        "Ask for explanations, examples, summaries, or exam-style practice.",
    )
    with st.form("tutor_form", clear_on_submit=False):
        tutor_topic = st.text_input(
            "Topic",
            placeholder="e.g., DFA, Moore machine, regression, SQL joins",
        )
        learner_level = st.selectbox(
            "Explanation level",
            ["Beginner", "Intermediate", "Exam preparation", "Advanced"],
        )
        tutor_question = st.text_area(
            "What would you like to understand?",
            placeholder="Explain the concept step by step with an example...",
            key="tutor_question_input",
        )
        submitted = st.form_submit_button("Ask EduMind AI", type="primary")
    if submitted:
        if not tutor_question.strip():
            st.warning("Enter a question first.")
        else:
            prompt = f"""
You are EduMind AI, a supportive tutor.
Topic: {tutor_topic or "Not specified"}
Learner level: {learner_level}
Student request: {tutor_question}

Respond with:
1. A clear definition or direct answer.
2. A step-by-step explanation.
3. A practical example when relevant.
4. A short recap.
5. Two quick self-check questions.
Be accurate, beginner-friendly, and do not claim that an uploaded source supports
something unless you have actually been given that source.
"""
            try:
                with st.spinner("Preparing your explanation..."):
                    answer = generate_text(prompt, temperature=0.35)
                st.markdown("### 🤖 Tutor response")
                st.markdown(answer)
                st.session_state["tutor_history"].append({
                    "question": tutor_question,
                    "answer": answer,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                })
            except Exception as exc:
                st.error(str(exc))
    if st.session_state["tutor_history"]:
        with st.expander("Recent tutor conversations"):
            for item in reversed(st.session_state["tutor_history"][-5:]):
                st.markdown(f"**You:** {item['question']}")
                st.markdown(item["answer"])
                st.divider()

# ------------------------- Adaptive Quiz ---------------------

elif page == "Adaptive Quiz":
    render_hero(
        "Adaptive Quiz 🧠",
        "Generate questions, check your answers, and get a personalized next step.",
    )
    with st.form("quiz_setup_form"):
        topic = st.text_input(
            "Quiz topic",
            value=st.session_state.get("quiz_topic") or "Theory of Automata and Formal Languages",
        )
        difficulty = st.selectbox(
            "Difficulty",
            ["Easy", "Medium", "Hard"],
            index=["Easy", "Medium", "Hard"].index(
                st.session_state.get("quiz_difficulty", "Medium")
                if st.session_state.get("quiz_difficulty", "Medium") in ["Easy", "Medium", "Hard"]
                else "Medium"
            ),
        )
        count = st.slider("Number of questions", 3, 10, 5)
        use_pdf = st.checkbox(
            "Use my uploaded PDF as the quiz source",
            value=bool(st.session_state["pdf_chunks"]),
            disabled=not bool(st.session_state["pdf_chunks"]),
        )
        make_quiz = st.form_submit_button("Generate Quiz", type="primary")

    if make_quiz:
        if not topic.strip():
            st.warning("Enter a topic.")
        else:
            try:
                with st.spinner("Creating your quiz..."):
                    generated = generate_quiz(topic.strip(), difficulty, count, use_pdf)
                st.session_state["quiz"] = generated
                st.session_state["quiz_topic"] = topic.strip()
                st.session_state["quiz_difficulty"] = difficulty
                st.session_state["quiz_submitted"] = False
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_results"] = None
                st.session_state["quiz_score"] = None
                st.success(f"Created {len(generated)} questions.")
            except Exception as exc:
                st.error(f"Could not generate the quiz: {exc}")

    quiz = st.session_state["quiz"]
    if quiz:
        st.divider()
        st.subheader(f"{st.session_state['quiz_topic']} · {st.session_state['quiz_difficulty']}")
        if not st.session_state["quiz_submitted"]:
            with st.form("answer_quiz_form"):
                answers = {}
                for idx, item in enumerate(quiz):
                    st.markdown(f"**Q{idx + 1}. {item['question']}**")
                    answers[str(idx)] = st.radio(
                        "Choose one answer",
                        options=["A", "B", "C", "D"],
                        format_func=lambda letter, q=item: f"{letter}. {q['options'][ord(letter) - ord('A')]}",
                        key=f"quiz_answer_{idx}",
                        index=None,
                    )
                    st.divider()
                submit_quiz = st.form_submit_button("Submit Quiz", type="primary")
            if submit_quiz:
                st.session_state["quiz_answers"] = {
                    key: value or "" for key, value in answers.items()
                }
                score, results = evaluate_quiz(
                    quiz, st.session_state["quiz_answers"]
                )
                st.session_state["quiz_score"] = score
                st.session_state["quiz_results"] = results
                st.session_state["quiz_submitted"] = True
                st.session_state["quiz_history"].append({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "topic": st.session_state["quiz_topic"],
                    "difficulty": st.session_state["quiz_difficulty"],
                    "score": score,
                    "questions": len(quiz),
                })
                st.rerun()
        else:
            score = st.session_state["quiz_score"]
            results = st.session_state["quiz_results"]
            correct_count = sum(1 for row in results["Result"] if row == "Correct")
            a, b, c = st.columns(3)
            a.metric("Score", f"{score}%")
            b.metric("Correct answers", f"{correct_count}/{len(quiz)}")
            c.metric("Learning level", build_learning_path(score, st.session_state["quiz_difficulty"])["learning_level"])
            st.progress(score / 100)
            st.subheader("Answer review")
            for idx, item in enumerate(quiz):
                row = results.iloc[idx]
                icon = "✅" if row["Result"] == "Correct" else "📌"
                with st.expander(f"{icon} Q{idx + 1}: {item['question']}"):
                    st.write(f"**Your answer:** {row['Your Answer']}")
                    st.write(f"**Correct answer:** {row['Correct Answer']}")
                    st.write(f"**Explanation:** {row['Explanation']}")
                    st.write(f"**Topic:** {row['Topic']}")
            path = build_learning_path(score, st.session_state["quiz_difficulty"])
            st.subheader("🎯 Your personalized learning path")
            st.info(f"Learning level: **{path['learning_level']}**")
            st.metric("Recommended next difficulty", path["recommended_difficulty"])
            for index, action in enumerate(path["actions"], start=1):
                st.write(f"{index}. {action}")
            wrong_topics = results.loc[results["Result"] != "Correct", "Topic"].value_counts()
            if not wrong_topics.empty:
                st.subheader("Topics to revise")
                st.dataframe(
                    wrong_topics.rename_axis("Topic").reset_index(name="Questions to review"),
                    use_container_width=True,
                    hide_index=True,
                )
            left, right = st.columns(2)
            with left:
                st.download_button(
                    "Download quiz review (CSV)",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="edumind_quiz_review.csv",
                    mime="text/csv",
                )
            with right:
                if st.button("Start Another Quiz", type="primary"):
                    st.session_state["quiz"] = []
                    st.session_state["quiz_submitted"] = False
                    st.session_state["quiz_answers"] = {}
                    st.session_state["quiz_results"] = None
                    st.session_state["quiz_score"] = None
                    st.rerun()

# ---------------------- Skill Gap Analysis -------------------

elif page == "Skill Gap Analysis":
    render_hero(
        "Skill-Gap Analysis 🎯",
        "Turn missed questions into a focused revision plan.",
    )
    results = st.session_state.get("quiz_results")
    if results is None or not isinstance(results, pd.DataFrame):
        st.info("Complete a quiz first to see your skill gaps.")
    else:
        gaps = results[results["Result"] != "Correct"].copy()
        total = len(results)
        correct = total - len(gaps)
        c1, c2, c3 = st.columns(3)
        c1.metric("Questions assessed", total)
        c2.metric("Correct", correct)
        c3.metric("Concepts to revisit", gaps["Topic"].nunique() if not gaps.empty else 0)
        if gaps.empty:
            st.success("Excellent! All answers were correct. Try a harder quiz to deepen your understanding.")
        else:
            st.warning("Focus your revision on the topics below.")
            st.dataframe(
                gaps[["Question", "Topic", "Your Answer", "Correct Answer", "Explanation"]],
                use_container_width=True,
                hide_index=True,
            )
            st.subheader("Suggested revision plan")
            topic_counts = gaps["Topic"].value_counts()
            for topic_name, missed_count in topic_counts.items():
                st.markdown(f"- **{topic_name}** — revisit this topic ({missed_count} missed question(s)).")
            st.caption("These recommendations are based on the latest quiz attempt, not a formal diagnosis of ability.")

# ----------------------- Learning Progress ------------------

elif page == "Learning Progress":
    render_hero(
        "Learning Progress 📈",
        "Review quiz attempts from the current app session and download a report.",
    )
    history = st.session_state["quiz_history"]
    if not history:
        st.info("Your progress will appear here after you submit a quiz.")
    else:
        df = pd.DataFrame(history)
        c1, c2, c3 = st.columns(3)
        c1.metric("Attempts", len(df))
        c2.metric("Average score", f"{df['score'].mean():.1f}%")
        c3.metric("Best score", f"{df['score'].max()}%")
        st.subheader("Score history")
        chart_df = df[["date", "score"]].copy()
        chart_df["Attempt"] = range(1, len(chart_df) + 1)
        st.line_chart(chart_df.set_index("Attempt")["score"])
        st.subheader("Recorded quiz history")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Progress Report",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="edumind_progress.csv",
            mime="text/csv",
        )
        st.caption("Progress currently lasts for this browser session. Configure a database for durable, per-user storage.")

# ------------------------- Study Materials -------------------

elif page == "Study Materials":
    render_hero(
        "AI Study Material Assistant 📚",
        "Upload a text-based PDF, process it, and ask questions grounded in its content.",
    )
    uploaded_file = st.file_uploader(
        "Upload your study material (PDF)",
        type=["pdf"],
        key="study_material_pdf_upload",
    )

    if uploaded_file is not None:
        st.caption(f"Selected file: {uploaded_file.name} · {uploaded_file.size:,} bytes")
        if st.button("Process PDF", type="primary", key="process_study_pdf"):
            try:
                with st.spinner("Extracting text and preparing study sections..."):
                    pdf_text = extract_pdf_text(uploaded_file)
                    chunks = split_into_chunks(pdf_text)
                if not chunks:
                    st.error("No usable text could be extracted from this PDF.")
                else:
                    st.session_state["pdf_name"] = uploaded_file.name
                    st.session_state["pdf_text"] = pdf_text
                    st.session_state["pdf_chunks"] = chunks
                    st.success(f"Processed {uploaded_file.name}: {len(chunks)} text sections ready.")
            except Exception as exc:
                st.error(f"Could not process this PDF: {exc}")

    if st.session_state["pdf_text"]:
        st.success(f"Loaded study material: {st.session_state['pdf_name']}")
        st.caption(f"{len(st.session_state['pdf_chunks'])} text sections available for retrieval.")
        with st.expander("Preview extracted text"):
            st.text(st.session_state["pdf_text"][:5000])

        st.divider()
        st.subheader("💬 Ask Questions About Your PDF")
        with st.form("pdf_question_form", clear_on_submit=False):
            pdf_question = st.text_area(
                "Enter your question",
                placeholder="Example: What is the difference between Moore and Mealy machines?",
                key="pdf_question_input",
            )
            ask_pdf = st.form_submit_button("Ask EduMind AI", type="primary")
        if ask_pdf:
            if not pdf_question.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    with st.spinner("Finding relevant sections and preparing an answer..."):
                        answer, cited_chunks = answer_from_pdf(pdf_question.strip())
                    st.markdown("### 🤖 EduMind AI Answer")
                    st.markdown(answer)
                    with st.expander("Study-material passages used"):
                        for index, chunk in enumerate(cited_chunks, start=1):
                            st.markdown(f"**Passage {index}**")
                            st.write(chunk)
                            st.divider()
                except Exception as exc:
                    st.error(f"Unable to answer your question: {exc}")
    else:
        st.info("Upload a PDF and click **Process PDF** to start asking questions.")

# --------------------------- Footer --------------------------

st.divider()
st.markdown(
    '<p class="small-note">EduMind AI · Adaptive Learning & Skill-Gap Detection Agent · SDG 4: Quality Education</p>',
    unsafe_allow_html=True,
)
