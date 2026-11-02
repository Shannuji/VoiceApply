
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from collections import Counter

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from docx import Document

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "voiceapply.db"

st.set_page_config(
    page_title="VoiceApply",
    page_icon="✍️",
    layout="wide"
)


# ==========================================
# DATABASE FUNCTIONS
# ==========================================

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            document_type TEXT NOT NULL,
            extracted_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS past_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            original_answer TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS generated_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            draft TEXT NOT NULL,
            final_answer TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS privacy_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_text TEXT NOT NULL UNIQUE,
            replacement_text TEXT NOT NULL
        );
        """)


# ==========================================
# DOCUMENT EXTRACTION
# ==========================================

def extract_text(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        try:
            reader = PdfReader(uploaded_file)
            extracted = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            ).strip()

            if extracted:
                return extracted

        except Exception:
            pass

        try:
            uploaded_file.seek(0)
            import pdfplumber

            with pdfplumber.open(uploaded_file) as pdf:
                pages_text = [
                    page.extract_text() or ""
                    for page in pdf.pages
                ]

                return "\n".join(pages_text).strip()

        except Exception as e:
            raise ValueError(
                f"Could not extract text from PDF: {e}"
            )

    if name.endswith(".docx"):
        doc = Document(uploaded_file)

        parts = [
            p.text for p in doc.paragraphs
            if p.text.strip()
        ]

        for table in doc.tables:
            for row in table.rows:
                parts.append(
                    " | ".join(cell.text for cell in row.cells)
                )

        return "\n".join(parts).strip()

    if name.endswith(".txt"):
        return uploaded_file.getvalue().decode(
            "utf-8",
            errors="replace"
        ).strip()

    raise ValueError(
        "Unsupported file type. Please upload PDF, DOCX or TXT."
    )


# ==========================================
# DATA RETRIEVAL
# ==========================================

def get_documents():
    with db() as conn:
        return conn.execute(
            "SELECT * FROM documents ORDER BY id DESC"
        ).fetchall()


def get_past_answers():
    with db() as conn:
        return conn.execute(
            "SELECT * FROM past_answers ORDER BY id DESC"
        ).fetchall()


def get_rules():
    with db() as conn:
        return conn.execute(
            "SELECT * FROM privacy_rules ORDER BY id DESC"
        ).fetchall()


# ==========================================
# PRIVACY SUBSTITUTIONS
# ==========================================

def anonymise(text):
    for rule in get_rules():
        text = re.sub(
            re.escape(rule["original_text"]),
            rule["replacement_text"],
            text,
            flags=re.IGNORECASE
        )

    return text


# ==========================================
# WRITING STYLE ANALYSIS
# ==========================================

def sentence_lengths(text):
    sentences = [
        s.strip()
        for s in re.split(r"[.!?]+", text)
        if s.strip()
    ]

    return [
        len(re.findall(r"\b[\w'-]+\b", sentence))
        for sentence in sentences
    ]


def build_style_profile(texts):
    combined = "\n".join(
        text for text in texts if text.strip()
    )

    if not combined.strip():
        return (
            "No writing samples uploaded yet. "
            "Use a clear, natural, professional tone."
        )

    lengths = sentence_lengths(combined)

    avg = (
        round(sum(lengths) / len(lengths), 1)
        if lengths else 0
    )

    words = re.findall(
        r"\b[a-zA-Z']+\b",
        combined.lower()
    )

    stop = {
        "the", "and", "for", "that", "with", "this",
        "was", "are", "have", "has", "from", "you",
        "your", "our", "but", "not", "they", "their",
        "into", "about", "which", "when", "were",
        "will", "would", "can", "could", "been",
        "being", "also", "very", "more", "than",
        "then", "what", "who", "how", "all", "use",
        "used", "using", "had", "its", "because",
        "through", "each", "such", "while", "where",
        "there", "here", "after", "before", "during",
        "over", "under", "an", "to", "of", "in",
        "on", "at", "is", "it", "as", "by", "or",
        "be", "i", "my", "me", "we", "us"
    }

    common = [
        word
        for word, count in Counter(
            word for word in words
            if word not in stop and len(word) > 3
        ).most_common(8)
    ]

    if avg < 12:
        sentence_style = "short, direct sentences"
    elif avg < 22:
        sentence_style = "medium-length sentences"
    else:
        sentence_style = "longer, more detailed sentences"

    return (
        f"Observed style from uploaded writing: usually "
        f"{sentence_style} (average about {avg} words per sentence). "
        f"Frequently used content words: "
        f"{', '.join(common) if common else 'not enough data'}. "
        "Treat this as a rough signal, not a definitive personality profile."
    )


# ==========================================
# QUESTION INTENT DETECTION
# ==========================================

def get_question_intent(question):
    q = question.lower().strip()

    categories = {
        "suitability": [
            "why should we consider",
            "why should we hire",
            "why are you suitable",
            "why are you a good fit",
            "why should we select",
            "right candidate",
            "ideal candidate",
            "value can you bring",
            "what makes you a good candidate"
        ],

        "strengths": [
            "what are your strengths",
            "your key strengths",
            "what makes you unique",
            "greatest strength",
            "strongest skills"
        ],

        "weaknesses": [
            "your weaknesses",
            "greatest weakness",
            "area of improvement",
            "what would you improve"
        ],

        "experience": [
            "professional experience",
            "work experience",
            "previous experience",
            "describe your experience",
            "tell us about your experience"
        ],

        "projects": [
            "tell us about a project",
            "describe a project",
            "project you worked on",
            "project experience",
            "explain your project"
        ],

        "skills": [
            "technical skills",
            "what skills",
            "skills do you possess",
            "your skill set",
            "programming skills"
        ],

        "motivation": [
            "why do you want to join",
            "why this company",
            "why do you want to work",
            "what motivates you",
            "why are you interested"
        ],

        "achievements": [
            "your achievements",
            "proudest achievement",
            "biggest accomplishment",
            "greatest achievement"
        ],

        "teamwork": [
            "teamwork",
            "worked in a team",
            "team conflict",
            "team collaboration",
            "team project"
        ]
    }

    for category, phrases in categories.items():
        if any(phrase in q for phrase in phrases):
            return category

    return "general"


# ==========================================
# RELEVANT PREVIOUS ANSWER RETRIEVAL
# ==========================================

def retrieve_similar(question, limit=3):

    rows = get_past_answers()

    if not rows:
        return []

    question_intent = get_question_intent(question)

    if question_intent == "general":
        return []

    matching_rows = [
        row for row in rows
        if get_question_intent(row["question"])
        == question_intent
    ]

    if not matching_rows:
        return []

    corpus = [question] + [
        row["question"] for row in matching_rows
    ]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(corpus)

        scores = cosine_similarity(
            matrix[0:1],
            matrix[1:]
        ).flatten()

        ranked = sorted(
            zip(matching_rows, scores),
            key=lambda item: item[1],
            reverse=True
        )

        return [
            (row, float(score))
            for row, score in ranked[:limit]
            if score >= 0.20
        ]

    except ValueError:
        return []


# ==========================================
# GEMINI PROMPT CREATION
# ==========================================

def make_prompt(question, facts, samples, style, anonymised):

    intent = get_question_intent(question)

    sample_text = "\n\n".join(
        f"Previous question: {row['question']}\n"
        f"Previous answer: {row['original_answer']}"
        for row, score in samples
    ) or "No relevant previous answers available."

    privacy_note = (
        "The supplied information has been anonymised."
        if anonymised
        else "The user has chosen not to anonymise the supplied information."
    )

    prompt = f"""
You are an accurate and personalised job application assistant.

YOUR MAIN OBJECTIVE:
Answer the exact question asked. Relevance is more important than
including every detail from the applicant's background.

QUESTION CATEGORY:
{intent}

STRICT RULES:

1. Understand exactly what the question is asking.
2. Answer the question directly in the opening sentence.
3. Use only information supported by the applicant's documents
   and relevant saved answers.
4. Include an experience or project ONLY when it directly supports
   the question.
5. Do not force project descriptions into questions about strengths,
   suitability, motivation or other topics.
6. Never invent skills, experience, achievements, qualifications,
   employers, responsibilities or results.
7. Previous answers are references, not mandatory content.
8. Ignore previous answers that do not directly help answer
   the current question.
9. Preserve the applicant's natural writing style.
10. Keep the answer concise, specific and professional.
11. Do not repeat the question.
12. If the available information is insufficient, do not guess.
    Ask one short clarification question instead.
13. Return only the answer or the clarification question.
14. {privacy_note}

SPECIAL INSTRUCTION FOR SUITABILITY QUESTIONS:

For questions such as "Why should we consider you for this role?",
focus on the applicant's relevant skills, strengths, qualifications
and suitability.

Use a project only as supporting evidence when it genuinely
demonstrates a relevant skill.

WRITING STYLE:
{style}

APPLICANT'S DOCUMENTS:
{facts[:12000] if facts else "No documents provided."}

RELEVANT PREVIOUS ANSWERS:
{sample_text[:8000]}

CURRENT QUESTION:
{question}

Now produce the most relevant answer supported by the available facts.
"""

    return prompt.strip()


# ==========================================
# GEMINI API GENERATION
# ==========================================

def generate_with_gemini(prompt):

    try:
        from dotenv import load_dotenv

        load_dotenv(
            APP_DIR / ".env",
            override=True
        )

        api_key = os.getenv(
            "GEMINI_API_KEY",
            ""
        ).strip()

        if not api_key:
            return None, (
                "GEMINI_API_KEY is missing or invalid. "
                "Add your API key to the .env file."
            )

        from google import genai

        client = genai.Client(
            api_key=api_key
        )

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 500
            }
        )

        answer = (
            response.text or ""
        ).strip()

        if not answer:
            return None, "Gemini returned an empty answer."

        return answer, None

    except Exception as exc:
        return None, f"Gemini API request failed: {exc}"


# ==========================================
# SAFE FALLBACK
# ==========================================

def fallback_draft(question, facts, samples):

    return (
        "I could not generate a personalised answer at this time. "
        "Please try again when the AI service is available. "
        "No previous answer has been automatically reused because "
        "its relevance to the current question has not been verified."
    )


# ==========================================
# INITIALISE APPLICATION
# ==========================================

init_db()

st.title("✍️ VoiceApply")

st.caption(
    "A privacy-aware assistant for job application answers "
    "that sound like you."
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("Project status")

    st.write("Local prototype • SQLite storage")

    st.info(
        "Upload writing samples first. "
        "Add a Gemini API key only if you want LLM-generated drafts."
    )

    st.markdown(
        "**Privacy note:** Uploaded content is stored in a local "
        "SQLite file on this computer. When using an LLM API, "
        "selected context is sent to that provider."
    )


# ==========================================
# APPLICATION TABS
# ==========================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Build your profile",
    "2. Add past answers",
    "3. Generate an answer",
    "4. Privacy rules",
    "5. History & report"
])


# ==========================================
# TAB 1: BUILD PROFILE
# ==========================================

with tab1:

    st.subheader(
        "Upload resume, cover letters or writing samples"
    )

    uploads = st.file_uploader(
        "Choose PDF, DOCX or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.button(
        "Extract and save documents",
        type="primary"
    ):

        if not uploads:
            st.warning("Choose at least one file first.")

        else:

            saved = 0

            for file in uploads:

                try:
                    content = extract_text(file)

                    if not content:
                        st.warning(
                            f"No text could be extracted from {file.name}."
                        )
                        continue

                    with db() as conn:
                        conn.execute(
                            """
                            INSERT INTO documents
                            (filename, document_type, extracted_text, created_at)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                file.name,
                                Path(file.name).suffix.lower(),
                                content,
                                datetime.now().isoformat(
                                    timespec="seconds"
                                )
                            )
                        )

                    saved += 1

                except Exception as exc:
                    st.error(
                        f"Could not read {file.name}: {exc}"
                    )

            st.success(
                f"Saved {saved} document(s)."
            )

    docs = get_documents()

    if docs:

        all_text = "\n\n".join(
            doc["extracted_text"]
            for doc in docs
        )

        st.write(
            f"**Documents saved:** {len(docs)}"
        )

        st.text_area(
            "Extracted text preview",
            all_text[:8000],
            height=220
        )

        st.markdown(
            "**Writing-style analysis**"
        )

        st.write(
            build_style_profile(
                [doc["extracted_text"] for doc in docs]
                + [
                    row["original_answer"]
                    for row in get_past_answers()
                ]
            )
        )

    else:

        st.info(
            "No documents yet. You can start with a resume "
            "and 2–3 pieces of your own writing."
        )


# ==========================================
# TAB 2: ADD PAST ANSWERS
# ==========================================

with tab2:

    st.subheader(
        "Store previous questions and answers"
    )

    st.write(
        "Saved answers are stored exactly as entered. "
        "They are not overwritten by anonymisation."
    )

    with st.form(
        "past_answer_form",
        clear_on_submit=True
    ):

        past_q = st.text_area(
            "Previous application question",
            placeholder="Tell us about a project you are proud of."
        )

        past_a = st.text_area(
            "Your original answer",
            placeholder="Paste the answer you actually wrote."
        )

        submitted = st.form_submit_button(
            "Save original answer"
        )

    if submitted:

        if not past_q.strip() or not past_a.strip():

            st.error(
                "Both the question and answer are required."
            )

        else:

            with db() as conn:
                conn.execute(
                    """
                    INSERT INTO past_answers
                    (question, original_answer, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (
                        past_q.strip(),
                        past_a,
                        datetime.now().isoformat(
                            timespec="seconds"
                        )
                    )
                )

            st.success(
                "Original answer saved exactly as entered."
            )

    saved_answers = get_past_answers()

    st.write(
        f"**Saved answers:** {len(saved_answers)}"
    )

    for row in saved_answers[:10]:

        with st.expander(
            row["question"][:100]
        ):

            st.write(
                row["original_answer"]
            )


# ==========================================
# TAB 3: GENERATE ANSWER
# ==========================================

with tab3:

    st.subheader(
        "Generate a personalised answer"
    )

    question = st.text_area(
        "Paste the job application's question",
        placeholder="Why do you want to join our company?"
    )

    anonymise_enabled = st.checkbox(
        "Apply my privacy substitutions before generating",
        value=True
    )

    docs = get_documents()
    past = get_past_answers()

    facts = "\n\n".join(
        doc["extracted_text"]
        for doc in docs
    )

    style = build_style_profile(
        [doc["extracted_text"] for doc in docs]
        + [
            row["original_answer"]
            for row in past
        ]
    )

    if st.button(
        "Find references and generate draft",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Enter a question first."
            )

        else:

            matches = retrieve_similar(
                question.strip()
            )

            st.markdown(
                "**Retrieved previous answers**"
            )

            if matches:

                for row, score in matches:

                    st.write(
                        f"- Similarity score: `{score:.2f}` "
                        f"— {row['question']}"
                    )

            else:

                st.write(
                    "No sufficiently similar saved question found."
                )

            prompt_facts = (
                anonymise(facts)
                if anonymise_enabled
                else facts
            )

            prompt_samples = []

            for row, score in matches:

                sample = dict(row)

                if anonymise_enabled:

                    sample["question"] = anonymise(
                        sample["question"]
                    )

                    sample["original_answer"] = anonymise(
                        sample["original_answer"]
                    )

                prompt_samples.append(
                    (sample, score)
                )

            prompt = make_prompt(
                question.strip(),
                prompt_facts,
                prompt_samples,
                style,
                anonymise_enabled
            )

            answer, error = generate_with_gemini(
                prompt
            )

            if answer is None:

                answer = fallback_draft(
                    question.strip(),
                    prompt_facts,
                    prompt_samples
                )

                st.warning(
                    "LLM generation was not available. "
                    + error
                )

            st.session_state["latest_question"] = (
                question.strip()
            )

            st.session_state["latest_draft"] = answer

            st.session_state["latest_prompt"] = prompt

    if st.session_state.get("latest_draft"):

        st.markdown(
            "### Draft — review before using"
        )

        edited = st.text_area(
            "Edit the answer",
            value=st.session_state["latest_draft"],
            height=260,
            key="edited_answer"
        )

        st.caption(
            "Check every claim. Remove placeholders and correct "
            "anything that does not reflect your real experience."
        )

        if st.button(
            "Save reviewed answer"
        ):

            with db() as conn:

                conn.execute(
                    """
                    INSERT INTO generated_answers
                    (question, draft, final_answer, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        st.session_state.get(
                            "latest_question", ""
                        ),
                        st.session_state["latest_draft"],
                        edited,
                        datetime.now().isoformat(
                            timespec="seconds"
                        )
                    )
                )

            st.success(
                "Reviewed answer saved."
            )


# ==========================================
# TAB 4: PRIVACY RULES
# ==========================================

with tab4:

    st.subheader(
        "Privacy substitutions"
    )

    st.write(
        "Example: replace a specific client name with "
        "'a client' or an internal project name with "
        "'an internal project'."
    )

    with st.form(
        "privacy_form",
        clear_on_submit=True
    ):

        original = st.text_input(
            "Exact text to replace",
            placeholder="Example Corp"
        )

        replacement = st.text_input(
            "Replacement text",
            placeholder="a previous employer"
        )

        add_rule = st.form_submit_button(
            "Add substitution"
        )

    if add_rule:

        if not original.strip() or not replacement.strip():

            st.error(
                "Enter both values."
            )

        else:

            try:

                with db() as conn:

                    conn.execute(
                        """
                        INSERT INTO privacy_rules
                        (original_text, replacement_text)
                        VALUES (?, ?)
                        """,
                        (
                            original.strip(),
                            replacement.strip()
                        )
                    )

                st.success(
                    "Privacy substitution saved."
                )

            except sqlite3.IntegrityError:

                st.warning(
                    "That exact substitution already exists."
                )

    rules = get_rules()

    if rules:

        st.write(
            "**Current rules**"
        )

        for rule in rules:

            c1, c2, c3 = st.columns([3, 3, 1])

            c1.write(
                rule["original_text"]
            )

            c2.write(
                "→ " + rule["replacement_text"]
            )

            if c3.button(
                "Delete",
                key=f"delete_rule_{rule['id']}"
            ):

                with db() as conn:

                    conn.execute(
                        "DELETE FROM privacy_rules WHERE id=?",
                        (rule["id"],)
                    )

                st.rerun()

    st.caption(
        "This is a simple exact-text substitution prototype, "
        "not a full named-entity recognition system. "
        "Review the output for missed identifiers."
    )


# ==========================================
# TAB 5: HISTORY AND REPORT
# ==========================================

with tab5:

    st.subheader(
        "Answer history"
    )

    with db() as conn:

        generated = conn.execute(
            """
            SELECT * FROM generated_answers
            ORDER BY id DESC
            """
        ).fetchall()

    st.write(
        f"**Generated/reviewed answers saved:** {len(generated)}"
    )

    for row in generated:

        with st.expander(
            row["question"][:100] or "Untitled question"
        ):

            st.markdown(
                "**Draft**"
            )

            st.write(
                row["draft"]
            )

            st.markdown(
                "**Reviewed answer**"
            )

            st.write(
                row["final_answer"] or "(Not reviewed)"
            )

            st.caption(
                row["created_at"]
            )

    st.markdown("---")

    st.subheader(
        "Assessment report checklist"
    )

    st.markdown("""
    - [x] Local database preserves original saved answers word for word
    - [x] Writing-style profile based on observable text patterns
    - [x] Similar-question retrieval using TF-IDF and cosine similarity
    - [x] Optional LLM generation with constraints against invented facts
    - [x] User review/edit before saving a final answer
    - [x] User-configured privacy substitutions
    - [ ] Test with several real application questions
    - [ ] Add authentication and per-user access controls before public deployment
    - [ ] Add deletion/export controls and document the data-retention policy
    - [ ] Test privacy substitutions and hallucination prevention
    """)
