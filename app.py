"""Run with: streamlit run app.py"""

import streamlit as st

from rag_core import (
    VectorStore,
    answer_question,
    chunk_pages,
    load_document,
    summarize_pages,
)

# ---------------- CONFIGURATION ----------------

SUMMARY_STYLE = "Bullet points"
CHUNK_SIZE = 800
OVERLAP = 150
TOP_K = 4

st.set_page_config(
    page_title="Document Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- SESSION STATE ----------------

if "store" not in st.session_state:
    st.session_state.store = VectorStore()
    st.session_state.pages = {}
    st.session_state.summaries = {}
    st.session_state.messages = []
    st.session_state.upload_key = 0

# ---------------- CSS ----------------
# Palette
#   Cream (main background)  #F2E8DA
#   Espresso (text, buttons) #382418
#   Sky (accent)             #8EC5FF
# Supporting tints derived from those three:
#   Cream light  #FAF5EC   Cream border  #DFD1BC   Muted text  #7A6454
#   Sky light    #D9ECFF   Sky dark      #5BA8F0   Espresso light #54392A

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #F2E8DA;
    --bg-light: #FAF5EC;
    --text: #382418;
    --text-hover: #54392A;
    --muted: #7A6454;
    --border: #DFD1BC;
    --accent: #8EC5FF;
    --accent-light: #D9ECFF;
    --accent-dark: #5BA8F0;
    --white: #FFFFFF;
}

/* Page */

.stApp {
    background:
        radial-gradient(circle at 0% 0%, #FAF3E7 0%, transparent 38%),
        linear-gradient(120deg, #F2E8DA, #EFE3D2);
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

.block-container {
    max-width: 1550px;
    padding: 1.8rem 2.2rem 2rem 2.2rem;
}

/* Typography */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

p {
    color: var(--text);
}

h1, h2, h3, h4 {
    color: var(--text) !important;
}

/* LEFT BRANDING PANEL */

.st-key-brand_panel {
    background: transparent;
    padding: 100px 18px 20px 10px;
    min-height: 650px;
}

.brand-icon {
    background: #8EC5FF;
    color: #382418;
    width: 55px;
    height: 55px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    margin-bottom: 18px;
    box-shadow: 0 5px 15px rgba(56,36,24,0.12);
}

.brand-title {
    font-size: clamp(35px, 4vw, 48px);
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: -1.8px;
    color: #382418;
    margin-bottom: 20px;
}

.brand-title span {
    color: #382418;
    background: linear-gradient(transparent 62%, #8EC5FF 62%);
}

.brand-description {
    font-size: 16px;
    line-height: 1.6;
    color: #7A6454;
    max-width: 300px;
    margin-bottom: 18px;
}

.brand-line {
    width: 48px;
    height: 3px;
    background: #8EC5FF;
    border-radius: 10px;
    margin: 20px 0;
}

.feature {
    display: flex;
    align-items: center;
    gap: 13px;
    margin: 17px 0;
}

.feature-icon {
    width: 46px;
    height: 46px;
    min-width: 46px;
    background: #D9ECFF;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 21px;
    color: #382418;
}

.feature-title {
    font-size: 13px;
    font-weight: 700;
    color: #382418;
}

.feature-description {
    font-size: 11px;
    color: #7A6454;
    margin-top: 4px;
}

/* Decorative illustration */

.illustration {
    margin-top: 45px;
    text-align: center;
    font-size: 95px;
    filter: drop-shadow(0 12px 15px rgba(56,36,24,0.15));
}

/* RIGHT SIDE CARDS */

.st-key-upload_panel,
.st-key-summary_panel,
.st-key-questions_panel {
    background: rgba(250,245,236,0.9);
    border: 1px solid #DFD1BC;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 7px 25px rgba(56,36,24,0.08);
    margin-bottom: 17px;
}

/* Card headers */

.card-heading {
    display: flex;
    align-items: center;
    gap: 13px;
    margin-bottom: 15px;
}

.card-icon {
    width: 45px;
    height: 45px;
    min-width: 45px;
    background: #D9ECFF;
    color: #382418;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 21px;
}

.card-title {
    color: #382418;
    font-size: 15px;
    font-weight: 700;
}

.card-subtitle {
    color: #7A6454;
    font-size: 11px;
    margin-top: 4px;
}

/* Upload area */

[data-testid="stFileUploader"] {
    color: var(--text);
}

[data-testid="stFileUploader"] section {
    background: #FDFAF4;
    border: 1.5px dashed #8EC5FF;
    border-radius: 14px;
    padding: 15px;
}

[data-testid="stFileUploader"] section:hover {
    border-color: #5BA8F0;
    background: #F5F9FF;
}

/* Upload text: "Drag and drop files here", size limit, file names */
[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploaderFile"] * {
    color: #382418 !important;
}

/* "Browse files" button and its inner text */
[data-testid="stFileUploader"] button {
    background: #382418 !important;
    border: 1px solid #382418 !important;
    border-radius: 22px !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button * {
    color: #F2E8DA !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #8EC5FF !important;
    border-color: #8EC5FF !important;
}

[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] button:hover * {
    color: #382418 !important;
}

/* Buttons */

.stButton > button,
.stFormSubmitButton > button {
    background: #382418 !important;
    color: #F2E8DA !important;
    border: 1px solid #382418 !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 9px 16px !important;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #8EC5FF !important;
    border-color: #8EC5FF !important;
    color: #382418 !important;
    box-shadow: 0 4px 12px rgba(142,197,255,0.4);
}

/* Button labels sit inside a <p>, which the global p color would override */
.stButton > button p,
.stFormSubmitButton > button p {
    color: #F2E8DA !important;
}

.stButton > button:hover p,
.stFormSubmitButton > button:hover p {
    color: #382418 !important;
}

.st-key-reset_button button p {
    color: #382418 !important;
}

.stButton > button:focus-visible,
.stFormSubmitButton > button:focus-visible {
    outline: 2px solid #8EC5FF !important;
    outline-offset: 2px;
}

/* Reset button */

.st-key-reset_button button {
    background: #FAF5EC !important;
    color: #382418 !important;
    border: 1px solid #DFD1BC !important;
}

.st-key-reset_button button:hover {
    background: #D9ECFF !important;
    border-color: #8EC5FF !important;
    color: #382418 !important;
}

/* Summary empty state */

.empty-summary {
    background: linear-gradient(120deg, #F2E8DA, #F7EFE3);
    border-radius: 14px;
    min-height: 115px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}

.empty-summary-icon {
    font-size: 45px;
    color: #8EC5FF;
    margin-bottom: 7px;
}

.empty-summary-text {
    color: #7A6454;
    font-size: 13px;
    font-weight: 600;
}

/* Summary content */

.st-key-summary_panel h3 {
    font-size: 15px !important;
    color: #382418 !important;
}

.st-key-summary_panel p,
.st-key-summary_panel li {
    font-size: 13px;
    line-height: 1.7;
    color: #382418 !important;
}

/* Questions */

.st-key-questions_panel [data-testid="stForm"] {
    background: transparent;
    border: none;
    padding: 0;
}

.st-key-questions_panel input {
    background: #FFFFFF !important;
    border: 1px solid #DFD1BC !important;
    border-radius: 25px !important;
    color: #382418 !important;
    padding: 11px 15px !important;
}

.st-key-questions_panel input:focus {
    border-color: #8EC5FF !important;
    box-shadow: 0 0 0 2px rgba(142,197,255,0.45) !important;
}

.st-key-questions_panel input::placeholder {
    color: #9A8574 !important;
}

.st-key-questions_panel label p {
    color: #382418 !important;
}

/* Chat messages */

[data-testid="stChatMessage"] {
    background: #F2E8DA;
    border: 1px solid #E3D6C2;
    border-radius: 13px;
    padding: 12px;
    margin: 8px 0;
}

[data-testid="stChatMessage"] p {
    color: #382418 !important;
    font-size: 13px;
}

/* Expanders */

[data-testid="stExpander"] {
    border: 1px solid #DFD1BC;
    border-radius: 10px;
    background: #FDFAF4;
}

[data-testid="stExpander"] summary {
    color: #382418 !important;
    font-weight: 600;
}

[data-testid="stExpander"] summary:hover {
    color: #5BA8F0 !important;
}

/* Alerts */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Captions */

[data-testid="stCaptionContainer"] {
    color: #7A6454 !important;
}

/* Responsive */

@media(max-width: 850px) {
    .block-container {
        padding: 1rem;
    }

    .st-key-brand_panel {
        padding: 15px 5px;
        min-height: auto;
    }

    .brand-title {
        font-size: 36px;
    }

    .illustration {
        display: none;
    }
}

</style>
""", unsafe_allow_html=True)


# ---------------- LEFT BRANDING ----------------

left, right = st.columns([0.85, 1.65], gap="large")

with left:

    with st.container(key="brand_panel"):

        st.markdown("""
        <div class="brand-title">
            Document<br>
            <span>Assistant</span>
        </div>

        <div class="brand-description">
            Upload, summarize and explore your documents
            with the power of AI.
        </div>

        <div class="brand-line"></div>

        <div class="feature">
            <div class="feature-icon">ϟ</div>
            <div>
                <div class="feature-title">Instant Summaries</div>
                <div class="feature-description">
                    Get key insights at a glance
                </div>
            </div>
        </div>

        <div class="feature">
            <div class="feature-icon">▢</div>
            <div>
                <div class="feature-title">Ask Anything</div>
                <div class="feature-description">
                    Dive deeper with questions
                </div>
            </div>
        </div>

        <div class="feature">
            <div class="feature-icon">♙</div>
            <div>
                <div class="feature-title">Secure & Private</div>
                <div class="feature-description">
                    Your data stays yours
                </div>
            </div>
        </div>

        <div class="illustration"></div>
        """, unsafe_allow_html=True)


# ---------------- RIGHT CONTENT ----------------

with right:

    # ================= UPLOAD CARD =================

    with st.container(key="upload_panel"):

        st.markdown("""
        <div class="card-heading">
            <div class="card-icon">☁</div>
            <div>
                <div class="card-title">Upload Your Documents</div>
                <div class="card-subtitle">
                    Support for PDF, DOCX, and TXT files
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        files = st.file_uploader(
            "Drag and drop files here",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key=f"document_uploader_{st.session_state.upload_key}"
        )

        col1, col2 = st.columns([3, 1])

        with col1:

            if st.button(
                "Process Documents",
                use_container_width=True
            ):

                if not files:

                    st.warning("Please choose a document first.")

                else:

                    with st.spinner(
                        "Reading, indexing and summarizing..."
                    ):

                        for f in files:

                            if f.name in st.session_state.pages:
                                continue

                            try:

                                pages = load_document(
                                    f.read(),
                                    f.name
                                )

                                chunks = chunk_pages(
                                    pages,
                                    f.name,
                                    CHUNK_SIZE,
                                    OVERLAP
                                )

                                st.session_state.store.add(chunks)
                                st.session_state.pages[f.name] = pages

                                st.success(
                                    f"{f.name}: {len(chunks)} chunks indexed"
                                )

                            except Exception as e:

                                st.error(f"{f.name}: {e}")
                                continue

                            try:

                                st.session_state.summaries[f.name] = (
                                    summarize_pages(
                                        pages,
                                        SUMMARY_STYLE
                                    )
                                )

                            except Exception as e:

                                st.session_state.summaries[f.name] = None

                                st.warning(
                                    f"{f.name}: Summary failed: {e}"
                                )

                    st.rerun()

        with col2:

            with st.container(key="reset_button"):

                if st.button(
                    "↻ Reset",
                    use_container_width=True
                ):

                    st.session_state.store = VectorStore()
                    st.session_state.pages = {}
                    st.session_state.summaries = {}
                    st.session_state.messages = []
                    st.session_state.upload_key += 1

                    st.rerun()


    # ================= SUMMARY CARD =================

    with st.container(key="summary_panel"):

        st.markdown("""
        <div class="card-heading">
            <div class="card-icon">▤</div>
            <div>
                <div class="card-title">Summary</div>
                <div class="card-subtitle">
                    Get a concise summary of your document
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.pages:

            st.markdown("""
            <div class="empty-summary">
                <div class="empty-summary-icon"></div>
                <div class="empty-summary-text">
                    Your summary will appear here
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            for name, doc_pages in st.session_state.pages.items():

                st.markdown(f"###  {name}")

                summary = st.session_state.summaries.get(name)

                if summary:

                    st.markdown(summary)

                else:

                    st.warning("Summary not available.")

                    if st.button(
                        "Try Again",
                        key=f"retry_{name}"
                    ):

                        with st.spinner("Generating summary..."):

                            try:

                                st.session_state.summaries[name] = (
                                    summarize_pages(
                                        doc_pages,
                                        SUMMARY_STYLE
                                    )
                                )

                                st.rerun()

                            except Exception as e:

                                st.error(f"LLM error: {e}")


    # ================= QUESTIONS CARD =================

    with st.container(key="questions_panel"):

        st.markdown("""
        <div class="card-heading">
            <div class="card-icon">?</div>
            <div>
                <div class="card-title">Ask Questions</div>
                <div class="card-subtitle">
                    Ask anything about your document
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.pages:

            st.caption(
                "Upload and process a document to start asking questions."
            )

        # Conversation history

        for m in st.session_state.messages:

            with st.chat_message(m["role"]):

                st.markdown(m["content"])

                if m.get("sources"):

                    with st.expander(" View Sources"):

                        for s in m["sources"]:

                            st.markdown(
                                f"**{s['source']}, Page {s['page']}** "
                                f"(Score: {s['score']:.2f})"
                            )

                            st.caption(s["text"])

        # Question input

        with st.form("question_form", clear_on_submit=True):

            question = st.text_input(
                "Ask a question",
                placeholder="Ask a question about your document..."
            )

            submitted = st.form_submit_button(
                "➤",
                use_container_width=True
            )

        if submitted:

            if not question.strip():

                st.warning("Please enter a question.")

            elif not st.session_state.pages:

                st.warning("Please process a document first.")

            else:

                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })

                try:

                    with st.spinner("AI is thinking..."):

                        answer, results = answer_question(
                            st.session_state.store,
                            question,
                            TOP_K
                        )

                except Exception as e:

                    answer = f"LLM error: {e}"
                    results = []

                sources = [
                    {
                        "source": c.source,
                        "page": c.page,
                        "score": score,
                        "text": c.text
                    }
                    for c, score in results
                ]

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

                st.rerun()