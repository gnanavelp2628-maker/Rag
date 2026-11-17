
import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


# ---------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="Multi-PDF RAG Application",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Application")
st.write(
    "Upload multiple PDF documents and ask questions "
    "across all documents with source references."
)

# ---------------------------------------
# 2. SESSION STATE
# ---------------------------------------

if "store" not in st.session_state:
    st.session_state.store = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ---------------------------------------
# 3. API KEY CHECK
# ---------------------------------------

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OpenAI API key not found. "
        "Please configure OPENAI_API_KEY and restart Streamlit."
    )
    st.stop()


# ---------------------------------------
# 4. MULTIPLE PDF UPLOAD
# ---------------------------------------

st.header("1. Upload PDF Documents")

uploaded_files = st.file_uploader(
    "Choose multiple PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    st.success(f"{len(uploaded_files)} PDF file(s) selected.")

    for uploaded_file in uploaded_files:
        st.write("📄", uploaded_file.name)


# ---------------------------------------
# 5. BUILD VECTOR DATABASE
# ---------------------------------------

if st.button("Build Knowledge Base", type="primary"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")

    else:

        try:

            all_documents = []
            all_chunks = []

            progress = st.progress(0)
            status = st.empty()

            for index, uploaded_file in enumerate(uploaded_files):

                status.info(
                    f"Processing {uploaded_file.name}..."
                )

                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                try:

                    # Load PDF
                    loader = PyPDFLoader(temp_path)
                    documents = loader.load()

                    # Preserve original filename
                    for document in documents:
                        document.metadata["source"] = (
                            uploaded_file.name
                        )

                    all_documents.extend(documents)

                finally:

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            # Split all documents into chunks
            status.info("Splitting documents into chunks...")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            all_chunks = text_splitter.split_documents(
                all_documents
            )

            if not all_chunks:
                st.error(
                    "No readable text found in the uploaded PDFs."
                )
                st.stop()

            # Create embeddings
            status.info("Creating embeddings and FAISS database...")

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key
            )

            # Store every document in ONE FAISS database
            vector_store = FAISS.from_documents(
                all_chunks,
                embeddings
            )

            # Save in Streamlit session state
            st.session_state.store = vector_store

            st.session_state.chunk_count = len(all_chunks)

            st.session_state.indexed_files = [
                f.name for f in uploaded_files
            ]

            status.success("Knowledge base created successfully!")

            st.success(
                f"Successfully indexed {len(all_chunks)} chunks!"
            )

            st.info(
                f"Total PDFs processed: {len(uploaded_files)}"
            )

        except Exception as e:

            st.error(f"Error while indexing documents: {e}")


# ---------------------------------------
# 6. DISPLAY INDEXED DOCUMENTS
# ---------------------------------------

if st.session_state.store is not None:

    st.divider()

    st.header("2. Knowledge Base Information")

    col1, col2 = st.columns(2)

    col1.metric(
        "PDF Documents",
        len(st.session_state.indexed_files)
    )

    col2.metric(
        "Indexed Chunks",
        st.session_state.chunk_count
    )

    st.write("### Indexed Files")

    for filename in st.session_state.indexed_files:
        st.write(f"📄 {filename}")


# ---------------------------------------
# 7. ASK QUESTIONS
# ---------------------------------------

st.divider()

st.header("3. Ask Questions")

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is the main topic of these documents?"
)

if st.button("Get Answer"):

    if st.session_state.store is None:

        st.warning(
            "Please upload PDFs and build the knowledge base first."
        )

    elif not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            with st.spinner("Searching documents and generating answer..."):

                # Retrieve relevant chunks
                retriever = st.session_state.store.as_retriever(
                    search_kwargs={"k": 4}
                )

                docs = retriever.invoke(question)

                if not docs:

                    st.warning(
                        "No relevant information was found."
                    )

                else:

                    # Prepare document context
                    context_parts = []

                    for doc in docs:

                        page_number = (
                            int(doc.metadata.get("page", 0)) + 1
                        )

                        filename = doc.metadata.get(
                            "source",
                            "Unknown file"
                        )

                        context_parts.append(
                            f"Source: {filename}, "
                            f"Page: {page_number}\n"
                            f"Content: {doc.page_content}"
                        )

                    context = "\n\n".join(context_parts)

                    # Generate answer
                    llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        temperature=0,
                        api_key=api_key
                    )

                    prompt = f"""
You are a helpful PDF document assistant.

Answer the question using ONLY the provided context.

If the documents do not contain the answer,
clearly say that the information was not found.

If multiple documents provide conflicting information,
explain the disagreement and identify the relevant sources.

Do not invent facts.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

                    response = llm.invoke(prompt)

                    # Display answer
                    st.subheader("Answer")

                    st.write(response.content)

                    # Display sources
                    st.subheader("Sources")

                    seen_sources = set()

                    for doc in docs:

                        filename = doc.metadata.get(
                            "source",
                            "Unknown file"
                        )

                        page_number = (
                            int(doc.metadata.get("page", 0)) + 1
                        )

                        source_key = (
                            filename,
                            page_number
                        )

                        if source_key not in seen_sources:

                            seen_sources.add(source_key)

                            st.caption(
                                f"📄 {filename} - Page {page_number}"
                            )

        except Exception as e:

            st.error(f"Error while answering question: {e}")


# ---------------------------------------
# 8. FOOTER
# ---------------------------------------

st.divider()

st.caption(
    "Multi-PDF RAG Application | "
    "Built with Python, LangChain, Streamlit and FAISS"
)
