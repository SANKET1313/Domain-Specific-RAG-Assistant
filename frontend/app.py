"""Streamlit frontend for Domain-Specific RAG Assistant.

Thin-client web interface that communicates with the FastAPI backend over HTTP.
All RAG pipeline execution, embedding models, vector database operations, and
secrets remain server-side.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict

import requests
import streamlit as st

logger = logging.getLogger("streamlit_frontend")

st.set_page_config(
    page_title="Domain RAG Assistant",
    page_icon="📚",
    layout="wide",
)
st.title("📚 Domain-Specific RAG Assistant")


# --- Configuration & Health Check -----------------------------------------
def get_backend_url() -> str:
    """Retrieve the backend API URL from the environment.
    
    In production (e.g., on Render), this reads the API_URL environment variable
    pointing to the deployed FastAPI service (e.g., https://domain-rag-backend.onrender.com).
    For local development, it defaults to http://127.0.0.1:8000.
    """
    raw_url = os.getenv("API_URL", "http://127.0.0.1:8000").strip()
    return raw_url.rstrip("/")


API_URL = get_backend_url()


def check_backend_health() -> tuple[bool, str]:
    """Check connectivity to the FastAPI backend /health endpoint.
    
    Returns:
        tuple of (is_healthy, diagnostic_message)
    """
    if not (API_URL.startswith("http://") or API_URL.startswith("https://")):
        return False, f"Invalid API_URL format: '{API_URL}'. Must begin with http:// or https://"
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.status_code == 200:
            return True, "Backend healthy"
        return False, f"Backend returned HTTP {resp.status_code} ({resp.reason})"
    except requests.exceptions.Timeout:
        return False, f"Connection timed out contacting backend at {API_URL} (timeout=3s)"
    except requests.exceptions.ConnectionError:
        return False, f"Connection refused or backend unavailable at {API_URL}"
    except requests.exceptions.RequestException as exc:
        return False, f"Network request error contacting {API_URL}: {type(exc).__name__}"
    except Exception as exc:
        return False, f"Unexpected error contacting backend: {type(exc).__name__}"


is_backend_online, health_diagnostic = check_backend_health()

if not is_backend_online:
    st.error(
        f"🚨 **FastAPI Backend Unreachable**\n\n"
        f"- **Target URL**: `{API_URL}`  \n"
        f"- **Diagnostic**: {health_diagnostic}\n\n"
        "**Troubleshooting Steps**:\n"
        "1. **Render / Production**: Verify that the backend service is deployed and healthy. "
        "In the frontend service's Environment settings on Render, set `API_URL` to the backend's URL "
        "(e.g., `https://domain-rag-backend.onrender.com`).\n"
        "2. **Local Development**: Ensure the backend server is running in a terminal:  \n"
        "   `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload` (or `python run.py`).\n"
        "3. **Docker Compose**: Verify that services are up and `API_URL=http://backend:8000` is set.",
        icon="🔌",
    )
else:
    st.sidebar.success(f"🟢 Backend Connected: `{API_URL}`")


# --- Helper Functions -----------------------------------------------------
def fetch_documents() -> list[dict]:
    """Fetch the list of indexed documents from the vector database."""
    try:
        resp = requests.get(f"{API_URL}/documents", timeout=15)
        if resp.ok:
            return resp.json()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch documents from %s: %s", API_URL, exc)
    return []


# --- Navigation Tabs ------------------------------------------------------
tab_upload, tab_docs, tab_qa, tab_eval = st.tabs(
    ["Upload & Index", "Indexed Documents", "Ask", "Evaluation"]
)


# --- 1. Upload & Index Tab ------------------------------------------------
with tab_upload:
    st.subheader("Upload & Index Documents")
    st.markdown(
        "Upload documents and images (`.pdf`, `.docx`, `.txt`, `.md`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`). "
        "The system supports digital PDFs, embedded tables/charts, standalone images, "
        "and scanned documents via hybrid Vision OCR."
    )

    uploaded_files = st.file_uploader(
        "Choose file(s) to index",
        type=["pdf", "txt", "md", "docx", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
        accept_multiple_files=True,
        key="main_document_uploader",
        help="Maximum 200 MB per file. Supports documents and images.",
    )

    if uploaded_files:
        st.write(f"Selected **{len(uploaded_files)}** file(s):")
        for f in uploaded_files:
            st.caption(f"📄 **{f.name}** ({f.size / 1024:.1f} KB)")

        if st.button("🚀 Ingest & Index Document(s)", type="primary", key="btn_ingest_docs"):
            progress_bar = st.progress(0)
            status_container = st.container()

            results = []
            for i, file_obj in enumerate(uploaded_files):
                with st.spinner(f"Processing '{file_obj.name}' ({i + 1}/{len(uploaded_files)})..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/upload",
                            files={"file": (file_obj.name, file_obj.getvalue())},
                            timeout=300,
                        )
                        if resp.ok:
                            data = resp.json()
                            results.append({
                                "File": file_obj.name,
                                "Status": "✅ Indexed",
                                "Document ID": data.get("document_id", "-"),
                                "Type": data.get("document_type", "digital"),
                                "Pages": data.get("total_pages", 1),
                                "Digital": data.get("pages_digital", 0),
                                "OCR": data.get("pages_ocr", 0),
                                "Chunks": data.get("chunk_count", 0),
                            })
                            if data.get("warnings"):
                                for w in data["warnings"]:
                                    st.warning(f"⚠️ {file_obj.name}: {w}")
                        elif resp.status_code in (400, 413):
                            results.append({
                                "File": file_obj.name,
                                "Status": f"❌ Validation Error: {resp.text[:100]}",
                                "Document ID": "-",
                                "Type": "-",
                                "Pages": 0,
                                "Digital": 0,
                                "OCR": 0,
                                "Chunks": 0,
                            })
                        elif resp.status_code == 422:
                            results.append({
                                "File": file_obj.name,
                                "Status": f"❌ Extraction Error: {resp.text[:100]}",
                                "Document ID": "-",
                                "Type": "-",
                                "Pages": 0,
                                "Digital": 0,
                                "OCR": 0,
                                "Chunks": 0,
                            })
                        else:
                            results.append({
                                "File": file_obj.name,
                                "Status": f"❌ Server Error ({resp.status_code}): {resp.text[:80]}",
                                "Document ID": "-",
                                "Type": "-",
                                "Pages": 0,
                                "Digital": 0,
                                "OCR": 0,
                                "Chunks": 0,
                            })
                    except requests.exceptions.Timeout:
                        results.append({
                            "File": file_obj.name,
                            "Status": "❌ Ingestion Timed Out (>300s)",
                            "Document ID": "-",
                            "Type": "-",
                            "Pages": 0,
                            "Digital": 0,
                            "OCR": 0,
                            "Chunks": 0,
                        })
                    except requests.exceptions.ConnectionError:
                        results.append({
                            "File": file_obj.name,
                            "Status": f"❌ Connection Error: Backend API at {API_URL} is offline",
                            "Document ID": "-",
                            "Type": "-",
                            "Pages": 0,
                            "Digital": 0,
                            "OCR": 0,
                            "Chunks": 0,
                        })
                    except Exception as exc:
                        results.append({
                            "File": file_obj.name,
                            "Status": f"❌ Failed: {type(exc).__name__}",
                            "Document ID": "-",
                            "Type": "-",
                            "Pages": 0,
                            "Digital": 0,
                            "OCR": 0,
                            "Chunks": 0,
                        })

                progress_bar.progress((i + 1) / len(uploaded_files))

            with status_container:
                st.success("Indexing complete!")
                st.dataframe(results, use_container_width=True)


# --- 2. Documents Tab -----------------------------------------------------
with tab_docs:
    st.subheader("Document Index Manager")

    docs = fetch_documents()

    total_docs = len(docs)
    total_chunks = sum(d.get("chunk_count", 0) for d in docs)

    # Detect duplicates
    docs_by_name = defaultdict(list)
    for d in docs:
        docs_by_name[d.get("document_name", "unknown")].append(d)

    duplicates = [items for items in docs_by_name.values() if len(items) > 1]
    duplicate_count = sum(len(items) - 1 for items in duplicates)

    # Top metrics bar
    m1, m2, m3 = st.columns(3)
    m1.metric("Indexed Documents", total_docs)
    m2.metric("Total Chunks in Vector DB", total_chunks)
    m3.metric("Duplicate Entries", duplicate_count)

    # Top Action Bar
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1.5, 2])
    with btn_col1:
        if st.button("🔄 Refresh List", use_container_width=True, key="btn_refresh_docs"):
            st.rerun()

    with btn_col2:
        if duplicate_count > 0:
            if st.button("🧹 Remove Duplicates", use_container_width=True, key="btn_remove_duplicates", help="Keeps newest version and removes duplicate older uploads"):
                deleted_dup_count = 0
                for items in duplicates:
                    sorted_items = sorted(items, key=lambda x: x.get("upload_time") or "", reverse=True)
                    for older_item in sorted_items[1:]:
                        try:
                            r = requests.delete(f"{API_URL}/documents/{older_item['document_id']}", timeout=30)
                            if r.ok:
                                deleted_dup_count += 1
                        except Exception:
                            pass
                st.success(f"Cleaned up {deleted_dup_count} duplicate document entries!")
                st.rerun()

    with btn_col3:
        with st.expander("⚠️ Clear Entire Index"):
            confirm_clear = st.checkbox("Confirm: Delete ALL documents from vector database?", key="chk_confirm_wipe_all")
            if confirm_clear and st.button("🗑️ Wipe All Documents", type="primary", key="btn_wipe_all_docs"):
                try:
                    r = requests.delete(f"{API_URL}/documents", timeout=60)
                    if r.ok:
                        st.success("All documents removed from vector index!")
                        st.rerun()
                    else:
                        st.error(f"Failed to clear index: {r.text}")
                except Exception as exc:
                    st.error(f"Error: {exc}")

    st.markdown("---")

    if not docs:
        st.info("No documents are currently indexed in the vector store. Upload files in the 'Upload & Index' tab to get started.")
    else:
        # Overview Table
        st.subheader("Current Documents Overview")
        st.dataframe(
            [
                {
                    "Name": d.get("document_name"),
                    "Uploaded": d.get("upload_time"),
                    "Status": d.get("status"),
                    "Chunks": d.get("chunk_count"),
                    "ID": d.get("document_id"),
                }
                for d in docs
            ],
            use_container_width=True,
        )

        # Inspection / Debugging Tool
        with st.expander("🔍 Inspect Document Content & Chunks (Diagnostics)", expanded=False):
            inspect_doc_options = {
                f"{d.get('document_name')} ({d.get('document_id')})": d.get("document_id")
                for d in docs
            }
            inspect_target = st.selectbox("Select document to inspect:", options=list(inspect_doc_options.keys()), key="select_inspect_doc")
            if inspect_target and st.button("Inspect Content", key="btn_inspect_doc"):
                target_id = inspect_doc_options[inspect_target]
                try:
                    r = requests.get(f"{API_URL}/documents/{target_id}/inspect", timeout=30)
                    if r.ok:
                        inspect_data = r.json()
                        st.write(f"**Document ID:** `{inspect_data['document_id']}` | **Total Chunks:** {inspect_data['chunk_count']}")
                        if inspect_data.get("original_file_path"):
                            st.caption(f"📁 **Source File:** `{inspect_data['original_file_path']}`")
                        if inspect_data.get("markdown_path"):
                            st.caption(f"📝 **Canonical Markdown:** `{inspect_data['markdown_path']}`")

                        # Generated Markdown Preview
                        md_content = inspect_data.get("markdown_content")
                        if md_content:
                            with st.expander("📝 View Generated Markdown Representation", expanded=False):
                                st.text_area("Markdown Source", md_content, height=300)

                        sections_list = inspect_data.get("sections", [])
                        if sections_list:
                            st.markdown("#### 📑 Detected Document Sections")
                            sec_rows = [
                                {
                                    "Section ID": s.get("section_id", "n/a"),
                                    "Heading": s.get("section_heading", "n/a"),
                                    "Pages": f"{s.get('first_page')} - {s.get('last_page')}",
                                    "Chunks": s.get("chunk_count", 0),
                                }
                                for s in sections_list
                            ]
                            st.dataframe(sec_rows, use_container_width=True)

                        st.markdown("#### 🧱 Chunk Details & Boundaries")
                        for ch in inspect_data.get("chunks", []):
                            heading_label = f" — [{ch.get('section_heading')}]" if ch.get('section_heading') else ""
                            sub_label = f" ({ch.get('subsection_heading')})" if ch.get('subsection_heading') else ""
                            p_start = ch.get("page_start", ch.get("page_number", "?"))
                            p_end = ch.get("page_end", p_start)
                            page_range = f"{p_start}–{p_end}" if p_start != p_end else str(p_start)
                            with st.expander(f"Chunk `{ch['chunk_id']}` (Page {page_range}{heading_label}{sub_label})"):
                                st.text(ch.get("chunk_text", ""))
                                st.caption(f"Position: {ch.get('chunk_position', 0)} | Tokens: {ch.get('token_count', 0)} | Section ID: `{ch.get('section_id', 'none')}`")
                    else:
                        st.error(f"Failed to inspect: {r.text}")
                except Exception as exc:
                    st.error(f"Inspection error: {exc}")

        st.markdown("### 🗑️ Delete Documents")

        # Multi-select Batch Delete
        doc_options = {
            f"📄 {d.get('document_name')} (ID: {d.get('document_id')}, {d.get('chunk_count')} chunks)": d.get("document_id")
            for d in docs
        }

        col_del1, col_del2 = st.columns([3, 1])
        with col_del1:
            selected_to_delete = st.multiselect(
                "Select one or more documents to delete:",
                options=list(doc_options.keys()),
                help="Choose documents to delete them together",
                key="select_batch_delete_docs",
            )
        with col_del2:
            st.write("")
            st.write("")
            if selected_to_delete:
                if st.button("🗑️ Delete Selected", type="primary", use_container_width=True, key="btn_batch_delete"):
                    doc_ids_to_del = [doc_options[k] for k in selected_to_delete]
                    try:
                        r = requests.post(
                            f"{API_URL}/documents/batch-delete",
                            json={"document_ids": doc_ids_to_del},
                            timeout=60,
                        )
                        if r.ok:
                            st.success(f"Successfully deleted {len(doc_ids_to_del)} document(s)!")
                            st.rerun()
                        else:
                            st.error(f"Failed: {r.text}")
                    except Exception as exc:
                        st.error(f"Error: {exc}")

        # Individual Document Actions
        st.markdown("### ⚙️ Individual Document Management")
        for d in docs:
            doc_id = d.get("document_id", "")
            doc_name = d.get("document_name", "Untitled")
            chunks = d.get("chunk_count", 0)
            upload_time = d.get("upload_time", "N/A")

            with st.container(border=True):
                c_info, c_reindex, c_delete = st.columns([3, 1, 1])
                with c_info:
                    st.markdown(f"**📄 {doc_name}**")
                    st.caption(f"ID: `{doc_id}` | Chunks: **{chunks}** | Uploaded: {upload_time}")
                with c_reindex:
                    if st.button("🔄 Re-index", key=f"reindex_{doc_id}", use_container_width=True, help="Re-extracts and re-indexes this document"):
                        with st.spinner(f"Reindexing {doc_name}..."):
                            try:
                                r = requests.post(f"{API_URL}/documents/{doc_id}/reindex", timeout=300)
                                if r.ok:
                                    st.success(f"Re-indexed '{doc_name}' ({r.json().get('chunk_count')} chunks)!")
                                    st.rerun()
                                else:
                                    st.error(f"Reindex failed: {r.text}")
                            except Exception as exc:
                                st.error(f"Error: {exc}")
                with c_delete:
                    if st.button("🗑️ Delete", key=f"del_{doc_id}", type="secondary", use_container_width=True, help="Deletes this document from the vector store"):
                        try:
                            r = requests.delete(f"{API_URL}/documents/{doc_id}", timeout=30)
                            if r.ok:
                                st.success(f"Deleted '{doc_name}'!")
                                st.rerun()
                            else:
                                st.error(f"Delete failed: {r.text}")
                        except Exception as exc:
                            st.error(f"Error: {exc}")


# --- 3. Ask Tab -----------------------------------------------------------
with tab_qa:
    st.subheader("Ask a question")

    docs = fetch_documents()
    filter_doc_id = None
    if docs:
        filter_options = {"All Documents (Global Search)": None}
        for d in docs:
            filter_options[f"📄 {d.get('document_name')} ({d.get('document_id')})"] = d.get("document_id")

        selected_filter = st.selectbox(
            "Filter Context by Document (Optional):",
            options=list(filter_options.keys()),
            key="select_filter_doc_qa",
        )
        filter_doc_id = filter_options[selected_filter]

    question = st.text_input("Your question", key="input_qa_question")
    if question and st.button("Get answer", type="primary", key="btn_get_answer"):
        with st.spinner("Searching documents & generating answer..."):
            try:
                payload = {"query": question}
                if filter_doc_id:
                    payload["document_id"] = filter_doc_id

                resp = requests.post(f"{API_URL}/query", json=payload, timeout=120)
                if resp.ok:
                    data = resp.json()
                    st.markdown("### Answer")
                    st.write(data["answer"])
                    if not data.get("sufficient_context", True):
                        st.warning("Insufficient context — answer not grounded in documents.")
                    st.markdown("### Sources")
                    if data.get("sources"):
                        for s in data["sources"]:
                            st.markdown(
                                f"- 📄 **{s['document']}** — Page {s.get('page', 'n/a')}"
                            )
                    else:
                        st.caption("No sources cited.")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except requests.exceptions.Timeout:
                st.error("Request timed out waiting for the LLM response (>120s).")
            except requests.exceptions.ConnectionError:
                st.error(f"Connection error: Could not reach backend at {API_URL}.")
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")


# --- 4. Evaluation Tab ----------------------------------------------------
with tab_eval:
    st.subheader("Evaluation & monitoring")
    try:
        m = requests.get(f"{API_URL}/metrics", timeout=15).json()
        summary = m.get("summary", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Queries served", summary.get("query_count", 0))
        c2.metric("Avg total latency (ms)", summary.get("avg_total_latency_ms") or 0)
        c3.metric("Avg cost / query ($)", summary.get("avg_estimated_cost_usd") or 0)
        c4, c5, c6 = st.columns(3)
        c4.metric("Avg retrieval latency (ms)", summary.get("avg_retrieval_latency_ms") or 0)
        c5.metric("Avg retrieved chunks", summary.get("avg_retrieved_chunks") or 0)
        c6.metric("Insufficient-context rate", summary.get("insufficient_context_rate") or 0)
        st.markdown("#### Active configuration")
        st.json(m.get("config", {}))
        st.markdown("#### Recent queries")
        st.dataframe(m.get("recent", []), use_container_width=True)
        st.info(
            "Ragas scores (Context Recall, Faithfulness, Answer Relevancy) are "
            "produced by running `python -m app.evaluation.ragas_eval` and are "
            "saved under data/evaluation/results/."
        )
    except requests.exceptions.ConnectionError:
        st.error(f"Could not reach backend at {API_URL}. Backend server is offline.")
    except requests.RequestException as exc:
        st.error(f"Could not retrieve metrics: {exc}")
