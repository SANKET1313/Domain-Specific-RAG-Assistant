"""Streamlit frontend.

This is a thin client: it only calls the FastAPI backend over HTTP. No RAG
logic, API keys, or secrets live here (keys stay server-side).

Run the backend first, then:
    streamlit run frontend/app.py
"""
from __future__ import annotations

import os
from collections import defaultdict

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Domain RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Domain-Specific RAG Assistant")

tab_upload, tab_docs, tab_qa, tab_eval = st.tabs(
    ["Upload & Index", "Indexed Documents", "Ask", "Evaluation"]
)


# --- Helper Functions -----------------------------------------------------
def fetch_documents() -> list[dict]:
    try:
        resp = requests.get(f"{API_URL}/documents", timeout=30)
        if resp.ok:
            return resp.json()
    except requests.RequestException:
        pass
    return []


# --- Upload & Index -------------------------------------------------------
with tab_upload:
    st.subheader("Upload & Index Documents")
    st.markdown(
        "Upload documents and images (`.pdf`, `.docx`, `.txt`, `.md`, `.png`, `.jpg`, `.jpeg`, `.webp`). "
        "The system supports digital PDFs, embedded tables/charts, standalone images, "
        "and scanned documents via hybrid Vision OCR."
    )

    uploaded_files = st.file_uploader(
        "Choose file(s) to index",
        type=["pdf", "txt", "md", "docx", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write(f"Selected **{len(uploaded_files)}** file(s):")
        for f in uploaded_files:
            st.caption(f"📄 **{f.name}** ({f.size / 1024:.1f} KB)")

        if st.button("🚀 Ingest & Index Document(s)", type="primary"):
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
                                "Document ID": data.get("document_id"),
                                "Type": data.get("document_type", "digital"),
                                "Pages": data.get("total_pages", 1),
                                "Digital": data.get("pages_digital", 0),
                                "OCR": data.get("pages_ocr", 0),
                                "Chunks": data.get("chunk_count", 0),
                            })
                            if data.get("warnings"):
                                for w in data["warnings"]:
                                    st.warning(f"⚠️ {file_obj.name}: {w}")
                        else:
                            results.append({
                                "File": file_obj.name,
                                "Status": f"❌ Error: {resp.text[:80]}",
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
                            "Status": f"❌ Failed: {exc}",
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


# --- Documents Tab --------------------------------------------------------
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
        if st.button("🔄 Refresh List", use_container_width=True):
            st.rerun()

    with btn_col2:
        if duplicate_count > 0:
            if st.button("🧹 Remove Duplicates", use_container_width=True, help="Keeps newest version and removes duplicate older uploads"):
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
            confirm_clear = st.checkbox("Confirm: Delete ALL documents from vector database?")
            if confirm_clear and st.button("🗑️ Wipe All Documents", type="primary"):
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

    # Quick Upload & Index Expander inside Documents Tab
    with st.expander("📥 Upload & Index Documents", expanded=False):
        quick_files = st.file_uploader(
            "Upload and index documents immediately",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            key="quick_upload_docs",
        )
        if quick_files and st.button("Index Uploaded Files", key="btn_quick_upload", type="primary"):
            with st.spinner("Indexing uploaded files..."):
                for qf in quick_files:
                    try:
                        requests.post(
                            f"{API_URL}/upload",
                            files={"file": (qf.name, qf.getvalue())},
                            timeout=300,
                        )
                    except Exception:
                        pass
                st.success("Upload and indexing completed!")
                st.rerun()

    if not docs:
        st.info("No documents are currently indexed in the vector store. Upload or index files above to get started.")
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
            inspect_target = st.selectbox("Select document to inspect:", options=list(inspect_doc_options.keys()))
            if inspect_target and st.button("Inspect Content"):
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
            )
        with col_del2:
            st.write("")
            st.write("")
            if selected_to_delete:
                if st.button("🗑️ Delete Selected", type="primary", use_container_width=True):
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


# --- Ask ------------------------------------------------------------------
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
        )
        filter_doc_id = filter_options[selected_filter]

    question = st.text_input("Your question")
    if question and st.button("Get answer", type="primary"):
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
                    if data["sources"]:
                        for s in data["sources"]:
                            st.markdown(
                                f"- 📄 **{s['document']}** — Page {s.get('page', 'n/a')}"
                            )
                    else:
                        st.caption("No sources cited.")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")


# --- Evaluation -----------------------------------------------------------
with tab_eval:
    st.subheader("Evaluation & monitoring")
    try:
        m = requests.get(f"{API_URL}/metrics", timeout=30).json()
        summary = m["summary"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Queries served", summary.get("query_count", 0))
        c2.metric("Avg total latency (ms)", summary.get("avg_total_latency_ms") or 0)
        c3.metric("Avg cost / query ($)", summary.get("avg_estimated_cost_usd") or 0)
        c4, c5, c6 = st.columns(3)
        c4.metric("Avg retrieval latency (ms)", summary.get("avg_retrieval_latency_ms") or 0)
        c5.metric("Avg retrieved chunks", summary.get("avg_retrieved_chunks") or 0)
        c6.metric("Insufficient-context rate", summary.get("insufficient_context_rate") or 0)
        st.markdown("#### Active configuration")
        st.json(m["config"])
        st.markdown("#### Recent queries")
        st.dataframe(m["recent"], use_container_width=True)
        st.info(
            "Ragas scores (Context Recall, Faithfulness, Answer Relevancy) are "
            "produced by running `python -m app.evaluation.ragas_eval` and are "
            "saved under data/evaluation/results/."
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach backend: {exc}")
