import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Multimodal RAG Assistant",
    layout="centered",
)
st.title("Multimodal RAG Assistant!")
mode = st.radio(
    "Select mode",
    ["Text RAG", "Image RAG", "SQL RAG"],
    horizontal=True,
)
st.divider()
if mode == "Text RAG":
    st.subheader("Text Retrieval (RAG)")
    question = st.text_area(
        "Enter your question",
        height=120,
    )

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                resp = requests.post(
                    f"{API_BASE}/ask",
                    params={"question": question},
                    timeout=600,
                )

                data = resp.json()
                st.success("Answer")
                st.write(data.get("answer", ""))
                if "score" in data:
                    st.text(f"Faithfulness score: {data['score']}")

elif mode == "Image RAG":
    st.subheader("Image-based RAG")
    question = st.text_area(
        "Optional question about the image",
        height=100,
    )
    image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
    )

    if st.button("Ask with Image"):
        if not image and not question.strip():
            st.warning("Provide at least a question or an image.")
        else:
            with st.spinner("Processing..."):
                data = {"top_k": 5}
                files = {}

                if question.strip():
                    data["question"] = question

                if image:
                    files["image"] = (
                        image.name,
                        image.getvalue(),
                        image.type,
                    )

                resp = requests.post(
                    f"{API_BASE}/ask-image",
                    data=data,
                    files=files,
                    timeout=1000,
                )

                result = resp.json()
                st.success("Answer")
                st.write(result.get("answer", ""))

                if "score" in result:
                    st.text(f"Faithfulness score: {result['score']}")

else:
    st.subheader("SQL Assistant")
    question = st.text_area(
        "Ask a question about the database",
        height=120,
    )

    if st.button("Run SQL Query"):
        with st.spinner("Running SQL query..."):
            resp = requests.post(
                f"{API_BASE}/ask-sql",
                params={"question": question},
                timeout=600,
            )
        if resp.status_code == 200:
            data = resp.json()
            st.subheader("Answer")
            st.write(data.get("summary", "No summary found"))
        else:
            st.error(f"Error: {resp.text}")
