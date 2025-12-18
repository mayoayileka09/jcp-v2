import streamlit as st
from adapters.milvus_adapter import fetch_vector_by_id, milvus_search
from adapters.metadata_store import get_store

st.title("JCP v2 — Demo Option C")

query_id = st.text_input("Query ID", "demo_0")
k = st.slider("Top-k", 5, 50, 10)

if st.button("Search"):
    qvec = fetch_vector_by_id(dataset="orf", id_=query_id)
    hits = milvus_search(dataset="orf", query_vector=qvec, k=k)
    ids = [i for i, _ in hits]

    meta = get_store().get_metadata(ids)
    st.dataframe(meta)