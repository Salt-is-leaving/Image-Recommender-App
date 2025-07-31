import faiss-cpu
print("FAISS imported successfully!")
print(f"FAISS can create index: {faiss.IndexFlatL2(128) is not None}")