from pymilvus import utility
from adapters.milvus_adapter import connect

def main():
    connect()
    name = "orf_profiles"
    if utility.has_collection(name):
        utility.drop_collection(name)
        print("✅ Dropped collection:", name)
    else:
        print("Collection not found:", name)

if __name__ == "__main__":
    main()