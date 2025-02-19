import json
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os

# Load environment variables (for OpenAI API key)
load_dotenv()

def preprocessing():
    with open("data/dev-v2.0.json", 'r') as f:
        data = json.load(f)

    # Establish client
    chroma_client = chromadb.PersistentClient(path="data/my_chromadb")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    # create a collection named squad_contexts, with a provided embedding function
    collection = chroma_client.get_or_create_collection(
        name="squad_contexts",
        embedding_function=openai_ef
    )

    # Initialize counter for unique ids
    id_counter = 0

    # Extract context chunks and store in Chroma
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            context_id = str(id_counter)
            collection.add(
                documents=[context],
                ids=[context_id]
            )
            # Increment the counter
            id_counter += 1
            print(f"Successfully added {id_counter}")

    print(f"Successfully stored {id_counter} context chunks in the Chroma database.")

preprocessing()