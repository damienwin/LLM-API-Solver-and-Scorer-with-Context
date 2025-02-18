import json
import os
import time

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv('.env')

openai_client = ChatCompletionsClient(
    endpoint=os.environ["AZURE_MLSTUDIO_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["AZURE_MLSTUDIO_KEY"]),
)

chroma_client = chromadb.PersistentClient(path="data/my_chromadb")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

collection = chroma_client.get_collection(
    name="squad_contexts",
    embedding_function=openai_ef
)

system_prompt = "You are a smart AI model. Answer this question correctly and keep it as short and concise as possible, prioritizing answering questions correctly."
user_prompt = """You are a smart AI assistant that answers questions using data returned by a search engine.

Guidelines:
\t1. You will be provided with a question by the user, you must answer that question, and nothing else.
\t2. Your answer should come directly from the provided context from the search engine.
\t3. Do not make up any information not provided in the context.
\t4. If the provided question does not contain the answers, respond with 'I am sorry, but I am unable to answer that question.'
\t5. Be aware that some chunks in the context may be irrelevant, incomplete, and/or poorly formatted.

Here is the provided context:
{context}

Here is the question: {question}

Your response: """

def retrieve_context(idx, question, n_results=5):
    print(f"Retrieving most relevant contexts for the question: {idx}")

    results = collection.query(query_texts=[question], n_results=n_results)
    return results["documents"][0]

def possible_questions():
    with open("data/dev-v2.0.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_responses = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if not qa["is_impossible"]:
                    valid_responses.append(qa["question"])
                if len(valid_responses) >= 500:
                    return valid_responses
    return valid_responses

def llama_answers(questions):
    tasks = []
    for idx, question in enumerate(questions):
        context_chunks = retrieve_context(idx, question, n_results=5)
        context = "\n\n".join(context_chunks)

        formatted_user_prompt = user_prompt.format(context=context, question=question)

        task = {
            "custom_id": f"question={question}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_user_prompt}
                ]
            }
        }
        tasks.append(task)

    # Write tasks to JSON file for batch processing
    with open("data/llama_input_batch.jsonl", 'w') as jf:
        for task in tasks:
            jf.write(json.dumps(task) + '\n')

    # Upload the batch file to OpenAI
    batch_file = openai_client.files.create(
        file=open("data/llama_input_batch.jsonl", 'rb'),
        purpose='batch'
    )

    # Run the batch job
    batch_job = openai_client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    # Wait for the batch job to complete
    complete = False
    while not complete:
        check = openai_client.batches.retrieve(batch_job.id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        time.sleep(1)

    print("Batch processing complete.")

    result = openai_client.files.content(check.output_file_id).content
    answered_questions = "data/llama_output.json"
    with open(answered_questions, 'wb') as file:
        file.write(result)

    res = []
    with open(answered_questions, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            res.append(json_object)

    return res

questions = possible_questions()
results = llama_answers(questions)

for item in results:
    print("Model's Response:")
    print('\t', item['response']['body']['choices'][0]['message']['content'])
