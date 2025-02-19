import json
import os
import time

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

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

def llama_answers(questions):
    """
    Generate answers using the Llama model via Azure's ChatCompletionsClient.

    This function reads questions from 'dev-v2.0.json', submits each question to the Llama model,
    and appends the responses to 'llama8Boutput.json'.
    """
    client = ChatCompletionsClient(
        endpoint=os.environ["AZURE_MLSTUDIO_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_MLSTUDIO_KEY"]),
    )

    # Open the output file in append mode
    with open('data/llama_output.json', 'a') as output_file:
        for idx, question in enumerate(questions, 1):
            # Submit the question to the Llama model
            context_chunks = retrieve_context(idx, question, n_results=5)
            context = "\n\n".join(context_chunks)

            formatted_user_prompt = user_prompt.format(context=context, question=question)

            response = client.complete(
                messages=[
                    SystemMessage(
                        content=system_prompt,
                    ),
                    UserMessage(content=formatted_user_prompt),
                ],
            )
            # Structure the result
            result = {
                "question": question,
                "response": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }

            print(f"{idx} / {len(questions)} Questions answered: {question}")
            output_file.write(json.dumps(result) + '\n')

    print("Model's Response:")
    print('\t', response.choices[0].message.content)
    print()
    print(f"Input Tokens:  {response.usage.prompt_tokens}")
    print(f"Output Tokens: {response.usage.completion_tokens}")
    print(f"Cost: ${response.usage.prompt_tokens * 0.0003 / 1000 + response.usage.completion_tokens * 0.00061 / 1000}")

questions = possible_questions()
llama_answers(questions)