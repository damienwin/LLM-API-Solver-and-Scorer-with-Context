import argparse
import json
import os
import re
import time

from openai import OpenAI
from dotenv import load_dotenv



def llama_grading():
    """
    Grade the Llama model's answers using OpenAI's batch API.

    This function reads the correct answers and Llama's responses, formats grading tasks,
    submits them as a batch to OpenAI, and saves the grading results.
    """

    load_dotenv('.env')


    # Load the dataset containing correct answers
    with open('data/dev-v2.0.json') as f:
        correct_data = json.load(f)

    # Load Llama's responses from the output file
    with open('data/llama_output.json') as f:
        llama_data = [json.loads(line) for line in f if line.strip()]

    correct_answers = []
    gpt_answers = []

    # Extract correct answers from the dataset
    for article in correct_data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if not qa["is_impossible"]:
                    if qa['answers']:
                        answer = qa['answers'][0]['text'].lower()
                        correct_answers.append(answer)
                if len(correct_answers) >= 500:
                    break
            if len(correct_answers) >= 500:
                break
        if len(correct_answers) >= 500:
            break

    # Extract questions and Llama's responses
    for entry in llama_data:
        question = entry["question"]
        response_content = entry["response"]
        gpt_answers.append({
            "question": question,
            "response": response_content
        })

    # Define prompts for grading
    system_prompt = (
        "You are a teacher tasked with determining whether a student's answer to a question was "
        "correct, based on a set of possible correct answers."
    )
    user_prompt = """Question: {question})
    Student's Response: {student_response}
    Possible Correct Answers: {correct_answers}
    Your response should be a valid JSON in the following format:
    {{
    "explanation": "A short explanation of why the student's answer was correct or incorrect.",
    "score": true or false (boolean)
    }}"""

    tasks = []

    # Create grading tasks for each question-response pair
    for idx, (correct_answer, gpt_answer) in enumerate(zip(correct_answers, gpt_answers), 1):
        if correct_answer is None:
            continue
        question = gpt_answer['question']
        student_response = gpt_answer['response']

        formatted_user_prompt = user_prompt.format(
            question=question,
            student_response=student_response,
            correct_answers=correct_answer
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_user_prompt}
        ]

        custom_id = f"{idx}. {question}"

        # Define the grading task with structured output
        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "messages": messages,
            }
        }

        tasks.append(task)

    # Write grading tasks to a JSONL file
    with open("data/llama_scoring_inpuit_batch.jsonl", "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    # Initialize the OpenAI client with the API key
    client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

    # Upload the batch file to OpenAI
    batch_file = client.files.create(
        file=open("data/llama_scoring_inpuit_batch.jsonl", 'rb'),
        purpose='batch'
    )

    # Create a batch job for grading
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    # Poll the batch job status until completion
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job.id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        time.sleep(1)

    print("Batch processing completed.")

    # Retrieve and save the grading results
    result = client.files.content(check.output_file_id).content
    output_file_name = "data/llama_scoring_results.jsonl"
    with open(output_file_name, 'wb') as file:
        file.write(result)

if __name__ == "__main__":
    llama_grading()