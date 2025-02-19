import json

for task in ['data/gpt4o_scoring_results.jsonl', 'data/llama_scoring_results.jsonl']:
    with open(task, 'r') as file:
        data = [json.loads(line) for line in file]

    correct_responses = 0
    total_responses = len(data)

    for entry in data:
        # Extract content from response
        response_content = entry["response"]["body"]["choices"][0]["message"]["content"]

        try:
            # Create search for word
            json_str = response_content[response_content.find('{'):response_content.rfind('}') + 1]
            score_data = json.loads(json_str)

            # If "score" is true, add 1 to the count
            if score_data.get("score", False):
                correct_responses += 1
        except json.JSONDecodeError:
            print(f"Error parsing JSON")

    accuracy = (correct_responses / total_responses) * 100

    print(task)
    print(f"Correct Responses: {correct_responses}")
    print(f"Total Responses: {total_responses}")
    print(f"Accuracy: {accuracy:.2f}%\n")