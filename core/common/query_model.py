import json
import re
import requests


def query_model(
    message,
    api_endpoint,
    model_name,
    temperature=0.2,
    max_tokens=1024,
    timeout_connection=20,
    timeout_read=120,
):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": message}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,  # Add this to ensure non-streaming response
    }

    # Add headers for proper API communication
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    try:
        # Make the POST request using requests library
        response = requests.post(
            api_endpoint,
            headers=headers,
            json=payload,  # requests will automatically handle JSON serialization
            timeout=(
                timeout_connection,
                timeout_read,
            ),  # (connect timeout, read timeout) in seconds
        )

        # Raise an exception for bad status codes
        response.raise_for_status()

        # Parse the JSON response
        if response.text:
            try:
                parsed_response = response.json()
                return parsed_response["choices"][0]["message"]["content"]
            except json.JSONDecodeError as e:
                print(f"Response text: {response.text}")
                print(f"Error parsing JSON response: {e}")
        else:
            print("Empty response received")
            print(f"Status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        if hasattr(e.response, "text"):
            print(f"Response text: {e.response.text}")

    return None


def decode_answer_true_false(model_answer):
    if model_answer:
        # Search for a True False Match
        match = re.search(r"<answer>(True|False)</answer>", model_answer)
        if match:
            model_choice = match.group(1).lower()
            if model_choice == "true":
                return True
            else:
                return False
        else:
            return "invalid"
    return "invalid"


def decode_answer(model_answer):
    if not model_answer:
        return "no_answer"
    start = "<answer>"
    end = "</answer>"
    if start in model_answer and end in model_answer:
        return model_answer.split(start, 1)[1].split(end, 1)[0].strip()
    # Fallback: model did not answer with what I required
    return "no_answer"


def extract_candidates_list(text):
    match = re.search(r"<answer>\s*\[(.*?)\]\s*</answer>", text)
    if match:
        words = match.group(1)
        return (
            [word.strip() for word in words.split(",") if word.strip()] if words else []
        )
    return []
