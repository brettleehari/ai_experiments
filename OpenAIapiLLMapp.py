import json
import time
import hashlib
import backoff
import os
import importlib
import litellm
import argparse

litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.drop_params = True
litellm._logging._disable_debugging()

# Retry exceptions
def retry_exceptions():
    import httpx
    import openai

    return (
        httpx.ConnectError,
        httpx.RemoteProtocolError,
        httpx.ReadTimeout,
        openai.APITimeoutError,
        openai.UnprocessableEntityError,
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APIError,
        openai.APIStatusError,
        openai.InternalServerError,
    )

# Send completion with retries
def send_completion(model_name, messages, stream, temperature=0, extra_params=None):
    kwargs = dict(
        model=model_name,
        messages=messages,
        stream=stream,
    )
    if temperature is not None:
        kwargs["temperature"] = temperature

    if extra_params is not None:
        kwargs.update(extra_params)

    key = json.dumps(kwargs, sort_keys=True).encode()
    hash_object = hashlib.sha1(key)

    res = litellm.completion(**kwargs)
    return hash_object, res

# Simple function to send with retries
def simple_send_with_retries(model_name, messages, extra_params=None):
    retry_delay = 0.125
    while True:
        try:
            kwargs = {
                "model_name": model_name,
                "messages": messages,
                "stream": False,
                "extra_params": extra_params,
            }

            _hash, response = send_completion(**kwargs)

            # Check if response has the expected structure
            if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'message'):
                content = response.choices[0].message.content
            else:
                content = None

            token_usage = getattr(response, 'usage', None)  # Capture token usage details
            return content, token_usage
        except retry_exceptions() as err:
            print(str(err))
            retry_delay *= 2
            if retry_delay > 60:
                print("Maximum retry limit reached. Exiting.")
                return None, None
            print(f"Retrying in {retry_delay:.1f} seconds...")
            time.sleep(retry_delay)
            continue
        except Exception as err:
            print(f"An unexpected error occurred: {err}")
            return None, None

class Model:
    DEFAULT_MODEL_NAME = "gpt-4o"

    def __init__(self, model_name):
        self.name = model_name or self.DEFAULT_MODEL_NAME
        self.info = self.get_model_info(model_name)

    def get_model_info(self, model_name):
        # Dummy implementation for model info
        return {"max_input_tokens": 4096}

    def token_count(self, messages):
        if isinstance(messages, str):
            messages = [messages]
        elif isinstance(messages, list) and isinstance(messages[0], dict):
            # Extract 'content' from message dictionaries
            messages = [msg['content'] for msg in messages]
        return sum(len(litellm.encode(model=self.name, text=msg)) for msg in messages)

class LLMWorkflow:
    def __init__(self, model_name):
        self.model = Model(model_name)
        self.messages = []  # Initialize conversation history

    def send_query(self, user_query):
        # Add the user query to the conversation history
        self.messages.append({"role": "user", "content": user_query})

        # Token count check on the full conversation
        token_count = self.model.token_count(self.messages)
        if token_count > self.model.info.get("max_input_tokens", 4096):
            print("Conversation exceeds the maximum token limit for the model.")
            self.clear_context()
            print("Conversation context cleared. Please retry your query.")
            return None, None

        # Send the conversation history
        response, token_usage = simple_send_with_retries(self.model.name, self.messages)

        # Handle cases where response might be None
        if response is None:
            print("Failed to get a response from the model.")
            return None, None

        # Add the assistant's response to the conversation history
        self.messages.append({"role": "assistant", "content": response})

        return response, token_usage

    def process_response(self, response, token_usage):
        # Process the response as needed
        if token_usage:
            # Display the token usage and other details
            print(f"Prompt tokens: {token_usage.get('prompt_tokens', 'N/A')}")
            print(f"Completion tokens: {token_usage.get('completion_tokens', 'N/A')}")
            print(f"Total tokens used: {token_usage.get('total_tokens', 'N/A')}")
        else:
            print("Token usage information is not available.")

        return response

    def clear_context(self):
        # Clear the conversation history
        self.messages = []

def main(api_key, endpoint, model_name):
    # Create the LLM workflow
    workflow = LLMWorkflow(model_name)
    
    print("Type 'exit' or 'quit' to exit, 'clear' or 'reset' to start a new conversation.")
    
    while True:
        message = input("You: ")
        if message.lower() in ['exit', 'quit']:
            print("Exiting chat. Goodbye!")
            break
        elif message.lower() in ['clear', 'reset']:
            workflow.clear_context()
            print("Conversation context cleared.")
            continue
        
        # Send the user's message and get the response and token usage
        response, token_usage = workflow.send_query(message)
        processed_response = workflow.process_response(response, token_usage)
        
        print("LLM:", processed_response)

if __name__ == "__main__":
    # Remove the pdb debugging statement
    # import pdb; pdb.set_trace()
    
    parser = argparse.ArgumentParser(description='Chat with OpenAI using litellm.')
    parser.add_argument('--api_key', type=str, help='Your API key.')
    parser.add_argument('--endpoint', type=str, default='https://api.openai.com/v1/chat/completions',
                        help='API endpoint for the OpenAI model.')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo',
                        help='Model name to use for the request.')
    
    args = parser.parse_args()

    # Use provided API key or fallback to environment variable
    api_key = args.api_key or os.getenv('OPENAI_API_KEY', 'your default api key')
    
    # Run the main function with the provided arguments
    main(api_key, args.endpoint, args.model_name)
