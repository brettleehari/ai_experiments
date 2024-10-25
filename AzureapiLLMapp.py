import argparse
import os
from openai import AzureOpenAI


class OpenAIChatClient:
    def __init__(self, endpoint: str, api_key: str, model_name: str):
        self.client = AzureOpenAI(azure_endpoint=endpoint, api_version="2024-05-13", api_key=api_key)
        self.model_name = model_name

    def get_completion(self, message: str):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": message}],
            )
            return response
        except Exception as e:
            print(f"Error while fetching completion: {e}")
            return None

    @staticmethod
    def print_completion_info(completion):
        if not completion:
            print("No completion data available.")
            return
        
        # Unpacking response data
        completion_id = completion.id
        choices = completion.choices
        created = completion.created
        model = completion.model
        object_type = completion.object
        system_fingerprint = completion.system_fingerprint
        usage = completion.usage
        prompt_filter_results = completion.prompt_filter_results

        # Print basic completion information
        print(f"ID: {completion_id}")
        print(f"Model: {model}")
        print(f"Created: {created}")
        print(f"Object Type: {object_type}")
        print(f"System Fingerprint: {system_fingerprint}")

        # Process and print each choice
        for idx, choice in enumerate(choices):
            print(f"Choice {idx + 1}:")
            print(f"  - Finish Reason: {choice.finish_reason}")
            print(f"  - Index: {choice.index}")
            print(f"  - Logprobs: {choice.logprobs}")
            print(f"  - Message Role: {choice.message.role}")
            print(f"  - Message Content: {choice.message.content}")
            print(f"  - Function Call: {choice.message.function_call}")
            print(f"  - Tool Calls: {choice.message.tool_calls}")

        # Print usage details
        print("Usage:")
        print(f"  - Completion Tokens: {usage.completion_tokens}")
        print(f"  - Prompt Tokens: {usage.prompt_tokens}")
        print(f"  - Total Tokens: {usage.total_tokens}")

        # Print prompt filter results
        print("Prompt Filter Results:")
        for prompt_result in prompt_filter_results:
            print(f"  - Prompt Index: {prompt_result['prompt_index']}")
            print(f"    Content Filter Results: {prompt_result['content_filter_results']}")


def main(api_key: str, endpoint: str, model_name: str):
    # Create the chat client
    chat_client = OpenAIChatClient(endpoint, api_key, model_name)
    
    while True:
        message = input("You: ")
        if message.lower() in ['exit', 'quit']:
            print("Exiting chat. Goodbye!")
            break
        
        # Fetch completion
        completion_response = chat_client.get_completion(message)
        
        # Print the information from the completion response
        chat_client.print_completion_info(completion_response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chat with OpenAI using Azure API.')
    parser.add_argument('--api_key', type=str, help='Your Azure API key.')
    parser.add_argument('--endpoint', type=str, default='https://ai-foundation-api.app/ai-foundation/chat-ai/gpt4',
                        help='API endpoint for the Azure OpenAI model.')
    parser.add_argument('--model_name', type=str, default='azure/gpt-4o',
                        help='Model name to use for the request.')

    args = parser.parse_args()

    # Use provided API key or fallback to environment variable
    api_key = args.api_key or os.getenv('AZURE_API_KEY', 'your default api key')

    # Run the main function with the provided arguments
    main(api_key, args.endpoint, args.model_name)

#usage sample : 
#Python 3.7.4
#pip install openai==1.30.1
#usage : python LLMapp.py --api_key <your azure key>
