import os
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import argparse
import re
import json

def save_conversation(conversation, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, ensure_ascii=False, indent=4)


def load_conversation(file_path):
    if not os.path.exists(file_path):
        print(f"Conversation file '{file_path}' not found.")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_response(text):
    text = text.split("[/INST]")[-1]
    text = text.split("</s>")[0]
    return text

def print_conversation(conversation):
    for message in conversation:
        # Ensure the message has both 'role' and 'content' keys
        if 'role' in message and 'content' in message:
            print(f"{message['role']}> {message['content']}")
        else:
            print("Error: Missing 'role' or 'content' in message.")
        print()

def debug(msg):
    print(msg)
    return None

def main(args):
    cmdparser = argparse.ArgumentParser(description='Interactive Mistral-7B')
    cmdparser.add_argument('-q', '--quit', action='store_true', help='exits the program')
    cmdparser.add_argument('-u', '--undo', action='store_true', help='removes the last assistant and user prompt')
    cmdparser.add_argument('-r', '--redo', action='store_true', help='removes the last assistant prompt and regenerates it')
    cmdparser.add_argument('-l', '--load', type=str, nargs='?', help='Path to a the conversation file to load')
    cmdparser.add_argument('-s', '--save', type=str, nargs='?', help='saves the current conversation to a new file')
    cmdparser.add_argument('user_input', nargs='?', help='String to pass to the LLM')

    print("Commands available at runtime:")
    try:
        cmdparser.parse_known_args(["-h"])
    except SystemExit as e:
        None

    messages = []

    # Load the conversation file if specified
    if args.load:
        conversation_file = args.load
        messages = load_conversation(conversation_file)
        print_conversation(messages)
    else:
        conversation_id = str(uuid.uuid4())
        conversation_file = f"{conversation_id}.json"

    local_path = "E:/LLMs/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2"
    model = AutoModelForCausalLM.from_pretrained(local_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)

    # device = "cpu"
    device = "cuda"
    model.half()
    model.to(device)

    max_token = 5000

    while True:
        if len(messages) == 0 or messages[-1]['role'] == 'assistant':
            user_input = input("\nuser> ")
            if (user_input[0] == '-'):
                cmdargs = None
                cmdunknown = None
                try:
                    cmdargs, cmdunknown = cmdparser.parse_known_args(user_input.split())
                except SystemExit:
                    continue

                if cmdunknown:
                    print("Command error: {}")
                    try:
                        cmdparser.parse_known_args(["-h"])
                    except SystemExit:
                        None
                    continue
                elif cmdargs.load:
                    # Handle loading a conversation from a file
                    print("Load")
                    conversation_file = cmdargs.load
                    messages = load_conversation(conversation_file)
                    print_conversation(messages)
                    continue
                elif cmdargs.save:
                    # Handle saving the current conversation to a file
                    print("Save")
                    conversation_file = cmdargs.save
                    save_conversation(messages, conversation_file)
                    continue
                elif cmdargs.undo:
                    # Handle removing the last assistant and user prompt
                    print("Undo")
                    messages = messages[:-2]
                    continue
                elif cmdargs.redo:
                    # Handle removing the last assistant prompt and regenerating it
                    print("Redo")
                    messages = messages[:-1]
                    continue
                elif cmdargs.quit:
                    # Handle removing the last assistant prompt and regenerating it
                    print("Exiting")
                    return
        
            messages.append({"role": "user", "content": user_input})
            save_conversation(messages, conversation_file)
        elif messages[-1]['role'] == 'user':
            encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

            model_inputs = encodeds.to(device)
            start_time = time.time()
            generated_ids = model.generate(
                model_inputs, 
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_token, 
                do_sample=True)

            print(f"Processing {model_inputs.size()[1]} input tokens.")
            decoded = tokenizer.batch_decode(generated_ids)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time} seconds.")
            # debug(decoded)
            response = extract_response(decoded[0])
            messages.append({"role": "assistant", "content": response})
            print(f"\nassistant> {response}")
            save_conversation(messages, conversation_file)
        else:
            print("FATAL: Unknown role {messages[-1]['role']}")
            return


    # while True:
    #     if len(messages) == 0 or messages[-1]['role'] == 'assistant':
    #         # Get user input
    #         user_input = input("\nuser> ")
    #         if user_input == 'quit' or user_input == '':
    #             break
    #         elif user_input == 'undo' and len(messages) >= 2:
    #             continue
    #         elif user_input == 'redo' and len(messages) >= 1:
    #             messages = messages[:-1]
    #             continue
    #         messages.append({"role": "user", "content": user_input})
    #         save_conversation(messages, conversation_file)
    #     elif messages[-1]['role'] == 'user':
    #         encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    #         model_inputs = encodeds.to(device)
    #         start_time = time.time()
    #         generated_ids = model.generate(
    #             model_inputs, 
    #             pad_token_id=tokenizer.eos_token_id,
    #             max_new_tokens=max_token, 
    #             do_sample=True)

    #         print(f"Processing {model_inputs.size()[1]} input tokens.")
    #         decoded = tokenizer.batch_decode(generated_ids)
    #         end_time = time.time()
    #         elapsed_time = end_time - start_time
    #         print(f"Elapsed time: {elapsed_time} seconds.")
    #         # debug(decoded)
    #         response = extract_response(decoded[0])
    #         messages.append({"role": "assistant", "content": response})
    #         print(f"\nassistant> {response}")
    #         save_conversation(messages, conversation_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive Mistral-7B')
    parser.add_argument('-l', '--load', type=str, nargs='?', help='Path to a the conversation file to load')
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='Path to the conversation file to continue')
    args = parser.parse_args()
    main(args)