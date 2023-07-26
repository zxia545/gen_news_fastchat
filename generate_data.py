import os
import json
import openai
import concurrent.futures
import time
from queue import Queue
import re
import threading
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Generate data using OpenAI.')
parser.add_argument('--input_files', type=str, nargs='+', help='Input file paths')
parser.add_argument('--output_folders', type=str, nargs='+', help='Output folder paths')
args = parser.parse_args()

# Verify that the number of input files and output folders are the same
assert len(args.input_files) == len(args.output_folders), "Mismatch between number of input files and output folders"

openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

retry_queue = Queue()

# shared flag variable
stop_threads = threading.Event()

def fetch_max_word_limit(s):
    try:
        word_limit = re.search(r'in length between (\d+) words and (\d+) words', s).groups()
        min_limit, max_limit = map(int, word_limit)
        return max_limit
    except AttributeError:
        return 200

def generate_text(i, prompt, max_tokens, output_name, model, output_folder, retry=False, retry_count=0):
    try:
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=40,
        )

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        file_name = os.path.join(output_folder, f'{output_name}.txt')
        with open(file_name, 'a', encoding='utf-8') as f:
            f.write("{}:\n{}".format("GPT-OUTPUT",completion.choices[0].message.content))
            f.write('\n\n\n')
    except Exception as e:
        print(f"Failed to generate text for prompt {i}, Number of Retry {retry_count}. Error: {e}")
        if retry_count < 2:
            print("Pausing execution for 3 seconds...")
            time.sleep(3)  # pause for 15 seconds
            retry_count += 1
            print("Resuming execution.")
            generate_text(i, prompt, max_tokens, output_name, model, output_folder, True, retry_count)
        else:
            print(f'Prompt: {output_name} still fail after retry')
            error_folder = os.path.join(output_folder, "file_cannot_generate_folder")
            if not os.path.exists(error_folder):
                os.makedirs(error_folder)
            file_name = os.path.join(error_folder, f'output_1.txt')
            with open(file_name, 'a', encoding='utf-8') as f:
                f.write("{},".format(output_name))
        raise

model = "13b"
BATCH_SIZE = 200

for input_file, output_folder in zip(args.input_files, args.output_folders):
    # Path to the output directory
    output_directory_path = output_folder

    # Get the list of existing file names in the output directory
    existing_files = os.listdir(output_directory_path)

    # Remove the '.txt' extension from the existing file names
    existing_files = [filename[:-4] for filename in existing_files if filename.endswith('.txt')]

    with open(input_file) as f:
        json_data = json.load(f)

    name_list = []
    prompt_list = []
    max_word_list = []

    for news_category in json_data:
        current_news_dict = json_data[news_category]
        for current_prompt_id in current_news_dict:
            file_name = f'{news_category}_{current_prompt_id}'
            if file_name not in existing_files:  # Check if the file already exists
                name_list.append(file_name)
                current_prompt_dict = current_news_dict[current_prompt_id]
                current_prompt = current_prompt_dict['prompt']
                prompt_list.append(current_prompt)
                max_word_list.append(200)

    example = """For the upcoming text generation task, please adhere to the following format:\n ##Title##: [Your specific topic here: Specific issue or question].\n[Your text goes here]\n(End of text)\nThis format includes a title, introduced by '##Title##', followed by the body of the text, and concludes with '(End of text)'. Please ensure your generated text follows this structure."""
    prompt_list = [ prompt  + '\n' +  example + '\nYour Answer:\n' for prompt in prompt_list]

    print(f'Len of the prompt list: {len(prompt_list)}')

    total_start = time.time()
    for batch_start in range(0, len(prompt_list), BATCH_SIZE):
        st_time = time.time()
        batch_end = min(batch_start + BATCH_SIZE, len(prompt_list))
        batch_prompt_list = prompt_list[batch_start:batch_end]
        batch_max_word_list = max_word_list[batch_start:batch_end]
        batch_name_list = name_list[batch_start:batch_end]

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(len(batch_prompt_list)):
                future = executor.submit(generate_text, i, batch_prompt_list[i], batch_max_word_list[i], batch_name_list[i], model, output_folder, False, 0)
                futures.append(future)
        
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"Generated an exception: {exc}")

        print(f'{BATCH_SIZE} prompts took: {time.time()-st_time} seconds')
    print(f'Total Time: {time.time()-total_start} seconds')
