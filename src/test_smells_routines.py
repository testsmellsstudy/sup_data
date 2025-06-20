import paramiko
import os
import subprocess
import csv
import shutil
import pandas as pd
import multiprocessing
from tqdm import tqdm
import chardet
import sys
import time
import math
import tiktoken
from datetime import datetime, timedelta
import re
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import json
from transformers import AutoTokenizer
import zipfile
# from pyjavaparser import JavaParser
# from huggingface_hub import login
import javalang
import requests
import shlex
import numpy as np
import time
import random
from openai import OpenAI
from openai import AsyncOpenAI
import concurrent.futures
from tqdm import tqdm
from GPT_test_smells_routines import Multiprocessor, DataframeManager, Timestamp, PyClassLocator
import final_report
import google.generativeai as genai
import ast
import traceback
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact

# login(token="")


DIR_ROOT_PATH = os.path.dirname(__file__)
DIR_ROOT_PATH_SUB = os.path.dirname(os.path.dirname(__file__))
DATASET_PATH = os.path.join(DIR_ROOT_PATH, "dataset_java")
GENERAL_DATASETS_REPO_PATH = os.path.join(DATASET_PATH, "Original_projects")
RPL_STR_ROOT_PATH = os.path.join(DATASET_PATH, "Replacement_structure")
AFTER_REPLACEMENT_PATH = os.path.join(DATASET_PATH, "After_replacement")
DATASET_TO_PROMPT_PATH = os.path.join(DIR_ROOT_PATH, "To_prompt")
DATASET_TO_REPLACE_PATH = os.path.join(DIR_ROOT_PATH, "To_replace")
DETECTOR_INPUT_PATH = os.path.join(DIR_ROOT_PATH_SUB, "GPT_test_smells", "PyNose_input")
DETECTOR_OUTPUT_PATH = os.path.join(DIR_ROOT_PATH_SUB, "GPT_test_smells", "PyNose_output")
PLUGIN_ROOT_PATH = os.path.join(DIR_ROOT_PATH_SUB, "PyNose")
DATASET_CSV_PATH = os.path.join(GENERAL_DATASETS_REPO_PATH, "dataset_java.csv")

PROMPTS_DIR_PATH = os.path.join(DIR_ROOT_PATH, "Prompts")

TS_DETECT_INITIAL_OUTPUT = os.path.join(DIR_ROOT_PATH, "TS_detect_initial_output")
TS_DETECT_REPLACED_OUTPUT = os.path.join(DIR_ROOT_PATH, "TS_detect_replaced_output")
TS_DETECT_FINAL_OUTPUT = os.path.join(DIR_ROOT_PATH, "TS_detect_final_output")
TS_DETECT_DATA_COMPARISON = os.path.join(DATASET_PATH, "Ts_detect_comparison")

NUM_WORKERS = 6
DISCOUNT_PROMPTING = 0
MAX_TOKENS_TO_PROMPT = 3200 # This is the target maximum number of output tokens expected from the API. This number should be small enough to make it unlikely that the output hits the maximum number of tokens supported by the currently used model
MAX_TOKENS_TO_REPLACE = 3000 # This is the maximum number of tokens the LLM output must have to be used in the replacement part. 
# MODEL_TO_USE = "google/gemma-7b"
# MODEL_TO_USE = "google/gemma-2b"
# MODEL_TO_USE = "meta-llama/Meta-Llama-3-70B"
# MODEL_TO_USE = "meta-llama/Meta-Llama-2-70B"
# MODEL_TO_USE = "meta-llama/Meta-Llama-2-70B-chat
MODEL_TO_USE = "llama3:70b"
# MODEL_TO_USE = "gpt-4-turbo"
MODEL_TO_USE_NAME = MODEL_TO_USE.replace(":","").replace(" ","").replace("/","")

LANGUAGE_TO_USE = "java"
MIN_N_TO_CONSIDER_EVALUATED_REF = 320
MIN_N_TO_CONSIDER_EVALUATED_DET = 320
MIN_N_TO_CONSIDER_EVALUATED = 320
# Maximum number of prompts for each prompting session
MAX_PROMPT_PER_SHOT = 1000
MAX_COST_PER_SHOT = 1000000
MAX_TOKENS_PER_SHOT = 10000 * ((100 / 3) + 100) * MAX_COST_PER_SHOT

ORIGINAL_RPL_STR_PATH = os.path.join(RPL_STR_ROOT_PATH, "Original")
SMELLS_RPL_STR_PATH = os.path.join(RPL_STR_ROOT_PATH, "Smells")

client = OpenAI()
client_async = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    )


API_KEY_GEMINI = ''
# API_KEY_GEMINI = ''
genai.configure(api_key=API_KEY_GEMINI)




def process_prompts_api(key, prompt):
    prompt, model_name = prompt
    model = genai.GenerativeModel(model_name)
    config = genai.GenerationConfig(temperature=0.00)
    
    messages = [
        {"role": "user", "parts": [prompt[0]]},
        {"role": "user", "parts": [prompt[1]]}
    ]
    
    # Attempt to generate content and handle possible errors
    try:
        response = model.generate_content(messages)
        # Check if response contains parts and is not empty
        if hasattr(response, 'parts') and response.parts:
            response_text = ' '.join([part.text for part in response.parts])
            tokens_received = model.count_tokens(response_text)
            tokens_sent = model.count_tokens(messages)
            print(f"This is the response: {response_text}")
            return (key, messages, response_text, tokens_received.total_tokens, tokens_sent.total_tokens)
        else:
            error_details = "No valid parts in response. Check safety ratings and other details."
            error_info = response.candidate.safety_ratings if hasattr(response, 'candidate') else "No candidate info available."
            messages.append({"role": "system", "parts": [error_details, error_info]})
            raise Exception(f"Response failed: No valid parts returned. Details added to messages.\n {[error_details, error_info]}\n\n#####{response}")
    except Exception as e:
        # Append exception details to messages and raise an exception with those details
        messages.append({"role": "system", "parts": [str(e)]})
        raise Exception(f"An error occurred: {e}. Details added to messages.")
    
def send_all_prompts(*args, **kwargs):
    mode = kwargs.get('mode', 'default')
    result = []
    if mode == "ollama":
        ssh_connection, user_prompt_list = args
        # send_ssh_command_setup(ssh_connection, model)
        for key, prompt_and_model in tqdm(user_prompt_list.items()):
            try:
                prompt, model = prompt_and_model
                system_prompt, user_prompt = prompt
                print(f"using model: {model}")
                # print(f"This is the prompt: {prompt} #####")
                # print(f"Prompting: {system_prompt}\n\n{user_prompt}")
                response = send_ssh_command_with_prompt(ssh_connection, model, system_prompt, user_prompt)
                result.append((response, key))
            except Exception as e:
                result.append((["", "", "", f"Prompt failed, detail: {e}"], key))
        return result     
    
    if mode == "openai":
        print(f"using model: {mode}")
        user_prompt_list = args[0]
        with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
            futures = {
                executor.submit(
                    lambda key, prompt, model_to_use: (
                        key,
                        prompt,
                        client.chat.completions.create(
                            model=model_to_use,
                            # model="gpt-4o",
                            messages=[
                                {"role": "system", "content": prompt[0]},
                                {"role": "user", "content": prompt[1]}
                            ],
                            temperature=0,
                            # temperature=.2,
                        )
                    ), key, prompt, model_to_use
                ): key for key, (prompt, model_to_use) in user_prompt_list.items()
            }
            
            result = []
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                key = futures[future]
                try:
                    key, prompt, response = future.result()
                    # print(f"This is the prompt: {prompt} #####")
                    resp = response.choices[0].message.content
                    tokens_received = response.usage.completion_tokens
                    tokens_sent = response.usage.prompt_tokens
                    prompt_status = "ok"
                    response_data = [str(resp), int(tokens_received), int(tokens_sent), str(prompt_status)]
                    result.append((response_data, key))
                except Exception as e:
                    result.append((["", "", "", f"Prompt failed, detail: {e} model: {mode}"], key))

        return result
    
    if mode == "gemini":
        print(f"using model: {mode}")
        user_prompt_list = args[0]
        print(f"Already inside: this is the len {len(user_prompt_list)} and the type {type(user_prompt_list)}")  
        # print(f"This is the content: {user_prompt_list}")  
        with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
            futures = {
                executor.submit(process_prompts_api, key, prompt): key for key, prompt in user_prompt_list.items()
            }

            result = []
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                key = futures[future]
                try:
                    # result.append((["", "", "", f"Prompt failed, detail: PROBLEMS WITH model BILLING: {mode}"], key))
                    key, prompt, response, tokens_received, tokens_sent = future.result()
                    # print(f"This is the prompt: {prompt} #####")
                    resp = response
                    prompt_status = "ok"
                    response_data = [str(resp), int(tokens_received), int(tokens_sent), str(prompt_status)]
                    result.append((response_data, key))
                except Exception as e:
                    result.append((["", "", "", f"Prompt failed, detail: {e} model: {mode}"], key))
                    # if response:
                    #     print(f"This is the response: {response}")
        return result
              
"""
if "history" in os.listdir(path_to_find):
    new_process = multiprocessing.Process(target=clean_data_files, args=(os.path.join(path_to_find,"history")))
    new_process.start() Decide in the future where to use this piece of code 
"""

def clean_data_files(history_path):

    def retain_files_based_on_interval(files, dirpath, minutes=0, hours=0, days=0):
        interval = timedelta(minutes=minutes, hours=hours, days=days)
        current_retention_datetime = None
        already_removed = []
        for file in tqdm(files, desc="cleaning history file by time"):
            if file in already_removed:
                continue
            # Use regex to extract the datetime part from the file name
            match = re.search(r'(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})', file)
            if match:
                date_str = match.group(1)
                try:
                    file_datetime = datetime.strptime(date_str, '%d-%m-%Y_%H-%M-%S')
                    if current_retention_datetime is None or file_datetime - current_retention_datetime >= interval:
                        current_retention_datetime = file_datetime
                    else:
                        file_path = os.path.join(dirpath, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        already_removed.append(file)
                except ValueError as e:
                    print(f"Error parsing date from filename {file}: {e}")
            else:
                print(f"No date found in filename {file}")
                
        parent_dir = os.path.dirname(history_path)
        filenames_to_check = []
        filenames_core = []
        for filename in os.listdir(parent_dir):
            if filename.endswith(".xlsx") and not filename.startswith('~$'):
                filenames_to_check.append(filename)
                
        for filename in filenames_to_check:
            filenames_core.append(filename.replace("history_","").replace("current_","")[:-25])
            
        filenames_core = list(set(filenames_core))   
        filenames_core_testing = filenames_core
        if len(filenames_core) >=1:
            for filename_core in filenames_core_testing:
                instances_of_filename_core = 0
                for filename in filenames_to_check:
                    if filename_core in filename:
                        instances_of_filename_core +=1  
                        
                if instances_of_filename_core <= 1:
                    filenames_core.remove(filename_core)          
            
        dataframes_newest = []
        possibly_deleted = []
        for filename_core in filenames_core:
            df_to_keep = DataframeManager.load(parent_dir, name_to_find=filename_core)
            new_name = "keep_" + filename_core
            print(f"this is the newname {new_name}")
            print(f"this is the path {parent_dir}")
            df_to_keep = DataframeManager.save(df_to_keep, parent_dir, name_to_find="keep_" + filename_core)
            if isinstance(df_to_keep, pd.DataFrame):
                dataframes_newest.append([new_name, filename_core])  
            else:
                possibly_deleted.append(filename_core)
                
                        
        for filename in filenames_to_check:    
            file_original_path = os.path.join(parent_dir, filename)
            file_history_path = os.path.join(dirpath, filename.replace("current_","history_"))
            try:
                if "keep_" not in filename:
                    shutil.copyfile(file_original_path, file_history_path)
            except:
                is_safe = False
                for filename_core in possibly_deleted:
                    if filename_core in filename:
                        is_safe = True
                if is_safe:
                    continue
                else:
                    print(f"FILE NOT FOUND: {file_original_path}")
            
            if os.path.isfile(file_history_path):
                if "keep_" not in filename:
                    subprocess.run(['cmd', '/c', 'del', '/q', file_original_path])
                
            else:
                print(f"Warning: could not copy file: \n{file_original_path} \nto \n{file_history_path}")

        for dataframe_newest in dataframes_newest:
            new_name = dataframe_newest[0]
            filename_core = dataframe_newest[1]
            df_to_keep = DataframeManager.load(parent_dir, name_to_find=new_name)
            DataframeManager.save(df_to_keep, parent_dir, name_to_find=filename_core)
            
        for filename in os.listdir(parent_dir):
            if "keep_" in filename:
                file_original_path = os.path.join(parent_dir, filename)
                subprocess.run(['cmd', '/c', 'del', '/q', file_original_path])
        
        filenames_to_check = [f for f in os.listdir(dirpath) if f.endswith(".xlsx") and not f.startswith('~$')]
        
        file_datetimes = {}
        
        for filename in filenames_to_check:
            filenames_core.append(filename.replace("history_","").replace("current_","")[:-25])
            
        filenames_core = list(set(filenames_core))   
        filenames_core_testing = filenames_core
        if len(filenames_core) >= 1:
            for filename_core in filenames_core_testing:
                instances_of_filename_core = 0
                for filename in filenames_to_check:
                    if filename_core in filename:
                        instances_of_filename_core +=1  
                        
                if instances_of_filename_core <= 1:
                    filenames_core.remove(filename_core)          
        
        for filename_core in filenames_core: 
            for file in filenames_to_check:
                if filename_core not in file:
                    continue
                # Extract the datetime part of the filename using regular expressions
                match = re.search(r'(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})', file)
                if match:
                    date_str = match.group(1)
                    try:
                        # Now that we have the date string, parse it with the consistent format
                        dt = datetime.strptime(date_str, '%d-%m-%Y_%H-%M-%S')
                        file_datetimes[file] = dt
                    except ValueError as e:
                        print(f"Error parsing date from filename {file}: {e}")
                else:
                    print(f"No date found in filename {file}")
            sorted_files = sorted(file_datetimes.keys(), key=lambda x: file_datetimes[x])
            
            if len(sorted_files) > 0:

                most_recent_date = file_datetimes[sorted_files[-1]].date()
                files_by_date = {}
                for file in sorted_files:
                    file_date = file_datetimes[file].date()
                    if file_date not in files_by_date:
                        files_by_date[file_date] = []
                    files_by_date[file_date].append(file)
                
                for file_date, files in files_by_date.items():
                    if file_date == most_recent_date:
                        # Keep the most recent file of every 10 minutes
                        retain_files_based_on_interval(files, dirpath, minutes=10)
                    elif file_date == most_recent_date - timedelta(days=1):
                        # Keep the most recent file of every 1 hour
                        retain_files_based_on_interval(files, dirpath, hours=1)
                    else:
                        # Keep the most recent file of the day
                        retain_files_based_on_interval(files, dirpath, days=1)
                
                if sorted_files:
                    oldest_file_path = os.path.join(dirpath, sorted_files[0])

def num_tokens_from_messages(messages, model=MODEL_TO_USE):
    if "gemma" in model.lower():
        return num_tokens_from_messages_GPT(messages, model="gpt-4")

    if "llama" in model.lower():
        return num_tokens_from_messages_GPT(messages, model="gpt-4")
    
    if "gpt" in model.lower():
        return num_tokens_from_messages_GPT(messages, model=model)

def num_tokens_from_messages_GPT(messages, model=MODEL_TO_USE):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages_GPT(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        #print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages_GPT(messages, model="gpt-4-0613")
    elif "gpt-4-1106-preview" in model:
        print(f"Warning: using gpt-4 instead of {model}")
        return num_tokens_from_messages_GPT(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    if len(messages) > 0:
        if isinstance(messages, str):
            messages = [
                            {"role": "system", "content": "You are an assistant"},
                            {"role": "user", "content": messages}
                        ]

        elif isinstance(messages, list):
            if isinstance(messages[0], str):
                new_messages = []
                for index, message in enumerate(messages):
                    if index == 0:
                        new_messages.append({"role": "system", "content": message})
                    else:
                        new_messages.append({"role": "user", "content": message})
                messages = new_messages
                
            elif isinstance(messages[0], list):
                messages_list = messages
                new_messages = []
                for messages_element in messages_list:
                    for index, message in enumerate(messages_element):
                        if index == 0:
                            new_messages.append({"role": "system", "content": message})
                        else:
                            new_messages.append({"role": "user", "content": message})
                messages = new_messages
            
        elif isinstance(messages, dict):
            messages_dict = messages
            new_messages = []
            for key, messages_element in messages_dict.items():
                for index, message in enumerate(messages_element):
                    if index == 0:
                        new_messages.append({"role": "system", "content": message})
                    else:
                        new_messages.append({"role": "user", "content": message})
            messages = new_messages

        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
    else:
        return 0

class SSHConnection:
    def __init__(self):
        # self.hostname = ''
        # self.ssh_port =
        # self.username = ''
        # self.password = ''
        # self.local_port =
        # self.remote_port =
        # self.ssh = paramiko.SSHClient()
        # self.channel = None
        
        self.hostname = ''
        self.username = ''
        self.password = ''
        self.local_port =
        self.remote_port =
        self.ssh = paramiko.SSHClient()
        self.channel = None
        
        
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # try:
        #     # print(f"Connecting to {self.hostname}:{self.ssh_port} as {self.username}...")
        #     self.ssh.connect(self.hostname, self.ssh_port, username=self.username, password=self.password)
        #     # print("Connection successful!")
            
        try: # Use this for the ssh addresses below
            # print(f"Connecting to {self.hostname}:{self.ssh_port} as {self.username}...")
            self.ssh.connect(self.hostname, username=self.username, password=self.password)
            # print("Connection successful!")
        
        except (paramiko.AuthenticationException, paramiko.SSHException, Exception) as e:
            print(f"Error connecting: {e}")

    def send_api_request(self, data):
        # Convert the dictionary to a properly formatted JSON string
        json_data = json.dumps(data)
        # Properly quote the JSON data for the shell command
        quoted_json_data = shlex.quote(json_data)
        command = f"curl -X POST http://localhost:{self.remote_port}/api/generate -d {quoted_json_data} -H 'Content-Type: application/json'"
        try:
            # print(f"Executing command: {command}")
            stdin, stdout, stderr = self.ssh.exec_command(command)
            output = stdout.read().decode()
            error = stderr.read().decode()



            # if error:
            #     print(f"Output: {stdout}")
            #     print(f"Error: {stderr}")
            #     print(f"Output: {output}")
            #     print(f"Error: {error}")
            #     print(f"ERROR: {error}")
            #     return None
            # else:
            #     print("API request successful!")
            return json.loads(output)  # Parse the JSON response
        except paramiko.SSHException as e:
            print(f"Error executing command: {e}")
            return None

    def close(self):
        self.ssh.close()

def send_ssh_command_with_prompt(ssh_connection, model, system_prompt, user_prompt):
    request_data = {
        "model": model,
        "prompt": user_prompt,
        "options": {
            "temperature": 0
        },
        "system": system_prompt,
        "stream": False
    }

    response_data = ssh_connection.send_api_request(request_data)

    if response_data is None:
        return None  # Handle the error more gracefully

    response = response_data.get('response', 'No response found')
    # print(f"Response: {response}")

    tokens_sent = response_data.get('prompt_eval_count', '')
    tokens_received = response_data.get('eval_count', '')
    status = "ok"

    if model.lower() != response_data.get('model', '').lower():  # Case-insensitive comparison
        error_msg = (
            f"Model mismatch: Expected '{model}', got '{response_data.get('model', '')}'"
        )
        print(error_msg)
        raise ValueError(error_msg)  # Raise a specific error for easier debugging

    return response, tokens_sent, tokens_received, status
    
def send_ssh_command_setup(ssh_connection, model):
    # Create Modelfile content
    print("Getting response")
    # modelfile_content = f'FROM {model}\n\nPARAMETER temperature 1\n\nSYSTEM """\n{systemprompt}\n"""'

    # Commands to pull the model, create a new custom model, and run it
    commands = (
        # f'ollama_esa pull {model} && '
        # 'echo \'' + modelfile_content + '\' > ./Modelfile && '
        # f'ollama_esa create {model} -f ./Modelfile && '
        f'ollama_esa create {model} && '
        f'ollama_esa run {model}'
    )

    # Execute the model creation and running process
    response = ssh_connection.send_command(commands)
    # time.sleep(2)
    print("got response")
    print(response)
 
def process_repository(repo_url, commit_hash):
    project_status = {'repo_url': repo_url, 'commit_hash': commit_hash.strip()}
    commit_hash = commit_hash.strip()
    if commit_hash and commit_hash != "":
        repo_name = repo_url.split('com/')[-1]
        directory_name = f"{repo_name.split('/')[0]}_{repo_name.split('/')[1]}_-_dataset_java"
        directory_path = os.path.join(GENERAL_DATASETS_REPO_PATH, directory_name)

        # Cloning phase
        if not os.path.exists(directory_path):
            try:
                result = subprocess.run(['git', 'clone', repo_url, directory_path], check=True, text=True, capture_output=True)
                project_status['cloning'] = 'ok'
            except subprocess.CalledProcessError as e:
                project_status['cloning'] = f"Failed: {e.stdout} | {e.stderr}"

                return project_status
        else:
            project_status['cloning'] = 'ok'
            # try:
            #     subprocess.run(['rm', '-rf', directory_path], cwd=GENERAL_DATASETS_REPO_PATH)
            # except Exception as cleanup_error:
            #     print(f"Error cleaning up directory {directory_path}: {cleanup_error}")

        # Checkout phase
        try:
            subprocess.run(['git', 'fetch', '--all'], check=True, cwd=directory_path, text=True, capture_output=True)
            result = subprocess.run(['git', 'checkout', commit_hash], check=True, cwd=directory_path, text=True, capture_output=True)
            project_status['checkout'] = 'ok'
        except subprocess.CalledProcessError as e:
            project_status['checkout'] = f"Failed: {e.stdout} | {e.stderr}"
            # subprocess.run(['rm', '-rf', item], cwd=dir_path)
            subprocess.run(['rm', '-rf', directory_path], cwd=GENERAL_DATASETS_REPO_PATH)
            return project_status

        # Zipping phase
        try:
            shutil.make_archive(directory_path, 'zip', directory_path)
            project_status['zipping'] = 'ok'
        except Exception as e:
            project_status['zipping'] = f"Exception: {str(e)}"
        
    return project_status 
    
def clone_and_checkout(csv_filename):
    

    
    tasks = []
    with open(csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tasks.append((row['url'], row['sha']))

    processor = Multiprocessor(process_repository, max_workers=4)
    processor.cache(tasks)
    results = processor.execute()

    # Process results and write to Excel
    df = pd.DataFrame(results['output'])
    excel_path = os.path.join(GENERAL_DATASETS_REPO_PATH, 'status_clone_and_checkout.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"Status Excel written at: {excel_path}")

def list_directories(base_path):
    
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and "dataset_java" in d]

def find_java_test_files(directory):
    test_files = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.java') and (file.lower().startswith('test') or file.lower().endswith('test.java')):
                # if "dns" in directory:
                #     print(f"in {directory} found this one:{file}")
                test_file_path = os.path.join(root, file)
                verification_result, code_content = verify_code(test_file_path, "test")
                if verification_result:
                    test_files[test_file_path] = code_content 
                # else:
                    # if "dns" in directory:
                    #     print(f"in {directory} found this one:{file}, did not pass verification process")
                    
    return test_files

def find_production_file(test_files, directory):
    production_files = {}
    production_code = {}

    test_variants = ['Test', 'test']
    production_names = {}
    

    for test_file in test_files:
        base_name = os.path.basename(test_file)
        production_file_name = base_name
        for variant in test_variants:
            if variant in production_file_name:
                production_file_name = production_file_name.replace(variant, '')
                if production_file_name.endswith('.java'):
                    production_names[test_file] = production_file_name
                break
        

    for root, dirs, files in os.walk(directory):
        for test_file, prod_name in production_names.items():
            if prod_name in files:
                prod_file_path = os.path.join(root, prod_name)
                verification_result, code_content = verify_code(prod_file_path, "prod")
                if verification_result:
                    production_files[test_file] = prod_file_path
                    production_code[test_file] = code_content 

    return production_files, production_code

def has_illegal_xml_characters(input_str):
    """Check for characters that are illegal in XML."""
    illegal_xml_ranges = [
        (0x00, 0x08),
        (0x0B, 0x0C),
        (0x0E, 0x1F),
        (0xD800, 0xDFFF),
        (0xFFFE, 0xFFFF),
        (0x1FFFE, 0x1FFFF),
        (0x2FFFE, 0x2FFFF),
        (0x3FFFE, 0x3FFFF),
        (0x4FFFE, 0x4FFFF),
        (0x5FFFE, 0x5FFFF),
        (0x6FFFE, 0x6FFFF),
        (0x7FFFE, 0x7FFFF),
        (0x8FFFE, 0x8FFFF),
        (0x9FFFE, 0x9FFFF),
        (0xAFFFE, 0xAFFFF),
        (0xBFFFE, 0xBFFFF),
        (0xCFFFE, 0xCFFFF),
        (0xDFFFE, 0xDFFFF),
        (0xEFFFE, 0xEFFFF),
        (0xFFFFE, 0xFFFFF),
        (0x10FFFE, 0x10FFFF)
    ]
    return any(ord(char) in range(start, end + 1) for start, end in illegal_xml_ranges for char in input_str)

def verify_code(path, file_type):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        if has_illegal_xml_characters(content):
            return (False, "Has illegal xml characters")
        
        tree = get_tree(content, "java")
        if file_type == "test":
            for _, class_decl in tree.filter(javalang.tree.ClassDeclaration):
                for method in class_decl.methods:
                    if (method.modifiers and 'public' in method.modifiers and
                        (any(isinstance(anno, javalang.tree.Annotation) and anno.name == 'Test' for anno in method.annotations) or
                        method.name.startswith('test'))):
                        return (True, content)
            return (False, "")
        else:
            return (True, content)

    except javalang.parser.JavaSyntaxError:
        return (False, "")

def filter_problematic_code(df):
    problematic_codes = ["""public class LossyCounting {"""]
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        original_code = row['Original_code']
        prod_code = row['Original_prod_code']
        
        # Check each problematic code snippet
        for code_snippet in problematic_codes:
            if code_snippet in original_code:
                df.at[index, 'Original_code'] = ""  # Clear problematic code
                df.at[index, 'Finding_class_status'] = "Error, couldn't save the code"
                break  # Stop checking after the first match

            if code_snippet in prod_code:
                df.at[index, 'Original_prod_code'] = ""  # Clear problematic code
                df.at[index, 'Finding_class_status'] = "Error, couldn't save the code"
                break  # Stop checking after the first match

    return df

def find_test_files(directory, base_path, mode="original"):
    if mode == "original":
        output_path = TS_DETECT_INITIAL_OUTPUT
        data_path = DATASET_PATH
        data_path_comparison = TS_DETECT_DATA_COMPARISON
        
    elif mode == "refactored":
        output_path = TS_DETECT_REPLACED_OUTPUT
        data_path = DATASET_PATH
        data_path_comparison = TS_DETECT_DATA_COMPARISON
        
    elif mode == "refactored_final":
        output_path = TS_DETECT_FINAL_OUTPUT
        data_path = DATASET_PATH
        data_path_comparison = TS_DETECT_DATA_COMPARISON
        
    
    # print(f"finding files for {directory}")
    """
    Vários arquivos de teste não foram incluidos: Arquivos de teste sem correspondente de produção encontrado, ou que o arquivo de produção continha algum problema citado abaixo. Arquivos de teste com mesmo nome de outro arquivo de teste do mesmo projeto. Arquivos de produção com mesmo nome de outo arquivo de produção. Arquivos de teste ou arquivos de produção que continham algum caracter de controle (\\x00 to \\x1F)
    """
    excel_path = os.path.join(data_path, f"{directory}.xlsx")
    csv_path = os.path.join(data_path, f"{directory}.csv")
    full_dir_path = os.path.join(base_path, directory)

    if os.path.isfile(excel_path):
        os.remove(excel_path)
    if os.path.isfile(csv_path):
        os.remove(csv_path)
    # Retrieve test files and their content
    test_files_and_content = find_java_test_files(full_dir_path)
    # Retrieve production files and their content
    production_files, production_files_content = find_production_file(list(test_files_and_content.keys()), full_dir_path)

    # Data for Excel, includes all details
    data_xlsx = {
                "Project": [], 
                "Path_to_file": [], 
                "File_name": [], 
                "Path_to_prod_file": [], 
                "Original_code": [], 
                "Original_prod_code": [],
                "Prompt_ID": [],
                "LLM_used": [],
                "Prompt_to_send": [],
                "Tokens_to_send": [],
                "Tokens_sent": [],
                "Finding_class_status": [],
                "Prompting_status": [],
                "Replacing_status": [],
                "Replaced_code" : [],
                "Class_refactored": [],
                }
    for test_file, test_content in test_files_and_content.items():
        prod_file = production_files.get(test_file, "")
        
        if prod_file == "":
            continue
        
        prod_content = production_files_content.get(test_file, "")
        data_xlsx["Project"].append(directory)
        data_xlsx["File_name"].append(os.path.basename(test_file))
        data_xlsx["Path_to_file"].append(test_file)
        data_xlsx["Path_to_prod_file"].append(prod_file)
        data_xlsx["Original_code"].append(test_content)
        data_xlsx["Original_prod_code"].append(prod_content)
        data_xlsx["Prompt_ID"].append("")
        data_xlsx["LLM_used"].append("")
        data_xlsx["Prompt_to_send"].append("")
        data_xlsx["Tokens_to_send"].append("")
        data_xlsx["Tokens_sent"].append("")
        data_xlsx["Finding_class_status"].append("ok")
        data_xlsx["Prompting_status"].append("")
        data_xlsx["Replacing_status"].append("")
        data_xlsx["Replaced_code"].append("")
        data_xlsx["Class_refactored"].append("")
        
    df_xlsx = pd.DataFrame(data_xlsx)
    df_xlsx = filter_problematic_code(df_xlsx)
    try:
        df_xlsx.to_excel(excel_path, index=False)
        data_csv = {
            "Project": data_xlsx["Project"],
            "Path_to_file": data_xlsx["Path_to_file"],
            "Path_to_prod_file": data_xlsx["Path_to_prod_file"]
        }
        df_csv = pd.DataFrame(data_csv)

        df_csv.to_csv(csv_path, index=False, header=False)
        # print(f"CSV file created without headers for {directory}: {csv_path}")
        run_ts_detect(csv_path, data_path, output_path, data_path_comparison, mode)
    except Exception as e:
        print(f"Exception: {e}")

def rename_test_smell(test_smell_name):
    dict_smell_names = {
    "Assertion Roulette":"AssertionRoulette",#
    "Conditional Test Logic":"ConditionalTestLogic",#
    "Constructor Initialization":"ConstructorInitialization",#
    "Default Test":"DefaultTest", #
    "EmptyTest":"EmptyTest",#
    "Exception Catching Throwing":"ExceptionHandling", # This#
    "General Fixture":"GeneralFixture",#
    "Mystery Guest":"MysteryGuest", # This#
    "Print Statement":"RedundantPrint", # This
    "Redundant Assertion":"RedundantAssertion", #
    "Sensitive Equality":"SensitiveEquality", # This#
    "Verbose Test":"VerboseTest", # This (TS detect tool detects but the paper does not cite it)
    "Sleepy Test":"SleepyTest", #
    "Eager Test":"EagerTest", # This#
    "Lazy Test":"LazyTest", # This#
    "Duplicate Assert":"DuplicateAssertion",#
    "Unknown Test":"UnknownTest",#
    "Ignored Test":"IgnoredTest", #
    "Resource Optimism":"ResourceOptimism", # This#
    "Magic Number Test":"MagicNumberTest",#
    "Dependent Test":"DependentTest" # This (TS detect tool detects but the paper does not cite it)
    }
    
    for smell_name_key, smell_name_value in dict_smell_names.items():
        if smell_name_key == test_smell_name:
            return test_smell_name.replace(smell_name_key, smell_name_value)
        
    return test_smell_name

def transfer_file_to_pynose(directory, base_path):
    pass

def get_test_smells_info(base_path, mode="original"):
    find_all_test_files(base_path, mode)
 
def find_all_test_files(base_path, mode="original"):
    # restore_original_files(base_path)
    directories = list_directories(base_path)
    condition_python_met = False

    if "original" !=  mode:
        subprocess.run(['python', 'runner.py'], cwd = PLUGIN_ROOT_PATH)
        subprocess.run(['python', 'get_csv_stats.py'], cwd = PLUGIN_ROOT_PATH)
        clear_rpl_str(TS_DETECT_REPLACED_OUTPUT)
        copy_files(DETECTOR_OUTPUT_PATH, TS_DETECT_REPLACED_OUTPUT)
        # pass

    with multiprocessing.Pool(processes=4) as pool:  
        for directory in directories:
                        
            condition_java = (
                        (f"_-_dataset_java" in directory) &
                        (f".zip" not in directory) &
                        (directory != "history")
                        )
            
            condition_python = (
                        (f"original" !=  mode) &
                        (f"_-_dataset_python" in directory) &
                        (f".zip" not in directory) &
                        (directory != "history")
                        )
            
            if condition_java and not condition_python:
                print(f"Beginning process for {directory}")
                pool.apply_async(find_test_files, args=(directory, base_path,), kwds={"mode": mode})
                
            # elif condition_python and not condition_java:
            #     print(f"Beginning process for {directory}")
            #     pool.apply_async(transfer_file_to_pynose, args=(directory, base_path,), kwds={"mode": mode})
            #     condition_python_met = True
            
            elif condition_java == condition_python and "original" !=  mode:
                print(f"Error: Condition java and condition python not excludent. {directory}")
            
        
            # pool.apply_async(run_pynose, args=(directory, base_path,), kwds={"mode": mode})        
        pool.close()
        pool.join()

        print("everything done")
        
def run_ts_detect(path_to_csv_or_dir, data_path, output_path, data_path_comparison, mode):
            
    if path_to_csv_or_dir.endswith(".csv"):
        # print(f"running ts detect on {path_to_csv_or_dir}")
        test_smell_detector_jar = os.path.join(output_path , "TestSmellDetector.jar")
        
        csv_file_name = os.path.basename(path_to_csv_or_dir)
        subdir_name = csv_file_name.replace('.csv', '-smells')
        subdir_path = os.path.join(output_path, subdir_name)
        
        # Create a subdirectory for running the detection
        os.makedirs(subdir_path, exist_ok=True)
        command = ['java', '-jar', test_smell_detector_jar, "--file", path_to_csv_or_dir, "-g", "boolean", "-t", "default"]
        # command = ['java', '-jar', test_smell_detector_jar, "--file", path_to_csv_or_dir, "-g", "boolean", "-t", "spadini"]

        try:
            # print("Running command:", ' '.join(command))
            result = subprocess.run(command, capture_output=True, text=True, cwd=subdir_path)
            # print("Command output:", result.stdout)  # Print the output of the Java tool
            if result.stderr:
                # print(f"Error output from TestSmellDetector.jar: {result.stderr}")
                pass
            
            file_output = os.listdir(subdir_path)
            
            if len(file_output) == 1:
                output_file_path = os.path.join(subdir_path, file_output[0])
                new_output_file_path = os.path.join(output_path, f"{subdir_name}.csv")
                shutil.move(output_file_path, new_output_file_path)
                
            else:
                raise FileNotFoundError("Output CSV file not created by Java application.")
            
            filename = os.path.basename(new_output_file_path).replace("-smells.csv", "")
            excel_path = os.path.join(data_path, f"{filename}.xlsx")
            print("UPDATING")
            update_xlsx(new_output_file_path, excel_path, data_path_comparison, mode)
            
        except Exception as e:
            # Capture detailed error information
            error_message = f"""
            Error: {str(e)}
            
            Command: {' '.join(command)}
            Return Code: {result.returncode if 'result' in locals() else 'No return code available'}
            Java STDOUT:
            {result.stdout if 'result' in locals() else 'No STDOUT available'}
            
            Java STDERR:
            {result.stderr if 'result' in locals() else 'No STDERR available'}
            """
            error_file_path = os.path.join(output_path, f"{subdir_name}.txt")
            with open(error_file_path, 'w') as error_file:
                error_file.write(error_message)
            
        finally:
            shutil.rmtree(subdir_path)
            
        
            
            
    else:
        print("something went wrong")
        
def update_xlsx(test_smell_output, xlsx_file_path, data_path_comparison, mode):
    """
    Reads test smell results and updates XLSX in long format (one row per smell)
    """
    # print(f"UPDATING XLSX {xlsx_file_path}")
    # Read XLSX into a DataFrame
    df_xlsx = pd.read_excel(xlsx_file_path)

    # Read test smell results (assuming modified output format)
    df_test_smells = pd.read_csv(test_smell_output)
    
    df_test_smells.drop(columns=['App','ProductionFilePath','RelativeTestFilePath','RelativeProductionFilePath','NumberOfMethods','TestClass'], inplace=True)

    # Melt the DataFrame for the desired format
    df_test_smells_long = df_test_smells.melt(
        id_vars='TestFilePath', 
        var_name='Smell',
        value_name='Is_smell_present_original'
    )

    df_test_smells_long.rename(columns={'TestFilePath': "Path_to_file"}, inplace=True)
    df_merged = df_xlsx.merge(df_test_smells_long, on="Path_to_file", how='outer')
    df_merged['Smell'] = df_merged['Smell'].apply(rename_test_smell)
    df_merged = transform_df(df_merged)
    print("SAVING")
    basename = os.path.basename(xlsx_file_path.replace(".xlsx", f"_clean_{mode}"))
    rootpath = os.path.dirname(xlsx_file_path.replace(".xlsx", f"_clean"))
    DataframeManager.save(df_merged, rootpath, name_to_find=basename)
    # basename = os.path.basename(xlsx_file_path.replace(".xlsx", f"_clean_{mode}"))
    # DataframeManager.save(df_merged, data_path_comparison, name_to_find=basename)

def process_csv(file_path):
    # Example processing logic adapted to the new CSV format
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Adapt this part based on the actual logic needed for the new CSV format
    # For example, if you need to filter rows, process columns, etc.
    # This is just a placeholder for the actual processing
    processed_data = df.apply(lambda x: x)  # Placeholder for actual processing logic
    
    # Save the processed data to a new CSV file (optional)
    output_path = file_path.replace('.csv', '_processed.csv')
    processed_data.to_csv(output_path, index=False)
    print(f"Processed {file_path} and saved to {output_path}")

def list_python_csv_files(dir_path):
    # List all files in the given directory with "python" in their name and ".csv" extension
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if "python" in f and f.endswith('.csv')]

def process_all_python_csv_files(dir_path):
    # Get the list of CSV files to process
    csv_files = list_python_csv_files(dir_path)
    
    # Use multiprocessing to process each file in parallel
    with mp.Pool(processes=NUM_WORKERS) as pool:
        pool.map(process_csv, csv_files)

def process_prompts(path_of_files_with_clean_generated, string_to_search, key_string, NUMBER_OF_SMELLS_LIST):
    prompts = []
    updated = False
    encodings = ['utf-8']
    df_code_to_prompt = DataframeManager.load(path_of_files_with_clean_generated, name_to_find=string_to_search)
    print(path_of_files_with_clean_generated)
    NumberOfIndividualPrompts = 0

    for dirpath, _, filenames in os.walk(PROMPTS_USED_DIR_PATH_JAVA):
        for filename in filenames:
            if filename.endswith('.txt') and MODEL_TO_USE_NAME in dirpath:
                encoding_found = False
                current_path = os.path.join(dirpath,filename)
                
                for enc in encodings:
                    try:
                        with open(current_path, "r", encoding=enc) as source:
                            content = source.read()
                            if "USERCONTENT" not in content:
                                continue
                            
                            else:
                                encoding_found = True
                                FIRSTcoord = content.find("---SYSTEMCONTENT---") + len("---SYSTEMCONTENT---")
                                SECONDcoord = content.find("---USERCONTENT---") + len("---USERCONTENT---")
                                smell_prompt, _ = os.path.splitext(filename[6:])
                                prompt = {"System_content": content[FIRSTcoord:(content.find("---USERCONTENT---"))],
                                    "User_content": content[SECONDcoord:],
                                    "Smell_prompt": smell_prompt.replace(MODEL_TO_USE_NAME,""),
                                    "Prompt_version": filename[:5],
                                    }
                                # print(f"This is the prompt: {prompt}")
                                prompts.append(prompt)
                                NumberOfIndividualPrompts += 1
                                break
                    except:
                        continue
                        
                if not encoding_found:  
                    try:  
                        with open(current_path, 'rb') as f:
                            result = chardet.detect(f.read()) 
                            encoding = result['encoding']
                            encodings.append(encoding)
                        if "USERCONTENT" in content:
                            FIRSTcoord = content.find("---SYSTEMCONTENT---") + len("---SYSTEMCONTENT---")
                            SECONDcoord = content.find("---USERCONTENT---") + len("---USERCONTENT---")
                            smell_prompt, _ = os.path.splitext(filename[6:])
                            prompt = {"System_content": content[FIRSTcoord:(content.find("---USERCONTENT---"))],
                                "User_content": content[SECONDcoord:],
                                "Smell_prompt": smell_prompt.replace(MODEL_TO_USE_NAME,""),
                                "Prompt_version": filename[:5]
                                }
                            prompts.append(prompt)
                            NumberOfIndividualPrompts += 1

                            
                    except:
                        continue
                    
    # try:              
    #     df_code_to_prompt["Is_smell_present_original"] = df_code_to_prompt["Is_smell_present_original"].astype("boolean")             
    # except Exception as e:
    #     print(e)    
    # df_code_to_prompt["Prompt_ID"] = df_code_to_prompt["Prompt_ID"].astype(str)
    
    prompts_done = set()
    function_counter = 0
    already_done = set()
    indices_to_delete = []
    new_lines = []
    
    if len(prompts) == 0:
        return
    
    # for index, row in tqdm(df_code_to_prompt.iterrows(), total=df_code_to_prompt.shape[0], file=sys.stdout, desc="Generating prompts", leave=True):
    #     if row["Prompt_ID"] and row["Prompt_ID"].split("_")[1].split(".")[0] == "01":
    #         prompt_ID_is_done = all([row["Prompt_ID"] in group_df["Prompt_ID"].values for _, group_df in df_code_to_prompt[df_code_to_prompt["Smell"] == row["Smell"]].groupby("Path_to_file")])
            
    #     elif row["Prompt_ID"] and row["Prompt_ID"].split("_")[1].split(".")[0] == "00":
    #         prompt_ID_is_done = all([row["Prompt_ID"] in group_df["Prompt_ID"].values for _, group_df in df_code_to_prompt[(df_code_to_prompt["Smell"] == row["Smell"]) & (df_code_to_prompt["Is_smell_present_original"])].groupby("Path_to_file")])
    #     else:
    #         print(f"New prompt type added, not 01. or 00.")   
              

    df_code_to_prompt["ID_smell"] = df_code_to_prompt["Path_to_file"] + df_code_to_prompt["Smell"]
        
    for index, row in tqdm(df_code_to_prompt.iterrows(), total=df_code_to_prompt.shape[0], file=sys.stdout, desc=f"Generating prompts for {os.path.basename(path_of_files_with_clean_generated)}", leave=True):
        values_dict = row.to_dict()
        if row["Prompt_ID"] in prompts_done:
            continue
        
        if row["Path_to_file"] + row["Smell"] not in already_done and "ok" == row["Finding_class_status"]:   
            smell = values_dict["Smell"]
            code = values_dict['Original_code']
            ID = values_dict["Path_to_file"]
            
            df_filtered_for_class = df_code_to_prompt[(df_code_to_prompt["Path_to_file"] == ID) & (df_code_to_prompt["Smell"] == smell)].copy()
            
            df_filtered_for_class["Prompt_ID"] = df_filtered_for_class["Prompt_ID"].fillna('')
            
            prompt_for_this_class = [prompt for prompt in prompts if prompt["Smell_prompt"] == smell]
            
            prompt_for_this_class_now = prompt_for_this_class
            for prompt in prompt_for_this_class_now:
                prompt_ID = prompt["Smell_prompt"] + "_" + prompt["Prompt_version"]
                if prompt_ID in prompts_done:
                
                    prompt_for_this_class.remove(prompt)
                    continue
                if prompt_ID not in df_filtered_for_class["Prompt_ID"].values:
                    if row["Is_smell_present_original"] == False and prompt["Prompt_version"][:2] != "01":
                        continue
                    if "DefaultTest_00.00" in prompt_ID:
                        print(f"This is the prompt ID: {prompt_ID}\n This smell exists on the current")
                    if pd.isna(row["Prompt_ID"]) or ((isinstance(row["Prompt_ID"], str) and not row["Prompt_ID"].strip())):
                        indices_to_delete.append(index)
                        
                    values_dict_new = {}
                    system_content = prompt['System_content']
                    user_content = prompt['User_content']
                    prompt_to_send = f"System: {system_content}\n User: {user_contentfrom
                    messages_to_count = [
                        {"role": "system", "content": prompt['System_content']},
                        {"role": "user", "content": prompt['User_content'] + code}
                    ]
                    tokens_to_send = num_tokens_from_messages(messages_to_count)
                    
                    values_dict_new["File_name"] = values_dict["File_name"]
                    values_dict_new["Project"] = values_dict["Project"]
                    values_dict_new["Path_to_file"] = values_dict["Path_to_file"]
                    values_dict_new["Path_to_prod_file"] = values_dict["Path_to_prod_file"]
                    values_dict_new["Original_code"] = values_dict["Original_code"]
                    values_dict_new["Original_prod_code"] = values_dict["Original_prod_code"]
                    values_dict_new["Is_smell_present_original"] = values_dict["Is_smell_present_original"]
                    values_dict_new["Smell"] = smell
                    values_dict_new['Original_code'] = code
                    values_dict_new['ID'] = ID
                    values_dict_new["Prompt_to_send"] = prompt_to_send
                    values_dict_new["LLM_used"] = MODEL_TO_USE
                    values_dict_new["Class_tokens"] = class_tokens
                    values_dict_new["Tokens_to_send"] = tokens_to_send
                    values_dict_new["Prompt_ID"] = prompt_ID
                    values_dict_new["Creating_prompt_status"] = "ok"
                    new_lines.append(values_dict_new)
                    
                    function_counter += 1
                    updated = True
                  
            already_done.add(row["Path_to_file"]+row["Smell"])
            
    print("UPDATED")
    df_code_to_prompt = df_code_to_prompt.drop(indices_to_delete)
    df_code_to_prompt.dropna(how='all', inplace=True)
    
    df_new_lines = pd.DataFrame(new_lines)
    df_new_lines.dropna(how='all', inplace=True)
    
    df_code_to_prompt = DataframeManager.concat([df_new_lines, df_code_to_prompt], ignore_index=True)
    filename = string_to_search.replace(f"_-_dataset_{LANGUAGE_TO_USE}","").replace(key_string,"")
    
    df_true = df_code_to_prompt[df_code_to_prompt['Is_smell_present_original'] == True]
    true_counts = df_true.groupby('Prompt_ID').size().reset_index(name='count_true')
    
    # Filter rows where 'Is_smell_present_original' is False
    df_false = df_code_to_prompt[df_code_to_prompt['Is_smell_present_original'] == False]
    false_counts = df_false.groupby('Prompt_ID').size().reset_index(name='count_false')
    
    # Merging the two counts into a single DataFrame
    counts = pd.merge(true_counts, false_counts, on='Prompt_ID', how='outer').fillna(0)
    counts["Project"] = string_to_search
    
    NUMBER_OF_SMELLS_LIST.append(counts)
    DataframeManager.save(df_code_to_prompt, DATASET_PATH, name_to_find=f"pre_ref_data_{LANGUAGE_TO_USE}_{filename}_-_{MODEL_TO_USE_NAME}")

def append_missing_lines(subdf, df_pre_ref, columns_to_empty):
    # Perform a right merge to get all entries from df_pre_ref and matched entries from subdf
    merged_df = pd.merge(subdf, df_pre_ref, how='outer', indicator=True)
    
    # Isolate rows that were only in df_pre_ref
    merged_df_filtered = merged_df[merged_df['_merge'] == 'right_only'].copy()

    # Drop the '_merge' indicator column and any unnecessary columns that aren't needed
    merged_df_filtered.drop(columns=['_merge'], inplace=True)

    # Clear specified columns in the missing rows
    for column in columns_to_empty:
        if column in merged_df_filtered.columns:
            merged_df_filtered[column] = ""
            
    merged_df_filtered = merged_df_filtered.drop_duplicates()
    # Append missing rows to subdf, resetting index to maintain consistency
    updated_subdf = pd.concat([subdf, merged_df_filtered], ignore_index=True)
    updated_subdf = updated_subdf[(updated_subdf["Smell"] != "VerboseTest") & (updated_subdf["Smell"] != "DependentTest")]
    return updated_subdf
    
    
def create_to_prompt_prompt_ID_dfs(key, subdf, string, key_string):
    print(000)
    
    general_string = (string.replace(key_string, "") + "_to_prompt_" + key).replace("_original","")
    print(1)
    try:
        # print(f"THIS IS THE GENERAL STRING {general_string}")
        df_current_info = DataframeManager.load(DATASET_TO_PROMPT_PATH, name_to_find=general_string)
        print(2)
        # print(f"THIS IS THE df {df_current_info.head()}")
        if isinstance(df_current_info, pd.DataFrame):
            print(20)
            df_current_info = df_current_info[df_current_info["Prompting_status"] == "ok"]
            print(200)
            print("Prompting_status" in df_current_info)
            print(len(df_current_info[df_current_info["Prompting_status"] == "ok"]) > 0)
            if "Prompting_status" in df_current_info and len(df_current_info[df_current_info["Prompting_status"] == "ok"]) > 0:
                print(2000)
                df_current_info['combined_key'] = df_current_info['Path_to_file'] + df_current_info['Smell'] + df_current_info['Prompt_ID'] + df_current_info['LLM_used']
                print(20000)
                subdf['combined_key'] = subdf['Path_to_file'] + subdf['Smell'] + subdf['Prompt_ID'] + subdf['LLM_used']
                for index, row in df_current_info.iterrows():
                    if row['combined_key'] in subdf['combined_key'].values:
                        print("INputting")
                        mask = subdf['combined_key'] == row['combined_key']
                        first_matching_index = subdf.index[mask][0] if any(mask) else None
                        subdf.loc[first_matching_index] = row
                    else:
                        print("Appending")
                        subdf._append(row, ignore_index=True)  
                        
                df_current_info.drop(columns=['combined_key'], inplace=True)  
                subdf.drop(columns=['combined_key'], inplace=True) 

                condition = (subdf["Prompting_status"] == "ok") & ((subdf["Prompt_received"].str.len() < 5) | (subdf["Tokens_sent"] < 2))
                subdf.loc[condition, "Prompting_status"] = ""
                subdf.loc[condition, "Prompt_received"] = ""
                subdf.loc[condition, "Tokens_sent"] = ""
            
                DataframeManager.save(subdf, DATASET_TO_PROMPT_PATH, name_to_find=general_string)
            else:
                condition = (subdf["Prompting_status"] == "ok") & ((subdf["Prompt_received"].str.len() < 5) | (subdf["Tokens_sent"] < 2))
                subdf.loc[condition, "Prompting_status"] = ""
                subdf.loc[condition, "Prompt_received"] = ""
                subdf.loc[condition, "Tokens_sent"] = ""
                DataframeManager.save(subdf, DATASET_TO_PROMPT_PATH, name_to_find=general_string)
        else: 
            condition = (subdf["Prompting_status"] == "ok") & ((subdf["Tokens_sent"] < 2))
            subdf.loc[condition, "Prompting_status"] = ""
            subdf.loc[condition, "Prompt_received"] = ""
            subdf.loc[condition, "Tokens_sent"] = ""
            DataframeManager.save(subdf, DATASET_TO_PROMPT_PATH, name_to_find=general_string)
    
    except Exception as e:
        print(f"An error occurred: {e}\n {general_string}")

def create_to_prompt_dfs(path_of_files_with_prompts_generated, string_to_search, key_string):
    df_pre_ref = DataframeManager.load(path_of_files_with_prompts_generated, name_to_find=string_to_search)
    df_pre_ref = transform_df(df_pre_ref)
    if isinstance(df_pre_ref, pd.DataFrame):
        subdfs_df_pre_ref = {prompt_ID: subdf for prompt_ID, subdf in df_pre_ref.groupby("Prompt_ID")}
        columns_to_empty = ['Creating_prompt_status', 'Prompt_to_send', 'LLM_used', 'Class_tokens', 'Tokens_to_send', 'Prompt_ID']
        for key, subdf in subdfs_df_pre_ref.items():
            updated_subdf = append_missing_lines(subdf, df_pre_ref, columns_to_empty)
            create_to_prompt_prompt_ID_dfs(key, updated_subdf, string_to_search, key_string)
    
def list_unique_xlsx_file_paths(dirpath, string):
    strings_to_search = [xl_file.replace("current_","")[:-25] for xl_file in os.listdir(dirpath) if string in xl_file and xl_file.endswith("xlsx") and "~$" not in xl_file]

    return list(set(strings_to_search))    
    
def create_to_prompt(path_of_files_with_clean_generated):
    key_string = "_clean"
    number_of_smells_list = []
    list_of_dfs = []
    strings_to_search = list_unique_xlsx_file_paths(path_of_files_with_clean_generated, key_string)
    # strings_already_done = list_unique_xlsx_file_paths(DATASET_TO_PROMPT_PATH, "to_prompt")
    # with multiprocessing.Pool(processes=NUM_WORKERS) as pool:  
    for string_to_search in strings_to_search:
        
        condition = (
                    (f"_-_dataset_java" in string_to_search) &
                    ("original" in string_to_search) &
                    ("pre_ref_data" not in string_to_search) &
                    ("_check_" not in string_to_search) &
                    ("00." not in string_to_search) &
                    ("01." not in string_to_search) 
                    )
        
        if condition:
            print(f"Beginning process for {string_to_search}, phase: {key_string}")
            # pool.apply_async(process_prompts, args=(path_of_files_with_clean_generated, string_to_search,key_string, NUMBER_OF_SMELLS_LIST,))   
            
            df_code_to_prompt = DataframeManager.load(path_of_files_with_clean_generated, name_to_find=string_to_search)
            print(f'loaded df: {string_to_search}')
            df_true = df_code_to_prompt[df_code_to_prompt['Is_smell_present_original'] == True]
            true_counts = df_true.groupby('Smell').size().reset_index(name='count_true')
            print(f'true contents: {string_to_search}')
            # Filter rows where 'Is_smell_present_original' is False
            df_false = df_code_to_prompt[df_code_to_prompt['Is_smell_present_original'] == False]
            false_counts = df_false.groupby('Smell').size().reset_index(name='count_false')
            print(f'false contents: {string_to_search}')
            # Merging the two counts into a single DataFrame
            counts = pd.merge(true_counts, false_counts, on='Smell', how='outer').fillna(0)
            print(f'merging: {string_to_search}')
            counts["Project"] = string_to_search
            list_of_dfs.append(df_code_to_prompt)
            number_of_smells_list.append(counts)
            print(f'appending: {string_to_search}')
                    
        # pool.close()
        # pool.join()
        
    aggregated_java = pd.concat(list_of_dfs, ignore_index=True)
    DataframeManager.save(aggregated_java, DATASET_TO_PROMPT_PATH, name_to_find=f"to_prompt_optimizat_java")
    
    aggregated_counts = pd.concat(number_of_smells_list, ignore_index=True)
    DataframeManager.save(aggregated_counts, DATASET_PATH, name_to_find=f"smell_per_project_{LANGUAGE_TO_USE}")
    # Group by 'Prompt_ID' and sum the counts
    aggregated_counts = aggregated_counts.groupby('Smell', as_index=False).sum()
    DataframeManager.save(aggregated_counts, DATASET_PATH, name_to_find=f"smell_total_{LANGUAGE_TO_USE}")
    
    
        
    # key_string = "pre_ref_data_"
    # strings_to_search = list_unique_xlsx_file_paths(path_of_files_with_clean_generated, key_string)
    # with multiprocessing.Pool(processes=NUM_WORKERS) as pool:  
    #     for string_to_search in strings_to_search:
            
    #         condition = (
    #                     (MODEL_TO_USE_NAME in string_to_search) & 
    #                     (LANGUAGE_TO_USE in string_to_search) &
    #                     ("original" in string_to_search) &
    #                     ("_check_" not in string_to_search) &
    #                     ("00." not in string_to_search) &
    #                     ("01." not in string_to_search) 
    #                     )
            
    #         if condition:
    #             print(f"Beginning process for {string_to_search}, phase: {key_string}")
    #             pool.apply_async(create_to_prompt_dfs, args=(path_of_files_with_clean_generated, string_to_search,key_string,))              
    #     pool.close()
    #     pool.join()
        
def get_token_time(prompt):
    approx_tokens = num_tokens_from_messages(prompt)
    max_time = (approx_tokens/10000)
    return max_time

def transform_df(old_df):
    # Define the mapping between old headers and expected new headers
    header_mapping = {
        'File_name': 'File_name',
        'Project': 'Project',
        'Path_to_file': 'Path_to_file',
        'Original_code': 'Original_code',
        'Is_smell_present_original': 'Is_smell_present_original',
        'Smell': 'Smell',
        'ID': 'ID',
        'Prompt_to_send': 'Prompt_to_send',
        'LLM_used': 'LLM_used',
        'Class_tokens': 'Class_tokens',
        'Tokens_to_send': 'Tokens_to_send',
        'Prompt_ID': 'Prompt_ID',
        'Creating_prompt_status': 'Creating_prompt_status',
        'Tokens_sent': 'Tokens_sent',
        'Finding_class_status': 'Finding_class_status',
        'Prompting_status': 'Prompting_status',
        'Replacing_status': 'Replacing_status',
        'Replaced_code': 'Replaced_code',
        'Class_refactored': 'Class_refactored',
        'ID_smell': 'ID_smell'
    }

    # List of all headers expected in the new DataFrame
    expected_headers = [
        'Project', 'ID', 'Dataset', 'File_name', 'Class_name', 'Path_to_file',
        'Original_code', 'Smell', 'Is_smell_present_original',
        'Is_smell_present_refactored', 'Prompt_ID', 'LLM_used',
        'Prompt_to_send', 'Tokens_to_send', 'Tokens_sent', 'Class_tokens',
        'Prompt_received', 'Response_time', 'Response_timestamp', 'Last_scan_timestamp',
        'Finding_class_status', 'Creating_prompt_status', 'Prompting_status',
        'Replacing_status', 'Replaced_code', 'Class_refactored',
        '%_lines_passed_original', '%_lines_passed_refactored', 'GPT_detected_right',
        'GPT_detection_status', 'ID_smell', 'Lines_executed_original', 'Lines_executed_refactored',
        'N_branch_cov_test_file_original', 'N_branch_cov_test_file_refactored', 'Tokens_received'
    ]
                

    new_df = old_df.rename(columns=header_mapping)

    # Add any missing columns from the expected headers list
    for column in expected_headers:
        if column not in new_df.columns:
            new_df[column] = pd.NA  # Using pandas NA for better compatibility with different data types
    return new_df

def create_to_replace(path_of_files_with_prompts_gathered, string_to_search, key_string,mode):
    df_to_replace_old = DataframeManager.load(path_of_files_with_prompts_gathered, name_to_find=string_to_search)
    df_to_replace = transform_df(df_to_replace_old)
    print(f"This is the key string :{key_string}")
    print(f"This is the string to search:{string_to_search}")
    
    df_to_replace_to_save = df_to_replace[df_to_replace["Det_or_ref"] == "Refatoração"]
    
    if mode == "optimization":
        string_to_search = string_to_search.replace(key_string, "") + f"_to_replace_{mode}"
    else:
        string_to_search = string_to_search.replace(key_string, "") + f"_to_replace"
        
    DataframeManager.save(df_to_replace_to_save.drop_duplicates(), DATASET_TO_REPLACE_PATH, name_to_find=string_to_search)      

def create_to_replace_all(path_of_files_with_prompts_gathered, mode="optimization"):
    del_files(DATASET_TO_REPLACE_PATH, mode="hard")
    if mode != "optimization":
        key_string = "to_prompt"
    else:
        key_string = "to_prompt_optimization_selected"
        
      
    strings_to_search = list_unique_xlsx_file_paths(path_of_files_with_prompts_gathered, key_string)
    print(f"not condition{strings_to_search}")   
    # with multiprocessing.Pool(processes=NUM_WORKERS) as pool:  
    for string_to_search in strings_to_search:
                        
            # condition = (
            #             (MODEL_TO_USE_NAME in string_to_search) & 
            #             (LANGUAGE_TO_USE in string_to_search) &
            #             ("01." not in string_to_search) 
            #             )
            
            # if condition:
        print(f"Beginning process for {string_to_search}, phase: {key_string}")
        # pool.apply_async(create_to_replace, args=(path_of_files_with_prompts_gathered, string_to_search,key_string,mode,))       
        create_to_replace(path_of_files_with_prompts_gathered, string_to_search, key_string, mode)    
                  
        # pool.close()
        # pool.join()
        
def duplicate_and_modify(df, path_to_prompt, new_llm_used, prompts):
                    
    # Copiar o DataFrame
    df_copy = df.copy()
    row_to_append = []
    # Renomear a coluna 'LLM_used' para o novo valor fornecido
    
    
    if new_llm_used != "gemma:7b":
        df_copy['LLM_used'] = new_llm_used
        columns_to_erase = [
            'Tokens_sent', 'Tokens_received', 'Last_scan_timestamp', "Prompt_to_send", 'Response_timestamp', 'Response_time', 'Prompt_received', 'Prompting_status', "GPT_detected_right", "GPT_detection_status"
        ]
        df_copy[columns_to_erase] = ''
        
    for index, row in df_copy.iterrows():

        prompt_ID_from_df = row["Prompt_ID"]
        smell = row["Smell"]
        code = row['Original_code']
        ID = row["Path_to_file"]
        llm_used = row["LLM_used"]
        if "00." in prompt_ID_from_df:
            cat = "00."
        elif "01." in prompt_ID_from_df:
            cat = "01."
        
        for prompt in prompts:
            prompt_ID = prompt["Smell_prompt"] + "_" + prompt["Prompt_version"]
            # print(f'LLM: {new_llm_used} --- Prompt ID from df {prompt_ID_from_df}')
            # print(f'LLM: {new_llm_used} --- Prompt ID {prompt_ID}')
            if cat in prompt_ID:
                
                if prompt_ID == prompt_ID_from_df:
                    print(f'LLM: {new_llm_used} --- Prompt ID in line: {prompt_ID}')
                    system_content = prompt['System_content']
                    user_content = prompt['User_content']
                    prompt_to_send = f"System: {system_content}\n User: {user_content + code }"
                    messages_to_count = [
                            {"role": "system", "content": ""},
                            {"role": "user", "content": code}
                        ]
                    class_tokens = num_tokens_from_messages(messages_to_count)
                        
                    messages_to_count = [
                            {"role": "system", "content": prompt['System_content']},
                            {"role": "user", "content": prompt['User_content'] + code}
                        ]
                    tokens_to_send = num_tokens_from_messages(messages_to_count)
                    
                    df_copy.at[index, "Prompt_to_send"] = prompt_to_send
                    
                elif prompt_ID not in df_copy["Prompt_ID"].values and smell in prompt_ID:
                    print(f'LLM: {new_llm_used} --- Prompt ID not in line: {prompt_ID}, appending')
                    new_row = row.copy()
                    system_content = prompt['System_content']
                    user_content = prompt['User_content']
                    prompt_to_send = f"System: {system_content}\n User: {user_content + code}"
                    messages_to_count = [
                            {"role": "system", "content": ""},
                            {"role": "user", "content": code}
                        ]
                    class_tokens = num_tokens_from_messages(messages_to_count)
                        
                    messages_to_count = [
                            {"role": "system", "content": prompt['System_content']},
                            {"role": "user", "content": prompt['User_content'] + code}
                        ]
                    tokens_to_send = num_tokens_from_messages(messages_to_count)
                    new_row["Prompt_ID"] = prompt_ID
                    new_row["Prompt_received"] = ""
                    new_row["Response_time"] = ""
                    new_row["Prompting_status"] = ""
                    new_row["Class_tokens"] = class_tokens
                    new_row["Tokens_to_send"] = tokens_to_send
                    new_row["Prompt_to_send"] = prompt_to_send
                    row_to_append.append(new_row)
                    
    df_rows_to_append = pd.DataFrame(row_to_append)
    df_copy = df_copy._append(df_rows_to_append, ignore_index=True)
    len(f"This is the len{df_copy}")
    df_copy = df_copy.drop_duplicates(subset=['ID', 'Prompt_ID', 'LLM_used'])
    return df_copy
       
def create_all_to_prompt_prompt_id_optimization_df(path_to_prompt, mode="soft"):
    prompts_llama = []
    encodings = ['utf-8']
    llm_used = "llama370b"
    for dirpath, _, filenames in os.walk(PROMPTS_USED_DIR_PATH_JAVA):
        for filename in filenames:
            if filename.endswith('.txt') and llm_used in dirpath:
                encoding_found = False
                current_path = os.path.join(dirpath,filename)
                
                for enc in encodings:
                    try:
                        with open(current_path, "r", encoding=enc) as source:
                            content = source.read()
                            if "USERCONTENT" not in content:
                                continue
                            
                            else:
                                encoding_found = True
                                FIRSTcoord = content.find("---SYSTEMCONTENT---") + len("---SYSTEMCONTENT---")
                                SECONDcoord = content.find("---USERCONTENT---") + len("---USERCONTENT---")
                                smell_prompt, _ = os.path.splitext(filename[6:])
                                prompt = {"System_content": content[FIRSTcoord:(content.find("---USERCONTENT---"))],
                                    "User_content": content[SECONDcoord:],
                                    "Smell_prompt": smell_prompt.replace(llm_used,""),
                                    "Prompt_version": filename[:5],
                                    }
                                # print(f"This is the prompt: {prompt}")
                                prompts_llama.append(prompt)
                                break
                    except:
                        continue
                        
                if not encoding_found:  
                    try:  
                        with open(current_path, 'rb') as f:
                            result = chardet.detect(f.read()) 
                            encoding = result['encoding']
                            encodings.append(encoding)
                        if "USERCONTENT" in content:
                            FIRSTcoord = content.find("---SYSTEMCONTENT---") + len("---SYSTEMCONTENT---")
                            SECONDcoord = content.find("---USERCONTENT---") + len("---USERCONTENT---")
                            smell_prompt, _ = os.path.splitext(filename[6:])
                            prompt = {"System_content": content[FIRSTcoord:(content.find("---USERCONTENT---"))],
                                "User_content": content[SECONDcoord:],
                                "Smell_prompt": smell_prompt.replace(llm_used,""),
                                "Prompt_version": filename[:5]
                                }
                            prompts_llama.append(prompt)

                            
                    except:
                        continue
    prompts_gpt = []                
    llm_used = "gpt-4-turbo"
    for dirpath, _, filenames in os.walk(PROMPTS_USED_DIR_PATH_JAVA):
        for filename in filenames:
            if filename.endswith('.txt') and llm_used in dirpath:
                encoding_found = False
                current_path = os.path.join(dirpath,filename)
                
                for enc in encodings:
                    try:
                        with open(current_path, "r", encoding=enc) as source:
                            content = source.read()
                            if "USERCONTENT" not in content:
                                continue
                            
                            else:
                                encoding_found = True
                                FIRSTcoord = content.find("---SYSTEMCONTENT---") + len("---SYSTEMCONTENT---")
                                SECONDcoord = content.find("---USERCONTENT---") + len("---USERCONTENT---")
                                smell_prompt, _ = os.path.splitext(filename[6:])
                                prompt = {"System_content": content[FIRSTcoord:(content.find("---USERCONTENT---"))],
                                    "User_content": content[SECONDcoord:],
                                    "Smell_prompt": smell_prompt.replace(llm_used,""),
                                    "Prompt_version": filename[:5],
                                    }
                                # print(f"This is the prompt: {prompt}")
                                prompts_gpt.append(prompt)
                                break
                    except:
                        continue
                        
                if not encoding_found:  
                    try:  
                        with open(current_path, 'rb') as f:
                            result = chardet.detect(f.read()) 
                            encoding = result['encoding']
                            encodings.append(encoding)
                        if "USERCONTENT" in content:
                            FIRSTcoord = content.find("---SYSTEMCONTENT---") + len("---SYSTEMCONTENT---")
                            SECONDcoord = content.find("---USERCONTENT---") + len("---USERCONTENT---")
                            smell_prompt, _ = os.path.splitext(filename[6:])
                            prompt = {"System_content": content[FIRSTcoord:(content.find("---USERCONTENT---"))],
                                "User_content": content[SECONDcoord:],
                                "Smell_prompt": smell_prompt.replace(llm_used,""),
                                "Prompt_version": filename[:5]
                                }
                            prompts_gpt.append(prompt)

                            
                    except:
                        continue
                    
    prompts_gemma= []                
    llm_used = "gemma7b"
    for dirpath, _, filenames in os.walk(PROMPTS_USED_DIR_PATH_JAVA):
        for filename in filenames:
            if filename.endswith('.txt') and llm_used in dirpath:
                encoding_found = False
                current_path = os.path.join(dirpath,filename)
                
                for enc in encodings:
                    try:
                        with open(current_path, "r", encoding=enc) as source:
                            content = source.read()
                            if "USERCONTENT" not in content:
                                continue
                            
                            else:
                                encoding_found = True
                                FIRSTcoord = content.find("---SYSTEMCONTENT---") + len("---SYSTEMCONTENT---")
                                SECONDcoord = content.find("---USERCONTENT---") + len("---USERCONTENT---")
                                smell_prompt, _ = os.path.splitext(filename[6:])
                                prompt = {"System_content": content[FIRSTcoord:(content.find("---USERCONTENT---"))],
                                    "User_content": content[SECONDcoord:],
                                    "Smell_prompt": smell_prompt.replace(llm_used,""),
                                    "Prompt_version": filename[:5],
                                    }
                                # print(f"This is the prompt: {prompt}")
                                prompts_gemma.append(prompt)
                                break
                    except:
                        continue
                        
                if not encoding_found:  
                    try:  
                        with open(current_path, 'rb') as f:
                            result = chardet.detect(f.read()) 
                            encoding = result['encoding']
                            encodings.append(encoding)
                        if "USERCONTENT" in content:
                            FIRSTcoord = content.find("---SYSTEMCONTENT---") + len("---SYSTEMCONTENT---")
                            SECONDcoord = content.find("---USERCONTENT---") + len("---USERCONTENT---")
                            smell_prompt, _ = os.path.splitext(filename[6:])
                            prompt = {"System_content": content[FIRSTcoord:(content.find("---USERCONTENT---"))],
                                "User_content": content[SECONDcoord:],
                                "Smell_prompt": smell_prompt.replace(llm_used,""),
                                "Prompt_version": filename[:5]
                                }
                            prompts_gemma.append(prompt)

                            
                    except:
                        continue
    
    file_to_create_based = None
        
    key_string = "_to_prompt"
    strings_to_search = list_unique_xlsx_file_paths(path_to_prompt, key_string)
    
    smells = [
        "AssertionRoulette",
        "ConditionalTestLogic",
        "ConstructorInitialization",
        "DefaultTest", 
        "EmptyTest",
        "ExceptionHandling", 
        "GeneralFixture",
        "MysteryGuest", 
        "RedundantPrint",
        "RedundantAssertion", 
        "SensitiveEquality", 
        "SleepyTest", 
        "EagerTest", 
        "LazyTest", 
        "DuplicateAssertion",
        "UnknownTest",
        "IgnoredTest", 
        "ResourceOptimism", 
        "MagicNumberTest",
        "LackCohesion", #Python exclusive
        "ObscureInLineSetup",#Python exclusive
        "SuboptimalAssert",#Python exclusive
        "TestMaverick",#Python exclusive
    ]
    ids = ["00.", "01."]
    
    for id in ids:
        for smell in smells:
            entered = False
            to_save = ""
            to_save_s = None
            final_df = pd.DataFrame()
            for string_to_search in strings_to_search:
                condition = (
                            (MODEL_TO_USE_NAME in string_to_search) & 
                            (LANGUAGE_TO_USE in string_to_search) &
                            (id in string_to_search) &
                            ("~$" not in string_to_search) &
                            (smell in string_to_search) 
                            )
                
                if condition:
                    entered = condition
                    print(f"Beginning process for {string_to_search}, phase: {key_string}")
                    final_df = DataframeManager.load(path_to_prompt, string_to_search)
                    # final_df = create_to_prompt_prompt_id_optimization_df(string_to_search, key_string, path_to_prompt, final_df)   
                    
                    true_count = final_df["Is_smell_present_original"].sum()
                    false_count = (final_df["Is_smell_present_original"] == False).sum()
                        
                    to_save = string_to_search
                    to_save_s = string_to_search
                    
                    if "optimization" in string_to_search and "01." not in string_to_search:
                        to_save_s = string_to_search
                        break
                    
                    elif "optimization" in string_to_search and "01." in string_to_search and true_count == 10 and false_count == 10:
                        to_save_s = string_to_search
                        break
                    
                    elif "optimization" in string_to_search:
                        to_save_s = string_to_search
                    
            if entered:
                # Count and print the number of True and False values for Is_smell_present_original if they are not both equal to 10
                true_count = final_df["Is_smell_present_original"].sum()
                false_count = (final_df["Is_smell_present_original"] == False).sum()
                if (true_count != 10 or false_count != 10) and "01." in string_to_search:
                    print(f"True count: {true_count}, False count: {false_count}")
                    
                final_df.drop(columns=['randomize'], inplace=True)
                # print(to_save)
                # process_prompts() ############### Adaptar process prompts to work here
                to_save_s = to_save_s.replace("_optimization", "").replace("_to_prompt", "_to_prompt_optimization")
                to_save = to_save.replace("_optimization", "").replace("_to_prompt", "_to_prompt_optimization")
                # print(to_save)
                # final_df_to_save = duplicate_and_modify(final_df, path_to_prompt, "gemma:7b", prompts_gemma)
                # final_d_ll = duplicate_and_modify(final_df, path_to_prompt, "llama3:70b", prompts_llama)
                # final_df_gpt = duplicate_and_modify(final_df, path_to_prompt, "gpt-4-turbo", prompts_gpt)
                # if to_save_s == None:
                #     DataframeManager.save(final_df_to_save, path_to_prompt, f"optimization_{smell}_gemma7b")     
                #     DataframeManager.save(final_d_ll, path_to_prompt, f"optimization_{smell}_llama370b") 
                #     DataframeManager.save(final_df_gpt, path_to_prompt, f"optimization_{smell}_gpt-4-turbo") 
                # else:
                #     DataframeManager.save(final_df_to_save, path_to_prompt, to_save_s) 
                #     DataframeManager.save(final_d_ll, path_to_prompt, to_save_s.replace("gemma7b","llama3:70b").replace(":","")) 
                #     DataframeManager.save(final_df_gpt, path_to_prompt, to_save_s.replace("gemma7b","gpt-4-turbo").replace(":",""))  
      
def create_to_prompt_prompt_id_optimization_df(string_to_search, key_string, path_to_prompt, final_df):
    df = DataframeManager.load(path_to_prompt, string_to_search)
    
    # Convert the 'Prompt_ID' column to string
    df['Prompt_ID'] = df['Prompt_ID'].astype(str)

    # Drop rows where the content of the column "Prompt_ID" is not a substring of string_to_search
    df = df[df['Prompt_ID'].apply(lambda x: x in string_to_search)]
    df = df[df['Tokens_to_send'] <= 2000]
    
    # Check if the "randomize" column exists, if not create it
    if 'randomize' not in df.columns:
        df['randomize'] = 0
    
    # Assign random values to "randomize" column based on the conditions
    if "_optimization_" in string_to_search:
        df['randomize'] = 999999999
    else:
        df.loc[df['Prompting_status'] == 'ok', 'randomize'] = df.loc[df['Prompting_status'] == 'ok', 'randomize'].apply(lambda x: random.randint(99999, 99999999))
        df.loc[df['Prompting_status'] != 'ok', 'randomize'] = df.loc[df['Prompting_status'] != 'ok', 'randomize'].apply(lambda x: random.randint(99999, 99999999))
        # Append the resulting dataframe to the final dataframe
    
    final_df = pd.concat([final_df, df])
    
    # Count non-null values in each row and add as a new column
    final_df['non_null_count'] = final_df.notnull().sum(axis=1)
    
    # Sort by the count of non-null values in descending order
    final_df = final_df.sort_values(by='non_null_count', ascending=False)
    
    # Drop duplicates based on 'ID', 'Prompt_ID', and 'LLM_used'
    final_df = final_df.drop_duplicates(subset=['ID', 'Prompt_ID', 'LLM_used'])
    
    # Drop the helper column 'non_null_count'
    final_df = final_df.drop(columns=['non_null_count'])
    
    final_df = final_df[final_df["Prompting_status"] != "Initially good, but not now"]
    final_df = final_df[final_df["Tokens_sent"].astype(str) != "1"]
    final_df = final_df[final_df["Tokens_sent"].astype(str) != "2"]
    
    if "01." in string_to_search:
        true_smell_df = final_df[final_df["Is_smell_present_original"] == True].nlargest(10, 'randomize')
        false_smell_df = final_df[final_df["Is_smell_present_original"] == False].nlargest(10, 'randomize')
        combined_df = pd.concat([true_smell_df, false_smell_df])
        
        if len(combined_df) < 20:
            remaining_df = final_df[~final_df.index.isin(combined_df.index)].nlargest(20 - len(combined_df), 'randomize')
            combined_df = pd.concat([combined_df, remaining_df])
        
        final_df = combined_df
    
    else:
        # Keep only the top 20 rows with the highest "randomize" values
        final_df = final_df.nlargest(20, 'randomize')
    
    return final_df
        
def start_prompting_all(path_to_files_to_prompt, n, n_max_uniform, mode = "optimization"):
    if mode == "optimization":
        key_string = "to_prompt_optimization_selected"   
    else:
        key_string = "to_prompt_final" 
        
    strings_to_search = list_unique_xlsx_file_paths(path_to_files_to_prompt, key_string)
    strings_to_search = sorted(strings_to_search)
    strings_to_search.reverse()
    print(f"these are the strings to search: {strings_to_search}")
    # ssh_connection = SSHConnection()
    ssh_connection = ""
    # for string_to_search in strings_to_search:
    #     print(f"Beginning process for {string_to_search}, phase: {key_string}-prompting")
    #     send_prompts_from_file(n, n_max_uniform, string_to_search, mode="prompt_smells_uniform_dataset")
        
    # with multiprocessing.Pool(processes=NUM_WORKERS-DISCOUNT_PROMPTING) as pool: 
    for string_to_search in strings_to_search:
        print(f"Beginning process for {string_to_search}, phase: {key_string}-prompting")
        send_prompts_from_file(ssh_connection, n, n_max_uniform, string_to_search, mode="prompt_smells_uniform_dataset_optimization")
        #         
            
            # condition = (
            #             (MODEL_TO_USE_NAME in string_to_search) & 
            #             (LANGUAGE_TO_USE in string_to_search)
            #             )
            
            # if condition:
                # print(f"Beginning process for {string_to_search}, phase: {key_string}-prompting")
                # send_prompts_from_file(ssh_connection, n, n_max_uniform, string_to_search, mode="prompt_smells_uniform_dataset")
        #         pool.apply_async(send_prompts_from_file, args=(ssh_connection, n, n_max_uniform, string_to_search), kwds={'mode': "prompt_smells_uniform_dataset"})            
        # pool.close()
        # pool.join()
        
def get_prompt_numbers(df):
    prompt_info = {}
    for index, row in df.iterrows():
        Prompt_ID = row["Prompt_ID"]
        Prompt_status = row["Prompting_status"]
        if Prompt_ID in prompt_info and Prompt_status == "ok":
            prompt_info[Prompt_ID] += 1
        elif Prompt_status == "ok":
            prompt_info[Prompt_ID] = 1
            
    prompt_info["Minimum to consider evaluated"] = MIN_N_TO_CONSIDER_EVALUATED
    
    for key, value in prompt_info.items():
        print(f"{key}: {value}")
        
def to_boolean(value):
    if pd.isna(value):
        return None
    
    if isinstance(value, str):
        if value.lower() in ('true', '1', 't', 'y', 'yes'):
            return True
        elif value.lower() in ('false', '0', 'f', 'n', 'no'):
            return False
        elif value.lower() in (''):
            return ""
    return bool(value)   

def send_the_scheduled_prompts(ssh_connection, df_code_to_prompt, prompt_list_to_send):
    # time_token = get_token_time(prompt_list_to_send)
    start_time = time.time()
    prompt_list_to_send_ollama = {}
    prompt_list_to_send_gpt = {}
    prompt_list_to_send_gemini = {}
    ollama = False
    gpt = False
    gemini = False
    print(f"- this is the len {len(prompt_list_to_send)}  and the type {type(prompt_list_to_send)}")  
    # for key, item in prompt_list_to_send.items():
    #     print(f"This is one of the items in prompt_list_to_send: key: {key}, item: {item}\n\n")
    # print(f"### Prompting ###")
    # print(f"# Prompt instances sent: {len(prompt_list_to_send)}")
    # print(f"# Prompt tokens: {total_tokens_prompts}")
    # print(f"# Prompt cost: $ {total_cost_prompts}")
    
    # print("##### LIST OF CURRENTLY SCHEDULED PROMPTS #####")
    # for prompt_scheduled in sorted(prompts_scheduled.keys()):
        # print(f"Scheduled: {prompt_scheduled}, number of prompts: {prompts_scheduled[prompt_scheduled]}")
    for key, prompt in prompt_list_to_send.items():
        prompt_info, model = prompt
        if "gpt" in model:
            prompt_list_to_send_gpt[key] = [prompt_info, model]
            # print(f"this is the gpt list: {prompt_list_to_send_gpt}")
            gpt = True
        elif "gemma" in model or "llama" in model:
            prompt_list_to_send_ollama[key] = [prompt_info, model]
            # print(f"this is the list: {prompt_list_to_send_ollama}")
            ollama = True     
        elif "gemini" in model:
            prompt_list_to_send_gemini[key] = [prompt_info, model]
            # print(f"this is the list gemini: {prompt_list_to_send_gemini}")
            gemini = True 
              
    if gpt:
        print("gpt, in")
        prompt_list_received_gpt = send_all_prompts(prompt_list_to_send_gpt, mode="openai")
        print("gpt, ok")
    else:
        prompt_list_received_gpt = []
        
    if ollama:
        print("ollama, in")
        ssh_connection = SSHConnection()
        prompt_list_received_ollama = send_all_prompts(ssh_connection, prompt_list_to_send_ollama, mode="ollama")
        print("ollama, ok")
    else:
        prompt_list_received_ollama = [] 
        
    if gemini:
        print("gemini, in")
        # prompt_list_received_gemini = send_all_prompts(prompt_list_to_send_gemini, mode="gemini")
        prompt_list_received_gemini = send_all_prompts(prompt_list_to_send_gemini, mode="openai")
        print("gemini, ok --------------------------------------------------")
        # print("gemini, ok")
        # print(f"gemini, DEACTIVATED, not promting number: {len(prompt_list_to_send_gemini)} -------- ")
        # prompt_list_received_gemini = []
    else:
        prompt_list_received_gemini = [] 
        
    prompt_list_received = prompt_list_received_gpt + prompt_list_received_gemini + prompt_list_received_ollama
    end_time = time.time()    
    time_to_sleep = end_time - start_time   
    response_time = time_to_sleep
       
    for prompt_info_received, index_df in prompt_list_received:
        prompt_received = prompt_info_received[0]
        tokens_received = prompt_info_received[1]
        tokens_sent = prompt_info_received[2]
        prompt_status = prompt_info_received[3]
        
        if prompt_status != "ok" or ((str(tokens_sent) == '1' or str(tokens_sent) == '2') and not ("yes" in prompt_received.lower() or "no" in prompt_received.lower())):
            print(f"Prompt status returned: {prompt_status}")
            prompt_status = f"Initially good, but not now, {prompt_status}"
            # df_code_to_prompt.at[index_df, 'Prompting_status'] = prompt_status
            # continue
        
        current_Prompt_ID = df_code_to_prompt.at[index_df, 'Prompt_ID']
        if current_Prompt_ID.split("_")[1].split(".")[0] != "01":
            print(f"current_prompt_ID: {current_Prompt_ID}, index_df: {index_df}")
            df_code_to_prompt.at[index_df, 'Prompt_received'] = prompt_received
            df_code_to_prompt.at[index_df, 'Tokens_received'] = tokens_received
            df_code_to_prompt.at[index_df, 'Response_time'] = response_time
            df_code_to_prompt.at[index_df, 'Tokens_sent'] = tokens_sent
            df_code_to_prompt.at[index_df, 'Prompting_status'] = prompt_status
            df_code_to_prompt.at[index_df, 'Response_timestamp'] = Timestamp.this()
            
        else:
            print(f"current_prompt_ID: {current_Prompt_ID}, index_df: {index_df}")
            df_code_to_prompt.at[index_df, 'Prompt_received'] = prompt_received
            df_code_to_prompt.at[index_df, 'Tokens_received'] = tokens_received
            df_code_to_prompt.at[index_df, 'Response_time'] = response_time
            df_code_to_prompt.at[index_df, 'Tokens_sent'] = tokens_sent
            df_code_to_prompt.at[index_df, 'Prompting_status'] = prompt_status
            df_code_to_prompt.at[index_df, 'Response_timestamp'] = Timestamp.this()
            
            if "true" in prompt_received.lower() or "YES" in prompt_received:
                df_code_to_prompt.at[index_df, 'GPT_detection_status'] = True
            elif "false" in prompt_received.lower() or "NO" in prompt_received:
                df_code_to_prompt.at[index_df, 'GPT_detection_status'] = False
            else:
                df_code_to_prompt.at[index_df, 'GPT_detection_status'] = "Not_detected"
                
            if df_code_to_prompt.at[index_df, 'GPT_detection_status'] != "" and df_code_to_prompt.at[index_df, 'GPT_detection_status'] != "Not_detected":
                df_code_to_prompt.at[index_df, 'GPT_detected_right'] = (df_code_to_prompt.at[index_df, 'GPT_detection_status'] == df_code_to_prompt.at[index_df, 'Is_smell_present_original'])
    print(f"Saving - {len(prompt_list_received)} - {len(prompt_list_to_send)}") 
            
    # end_time = time.time()
    # time_to_sleep = end_time - start_time     
    # # get_prompt_numbers(df_code_to_prompt)    
    # # print(f"Sleeping: {max(time_token - time_to_sleep, 1)}")   
    # time.sleep(max(time_token - time_to_sleep, 1))
    modified = True
    if len(prompt_list_to_send) <= 0:
        modified = False
    return df_code_to_prompt, modified

def map_tokens():
    df = DataframeManager.load(DATASET_TO_PROMPT_PATH, name_to_find="to_prompt_optimization")
    df_copy = df
    # df["Class_tokens"] = 0
    for index, row in df.iterrows():
        if row["Finding_class_status"] == "ok" and row["Class_tokens"] == 0:
            
            df_copy.loc[index,"Class_tokens"] = num_tokens_from_messages(row["Original_code"])
            print(f"this is the current index {index} max:{len(df)}\n{row['Class_tokens']}")
        
    
    DataframeManager.save(df_copy, DATASET_TO_PROMPT_PATH, name_to_find="to_prompt_oprimization")

def assemble_prompts(prompt_list_to_send, df_prompt_matrix, errors):
    prompt_list_to_send_assembled = {}
    for key, prompt_info in prompt_list_to_send.items():
        try:
            original_code = prompt_info["Code"] 
            smell = prompt_info["Smell"]
            category = prompt_info["Category"]
            model = prompt_info["Model"]
            language = prompt_info["Dataset"]
            prompt_ID = prompt_info["Prompt_ID"]
            
            query_string = f"Smell == @smell and Category == @category and Model == @model and Dataset == @language and Prompt_ID == @prompt_ID"
            filtered_df = df_prompt_matrix.query(query_string)
            
            if len(filtered_df) == 1:
                row = filtered_df.iloc[0]
                prompt_string_to_split= row['String_to_split_prompt']
                prompt_string = row['Prompt_string']
                string_to_definition = row['String_to_definition']
                definition = row['Definition']
                string_to_code = row['String_to_code']
                code = original_code
                string_to_example = row['String_to_example']
                example = row['Example']
                string_to_COT = row['String_to_COT']
                COT = row['COT']
                
                # if model == "gemma:7b":
                if prompt_string_to_split != False and str(prompt_string_to_split) != "nan":
                    if string_to_definition:
                        prompt_string = prompt_string.replace(string_to_definition, definition)
                    if string_to_COT:
                        prompt_string = prompt_string.replace(string_to_COT, COT)
                    if string_to_example:
                        prompt_string = prompt_string.replace(string_to_example, example)
                    if string_to_code:
                        prompt_string = prompt_string.replace(string_to_code, code)
                    
                    systemcontent, usercontent = prompt_string.split(prompt_string_to_split)
                    prompt = [systemcontent, usercontent]
                    prompt_list_to_send_assembled[key] = [prompt, model]

                else:
                    errors[key] = [False, "False to use this prompt"]
            else:
                errors[key] = [False, "Search resulted in 0 or > 1 results"]
                    # else: 
                        # print(f"EXCEPTION: No unique row found for query: {query_string}_onelse")
                        # print(f"Values - Smell: {smell}, Category: {category}, Model: {model}, Dataset: {language}")
                        # print(f"EXCEPTION Non-string value found in one of the required fields for key: {key}---")
                        # print(f"Values - Smell: {smell}, Category: {category}, Model: {model}, Dataset: {language}")
                        # print(f"Debug Info for key {key}:")
                        # print(f"prompt_string: {prompt_string}")
                        # print(f"prompt_string_to_split: {prompt_string_to_split}")
                        # print(f"string_to_definition: {string_to_definition}")
                        # print(f"definition: {definition}")
                        # print(f"string_to_code: {string_to_code}")
                        # print(f"code: {code}")
                        # print(f"string_to_example: {string_to_example}")
                        # print(f"example: {example}")
                        # print(f"string_to_COT: {string_to_COT}")
                        # print(f"COT: {COT}")
        except KeyError as e:
            print(f"EXCEPTION {e} for key: {key}")
        except Exception as e:
            print(f"EXCEPTION {e} for key: {key}")
            print(f"EXCEPTION {e} for key: {key}")
            print(f"EXCEPTION Non-string value found in one of the required fields for key: {key}---")
            print(f"Values - Smell: {smell}, Category: {category}, Model: {model}, Dataset: {language}")
            print(f"Debug Info for key {key}:")
            # print(f"prompt_string: {prompt_string}")
            print(f"prompt_string_to_split: {prompt_string_to_split}")
            print(f"string_to_definition: {string_to_definition}")
            print(f"definition: {definition}")
            print(f"string_to_code: {string_to_code}")
            print(f"code: {code}")
            print(f"string_to_example: {string_to_example}")
            print(f"example: {example}")
            print(f"string_to_COT: {string_to_COT}")
            print(f"COT: {COT}")

        
    print(1) 
    return prompt_list_to_send_assembled, errors

def get_prompt_info_progress(df_to_prompt, path_to_save):
    
    df_to_prompt["Cathegory_ID"] = "None"
    df_to_prompt = df_to_prompt.sample(frac=1).reset_index(drop=True) ## Randomize the dataframe
    tokens = 0
    output_list = []
    # for index, row in tqdm(df_to_prompt.iterrows()):
    #     if str(row["Is_smell_present_original"]) == "1":
    #         df_to_prompt.loc[index, "Cathegory_ID"] = str(row["LLM_used"]) + str(row["Is_smell_present_original"]) + str(row["Smell"]) + str(row["Prompt_ID"]) + str(row["Det_or_ref"])
    #     elif str(row["Is_smell_present_original"]) == "0" and str(row["Det_or_ref"]) == "Detecção":
    #         df_to_prompt.loc[index, "Cathegory_ID"] = str(row["LLM_used"]) + str(row["Is_smell_present_original"]) + str(row["Prompt_ID"]) + str(row["Det_or_ref"])
    # df_to_prompt['Is_Selected'] = (df_to_prompt['Optimization'] == 'Selected').astype(int)
    
    condition_1 = df_to_prompt['Is_smell_present_original'].astype(str) == '1'
    condition_2 = (df_to_prompt['Is_smell_present_original'].astype(str) == '0') & (df_to_prompt['Det_or_ref'] == 'Detecção')

    # Use vectorized operations to assign 'Cathegory_ID'
    df_to_prompt.loc[condition_1, 'Cathegory_ID'] = (
        df_to_prompt['LLM_used'].astype(str) +
        df_to_prompt['Is_smell_present_original'].astype(str) +
        df_to_prompt['Smell'].astype(str) +
        df_to_prompt['Prompt_ID'].astype(str) +
        df_to_prompt['Det_or_ref'].astype(str)
    )

    df_to_prompt.loc[condition_2, 'Cathegory_ID'] = (
        df_to_prompt['LLM_used'].astype(str) +
        df_to_prompt['Is_smell_present_original'].astype(str) +
        df_to_prompt['Prompt_ID'].astype(str) +
        df_to_prompt['Det_or_ref'].astype(str)
    )

    # Assign 'Is_Selected' based on the 'Optimization' column
    df_to_prompt['Is_Selected'] = (df_to_prompt['Optimization'] == 'Selected').astype(int)
    
    cathegory_counts = df_to_prompt["Cathegory_ID"].value_counts()
    selected_counts = df_to_prompt.groupby('Cathegory_ID')['Is_Selected'].sum()
    
    output_messages = []  # Initialize an empty string to accumulate the message
    outputs = []  # Initialize an empty string to accumulate the message
    
    for prompt_cathegory, count in cathegory_counts.items():
        if prompt_cathegory != "None":
            selected_count = selected_counts.get(prompt_cathegory, 0)
            outputs.append(f'Category: ---{prompt_cathegory}|| - Total: ---{count}||, Selected: ---{selected_count}||\n')
            if count != selected_count:
                output_messages.append(f'Category: ---{prompt_cathegory}|| Total: ---{count}||, Selected: ---{selected_count}||\n')
    
    pattern = r'---(.*?)\|\|'

    for output_message in output_messages:
        matches = re.findall(pattern, output_message)
        if matches and len(matches) == 3:  # Ensure we have exactly three matches
            category, total, selected = matches
            output_list.append({"Category": category, "Total": int(total), "Selected": int(selected)})
        else:
            print(f"Unexpected format in message: {output_message}")
            
    for output in outputs:
        matches = re.findall(pattern, output)
        if matches and len(matches) == 3:  # Ensure we have exactly three matches
            category, total, selected = matches
            output_list.append({"Category": category, "Total": int(total), "Selected": int(selected)})
        else:
            print(f"Unexpected format in message: {output}")

    print(output_messages)
    print(outputs)
    # output_list.append({"Information": output_message} for output_message in output_messages)
    # output_list.append({"Information": output_message} for output_message in output_messages)

    # for prompt_cathegory, count in cathegory_counts.items():
    #     if prompt_cathegory != "None":
    #         print(f'Prompt category: {prompt_cathegory} - {count}')
            
    # # Display the counts for each category
    # for category, count in selected_counts.items():
    #     if category != "None":
    #         print(f'Category {category} has {count} selections.')
            
    # Add entries as 'Selected' if they do not meet the minimum required selections
    for category, count_selected in selected_counts.items():
        if "Detecção" in category:
            needed = MIN_N_TO_CONSIDER_EVALUATED_DET - count_selected
            if needed > 0:
                # Retrieve all rows in this category
                category_rows = df_to_prompt[df_to_prompt['Cathegory_ID'] == category]
                # Group by 'Smell' and count selected within each group
                smell_counts = category_rows.groupby('Smell')['Is_Selected'].sum()
                for smell, smell_count in smell_counts.items():
                    # Calculate needed per smell to balance the selection
                    smell_needed = (needed // len(smell_counts)) + (1 if needed % len(smell_counts) > 0 else 0)
                    mask = (category_rows['Smell'] == smell) & (df_to_prompt['Optimization'] != 'Selected')
                    indices_to_select = df_to_prompt[mask].index[:smell_needed]
                    df_to_prompt.loc[indices_to_select, 'Optimization'] = 'Selected'
                    needed -= smell_needed
                    if needed <= 0:
                        break
        # if "Detecção" in category and count_selected < MIN_N_TO_CONSIDER_EVALUATED_DET:
        #     # Select more entries to reach the minimum
        #     needed = MIN_N_TO_CONSIDER_EVALUATED_DET - count_selected
        #     mask = (df_to_prompt['Cathegory_ID'] == category) & (df_to_prompt['Optimization'] != 'Selected')
        #     indices_to_select = df_to_prompt[mask].index[:needed]
        #     df_to_prompt.loc[indices_to_select, 'Optimization'] = 'Selected'
            
        elif "Refatoração" in category and count_selected < MIN_N_TO_CONSIDER_EVALUATED_REF:
            needed = MIN_N_TO_CONSIDER_EVALUATED_REF - count_selected
            mask = (df_to_prompt['Cathegory_ID'] == category) & (df_to_prompt['Optimization'] != 'Selected')
            indices_to_select = df_to_prompt[mask].index[:needed]
            df_to_prompt.loc[indices_to_select, 'Optimization'] = 'Selected'

    # Re-calculate the selection counts after updates
    df_to_prompt['Is_Selected'] = (df_to_prompt['Optimization'] == 'Selected').astype(int)
    new_selected_counts = df_to_prompt.groupby('Cathegory_ID')['Is_Selected'].sum()
    
    for category, new_count in new_selected_counts.items():
        if category != "None":
            print(f'Updated category {category} now has {new_count} selections.')
            output_list.append({"Category": category, "Total": int(0), "Selected":  int(new_count)})  # Store the message in the list
    
            
    df_selected = df_to_prompt[df_to_prompt['Optimization'] == "Selected"]
    tokens = df_selected['Class_tokens'].sum()
    tokens_t = df_to_prompt['Class_tokens'].sum()
    print(f"These are the tokens of the selected {tokens}, using GPT pricing: {4 * tokens * 1.1 * 0.00001}")
    # output_list.append({"Information": f"These are the tokens of the selected {tokens}, using GPT pricing: {4 * tokens * 1.1 * 0.00001}"})  # Store the message in the list
    print(f"These are the tokens total {tokens_t}, using GPT pricing: {4 * tokens_t * 1.1 * 0.00001}")
    # output_list.append({"Information": f"These are the tokens total {tokens_t}, using GPT pricing: {4 * tokens_t * 1.1 * 0.00001}"})  # Store the message in the list
    output_df = pd.DataFrame(output_list)
    
    DataframeManager.save(output_df,  path_to_save, name_to_find="Relevant_info")
    return df_to_prompt
    ## ADD HERE LOGIC TO COUNT AND CALCULATE HOW MANY PROMPTS HAVE ALREADY BEEN DONE FOR THIS PROMPTID TEST SMELL, LANGUAGE, IS_PRESENT_ORIGINAL, MODEL, DETECTION/REFACTORING, TOKENCOUNT
    

def send_prompts_from_file(ssh_connection, n, n_max_uniform, key_string, mode="prompt_smells_uniform_dataset_optimization"):
    errors = {}
    just_checking = False
    # df_prompt_report = DataframeManager.load(POST_REF_DATA_PATH, name_to_find="FINAL_report")
    # df_prompt_report_det = DataframeManager.load(POST_REF_DATA_PATH, name_to_find="FINAL_detection_final")
    df_prompt_matrix = DataframeManager.load(PROMPTS_DIR_PATH, name_to_find="prompt_matrix")
    print(f"this is the new keystring: {key_string}")
    df_code_to_prompt = DataframeManager.load(DATASET_TO_PROMPT_PATH, name_to_find=key_string)
    
    if "selected_6" in key_string:
        
        replacements = {
            "Prompt_j_01.00": "Prompt_j_01.02",
            "Prompt_j_00.00": "Prompt_j_00.02",
            "Prompt_p_01.00": "Prompt_p_01.02",
            "Prompt_p_00.00": "Prompt_p_00.01"
        }
        
        df_code_to_prompt_c = df_code_to_prompt.copy()
        # Replace the values in the 'Prompt_ID' column
        df_code_to_prompt_c['Prompt_ID'] = df_code_to_prompt_c['Prompt_ID'].replace(replacements)
        
        df_code_to_prompt_f = df_code_to_prompt_c.copy()
        if not any(df_code_to_prompt["LLM_used"] == "gemini-1.5-pro"):
            df_code_to_prompt_c_2 = df_code_to_prompt_c.copy()
            df_code_to_prompt_c_2["LLM_used"] = df_code_to_prompt_c_2["LLM_used"].replace("llama3:70b", "gemini-1.5-pro")
            df_code_to_prompt_f = pd.concat([df_code_to_prompt_f, df_code_to_prompt_c_2])
            
        if not any(df_code_to_prompt["LLM_used"] == "gpt-4-turbo"):
            df_code_to_prompt_c_3 = df_code_to_prompt_c.copy()
            df_code_to_prompt_c_3["LLM_used"] = df_code_to_prompt_c_3["LLM_used"].replace("llama3:70b", "gpt-4-turbo")
            df_code_to_prompt_f = pd.concat([df_code_to_prompt_f, df_code_to_prompt_c_3])
        
        df_code_to_prompt = df_code_to_prompt_f[~(
            ((df_code_to_prompt_f['Prompt_ID'] == 'Prompt_j_00.02') | (df_code_to_prompt_f['Prompt_ID'] == 'Prompt_p_00.01')) & 
            (df_code_to_prompt_f['Is_smell_present_original'] == False)
        )]
        # df_code_to_prompt["Prompting_status"] = ""
        
    df_code_to_prompt_c = df_code_to_prompt.copy()
    # df_code_to_prompt = df_code_to_prompt.sample(frac=1).reset_index(drop=True)
    # df_code_to_prompt_filtered = df_code_to_prompt[df_code_to_prompt["Prompting_status"] == "ok"].copy()
    if mode == "prompt_smells_uniform_dataset_optimization":
        number_of_selected = len(df_code_to_prompt[df_code_to_prompt["Optimization"] == "Selected"].copy())
        print(f"This is the number of selected: {number_of_selected}")
        # df_code_to_prompt = df_code_to_prompt[df_code_to_prompt["Optimization"] == "Selected"].copy()
    else:
        number_of_selected = len(df_code_to_prompt[df_code_to_prompt["Optimization"] == "Selected"].copy())
        print(f"This is the number of selected: {number_of_selected}_2")
        
    # df_code_to_prompt = df_code_to_prompt.sort_values(
    #     by="Optimization",
    #     key=lambda col: col != "selected"
    # )
    # Iterate over the rows in the Code to prompt dataframe and generate prompts
    prompts_done = (df_code_to_prompt["Prompting_status"] == "ok").sum()
    total_prompts = (df_code_to_prompt["Finding_class_status"] == "ok").sum()
    n_ = 0
    
    if prompts_done == total_prompts:
        print(
            f"#### OVERVIEW: {key_string}",
            f"Total prompts: {total_prompts}"
            f"Done: {prompts_done}"
        )
        return
    
    prompts_scheduled = {}
    get_prompt_numbers(df_code_to_prompt)
    prompt_list_to_send = {}
    last_index = df_code_to_prompt.index[-1]
    
    df_code_to_prompt = df_code_to_prompt[df_code_to_prompt["Finding_class_status"] == "ok"]
    # df_code_to_prompt = df_code_to_prompt[df_code_to_prompt["Optimization"] == "Selected"]
    
    if just_checking:
        df_code_to_prompt = get_prompt_info_progress(df_code_to_prompt, DATASET_TO_PROMPT_PATH)
        print(f"Before saving, here is the len of the df {len(df_code_to_prompt)}")
        DataframeManager.save(df_code_to_prompt, DATASET_TO_PROMPT_PATH, name_to_find=key_string)
        ## ADD HERE LOGIC TO COUNT AND CALCULATE HOW MANY PROMPTS HAVE ALREADY BEEN DONE FOR THIS PROMPTID TEST SMELL, LANGUAGE, IS_PRESENT_ORIGINAL, MODEL, DETECTION/REFACTORING, TOKENCOUNT
        return
    else:
        df_code_to_prompt = get_prompt_info_progress(df_code_to_prompt, DATASET_TO_PROMPT_PATH)
        # df_code_to_prompt = get_prompt_info_progress(df_code_to_prompt)  
        df_code_to_prompt = df_code_to_prompt.sort_values(by='LLM_used')
        df_code_to_prompt_c = df_code_to_prompt.copy()
        df_code_to_prompt = df_code_to_prompt[df_code_to_prompt['Optimization'] == "Selected"]
        df_code_to_prompt = df_code_to_prompt[df_code_to_prompt["Prompting_status"] != "ok"]
        df_done = df_code_to_prompt[df_code_to_prompt["Prompting_status"] == "ok"]
        print(f"This is the number of selected and not ok = {len(df_code_to_prompt)}\nand this is the number of done: {len(df_done)}")
          
    for index, row in  tqdm(df_code_to_prompt.iterrows(), total=df_code_to_prompt.shape[0], file=sys.stdout, desc=f"Sending {key_string}", leave=True):
        n_ += 1
        is_selected = row["Optimization"] == "Selected"
        if not is_selected:
            continue
        
        current_Prompt_ID = row["Prompt_ID"]
        current_LLM_used = row["LLM_used"]
        tokens_to_send = row["Class_tokens"]
        smell_presence_status = row["Is_smell_present_original"]
        
        if isinstance(current_Prompt_ID, str):
            if current_Prompt_ID not in prompts_scheduled:
                prompts_scheduled[current_Prompt_ID] = 0   
        
        if isinstance(current_Prompt_ID, float) or current_Prompt_ID.split("_")[1].split(".")[0] == "00" and not smell_presence_status:
            if n_ >= n or n_ >= number_of_selected:
                errors[index] = [False, "False to use this prompt n_ >= number_of_selected or n_ >= n"]
                break
            continue
            
        if pd.isna(tokens_to_send):
            if n_ >= n or n_ >= number_of_selected:
                errors[index] = [False, "False to use this prompt n_ >= number_of_selected or n_ >= n"]
                break
            continue
        
        if mode == "prompt_smells_uniform_dataset" or mode == "prompt_smells_uniform_dataset_optimization": 
            # Counting occurrences
            condition = (
                (df_code_to_prompt["Prompt_ID"] == current_Prompt_ID) & 
                (df_code_to_prompt["Prompting_status"] == "ok") &
                (df_code_to_prompt["LLM_used"] == current_LLM_used) &
                (df_code_to_prompt["Is_smell_present_original"] == smell_presence_status)
                )
            
            prompts_scheduled_count = prompts_scheduled[current_Prompt_ID]
            
            count = df_code_to_prompt[condition].shape[0] + prompts_scheduled_count

        if count > n_max_uniform:
            if n_ >= n or n_ >= number_of_selected:
                # errors[index] = [False, "False to use this prompt n_ >= number_of_selected or n_ >= n"]
                break
            # errors[index] = [False, "Continued, n_max_uniform"]
            continue
        
        if mode == "prompt_smells_uniform_dataset_optimization" and row['Optimization'] != "Selected":
            if n_ >= n or n_ >= number_of_selected:
                # errors[index] = [False, "False to use this prompt n_ >= number_of_selected or n_ >= n"]
                break
            # errors[index] = [False, "Continued, not selected"]
            continue

        # elif count > MIN_N_TO_CONSIDER_EVALUATED_REF and current_Prompt_ID.split("_")[1].split(".")[0] == "00" and smell_presence_status == True:
            
            # df_prompt_report_filtered = df_prompt_report[df_prompt_report["Prompt_ID"].str.contains(current_Prompt_ID[:-6], na=False)].copy()
            
            # selected_status = (df_prompt_report_filtered["Selected"] == "Selected").any()
            # if not selected_status:
            #     continue
            
            # if "Selected" not in df_prompt_report:
            #     continue

            # index_current_Prompt_ID = df_prompt_report_filtered[df_prompt_report_filtered["Prompt_ID"].str.contains(current_Prompt_ID, na=False)].index
            
            # if len(index_current_Prompt_ID) > 0:
            
            #     if df_prompt_report_filtered.at[index_current_Prompt_ID[0], "Selected"] != "Selected":
            #         continue
            #     else:
            #         print(f'################# This is the selected smell: {current_Prompt_ID} ####################')
            # else:    
            #     continue
        
            
            
                
        # elif count > MIN_N_TO_CONSIDER_EVALUATED_DET and current_Prompt_ID.split("_")[1].split(".")[0] == "01":    
        # elif current_Prompt_ID.split("_")[1].split(".")[0] == "01":
        #     continue
            
            # df_prompt_report_det_filtered = df_prompt_report_det[df_prompt_report_det["Prompt_ID"].str.contains(current_Prompt_ID[:-6], na=False)].copy()
            
            # selected_status = (df_prompt_report_det_filtered["Selected"] == "Selected").any()
            # if not selected_status:
            #     continue
            
            # if "Selected" not in df_prompt_report_det:
            #     continue

            # index_current_Prompt_ID = df_prompt_report_det_filtered[df_prompt_report_det_filtered["Prompt_ID"].str.contains(current_Prompt_ID, na=False)].index
            
            # if len(index_current_Prompt_ID) > 0:
            
            #     if df_prompt_report_det_filtered.at[index_current_Prompt_ID[0], "Selected"] != "Selected":
            #         continue
                
            #     else:
            #         print(f'################# This is the selected smell: {current_Prompt_ID} ####################')
            # else:    
            #     continue
            # pass
        
            
        if tokens_to_send <= MAX_TOKENS_TO_PROMPT and row["Finding_class_status"] == "ok" and (row["Prompting_status"] != "ok" or str(row["Tokens_sent"]) == '1'):
            prompt = row['Original_code']
            tokens_to_send = int(tokens_to_send)
            
            prompts_remaining = total_prompts - prompts_done
            # print(f"NUMBER OF PROMPTS ALREADY DONE: {prompts_done}")
            # print(f"PROMPTS DONE THIS RUN: {n_} STOPPING WHEN == {n}")
            # print(f"TOTAL PROMPTS REMAINING: {prompts_remaining}")
            # print(f"PROMPT ID NOW: {current_Prompt_ID} tokens: {tokens_to_send}")
            
            # if current_Prompt_ID.split("_")[1].split(".")[0] == "01":
            #     sum_of_the_n_of_is_present_status = (
            #             (df_code_to_prompt["Prompt_ID"] == current_Prompt_ID) &
            #             (df_code_to_prompt["Is_smell_present_original"] == True) &
            #             (df_code_to_prompt["Prompting_status"] == "ok")
            #                 ).sum()
                
            #     sum_of_the_n_of_is_not_present_status = (
            #             (df_code_to_prompt["Prompt_ID"] == current_Prompt_ID) &
            #             (df_code_to_prompt["Is_smell_present_original"] == False) &
            #             (df_code_to_prompt["Prompting_status"] == "ok")
            #                 ).sum()
                
            #     current_status = to_boolean(row["Is_smell_present_original"])
                
            #     total = sum_of_the_n_of_is_present_status - sum_of_the_n_of_is_not_present_status 
                # if total > 0 and current_status:
                #     continue
                # if total < 0 and not current_status:
                #     continue    
                  
                
            # # Capture the start time
            
            # # Execute the function
            # prompt_received, tokens_received, tokens_sent, prompt_status = generate_prompt(prompt, mode="default")
            # # Capture the end time
            # 
            
            prompt_list_to_send[index] = {"Code": prompt, "Smell": row['Smell'], "Category": row["Det_or_ref"], "Model": row["LLM_used"], "Dataset": row["Dataset"], "Prompt_ID": row["Prompt_ID"]}

            prompts_scheduled[current_Prompt_ID] += 1
            if prompts_scheduled[current_Prompt_ID] % MIN_N_TO_CONSIDER_EVALUATED_REF == 0:
                total_scheduled = 0
                # print("### LIST OF CURRENTLY SCHEDULED PROMPTS:")
                for prompt_scheduled in sorted(prompts_scheduled.keys()):
                    # print(f"Scheduled: {prompt_scheduled}, number of prompts: {prompts_scheduled[prompt_scheduled]}")
                    total_scheduled += prompts_scheduled[prompt_scheduled]
                # print(f"Scheduled total: {total_scheduled}")

        total_tokens_prompts = num_tokens_from_messages(prompt_list_to_send, model=MODEL_TO_USE)
        total_cost_prompts =  (1/1000) * ((total_tokens_prompts/2) * (1/100) + (total_tokens_prompts/2) * (3/100))
        if len(prompt_list_to_send) >= MAX_PROMPT_PER_SHOT or total_tokens_prompts >= MAX_TOKENS_PER_SHOT or index == last_index or (row["LLM_used"] == "llama3:70b" and len(prompt_list_to_send) >= MAX_PROMPT_PER_SHOT/10):
            try:
                print(f"HERE JUST TO CHECK_before: {len(prompt_list_to_send)}")
                prompt_list_to_send, errors = assemble_prompts(prompt_list_to_send, df_prompt_matrix, errors)
                print(f"HERE JUST TO CHECK: {len(prompt_list_to_send)}")
                df_code_to_prompt_c, modified = send_the_scheduled_prompts(ssh_connection, df_code_to_prompt_c, prompt_list_to_send)
                
                for index_df, error_status in errors.items():
                    df_code_to_prompt_c.at[index_df, 'Prompting_status'] = error_status[1]
                    
                # if modified:
                DataframeManager.save(df_code_to_prompt_c, DATASET_TO_PROMPT_PATH, name_to_find=key_string)
                errors = {}
            except Exception as e:
                print(f"EXCEPTION {e}")
            prompt_list_to_send = {}
            prompts_scheduled = {}
        if n_ >= n or n_ >= number_of_selected:
            break
        
    prompt_list_to_send, errors = assemble_prompts(prompt_list_to_send, df_prompt_matrix, errors)
    df_code_to_prompt_c, modified  = send_the_scheduled_prompts(ssh_connection, df_code_to_prompt_c, prompt_list_to_send)
    for index_df, error_status in errors.items():
        df_code_to_prompt_c.at[index_df, 'Prompting_status'] = error_status[1]
    
    DataframeManager.save(df_code_to_prompt_c, DATASET_TO_PROMPT_PATH, name_to_find=key_string)
##################   

class JavaClassLocator:
    
    def __init__(self, target_name, mode="default"):
        self.target_name = target_name
        self.mode = mode
        self.found_class = None
        self.found_node = None

    def visit(self, node):
        print(f"Visiting node: {type(node).__name__}")  # Debug: Print the type of each node visited
        
        # Handle node types
        if isinstance(node, javalang.tree.ClassDeclaration):
            print(f"Found Class: {node.name}")  # Debug: Print name of each class found
            if node.name == self.target_name:
                self.found_class = node
        elif isinstance(node, javalang.tree.MethodDeclaration):
            print(f"Found Method: {node.name}")  # Debug: Print name of each method found
            if node.name == self.target_name:
                self.found_node = node
                if not self.found_class:
                    self.found_class = self.get_parent_class(node)

        # Recursively visit children of the current node
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, javalang.tree.Node):
                    self.visit(item)
                else:
                    print(f"Non-node item in list: {item}")  # Debug: Identify non-node items

        elif hasattr(node, 'children'):
            for child in node.children:
                self.visit(child)

    def get_parent_class(self, node):
        # Navigate up in the tree to find the parent class of a method
        parent = node
        while not isinstance(parent, javalang.tree.ClassDeclaration):
            parent = parent._position.node
            if parent is None:
                return None
        return parent

    def parse_java(self, java_code):
        tree = javalang.parse.parse(java_code)
        self.visit(tree)

def is_valid_java(code: str) -> bool:
    try:
        # Use javalang to parse the Java code
        javalang.parse.parse(code)
        return True
    except javalang.parser.JavaSyntaxError as e:
        print("_________________________________________ Syntax Error:")
        traceback.print_exc()  # This will print the traceback of the exception
        return False
    except Exception as e:
        print("_________________________________________ Exception occurred:")
        traceback.print_exc()  # This will print the traceback of the exception
        print(f"THIS IS THE CURRENT CODE WITH PROBLEMS:\n\n{code}\n\n")
        return False
    
def get_end_lineno_java(node):
    """Recursively get the last line number of a node, accounting for potential absence of 'position'."""
    lineno = 0  # Default line number if none found

    # Check if the node has a 'position' attribute and if it's not None
    if hasattr(node, 'position') and node.position:
        # Update lineno if a valid line number is available
        lineno = node.position.line if node.position.line else lineno

    # Recursively explore children nodes to find the maximum line number
    # Ensure the node has 'children' and that it's not None
    if hasattr(node, 'children') and node.children:
        for child in node.children:
            if isinstance(child, list):  # Children could be a list of nodes
                for subchild in child:
                    if isinstance(subchild, javalang.tree.Node):
                        lineno = max(lineno, get_end_lineno_java(subchild))
            elif isinstance(child, javalang.tree.Node):  # Direct node children
                lineno = max(lineno, get_end_lineno_java(child))

    return lineno

def get_end_lineno_python(node):
    """Recursively get the last line number of a node."""
    if hasattr(node, "end_lineno"):
        return node.end_lineno
    elif hasattr(node, "lineno"):
        return node.lineno
    else:
        max_lineno = 0
        for child_node in ast.iter_child_nodes(node):
            max_lineno = max(max_lineno, get_end_lineno_python(child_node))
        return max_lineno

def replace_class_content(source_code, class_name, new_class_content, language):
    print(f"This is the classname: {class_name} and the language is {language}")
    if language == "java":
        msg, status = replace_class_content_java(source_code, class_name, new_class_content)
    elif language == "python":
        msg, status = replace_class_content_python(source_code, class_name, new_class_content)
    else:
        raise Exception(f"Wrong language sent. Language: {language}")
    
    return msg, status 

def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def replace_class_content_python(source_code, class_name, new_class_content):    
    locator = PyClassLocator(class_name)
    locator.visit(ast.parse(source_code))
    
    if locator.found_class:
        source_lines = source_code.splitlines()
        start_pos = locator.found_class.lineno - 1
        end_pos = get_end_lineno_python(locator.found_class)
        before_class = "\n".join(source_lines[:start_pos])
        after_class = "\n".join(source_lines[end_pos:])
        new_code = f"{before_class}\n{new_class_content}\n{after_class}"
        
        if not is_valid_python(new_code):
            msg = f"Error: The resulting code after replacing class {class_name} is invalid. The code:\n\n {new_code}"
            return msg, False

        return new_code, True
    else:
        msg = f"Class {class_name} not found in the source code!"
        return msg, False    

def replace_class_content_java(source_code, class_name, new_class_content):
    # print("-")
    locator = JavaClassLocator(class_name)
    # print("--")
    locator.visit(javalang.parse.parse(source_code))
    # print("---")
    if locator.found_class:
        # print("----")
        source_lines = source_code.splitlines()
        # print("-----")
        start_pos = locator.found_class.position.line - 1  # Correctly access start line
        end_pos = get_end_lineno_java(locator.found_class)
        before_class = "\n".join(source_lines[:start_pos])
        after_class = "\n".join(source_lines[end_pos:])
        # new_code = f"{before_class}\n{new_class_content}\n{after_class}"
        new_code = f"{new_class_content}"
        if not is_valid_java(new_code):
            msg = f"Error: The resulting code after replacing class {class_name} is invalid. The code:\n\n {new_code}"
            return msg, False

        return new_code, True
    else:
        msg = f"Class {class_name} not found in the source code!"
        return msg, False

def get_current_language(prompt_ID):
    if "Prompt_p" in prompt_ID:
        return "python"
    elif "Prompt_j" in prompt_ID:
        return "java"
    else:
        raise Exception(f"Wrong Prompt_ID format sent. Prompt_ID: {prompt_ID}")
    
def safe_split_and_process(prompt_received, language):
    if "java" in language:
        prompt_parts_return = []
        # Define a dictionary of delimiters and their safe replacements
        # delimiter_subs = {
        #     '```java': '|||JAVA_DELIM|||',
        #     '```': '|||JAVA_DELIM|||',
        #     '####': '|||JAVA_DELIM|||'
        # }
        
        # # Substitute delimiters with unique markers
        # safe_prompt = prompt_received
        # for delim, marker in delimiter_subs.items():
        #     safe_prompt = prompt_received
        #     safe_prompt = safe_prompt.replace(delim, marker)
        
        #     # Now split using the unique markers
        #     prompt_parts = re.split(r'\|\|\|JAVA_DELIM\|\|\||\|\|\|JAVA_DELIM\|\|\||\|\|\|JAVA_DELIM\|\|\|', safe_prompt)

            
        #     refactored_code = ""
        #     not_valid_code = []
            
        #     for prompt_part in prompt_parts:
        #         if len(prompt_part) > len(refactored_code) and is_valid_java(prompt_part):
        #             refactored_code = prompt_part

        refactored_code = ""
        split_pattern = r"```java|```"
        prompt_parts = re.split(split_pattern, prompt_received)
        
        for prompt_part in prompt_parts:
            if len(prompt_part) > len(refactored_code) and is_valid_java(prompt_part.replace("`","'")):
                refactored_code = prompt_part

        prompt_parts_return = prompt_parts + prompt_parts_return 

        split_pattern = r"####"
        prompt_parts = re.split(split_pattern, prompt_received)
        
        for prompt_part in prompt_parts:
            if len(prompt_part) > len(refactored_code) and is_valid_java(prompt_part.replace("`","'")):
                refactored_code = prompt_part

        prompt_parts_return = prompt_parts + prompt_parts_return        

        # for part in prompt_parts:
        #     if is_valid_java(part) and len(part) > len(refactored_code):
        #         refactored_code.append(part)
        #         not_valid_code.append(part)
        # for 
        # if len(refactored_code) == 0:
        #     if len(prompt_parts) == 1:
        #         refactored_code.append(prompt_parts[0])
        #     else:
        #         raise Exception(f"Valid java code not found:\n\n{not_valid_code}\n\n{len(prompt_parts)} \n\n{prompt_parts} ")
            
        # for prompt_part in prompt_parts:
        #     if len(prompt_part) > len(refactored_code):
        #         refactored_code = prompt_part
        #     else:
                # print("no code found")
                
        # if not is_valid_java(refactored_code):
        #     raise Exception(f"Not replaced: Valid {language} code not found")  

        if len(refactored_code) < 20:
            raise Exception(f"refactored code too short: {refactored_code}, these are the parts:{prompt_parts_return}") 
    
    elif "python" in language:
        refactored_code = ""
        
        split_pattern = r"```python|```"
        prompt_parts = re.split(split_pattern, prompt_received)
        
        for prompt_part in prompt_parts:
            if len(prompt_part) > len(refactored_code) and is_valid_python(prompt_part):
                refactored_code = prompt_part
        
        split_pattern = r"####"
        prompt_parts = re.split(split_pattern, prompt_received)
        
        for prompt_part in prompt_parts:
            if len(prompt_part) > len(refactored_code) and is_valid_python(prompt_part):
                refactored_code = prompt_part
    else:
        raise Exception(f"Wrong language sent. Language: {language}")
    
    return refactored_code

def get_tree(content, language):
    if language == "java":
        tree = javalang.parse.parse(content)
    elif language == "python":
        tree = ast.parse(content)
    else:
        raise Exception(f"Wrong language sent. Language: {language}")
    
    return tree
    
def replace_code_project(df_replace_project, LLM_used, path_to_replace, mode="default"):
    
    if mode == "final":
        sign = "final"
    else:
        sign = "optimization"

    df_replace_project["Replaced_code"] = ""

    visited_files = {}
    for index, row in tqdm(df_replace_project.iterrows(), total=df_replace_project.shape[0], file=sys.stdout, desc="Replacing code", leave=True):
        prompt_ID = row['Prompt_ID']
        filename = row["File_name"]
        LLM_used = row["LLM_used"].replace(":","").replace(" ","").replace("/","")
        prompt_status = str(row['Prompting_status'])
        try:
            tkns_sent = int(row['Tokens_sent'])
        except:
            continue
        path_to_file = row["Path_to_file"].replace("\\","/")
        project = row["Project"]
        prompt_received = row["Prompt_received"]
        smell_type = row['Smell']
        class_name = row["Class_name"]
        language = get_current_language(prompt_ID)

        if isinstance(prompt_ID, str) and prompt_status == "ok":
            found = False

            if tkns_sent > MAX_TOKENS_TO_REPLACE:
                df_replace_project.loc[index, "Replacing_status"] = f"Not replaced: Original prompt > {MAX_TOKENS_TO_REPLACE}"
                continue
            
            file_id = path_to_file + filename + LLM_used + prompt_ID + smell_type
            print(f"This is the path to file: {path_to_file}")
            # print(f"This is the path to general_dataset_repo_path: {GENERAL_DATASETS_REPO_PATH}")
            
            relative_path = os.path.join(project, path_to_file[1:])
            relative_path_to_file = path_to_file[1:]
            # relative_path = os.path.relpath(path_to_file, GENERAL_DATASETS_REPO_PATH)
            # relative_path_to_file = os.path.relpath(path_to_file, os.path.join(GENERAL_DATASETS_REPO_PATH, project))
            # print(f"This is the relative_path: {relative_path}")
            original_file_dataset = os.path.join(GENERAL_DATASETS_REPO_PATH, relative_path.replace("\\","/"))
            
            original_file_rpl_str_path = os.path.join(ORIGINAL_RPL_STR_PATH, relative_path)
            
            if mode == "final":
                refactored_file_rpl_str_path = os.path.join(SMELLS_RPL_STR_PATH, "Final", LLM_used + '_' + prompt_ID + '_' + project, relative_path_to_file)
            
            else:
                refactored_file_rpl_str_path = os.path.join(SMELLS_RPL_STR_PATH, smell_type + '_' + LLM_used + '_' + prompt_ID + '_' + sign + '_' + project, relative_path_to_file)
                # print(f"CHECKING FILE_PATH {refactored_file_rpl_str_path}\n\n\n {SMELLS_RPL_STR_PATH}")
                
            if file_id in visited_files:
                try:
                    with open(refactored_file_rpl_str_path, 'r', encoding='utf-8') as file:  
                        content = file.read()
                        tree = get_tree(content, language)
                    found = True
                    
                except Exception as e:
                    # print(e)
                    df_replace_project.loc[index, "Replacing_status"] = f"Not replaced: Could not reopen directory: {refactored_file_rpl_str_path}"
                    # print("------------------1")
                    continue

            else:
                try:
                    with open(original_file_dataset, 'r', encoding='utf-8') as file:
                        content = file.read()
                        tree = get_tree(content, language)
                     
                except Exception as e:
                    df_replace_project.loc[index, "Replacing_status"] = f"Not replaced: Could not open and read directory: {original_file_dataset}"
                    # print(f"------------------2{original_file_dataset}\n {e}")
                    continue
                
                try:
                    os.makedirs(os.path.dirname(original_file_rpl_str_path), exist_ok=True)
                    with open(original_file_rpl_str_path, 'w', encoding='utf-8') as file:
                        file.write(content)
                        
                except Exception as e:
                    df_replace_project.loc[index, "Replacing_status"] = f"Not replaced: Could not open and write directory: {original_file_rpl_str_path}"
                    # print("------------------3")
                    continue
                
                try:
                    os.makedirs(os.path.dirname(refactored_file_rpl_str_path), exist_ok=True)
                    with open(refactored_file_rpl_str_path, 'w', encoding='utf-8') as file:
                        file.write(content)
                    visited_files[file_id] = True
                    found = True
                    
                except Exception as e:
                    df_replace_project.loc[index, "Replacing_status"] = f"Not replaced: Could not open and write directory: {refactored_file_rpl_str_path}"
                    # print("------------------4")
                    continue

            if found:
                
                try:
                    refactored_code = ""
                    refactored_code  = safe_split_and_process(prompt_received, language)
                    
                    # print(200000)
                    output, status = replace_class_content(content, class_name, refactored_code, language)
                    # print(3)
                    if status:
                        modified_code = output
                        df_replace_project.loc[index, "Replacing_status"] = f"ok"
                    else:
                        raise Exception(f"Not replaced: {output}")
                    # print(30)
                    if content == modified_code:
                        df_replace_project.loc[index, "Replacing_status"] = f"Not replaced: Refactored code == original code"
                        # print(300)
                    else:
                        # print(300)
                        if len(modified_code) > 10:
                            # print(3000)
                            try:
                                # os.remove(refactored_file_rpl_str_path)
                                with open(refactored_file_rpl_str_path, 'w', encoding='utf-8') as file:
                                    # print(30000)
                                    df_replace_project.loc[index, "Replaced_code"] = modified_code
                                    file.write(modified_code)
                                    # print(300000, refactored_file_rpl_str_path)
                            except Exception as e:
                                df_replace_project.loc[index, "Replacing_status"] = f"Not replaced: {e}\nThis is the whole code: {output}"
                                continue
                                
                        else:
                            raise Exception(f"Not replaced: Code too short")
    
                    # df_replace_project.loc[index, "Replacing_status"] = f"ok"
                    # if "@mock.patch('cloudinit.sources.helpers.netlink.socket.socket')" in output:
                    #     df_replace_project.loc[index, "Replaced_code"] = "ok, not here to prevent errors"
                    # else:
                    #     df_replace_project.loc[index, "Replaced_code"] = output
                
                except Exception as e:
                    df_replace_project.loc[index, "Replacing_status"] = e
            
            if not found:
                df_replace_project.loc[index, "Replacing_status"] = f"file not found"      
    # del_files(GENERAL_DATASETS_REPO_PATH, mode="directed", string=os.path.basename(project_path), ignore_string=".zip")
    return df_replace_project

# def wrapp_replace_code_project(subdf, path, mode):
#     # Call the actual function with unpacked arguments
#     return replace_code_project(subdf, path, mode=mode)

def del_files(dir_path, mode="soft", string="", ignore_string=""):
    # Erase everything except .zip or .txt or .xlsx
    if mode == "soft":
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if not item_path.endswith('.zip') and not item_path.endswith('.txt') and not item_path.endswith('.xlsx') and not item_path.endswith('.gitignore'):
                if os.path.isdir(item_path):
                    subprocess.run(['rm', '-rf', item], cwd=dir_path)
                elif os.path.isfile(item_path):
                    subprocess.run(['rm', '-f', item], cwd=dir_path)

    
    if mode == "hard":
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if not item_path.endswith('.gitignore'):
                if os.path.isdir(item_path):
                    subprocess.run(['rm', '-rf', item], cwd=dir_path)
                elif os.path.isfile(item_path):
                    subprocess.run(['rm', '-f', item], cwd=dir_path)

    
    if mode == "directed":
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if not item_path.endswith('.txt') and not item_path.endswith('.xlsx') and not item_path.endswith('.gitignore'):
                if string in os.path.basename(item_path) and ignore_string not in os.path.basename(item_path):
                    if os.path.isdir(item_path):
                        subprocess.run(['rm', '-rf', item], cwd=dir_path)
                    elif os.path.isfile(item_path):
                        subprocess.run(['rm', '-f', item], cwd=dir_path)

                        
    if mode == "directed-excel":
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if not item_path.endswith('.txt') and not item_path.endswith('.gitignore'):
                if string in os.path.basename(item_path) and ignore_string not in os.path.basename(item_path):
                    if os.path.isdir(item_path):
                        subprocess.run(['rm', '-rf', item], cwd=dir_path)
                    elif os.path.isfile(item_path):
                        subprocess.run(['rm', '-f', item], cwd=dir_path)


def extract_item_default(zip_file_path, item, dir_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(dir_path, item.replace(".zip", "")))
        
def extract_item_directed(zip_file_path, item, dir_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(dir_path, item))

def remove_item(item_path, item, dir_path):
    if os.path.isdir(item_path):
        subprocess.run(['rm', '-rf', item], cwd=dir_path)
    elif os.path.isfile(item_path):
        subprocess.run(['rm', '-f', item], cwd=dir_path)


def restore_original_files(dir_path, mode="default", files=[]):
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        if mode == "directed":
            for item in tqdm(files, total=len(files), file=sys.stdout, desc="Cleaning dir directed", leave=True):
                item_path = os.path.join(dir_path, item)
                if not item_path.endswith('.zip') and not item_path.endswith('.txt') and not item_path.endswith('.xlsx') and not item_path.endswith('.gitignore'):
                    # if os.path.isdir(item_path):
                    #     subprocess.run(['rm', '-rf', item], cwd=dir_path)
                    # elif os.path.isfile(item_path):
                    #     subprocess.run(['rm', '-f', item], cwd=dir_path)

                    remove_item(item_path, item, dir_path)
                    
            for item in tqdm(files, total=len(files), file=sys.stdout, desc="Restoring files directed", leave=True):
                if os.path.isfile(os.path.join(dir_path, item + ".zip")):
                    zip_file_path = os.path.join(dir_path, item + ".zip")
                    # with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    #     zip_ref.extractall(os.path.join(dir_path, item))
                    extract_item_directed(zip_file_path, item, dir_path)

            
        elif mode == "default":
            for item in tqdm(os.listdir(dir_path), total=len(os.listdir(dir_path)), file=sys.stdout, desc="Cleaning dir", leave=True):
                item_path = os.path.join(dir_path, item)
                if not item_path.endswith('.zip') and not item_path.endswith('.txt') and not item_path.endswith('.xlsx') and not item_path.endswith('.gitignore'):
                    # if os.path.isdir(item_path):
                    #     subprocess.run(['rm', '-rf', item], cwd=dir_path)
                    # elif os.path.isfile(item_path):
                    #     subprocess.run(['rm', '-f', item], cwd=dir_path)

                    remove_item(item_path, item, dir_path)
                    
            for item in tqdm(os.listdir(dir_path), total=len(os.listdir(dir_path)), file=sys.stdout, desc="Restoring files", leave=True):
                if item.endswith('.zip'):
                    zip_file_path = os.path.join(dir_path, item)
                    # with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    #     zip_ref.extractall(os.path.join(dir_path, item.replace(".zip", "")))
                    extract_item_default(zip_file_path, item, dir_path)
                    
        pool.close()
        pool.join()

def clear_replaced():
        removido = 0
        for rootname, dirnames, filenames in os.walk(DATASET_TO_REPLACE_PATH):
            for filename in filenames:
                if "replaced" in filename:
                    os.remove(os.path.join(rootname, filename))
                    removido += 1
        print(f"Removidos: {removido}")
        return removido

def get_sample_size(population_size, margin_of_error, confidence_level, proportion=0.5):
    # Z-scores for the corresponding confidence levels
    z_scores = {0.80: 1.28, 0.85: 1.44, 0.90: 1.65, 0.95: 1.96, 0.99: 2.58}
    
    # Find the appropriate z-score based on the desired confidence level
    z = z_scores.get(confidence_level)
    
    if not z:
        raise ValueError("Unsupported confidence level. Use one of the following: 80%, 85%, 90%, 95%, 99%")
    
    # Calculate the first part of the formula (numerator of both)
    numerator_top = (z ** 2) * proportion * (1 - proportion)
    denominator_top = margin_of_error ** 2
    
    numerator_bot = numerator_top
    denominator_bot = (margin_of_error ** 2) * population_size

    # Apply finite population correction (FPC)
    corrected_sample_size = (numerator_top / denominator_top) / (1 + (numerator_bot / denominator_bot))
    
    return math.ceil(corrected_sample_size) 

def calculate_margin_of_error_proportion(sample_size, population_size, confidence_level, proportion=0.5):
    z_scores = {0.80: 1.28, 0.85: 1.44, 0.90: 1.65, 0.95: 1.96, 0.99: 2.58}
    
    # Find the appropriate z-score based on the desired confidence level
    z_score = z_scores.get(confidence_level)
    # Apply finite population correction (FPC) if necessary
    if population_size is not None and population_size > 0:
        fpc = math.sqrt((population_size - sample_size) / (population_size - 1))
    else:
        fpc = 1  # No correction needed for large populations
    
    # Calculate margin of error for proportions
    moe = z_score * math.sqrt((proportion * (1 - proportion)) / sample_size) * fpc
    return moe

def calculate_moe_for_groups(df_prompted, df_to_prompt):
    # Group by the desired columns
    group_columns = ["Smell", "Prompt_ID", "LLM_used", "Is_smell_present_original"]
    
    df_prompted["Smell"] = df_prompted.apply(
        lambda row: "other smell" if str(row["Is_smell_present_original"]) == "0" else row["Smell"], axis=1
    )
    df_to_prompt["Smell"] = df_to_prompt.apply(
        lambda row: "other smell" if str(row["Is_smell_present_original"]) == "0" else row["Smell"], axis=1
    )
    
    grouped_prompted = df_prompted.groupby(group_columns)
    grouped_to_prompt = df_to_prompt.groupby(group_columns)
    

    results = []

    for name, group_prompted in grouped_prompted:
        # Sample size is the size of the prompted group
        smell, prompt_id, llm_used, is_smell_present_original = name
        sample_size = len(group_prompted)
        if sample_size == 0:
            continue
        # Population size is the size of the corresponding group in df_to_prompt
        if name in grouped_to_prompt.groups:
            population_size = len(grouped_to_prompt.get_group(name))
        else:
            continue  # Skip if no corresponding group
        
        # Calculate margin of error for 95% and 99% confidence levels
        moe_95 = calculate_margin_of_error_proportion(sample_size, population_size, 0.95)
        moe_99 = calculate_margin_of_error_proportion(sample_size, population_size, 0.99)
        
        results.append({
            "Smell": smell,
            "Prompt_ID": prompt_id,
            "LLM_used": llm_used,
            "Is_smell_present_original": is_smell_present_original,
            "Sample_Size": sample_size,
            "Population_Size": population_size,
            "MOE_95": moe_95,
            "MOE_99": moe_99
        })

    return pd.DataFrame(results)

def figure_out_precision():
    key_string = "_to_prompt_optimization_selected_6"
    df_to_prompt = DataframeManager.load(DATASET_TO_PROMPT_PATH, name_to_find=key_string)    
    # strings_to_search = list_unique_xlsx_file_paths(DATASET_TO_PROMPT_PATH, key_string)
    # if len(strings_to_search) == 1:
    #     string_to_search = strings_to_search[0]
    #     df_to_prompt = DataframeManager.load(DATASET_TO_PROMPT_PATH, name_to_find=string_to_search)
    # else:
    #     raise Exception(f"An error has occurred in figuring out precision with key string:{key_string}")
    
    df_prompted = df_to_prompt[df_to_prompt["Prompting_status"] == "ok"].copy()
    
    # Call the function to calculate MOE for groups
    moe_results = calculate_moe_for_groups(df_prompted, df_to_prompt)
    
    # Display or save results
    print(moe_results)
    
    DataframeManager.save(
        moe_results, 
        DATASET_TO_PROMPT_PATH, 
        name_to_find=key_string.replace("_to_prompt", "_MOE_results")
    )
    

def replace_all_code(path_to_replace, REPLACEMENT_STATUS_LIST, mode="optimization"):
    if mode != "optimization":
        key_string = "_to_replace"
    else:
        key_string = "_to_replace_optimization"
        
    strings_to_search = list_unique_xlsx_file_paths(path_to_replace, key_string)
    clear_replaced()
    restore_original_files(GENERAL_DATASETS_REPO_PATH)
    clear_rpl_str(RPL_STR_ROOT_PATH)
    if len(strings_to_search) == 1:
        string_to_search = strings_to_search[0]
        df_replace = DataframeManager.load(path_to_replace, name_to_find=string_to_search)
        
    else:
        raise Exception(f"More than one excel file with the following key string: {key_string}")
    
    subdfs_grouped = df_replace.groupby(["Prompt_ID", "LLM_used"])
    # with multiprocessing.Pool(processes=1) as pool:  
    for (prompt_ID, LLM_used), df_subdf_prompt_ID_LLM_used in subdfs_grouped:
        print(f"Beginning process for {prompt_ID, LLM_used}, phase: {key_string}")    
    #         condition = (
    #                     (MODEL_TO_USE_NAME in string_to_search) & 
    #                     (LANGUAGE_TO_USE in string_to_search) &
    #                     ("01." not in string_to_search) &
    #                     ("replaced" not in string_to_search)
    #                     )
        replace_code(df_subdf_prompt_ID_LLM_used, prompt_ID, LLM_used, path_to_replace, string_to_search)
        #     pool.apply_async(replace_code, args=(df_subdf_prompt_ID_LLM_used, prompt_ID, LLM_used, path_to_replace, string_to_search))              
        # pool.close()
        # pool.join()
    # aggregated_counts = pd.concat(REPLACEMENT_STATUS_LIST, ignore_index=True)
    # DataframeManager.save(aggregated_counts, path_to_replace, name_to_find=f"replacement_status_{LANGUAGE_TO_USE}")

def clear_rpl_str(dir_to_clear):
    for dirname in os.listdir(dir_to_clear):
        if dirname.endswith(".jar"):
            continue
        dirpath = os.path.join(dir_to_clear, dirname)
        if os.path.isdir(dirpath):
            subprocess.run(['rm', '-rf', dirpath], cwd=dir_to_clear)
        elif os.path.isfile(dirpath):
            subprocess.run(['rm', '-f', dirpath], cwd=dir_to_clear) 
    return 
    
def replace_code(df_replace, prompt_ID, LLM_used, path_to_replace, string_to_search):
    def clear_rpl_str(prompt_ID, LLM_used_name):
        removido = 0
        if "Prompt_p":
            extension = ".py"
        elif "Prompt_j":
            extension = ".java"
        for rootname, dirnames, filenames in os.walk(RPL_STR_ROOT_PATH):
            if "Smells" in rootname and LLM_used_name in rootname and prompt_ID in rootname:
                for filename in filenames:
                    os.remove(os.path.join(rootname, filename))
                    removido += 1
        print(f"Removidos: {removido}")
        return removido
    
    # if mode == "final":
    #     df_prompt_report = DataframeManager.load(POST_REF_DATA_PATH, name_to_find="FINAL_report")
    #     df_prompt_report_filtered = df_prompt_report[df_prompt_report["Selected"] == "Selected"]
    #     print(f"len of the df selected: {df_prompt_report_filtered.shape[0]}, {df_prompt_report_filtered.shape[1]}")
    # not_finished_before = True
    
    try:
        df_replace = df_replace.sort_values(by=['ID'])
        df_replace = df_replace.sort_values(by=['Prompting_status'])
        LLM_used_name = LLM_used.replace(":","").replace(" ","").replace("/","")
        print(f"This is the llm used: {LLM_used}")
        print(f"len of the df selected: {df_replace.shape[0]}, {df_replace.shape[1]}")
        # clear_rpl_str(prompt_ID, LLM_used_name)
        df_replace = replace_code_project(df_replace, LLM_used, path_to_replace, mode="default")
        
    except Exception as e:
        print(f"exception: {e}")
        
    DataframeManager.save(df_replace, DATASET_TO_REPLACE_PATH, name_to_find=string_to_search.replace("_to_replace","_replaced") + "_" + LLM_used_name + "_" + prompt_ID)

    # df_true = df_replace [df_replace ['Is_smell_present_original'] == True]
    # true_counts = df_true.groupby('Prompt_ID').size().reset_index(name='count_true')
    
    # # Filter rows where 'Is_smell_present_original' is False
    # df_false = df_replace [df_replace ['Is_smell_present_original'] == False]
    # false_counts = df_false.groupby('Prompt_ID').size().reset_index(name='count_false')
    
    # # Merging the two counts into a single DataFrame
    # counts = pd.merge(true_counts, false_counts, on='Prompt_ID', how='outer').fillna(0)
    # counts["Project"] = string_to_search
    
    # NUMBER_OF_SMELLS_LIST.append(counts)


def create_all_refactored_copies(path_to_rpl_str, mode="optimization"):
    original_project_names = [repo for repo in os.listdir(GENERAL_DATASETS_REPO_PATH) if not repo.endswith(".zip")]
    rpl_str_project_names = [repo for repo in os.listdir(path_to_rpl_str) if not repo.endswith(".zip")]
    restore_original_files(GENERAL_DATASETS_REPO_PATH)
    clear_rpl_str(AFTER_REPLACEMENT_PATH)
    clear_rpl_str(DETECTOR_OUTPUT_PATH)
    clear_rpl_str(DETECTOR_INPUT_PATH)
    with multiprocessing.Pool(processes=2) as pool:
        for original_project_name in original_project_names:
            original_project_path = os.path.join(GENERAL_DATASETS_REPO_PATH, original_project_name)
            new_project_paths = [(os.path.join(path_to_rpl_str, repo), os.path.join(AFTER_REPLACEMENT_PATH, repo)) for repo in rpl_str_project_names if original_project_name in repo]
            # print(f"##### FOR THIS PROJECT: {original_project_name}\n\nIN PATH {original_project_path}\n\nNEW_FILES:{new_project_paths}")
            for src_project_path, dest_project_path in new_project_paths:
        
            
                pool.apply_async(create_refactored_copies, args=(dest_project_path, original_project_path, src_project_path))
                
            pool.apply_async(create_refactored_copies, args=(os.path.join(AFTER_REPLACEMENT_PATH, original_project_name + "_check"), original_project_path, "check"))              
        pool.close()
        pool.join()
    # subprocess.run(['python', 'runner.py'], cwd = PLUGIN_ROOT_PATH, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # subprocess.run(['python', 'get_csv_stats.py'], cwd = PLUGIN_ROOT_PATH, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # subprocess.run(['python', 'runner.py'], cwd = PLUGIN_ROOT_PATH)
    # subprocess.run(['python', 'get_csv_stats.py'], cwd = PLUGIN_ROOT_PATH)
    # with multiprocessing.Pool(processes=NUM_WORKERS*2) as pool:  
    #     for string_to_search in strings_to_search:
    #         print(f"Beginning process for {string_to_search}, phase: {key_string}")
    #         pool.apply_async(create_refactored_copies, args=(string_to_search, key_string, path_to_rpl_str), kwds={"mode":"soft", "restart":False,})              
    #     pool.close()
    #     pool.join()

def create_refactored_copies(dest_project_path, original_project_path, src_project_path):
    
    if src_project_path != "check": 
        copy_files(original_project_path, dest_project_path)
        copy_files(src_project_path, dest_project_path)
        if "dataset_python" in original_project_path:
            dest_project_file = os.path.basename(dest_project_path)
            dest_pynose_input_path = os.path.join(DETECTOR_INPUT_PATH, dest_project_file)
            copy_files(original_project_path, dest_pynose_input_path)
            copy_files(src_project_path, dest_pynose_input_path)
            
    else:
        copy_files(original_project_path, dest_project_path)
        if "dataset_python" in dest_project_path:
            dest_project_file = os.path.basename(dest_project_path)
            dest_pynose_input_path = os.path.join(DETECTOR_INPUT_PATH, dest_project_file)
            copy_files(original_project_path, dest_pynose_input_path)
    
def copy_files(src, dest):
    # Ensure the source directory exists
    if not os.path.exists(src):
        print(f"The source directory {src} does not exist.")
        return

    # Ensure the destination directory exists, create if it does not
    if not os.path.exists(dest):
        os.makedirs(dest)
        # print(f"Created the destination directory {dest}")

    # Walk through the source directory
    for root, dirs, files in os.walk(src):
        # For each directory in the source, create a corresponding directory in the destination
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            dest_dir_path = os.path.join(dest, os.path.relpath(dir_path, src))
            if not os.path.exists(dest_dir_path):
                os.makedirs(dest_dir_path)
                # print(f"Created directory {dest_dir_path}")

        # For each file in the source, copy it to the corresponding location in the destination
        for file in files:
            file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest, os.path.relpath(file_path, src))
            shutil.copy2(file_path, dest_file_path)
            # print(f"Copied {file_path} to {dest_file_path}")     
    
def generate_report_ref(original_project_file, replaced_project_file, path_to_compare, RESULTS_LIST): 
    df_original_stats = DataframeManager.load(DATASET_TO_REPLACE_PATH, name_to_find=original_project_file)
    df_replaced_stats = DataframeManager.load(path_to_compare, name_to_find=replaced_project_file)

    # Apply the to_bool function to specific columns
    # Convert the Project column in df_replaced_stats to a list for efficient membership testing
    projects_list = df_replaced_stats['Project'].tolist()

    # Filter df_original_stats
    # df_original_stats = df_original_stats[df_original_stats["Replacing_status"]=="ok"]
    df_original_stats = df_original_stats[df_original_stats.apply(lambda row: any(row['Prompt_ID'] in project for project in projects_list), axis=1)]

    df_original_stats['Is_smell_present_original'] = df_original_stats['Is_smell_present_original'].apply(to_boolean)
    df_replaced_stats['Is_smell_present_original'] = df_replaced_stats['Is_smell_present_original'].apply(to_boolean)
    
    df_original_stats['combined_key'] = df_original_stats['Path_to_file'] + df_original_stats['Smell']
    df_original_stats['combined_key'] = df_original_stats.apply(
        lambda row: (row['Path_to_file'] + row['Smell']).split(row['Project'])[1] if row['Project'] in row['Path_to_file'] + row['Smell'] else None,
        axis=1
    )
    
    df_replaced_stats['combined_key'] = df_replaced_stats['Path_to_file'] + df_replaced_stats['Smell']
    df_replaced_stats['combined_key'] = df_replaced_stats.apply(
        lambda row: (row['Path_to_file'] + row['Smell']).split(row['Project'])[1] if row['Project'] in row['Path_to_file'] + row['Smell'] else None,
        axis=1
    )
    
    # Create a dictionary from df_original_stats for fast lookup using the combined key
    original_smell_presence = df_replaced_stats.set_index('combined_key')['Is_smell_present_original'].to_dict()  
    # Apply the values from df_original_stats to df_replaced_stats based on the combined key,
    df_original_stats['Is_smell_present_refactored'] = df_original_stats['combined_key'].map(original_smell_presence)
    # Clean up temporary columns created for this operation
    df_original_stats.drop(columns=['combined_key'], inplace=True)
    df_replaced_stats.drop(columns=['combined_key'], inplace=True)
    
    smells = df_original_stats['Smell'].unique()
    Prompts_ID = df_original_stats['Prompt_ID'].unique()
    print(f'smells: {smells},     {len(smells)}')
    print(f'prompt_ids: {Prompts_ID},     {len(Prompts_ID)}')
    # total_lines_for_this_prompt_version = len(df_replaced_stats)
    # prompted_lines_for_this_prompt_prompt_version = len(df_replaced_stats)
    
    for Prompt_ID in Prompts_ID:
        df_original_stats_filtered = df_original_stats[df_original_stats['Prompt_ID']==Prompt_ID]
        
        for current_smell in smells:
            df_original_stats_filtered = df_original_stats_filtered[df_original_stats_filtered['Smell']==current_smell]
            print(f"INFO DATAAFRAME: len {len(df_original_stats_filtered)}")
            total_classes_to_prompt = len(df_original_stats_filtered)
            # total_classes_to_prompt = (((df_original_stats_filtered['Is_smell_present_original']==True) & ((df_original_stats_filtered['Is_smell_present_refactored'] == False) | (df_original_stats_filtered['Is_smell_present_refactored'] == True))) & (df_original_stats_filtered['Creating_prompt_status']=="ok")).sum()
            # total_classes_replaced = ((df_original_stats_filtered['Is_smell_present_original']==True) & (df_original_stats_filtered['Prompting_status']=="ok")).sum()
            total_classes_replaced = (((df_original_stats_filtered['Is_smell_present_original']==True) & ((df_original_stats_filtered['Is_smell_present_refactored'] == False) | (df_original_stats_filtered['Is_smell_present_refactored'] == True))) & (df_original_stats_filtered['Replacing_status']=="ok")).sum()
           
            # Calculate comparison metrics
            smell_removed = ((df_original_stats_filtered['Is_smell_present_original'] == True) & (df_original_stats_filtered['Is_smell_present_refactored'] == False)).sum()
            smell_added = ((df_original_stats_filtered['Is_smell_present_original'] == False) & (df_original_stats_filtered['Is_smell_present_refactored'] == True)).sum()
            smell_kept = ((df_original_stats_filtered['Is_smell_present_original'] == True) & (df_original_stats_filtered['Is_smell_present_refactored'] == True)).sum()
            smell_not_present = ((df_original_stats_filtered['Is_smell_present_original'] == False) & (df_original_stats_filtered['Is_smell_present_refactored'] == False)).sum()
            # total_classes_replaced = smell_removed + smell_added + smell_kept + smell_not_present
            # Append results to global list
            RESULTS_LIST.append({
                'Original Project': original_project_file.replace("_clean_original",""),
                'Replaced Project': replaced_project_file.replace("_clean_refactored",""),
                'Prompt_ID': Prompt_ID,
                'Classes prompted': total_classes_to_prompt,
                # 'Classes prompted': total_classes_prompted,
                'Classes replaced': total_classes_replaced,
                'Smell Present Removed': smell_removed,
                'Smell Not Present Added': smell_added,
                'Smell Present Not Removed': smell_kept,
                'Smell Not Present Not Added': smell_not_present
            })
    DataframeManager.save(df_original_stats, TS_DETECT_DATA_COMPARISON, name_to_find=(replaced_project_file.replace("check_","").replace("dataset_java_", "dataset_java_comp_").replace("clean","").replace("__","_")))

def clean_the_unclean(path_to_delete):
    total = 0
    removed = 0
    not_removed = 0

    for root, dirs, files in os.walk(path_to_delete, topdown=False):
        for file in files:
            if file.endswith("xlsx"):
                total += 1
                try:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)  # Correct way to remove a file
                    removed += 1
                    print(f"REMOVED: {file_path}")
                except Exception as e:
                    print(f"NOT REMOVED: {file_path} due to {e}")
                    not_removed += 1

        for dir in dirs:
            if dir.endswith("xlsx"):
                total += 1
                try:
                    dir_path = os.path.join(root, dir)
                    shutil.rmtree(dir_path)  # Correct way to remove a directory
                    removed += 1
                    print(f"REMOVED: {dir_path}")
                except Exception as e:
                    print(f"NOT REMOVED: {dir_path} due to {e}")
                    not_removed += 1

    print(f"Total items processed: {total}")
    print(f"Successfully removed: {removed}")
    print(f"Failed to remove: {not_removed}")

    # print(f"#### STATUS REMOVED #### \nTotal: {total}\nRemoved: {removed}\nNot removed: {not_removed}")

def parse_original_filename(filename):
    try:
        # Splitting by '_' and taking the first part to get the Dataset_name
        dataset_name = filename.split('_')[0]
        
        # Finding 'Project_name' which appears after '-_'
        project_name = filename.split('_-_')[0].split('_')[1]
        
        # Extracting the 'Prompt_ID' which contains '00.' string
        prompt_id_n = filename.split("_00.")[1].split("_")[0]
        prompt_id_name = filename.split("_00.")[0].split("_")[-1]
        # prompt_id = prompt_id_name + "_00." + prompt_id_n
        prompt_id = prompt_id_name
        
        return dataset_name, project_name, prompt_id
    except Exception as e:
            print(f"original: {filename} - {e}")

def parse_replaced_filename(filename):
    try:
        # Splitting by '_' and taking the first part to get the Dataset_name
        dataset_name = filename.split('dataset_')[1].split('_')[0]
        
        # Finding 'Project_name' which appears after '-_'
        project_name = filename.split('_-_')[0].split('_')[-1]
        
        # Extracting the 'Prompt_ID' which contains '00.' string
        prompt_id_n = filename.split("_00.")[1].split("_")[0]
        prompt_id_name = filename.split("_00.")[0].split("_")[-1]
        # prompt_id = prompt_id_name + "_00." + prompt_id_n
        prompt_id = prompt_id_name
        
        return dataset_name, project_name, prompt_id
    except Exception as e:
        print(f"replaced: {filename} - {e}")

def read_and_prepare_data(csv_path, output_path):
    # List to hold data from all files
    all_data = []
    all_paths = os.listdir(csv_path)
    # Process each file path in the list
    data_python = ""
    aggregated_data_java_old = ""
    for dirname in all_paths:
        path = os.path.join(csv_path, dirname)    
        # Read the CSV file
        if "dataset_java" in path and path.endswith("csv"):
            data = pd.read_csv(path)
            if "dropwizard" in path:
                print(f"This is the path: {path}")
            # Extract relevant columns (modify this based on your needs)
            relevant_data = data[['App', 'NumberOfMethods', 'Assertion Roulette', 'Conditional Test Logic',
                                'Constructor Initialization', 'Default Test', 'EmptyTest', 'Exception Catching Throwing',
                                'General Fixture', 'Mystery Guest', 'Print Statement', 'Redundant Assertion',
                                'Sleepy Test', 'Eager Test', 'Lazy Test', 'Duplicate Assert', 'Unknown Test',
                                'IgnoredTest', 'Resource Optimism', 'Magic Number Test', 'Dependent Test']]
            
            # Convert boolean columns to integer for easier summing
            bool_columns = relevant_data.select_dtypes(include=['bool']).columns
            relevant_data[bool_columns] = relevant_data[bool_columns].astype(int)
            
            # Append the processed data to the list
            all_data.append(relevant_data)
        
        elif "aggregated.csv" in path:
            data_python = pd.read_csv(path)
        elif "aggregated_java.csv" in path:
            aggregated_data_java_old = pd.read_csv(path)
            
    
    # Combine all DataFrames into one
    combined_data = pd.concat(all_data)
    
    if isinstance(aggregated_data_java_old, pd.DataFrame):
        aggregated_data_java = aggregated_data_java_old
    
    else:
        # Aggregate the data by 'App', summing up methods and each test smell
        aggregated_data_java = combined_data.groupby('App').agg({
            'NumberOfMethods': 'sum',
            **{col: 'sum' for col in bool_columns}
        }).reset_index()

    if isinstance(data_python, pd.DataFrame):
        # Output the aggregated data to a CSV file
        output_path_java = os.path.join(output_path,"aggregated_java.csv")
        output_path_python = os.path.join(output_path,"aggregated_python.csv")
        aggregated_data_java.to_csv(output_path_java, index=False)
        data_python.to_csv(output_path_python, index=False)
        return aggregated_data_java, data_python
    else:
        # Output the aggregated data to a CSV file
        output_path_java = os.path.join(output_path,"aggregated_java.csv")
        aggregated_data_java.to_csv(output_path_java, index=False)
        return aggregated_data_java, data_python


def compare_data_ref(df_replace_filtered, row_original, row_refactored, refactored_project, original_project):
    """Compares the original and refactored data for a given project, calculating the differences for each smell."""
    # Extract the app name (assuming it's the same for both)
    app_name = refactored_project
    print(f"THIS IS THE CURRENT LENGH: {len(df_replace_filtered)}")
    # Create a DataFrame to store the results, including the 'App' column
    result_df = pd.DataFrame(columns=["Refactored_project"])

    # Create a list to hold smell data
    smell_data = []
    
    # Calculate total smells
    total_original_smells = row_original[1:].sum()
    total_refactored_smells = row_refactored[1:].sum()

    # Iterate over smells, creating a row per smell
    for smell in row_original.index[1:]:
        if "test_" not in smell and "_count" not in smell and "NumberOfMethods" not in smell:
            smell_to_put = rename_test_smell(smell)
            # print(f"this is the refactored projetc: {app_name}")
            df_replace_filtered_further = df_replace_filtered[df_replace_filtered["Smell"] == smell_to_put]
            smell_data.append({
                "Original_project": original_project,
                "Refactored_project": app_name,
                "Smell": smell_to_put,
                "LLM_used": app_name.split("_")[1].split("_")[0],
                "Prompt_ID": ("Prompt_" + str(app_name.split("_Prompt_")[1].split("_o")[0])),
                "Original_project_smell_count": len(df_replace_filtered_further),
                "Refactored_project_smell_count": len(df_replace_filtered_further) - (row_original[smell] - row_refactored[smell]),
                "Smells_removed": row_original[smell] - row_refactored[smell],
                # "row_refactored[smell]": row_refactored[smell],
                # "row_original[smell]": row_original[smell]
            })
    return pd.DataFrame(smell_data)

def condition_java(refactored_project, original_project):
    # print(f"Java: This is the refactored project - {refactored_project}\n And this is the original project: {original_project}")
    if (original_project in refactored_project and 
        "check" not in refactored_project and "optimization" in refactored_project):
        print(f"TRUE")
        print(f"Java: This is the refactored project - {refactored_project}\n And this is the original project: {original_project}")

        return True
    # print(f"FALSE")
    return False

def condition_python(refactored_project, original_project):
    # print(f"Python: This is the refactored project - {refactored_project}\n And this is the original project: {original_project}")
    if (original_project in refactored_project and 
        "check" not in refactored_project):
        return True
    return False

def calcular_efeitos(df_dados):
    df_dados_copied = df_dados.copy()
    df_dados_copied["Effect"] = ""
    for index, row in df_dados.iterrows():
        project = row["Refactored_project"]
        smell = row["Smell"]
        if smell in project:
            df_dados_copied.loc[index, "Effect"] = "Direct"
        else:
            df_dados_copied.loc[index, "Effect"] = "Collateral"
    return df_dados_copied

def segment_and_aggregate(df_dados):
    df_dados_direct_detailed = df_dados[df_dados["Effect"] == "Direct"]
    df_dados_collateral_detailed = df_dados[df_dados["Effect"] == "Collateral"]

    columns_to_exclude = ['Effect']

    # Create a list of numeric columns excluding the ones we don't want to sum
    numeric_columns_direct = df_dados_direct_detailed.select_dtypes(include='number').columns.difference(columns_to_exclude)
    numeric_columns_collateral = df_dados_collateral_detailed.select_dtypes(include='number').columns.difference(columns_to_exclude)

    # Group by and sum only the desired numeric columns
    # df_dados_direct = df_dados_direct_detailed.groupby(['Smell', "Prompt_ID", 'LLM_used'])[numeric_columns_direct].sum().reset_index()
    # df_dados_collateral = df_dados_collateral_detailed.groupby(['Smell', "Prompt_ID", 'LLM_used'])[numeric_columns_collateral].sum().reset_index()

    
    df_dados_direct = df_dados_direct_detailed.groupby(["Prompt_ID", 'LLM_used'])[numeric_columns_direct].sum().reset_index()
    df_dados_collateral = df_dados_collateral_detailed.groupby(["Prompt_ID", 'LLM_used'])[numeric_columns_collateral].sum().reset_index()

    # Add the 'Effect' column back
    df_dados_direct['Effect'] = "Direct"
    df_dados_collateral['Effect'] = "Collateral"

    # Concatenate the dataframes using pandas
    df_to_return = pd.concat([df_dados_direct, df_dados_collateral], ignore_index=True)

    # Drop the 'Refactored_project' column
    if 'Refactored_project' in df_to_return.columns:
        df_to_return = df_to_return.drop(columns=['Refactored_project'])

    return df_to_return

def process_aggregates(df_dados):
    df_dados_effects = calcular_efeitos(df_dados)
    df_dados_updated = segment_and_aggregate(df_dados_effects)
    return df_dados_updated

def contains_substring(row, refactored_project):
    i = 0
    columns_to_check = ['Dataset', 'Prompt_ID', 'LLM_used', 'Project']
    for col in columns_to_check:
        if row[col].replace(":","") in refactored_project:
            i += 1
    if i == len(columns_to_check):
        return True
    return False

def generate_report_ref_all(path_to_compare, mode="refactored"):
    
    df_data_java_original, df_data_python_original = read_and_prepare_data(TS_DETECT_INITIAL_OUTPUT, TS_DETECT_INITIAL_OUTPUT)
    df_data_java_replaced, df_data_python_replaced = read_and_prepare_data(TS_DETECT_REPLACED_OUTPUT, TS_DETECT_REPLACED_OUTPUT)
    df_replace = DataframeManager.load(DATASET_TO_REPLACE_PATH, name_to_find="to_replace_optimization")
    java_present = False
    python_present = False
    done = []
    # print(f"This is the replaced instance: {df_data_python_replaced}")
    df_results_ref_java = pd.DataFrame()
    for _, row_original_java in df_data_java_original.iterrows():
        original_project = row_original_java["App"]
        # print(f"This is the original_project {original_project}")
        for _, row_refactored_java in df_data_java_replaced.iterrows():
            refactored_project = row_refactored_java["App"]
            if condition_java(refactored_project, original_project):
                print(f"Java: This is the refactored project - {refactored_project}\n And this is the original project: {original_project}")
                df_replace_filtered = df_replace[df_replace.apply(lambda row: contains_substring(row, refactored_project), axis=1)]
                project_result_df_java = compare_data_ref(df_replace_filtered, row_original_java, row_refactored_java, refactored_project, original_project)
                df_results_ref_java = pd.concat([df_results_ref_java, project_result_df_java], ignore_index=True)
                java_present = True

    
    df_results_ref_python = pd.DataFrame()
    if isinstance(df_data_python_replaced, pd.DataFrame):
        for _, row_original_python in df_data_python_original.iterrows():
            original_project = row_original_python["repo_name"]
            for _, row_refactored_python in df_data_python_replaced.iterrows():
                refactored_project = row_refactored_python["repo_name"]
                if condition_python(refactored_project, original_project):
                    df_replace_filtered = df_replace[df_replace.apply(lambda row: contains_substring(row, refactored_project), axis=1)]
                    project_result_df_python = compare_data_ref(df_replace_filtered, row_original_python, row_refactored_python, refactored_project, original_project)
                    df_results_ref_python = pd.concat([df_results_ref_python, project_result_df_python], ignore_index=True)
                    python_present = True

    # Aggregate results
    if java_present:
        df_agg_ref_java  = process_aggregates(df_results_ref_java)
        DataframeManager.save(df_results_ref_java, TS_DETECT_DATA_COMPARISON, name_to_find="Detailed_refactoring_report_java")
        DataframeManager.save(df_agg_ref_java, TS_DETECT_DATA_COMPARISON, name_to_find="Aggregated_refactoring_report_java")

    if python_present:
        df_agg_ref_python = process_aggregates(df_results_ref_python)
        DataframeManager.save(df_results_ref_python, TS_DETECT_DATA_COMPARISON, name_to_find="Detailed_refactoring_report_python")
        DataframeManager.save(df_agg_ref_python, TS_DETECT_DATA_COMPARISON, name_to_find="Aggregated_refactoring_report_python")
    
    print(f"Is java present: {java_present} and Python: {python_present}")
    # return None, df_results_ref_java, df_agg_ref_python, df_results_ref_python 

    return df_agg_ref_java, df_results_ref_java, df_agg_ref_python, df_results_ref_python 
    # Save the results using DataframeManager


    # if mode == "refactored":
    #     original_project_files = list(set([repo.replace("current_","")[:-25] for repo in os.listdir(DATASET_TO_REPLACE_PATH) if repo.endswith(".xlsx") and "replaced" in repo and "optimization" in repo and LANGUAGE_TO_USE in repo and MODEL_TO_USE_NAME in repo and "00." in repo and "_check_" not in repo and "_comp_" not in repo and "report" not in repo]))
    #     replaced_project_files = list(set([repo.replace("current_","")[:-25] for repo in os.listdir(path_to_compare) if repo.endswith(".xlsx") and f"clean_{mode}" in repo and "optimization" in repo and LANGUAGE_TO_USE in repo and MODEL_TO_USE_NAME in repo and "_comp_" not in repo and "report" not in repo and ("00." in repo or "_check_" in repo)]))
    
    # else:
    #     original_project_files = list(set([repo.replace("current_","")[:-25] for repo in os.listdir(DATASET_TO_REPLACE_PATH) if repo.endswith(".xlsx") and "replaced" in repo and LANGUAGE_TO_USE in repo and MODEL_TO_USE_NAME in repo and "00." in repo and "_check_" not in repo and "_comp_" not in repo and "report" not in repo]))
    #     replaced_project_files = list(set([repo.replace("curreWnt_","")[:-25] for repo in os.listdir(path_to_compare) if repo.endswith(".xlsx") and f"clean_{mode}" in repo and LANGUAGE_TO_USE in repo and MODEL_TO_USE_NAME in repo and "_comp_" not in repo and "report" not in repo and ("00." in repo or "_check_" in repo)]))
    # # print(f"this is the first len: {len(original_project_files)}")
    # # print(f"this is the second len: {len(replaced_project_files)}")
    
    # with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
    #     for original_project_file in original_project_files:
    #         for replaced_project_file in replaced_project_files:
    #             dataset_name_original, project_name_original, prompt_id_original = parse_original_filename(original_project_file)
    #             dataset_name_replaced, project_name_replaced, prompt_id_replaced = parse_replaced_filename(replaced_project_file)
                                                                        
    #             condition = (
    #                         (dataset_name_original == dataset_name_replaced)  &
    #                         # (project_name_original == project_name_replaced) &
    #                         (prompt_id_original == prompt_id_replaced)
    #                         )
                
    #             if condition:
    #                 pool.apply_async(generate_report_ref, args=(original_project_file, replaced_project_file, path_to_compare, RESULTS_REF_LIST))
    #             # else:
    #             #     print(f"""FAILED:\n{original_project_file} 
    #             #           \n{replaced_project_file}\n\n""")
    #     pool.close()
    #     pool.join()
    #      # Create DataFrame from results and save
    #     print(f"Results_REF_list: {RESULTS_REF_LIST}")
    #     final_df = pd.DataFrame(list(RESULTS_REF_LIST))
        
    #     columns_to_sum = [
    #         "Classes replaced", "Smell Present Removed", 
    #         "Smell Not Present Added", "Smell Present Not Removed", "Smell Not Present Not Added"
    #     ]
    #     final_df = final_df[final_df.apply(lambda row: row['Prompt_ID'] in row['Replaced Project'], axis=1)]
    #     # Agrupando por "smell" e somando as colunas desejadas
    #     final_df = final_df.groupby('Prompt_ID')[columns_to_sum].sum().reset_index()

    #     # Dropando as colunas indesejadas
    #     final_df = final_df.drop(columns=["Original Project", "Replaced Project"], errors='ignore')
    #     final_df.drop_duplicates(inplace=True)
        # DataframeManager.save(final_df, path_to_compare, name_to_find="General_refactoring_report")

def generate_report_det_all(path_to_compare, mode="det_ref"):  
    # if mode == "det_ref":op
    #     to_prompt_files = list(set([repo.replace("current_","")[:-25] for repo in os.listdir(DATASET_TO_PROMPT_PATH) if repo.endswith(".xlsx") and "to_prompt" in repo and "01." in repo and "00." not in repo and "optimization" in repo]))
    # else:
    #     to_prompt_files = list(set([repo.replace("current_","")[:-25] for repo in os.listdir(DATASET_TO_PROMPT_PATH) if repo.endswith(".xlsx") and "to_prompt" in repo and "01." in repo and "00." not in repo]))
    strings_to_search = list_unique_xlsx_file_paths(path_to_compare, "optimization_selected")
    list_dfs_to_concat = []
    for string_to_search in strings_to_search:
        df_prompted_stats = DataframeManager.load(path_to_compare, name_to_find=string_to_search)
        list_dfs_to_concat.append(df_prompted_stats)
    df_concatenated = DataframeManager.concat(list_dfs_to_concat)
    aggregated_df, results_df = generate_report_det(df_concatenated, TS_DETECT_DATA_COMPARISON)
    
    # with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        # for to_prompt_file in to_prompt_files:
            # print(f"{to_prompt_file}\n\n")
            # condition = (
            #             (MODEL_TO_USE_NAME in to_prompt_file) &
            #             (LANGUAGE_TO_USE in to_prompt_file) 
            #             )
            # if condition:
            
                # pool.apply_async(generate_report_det, args=(to_prompt_file, DATASET_TO_PROMPT_PATH, RESULTS_DET_LIST))
        # pool.close()
        # pool.join()
         # Create DataFrame from results and save
        # print(f"Results_DET_list: {RESULTS_DET_LIST}")
        # final_df = pd.DataFrame(list(RESULTS_DET_LIST))
        # final_df.drop_duplicates(inplace=True)
        # columns_to_sum = [
        #     "Prompted Lines", "True Positives", "True Negatives", 
        #     "False Positives", "False Negatives"
        # ]
        # # # Filtrar as linhas onde 'Smell' não é uma substring de 'Prompted Project'
        # # final_df = final_df[final_df.apply(lambda row: row['Prompt_ID'] in row['Prompted Project'], axis=1)]
        # final_df = final_df.groupby('Prompt_ID')[columns_to_sum].sum().reset_index()
        # # # Agrupando por "smell" e somando as colunas desejadas
        # # aggregated_df = filtered_df.groupby('Smell')[columns_to_sum].sum().reset_index()

        # # # Dropando as colunas indesejadas
        # # final_df = final_df.drop(columns=["Prompted Project"], errors='ignore')
        # DataframeManager.save(final_df, path_to_compare, name_to_find="General_detection_report")
    return aggregated_df, results_df
    
    
def get_gpt_detection_status(prompt_received):
    """Determines GPT detection status based on the prompt received."""
    if ((prompt_received.find("YES") != -1) or
        (prompt_received.lower().find("yes") != -1 and prompt_received.lower().find("no") == -1)):
        return True  

    elif ((prompt_received.find("NO") != -1) or (prompt_received.lower().find("no") != -1 and prompt_received.lower().find("yes") == -1)):
        return False  
    
    else:
        return ""

def append_fisher_pvalue(df):
    """
    Adds p-values using Fisher's Exact Test to the DataFrame, along with a 'significant' flag.
    """
    def calculate_p_value(row):
        # Create a 2x2 contingency table for Fisher's Exact Test
        table = [[row['True Positives'], row['False Negatives']],
                 [row['False Positives'], row['True Negatives']]]

        if sum(table[0]) == 0 or sum(table[1]) == 0:
            return None  # Invalid case for Fisher's Exact Test

        # Apply Fisher's Exact Test (use 'two-sided' for a general test)
        _, p_value = fisher_exact(table, alternative='two-sided')
        return p_value

    # Apply Fisher's test for each row in the DataFrame
    df['p_value'] = df.apply(calculate_p_value, axis=1)
    
    # Flag rows as significant if p_value < 0.05
    df['significant'] = df['p_value'] < 0.05
    return df

    # Define a function to calculate the metrics
    
def calculate_classification_metrics(df):
    # Define a function to calculate the metrics for a given row or group of rows
    def calculate_row_metrics(row):
        tp = row['True Positives']
        tn = row['True Negatives']
        fp = row['False Positives']
        fn = row['False Negatives']
        
        # Calculate accuracy, precision, recall, and F1-Score
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return pd.Series({
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score
        })

    # Apply the metrics calculation function to each row in the DataFrame
    metrics_df = df.apply(calculate_row_metrics, axis=1)
    
    # Combine the calculated metrics with the original DataFrame
    df_with_metrics = pd.concat([df, metrics_df], axis=1)
    
    return df_with_metrics

def generate_report_det(df_prompted_stats, path_to_compare):
    # Filter the DataFrame to include only valid entries
    df_prompted_stats = transform_df(df_prompted_stats)
    df_prompted_stats = df_prompted_stats[df_prompted_stats["Prompting_status"] == "ok"]
    df_prompted_stats = df_prompted_stats[df_prompted_stats["Det_or_ref"] == "Detecção"]
    
    df_prompted_stats['GPT_detection_status'] = df_prompted_stats['Prompt_received'].apply(get_gpt_detection_status)

    df_prompted_stats['Is_smell_present_original'] = df_prompted_stats['Is_smell_present_original'].apply(to_boolean)
    df_prompted_stats['GPT_detection_status'] = df_prompted_stats['GPT_detection_status'].apply(to_boolean)

    # Define a function to calculate the metrics
    def calculate_metrics(group):
        tp = ((group['Is_smell_present_original'] == True) & (group['GPT_detection_status'] == True)).sum()
        tn = ((group['Is_smell_present_original'] == False) & (group['GPT_detection_status'] == False)).sum()
        fp = ((group['Is_smell_present_original'] == False) & (group['GPT_detection_status'] == True)).sum()
        fn = ((group['Is_smell_present_original'] == True) & (group['GPT_detection_status'] == False)).sum()
        return pd.Series({
            'True Positives': tp,
            'True Negatives': tn,
            'False Positives': fp,
            'False Negatives': fn
        })

    # Apply the function to each group
    results_df = df_prompted_stats.groupby(['Smell', 'Project', 'Prompt_ID', 'LLM_used']).apply(calculate_metrics).reset_index()
    aggregated_df = results_df.groupby(['Smell', 'Prompt_ID', 'LLM_used']).sum().reset_index()
    
    results_df = calculate_classification_metrics(results_df)
    aggregated_df = calculate_classification_metrics(aggregated_df)
    
    # Save the results using DataframeManager
    DataframeManager.save(results_df, TS_DETECT_DATA_COMPARISON, name_to_find="Detailed_detection_report")
    DataframeManager.save(aggregated_df, TS_DETECT_DATA_COMPARISON, name_to_find="Aggregated_detection_report")
    
    results_df['Language'] = results_df['Prompt_ID'].apply(lambda x: 'Java' if '_j_' in x else 'Python')

    identified_summary = results_df.groupby(['LLM_used', 'Language', 'Smell'])[['True Positives', 'False Negatives']].sum()
    identified_summary['Percent Identified'] = (identified_summary['True Positives'] / (identified_summary['True Positives'] + identified_summary['False Negatives'])) * 100
    identified_summary['Percent Not Identified'] = (identified_summary['False Negatives'] / (identified_summary['True Positives'] + identified_summary['False Negatives'])) * 100

    # Round percentages to 1 decimal place
    identified_summary['Percent Identified'] = identified_summary['Percent Identified'].round(1)
    identified_summary['Percent Not Identified'] = identified_summary['Percent Not Identified'].round(1)

    # Pivot the table to get the structure needed for Python / Java percentages in one column
    table_1 = identified_summary.reset_index().pivot_table(index='Smell', columns=['LLM_used', 'Language'], values=['Percent Identified']).reset_index()

    # Combine Python and Java percentages into a single column
    for llm in table_1.columns.levels[1]:  # Iterate over LLMs (columns)
        print(f'{llm} THIS IS THE ONE')
        table_1[f'{llm}_combined'] = table_1[('Percent Identified', llm, 'Python')].fillna('-').astype(str) + ' / ' + table_1[('Percent Identified', llm, 'Java')].fillna('-').astype(str)

    # Drop the separate Python/Java columns, keeping only the combined column
    columns_to_keep = ['Smell'] + [col for col in table_1.columns if 'combined' in col]
    table_1 = table_1[columns_to_keep]

    # Save Table 1 DataFrame
    DataframeManager.save(table_1, TS_DETECT_DATA_COMPARISON, name_to_find="Table_1_detection_report")

    return aggregated_df, results_df
    
def generate_reports_all(path_to_compare, mode="refactored"):
    if mode == "refactored":
        clean_the_unclean(path_to_compare)
        # generate_report_ref_all(path_to_compare, mode="refactored")
        generate_report_det_all(DATASET_TO_PROMPT_PATH, mode="det_ref")
        load_and_drop_paths_report()
        
    elif mode == "refactored_final":
        df_results_ref_java = None
        df_results_ref_python = None
        df_agg_ref_java = None
        df_agg_ref_python = None
        clean_the_unclean(path_to_compare)
        mode = "refactored"
        df_agg_ref_java, df_results_ref_java, df_agg_ref_python, df_results_ref_python = generate_report_ref_all(path_to_compare, mode="refactored")
        df_agg_det, df_detailed_det = generate_report_det_all(DATASET_TO_PROMPT_PATH, mode="det_ref")
        df_concat_to_prompt = load_and_drop_paths_report()
        
        if isinstance(df_results_ref_java, pd.DataFrame):
            df_results_ref_java = DataframeManager.load(TS_DETECT_DATA_COMPARISON, name_to_find="Detailed_refactoring_report_java")
        if isinstance(df_results_ref_python, pd.DataFrame):
            df_results_ref_python = DataframeManager.load(TS_DETECT_DATA_COMPARISON, name_to_find="Detailed_refactoring_report_python")
        if isinstance(df_agg_ref_java, pd.DataFrame):
            df_agg_ref_java = DataframeManager.load(TS_DETECT_DATA_COMPARISON, name_to_find="Aggregated_refactoring_report_java")
        if isinstance(df_agg_ref_python, pd.DataFrame):
            df_agg_ref_python = DataframeManager.load(TS_DETECT_DATA_COMPARISON, name_to_find="Aggregated_refactoring_report_python")
        print(f"FINAL REPORT {df_results_ref_java}")
        final_report.generate_report(df_results_ref_java, "final_comparison_ref_java", TS_DETECT_DATA_COMPARISON)
        final_report.generate_report(df_results_ref_python, "final_comparison_ref_python", TS_DETECT_DATA_COMPARISON)
        # print(f"Running now the aggregates")
        
        # final_report.generate_report(df_agg_ref_java, "final_comparison_agg_ref_java", TS_DETECT_DATA_COMPARISON)
        # final_report.generate_report(df_agg_ref_python, "final_comparison_agg_ref_python", TS_DETECT_DATA_COMPARISON)
        
    else:
        print(f"Error: INCORRECT MODE")

def load_and_drop_paths_report():
    df_manager = DataframeManager()
    # Load the concatenated DataFrame
    df_concat_to_prompt = df_manager.load_concat(DATASET_TO_PROMPT_PATH, "to_prompt_optimization_selected")

    # List of columns to drop
    columns_to_drop = [
        "ID", "Optimization", "Finding_class_status", "Class_tokens", "Ready_to_prompt",
        "Prompting_status", "Replacing_status", "Prompt_sent", "Tokens_received", "Response_time",
        "Tokens_sent", "Response_timestamp", "Random", "ideal_number_with_dist",
        "ideal_number_without_dist", "current_number_present", "current_number_not_present",
        "Select?", "Unnamed: 30", "ID_2"
    ]

    # Drop the specified columns
    df_concat_to_prompt.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Save the modified DataFrame using the save method of df_manager
    df_manager.save(df_concat_to_prompt, TS_DETECT_DATA_COMPARISON, "file_paths_and_prompts_received")
    
    return df_concat_to_prompt
    
def zip_directories(path):
    # Substrings to check and their corresponding directories
    substrings_language = {"dataset_java": "Java", "dataset_python": "Python"}
    substrings = {"_gemini": "gemini", "_gpt-4": "gpt-4", "_llama": "llama"}

    for substring_l, subdir_l in substrings_language.items():
        subdir_l = os.path.join(path, subdir_l)
        try:
            clear_rpl_str(subdir_l)
        except Exception as e:
            print(f"Tried to erase and failed, path to erase: {subdir_l}")
    # Pattern to match ".XY_" where XY are two digits
    pattern = re.compile(r'\.(\d{2})_')

    # Iterate over all items in the given path
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        # Check if the item is a directory
        if os.path.isdir(item_path):
            # Determine the directory for zipping based on the substrings
            for substring_l, subdir_l in substrings_language.items():
                if substring_l in item.lower():
                    subdir_l = os.path.join(path, subdir_l)

                    for substring, subdir in substrings.items():
                        if substring in item.lower():
                            subdir_path = os.path.join(subdir_l, subdir)
                            # Create subdir if it does not exist
                            if not os.path.exists(subdir_path):
                                os.makedirs(subdir_path)

                            # Search for the pattern in the item name
                            match = pattern.search(item)
                            if match:
                                # Extract the digits and create a version subdirectory
                                version_subdir = 'V' + match.group(1)
                                version_path = os.path.join(subdir_path, version_subdir)
                                if not os.path.exists(version_path):
                                    os.makedirs(version_path)
                                # Set the archive base name to the version path
                                archive_base = os.path.join(version_path, item)
                                # Create a zip file with the appropriate base path
                                shutil.make_archive(archive_base, 'zip', item_path)
                                # print(f"Zipped {archive_base}")
                                break
                            else:
                                print(f"ERROR: THIS WAS NO MATCH: {match}")
    

def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return total_size
                            
def compare_zip_contents(root_path):
    # Define the directories to check
    directories_to_check = ["Java", "Python"]
    results = []
    results_g = []
    
    # Iterate over each directory to check
    for language_dir in directories_to_check:
        language_path = os.path.join(root_path, language_dir)
        
        # Walk through the directory structure
        for dirpath, dirnames, filenames in os.walk(language_path):
            number_of_zipfiles = 0
            for filename in filenames:
                # Check if the file is a zip file
                if filename.endswith('.zip'):
                    number_of_zipfiles += 1
                    zip_path = os.path.join(dirpath, filename)
                    # Assuming original directories are structured similarly, adjust as necessary
                    original_dir = os.path.join(root_path, filename[:-4])  # Adjust this to match your directory structure
                    
                    # Open the zip file
                    with zipfile.ZipFile(zip_path, 'r') as zfile:
                        # List of all files and directories in the zip
                        zip_content_set = set(zfile.namelist())

                        # List of all files and directories in the original directory
                        original_content_set = set()
                        for root, dirs, files in os.walk(original_dir):
                            for dir in dirs:
                                dir_path = os.path.join(root, dir)
                                relative_path = os.path.relpath(dir_path, start=original_dir) + '/'
                                if '.git' not in relative_path:
                                    original_content_set.add(relative_path)
                            for file in files:
                                file_path = os.path.join(root, file)
                                relative_path = os.path.relpath(file_path, start=original_dir)
                                if '.git' not in relative_path:
                                    original_content_set.add(relative_path)

                        # Normalize directory entries in zip_content_set
                        zip_content_set = {x + '/' if not x.endswith('/') and x.count('/') == original_content_set else x for x in zip_content_set if '.git' not in x}
                        
                        data = {
                            "zip_path": zip_path,
                            "original_dir": original_dir,
                            "number_of_files_zip_path": len([x for x in zip_content_set if not x.endswith('/')]),
                            "number_of_files_original_dir": len([x for x in original_content_set if not x.endswith('/')])
                        }
                        results.append(data)

                        # Compare the sets
                        if zip_content_set == original_content_set:
                            # print(f"Match: Contents of {zip_path} are identical to original {original_dir}\n\n")
                            pass
                        else:
                            print(f"Mismatch: Contents of {zip_path} do not match original {original_dir}\n\n")
                            print("-Extra in zip:", zip_content_set - original_content_set,"\n\n")
                            print("-Missing in zip:", original_content_set - zip_content_set,"\n\n")
                            print("--in zip:", zip_content_set,"\n\n")
                            print("--in original:", original_content_set,"\n\n")
            
            version_of_prompt = os.path.basename(dirpath)
            basename_model = os.path.dirname(dirpath)
            model_of_prompt = os.path.basename(basename_model)
            basename_language = os.path.dirname(basename_model)
            language = os.path.basename(basename_language)
            if language == language_dir:
                data_g = {
                        "version_of_prompt": version_of_prompt,
                        "model_of_prompt": model_of_prompt,
                        "language_of_prompt": language_dir,
                        "number_zip_files": number_of_zipfiles,
                        "size": get_directory_size(dirpath)
                        }
                
                results_g.append(data_g)


    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    excel_path = os.path.join(root_path, 'zip_comparison_results.xlsx')
    df.to_excel(excel_path, index=False)
    # print(f"Data saved to {excel_path}")

    # Create DataFrame
    df = pd.DataFrame(results_g)

    # Save to CSV
    excel_path = os.path.join(root_path, 'general_results.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"Data saved to {excel_path} DONT FORGET TO SEND IT")


            
def create_results_tables():
    
    results_df = DataframeManager.load(TS_DETECT_DATA_COMPARISON, name_to_find="Aggregated_detection_report")
        
    results_df['Language'] = results_df['Prompt_ID'].apply(lambda x: 'Java' if '_j_' in x else 'Python')

    identified_summary = results_df.groupby(['LLM_used', 'Language', 'Smell'])[['True Positives', 'False Negatives']].sum()
    identified_summary['Percent Identified'] = (identified_summary['True Positives'] / (identified_summary['True Positives'] + identified_summary['False Negatives'])) * 100
    identified_summary['Percent Not Identified'] = (identified_summary['False Negatives'] / (identified_summary['True Positives'] + identified_summary['False Negatives'])) * 100

    # Round percentages to 1 decimal place
    identified_summary['Percent Identified'] = identified_summary['Percent Identified'].round(1)
    identified_summary['Percent Not Identified'] = identified_summary['Percent Not Identified'].round(1)

    # Pivot the table to get the structure needed for Python / Java percentages in one column
    table_1 = identified_summary.reset_index().pivot_table(index='Smell', columns=['LLM_used', 'Language'], values=['Percent Identified']).reset_index()

    # Combine Python and Java percentages into a single column
    for llm in table_1.columns.levels[1]:  # Iterate over LLMs (columns)
        print(f'{llm} THIS IS THE ONE')
        table_1[f'{llm}_combined'] = table_1[('Percent Identified', llm, 'Python')].fillna('-').astype(str) + ' / ' + table_1[('Percent Identified', llm, 'Java')].fillna('-').astype(str)

    # Drop the separate Python/Java columns, keeping only the combined column
    columns_to_keep = ['Smell'] + [col for col in table_1.columns if 'combined' in col]
    table_1 = table_1[columns_to_keep]

    # Save Table 1 DataFrame
    DataframeManager.save(table_1, TS_DETECT_DATA_COMPARISON, name_to_find="Table_1_detection_report")


    


def main():

    RESULTS_REF_LIST = Manager().list()
    RESULTS_DET_LIST = Manager().list()
    NUMBER_OF_SMELLS_LIST = Manager().list()
    REPLACEMENT_STATUS_LIST = Manager().list()

    operation = ""

    if len(sys.argv) >= 2:
        operation = sys.argv[1]
        
    if operation == "clone":
        clone_and_checkout(DATASET_CSV_PATH)

    elif operation == "restore_files":
        restore_original_files(GENERAL_DATASETS_REPO_PATH)

    elif operation == "initial_smell_info":
        get_test_smells_info(GENERAL_DATASETS_REPO_PATH)

    elif operation == "create_to_prompt":
        create_to_prompt(DATASET_PATH) 

    elif operation == "prompt_selected":
        # for n_of_prompts in range(5):
        # print(f"########### Prompt for {n_of_prompts}")
        n = 100000000
        n_max_uniform = 500000000
        # n_max_uniform = 100
        start_prompting_all(DATASET_TO_PROMPT_PATH, n, n_max_uniform, mode="optimization")

    elif operation == "replace_create":
        create_to_replace_all(DATASET_TO_PROMPT_PATH)

    elif operation == "replace":
        replace_all_code(DATASET_TO_REPLACE_PATH, REPLACEMENT_STATUS_LIST)
        create_all_refactored_copies(SMELLS_RPL_STR_PATH)

    elif operation == "smell_info":
        find_all_test_files(AFTER_REPLACEMENT_PATH, mode="refactored")

    elif operation == "report":
        generate_reports_all(TS_DETECT_DATA_COMPARISON, mode="refactored")

    elif operation == "pack_refactored":
        zip_directories(AFTER_REPLACEMENT_PATH)
        compare_zip_contents(AFTER_REPLACEMENT_PATH)

    elif operation == "pack_check":
        compare_zip_contents(AFTER_REPLACEMENT_PATH)
        
    elif operation == "report_final":
        generate_reports_all(TS_DETECT_DATA_COMPARISON, mode="refactored_final")
    
    elif operation == "get_sample_size":
        
        
        figure_out_precision()
        # result = calculate_margin_of_error_proportion(sample_size, population_size, confidence_level, proportion=0.5)    
        # print(f"This is the result {result}")

    elif operation == "replace_and_pack":

        # replace_all_code(DATASET_TO_REPLACE_PATH, REPLACEMENT_STATUS_LIST)
        # create_all_refactored_copies(SMELLS_RPL_STR_PATH)

        # find_all_test_files(AFTER_REPLACEMENT_PATH, mode="refactored")

        generate_reports_all(TS_DETECT_DATA_COMPARISON, mode="refactored_final")

        # zip_directories(AFTER_REPLACEMENT_PATH)

        # compare_zip_contents(AFTER_REPLACEMENT_PATH)
    elif operation == "create_results_tables":
        create_results_tables()
    else:
        print("""
        select an option:
            clone
            restore_files
            initial_smell_info
            create_to_prompt
            prompt_selected
            replace_create
            replace
            smell_info
            report
            report_final
            pack_refactored
            replace_and_pack
        """)
        operation = "wait"

if __name__ == "__main__":
    main()

    # clone_and_checkout(DATASET_CSV_PATH)
    # restore_original_files(GENERAL_DATASETS_REPO_PATH)
    # get_test_smells_info(GENERAL_DATASETS_REPO_PATH)
    # create_to_prompt(DATASET_PATH) # BE CAREFUL, Remember this will create a more recent to prompt, that might lead to redundant prompting
    # map_tokens()
    # create_all_to_prompt_prompt_id_optimization_df(DATASET_TO_PROMPT_PATH, mode="soft")
    # start_prompting_all(DATASET_TO_PROMPT_PATH, n, n_max_uniform, mode="optimization")
    # n = 4
    # for times in range(n):
        # create_all_to_prompt_prompt_id_optimization_df(DATASET_TO_PROMPT_PATH, mode="soft")
        
    # n = 10000
    # n_max_uniform = 500
    # start_prompting_all(DATASET_TO_PROMPT_PATH, n, n_max_uniform, mode="optimization")
    # restore_original_files(GENERAL_DATASETS_REPO_PATH)
    # create_to_replace_all(DATASET_TO_PROMPT_PATH)
    
    # replace_all_code(DATASET_TO_REPLACE_PATH, REPLACEMENT_STATUS_LIST)
    # create_all_refactored_copies(SMELLS_RPL_STR_PATH)
    # find_all_test_files(AFTER_REPLACEMENT_PATH, mode="refactored")

    # generate_reports_all(TS_DETECT_DATA_COMPARISON, mode="refactored")
    pass
    
