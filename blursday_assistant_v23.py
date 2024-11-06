##### ------ IMPORT PACKAGES + SETUP ------- ####

from pydantic import BaseModel
from openai import OpenAI
import fitz  # PyMuPDF for PDF handling
import io
import os
from PIL import Image # pillow
import base64
import json
#import psycopg2
import pandas as pd
import numpy as np
import re
import time
import openpyxl

# See reference github repo:  https://github.com/pixegami/openai-assistants-api-demo
# See also OpenAI reference documentation:  ttps://platform.openai.com/docs/assistants/how-it-works

# Enter your Assistant ID here.
ASSISTANT_ID = "asst_DmcouHcpU3rAZ4MV2FbwRrkR" # Old (superseded on 10 April - accidental delete): "asst_wWt15CA9kKqTI79SLQDPlGWm"

# Make sure your API key is set as an environment variable.
client = OpenAI()

######### ------ START OF NEW CLASSIFICATIONS CODE (No Assistants API - Directly using the Structured Outputs with Chat Completions API) ADDED ON 19 OCTOBER (UNUSED SO FAR) ------ #########

##### ------ DEFINE FUNCTIONS ------- ####

# Define BASIC item schema for coding responses
class ItemBasic(BaseModel):
    original_response: str
    categories_selected: list[str]
    ranking: list[str]
    activity_labels: list[str]
    flagged: bool
    justification_comments: str


# List of BASIC item schema for coding multiple responses
class QuestionExtraction(BaseModel):
    items: list[ItemBasic]


# Encode image as base64 encoded image
@staticmethod
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Convert pdf to base64 encoded images
def pdf_to_base64_images(pdf_path):
    #Handles PDFs with multiple pages
    pdf_document = fitz.open(pdf_path)
    base64_images = []
    temp_image_paths = []
    total_pages = len(pdf_document)
    for page_num in range(total_pages):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        temp_image_path = f"temp_page_{page_num}.png"
        img.save(temp_image_path, format="PNG")
        temp_image_paths.append(temp_image_path)
        base64_image = encode_image(temp_image_path)
        base64_images.append(base64_image)
    for temp_image_path in temp_image_paths:
        os.remove(temp_image_path)
    return base64_images


def categorize_blursday_responses(text_messages, base64_images, temperature_setting=0.8, max_tokens_setting=None, top_p_setting=1, presence_penalty_setting=0, n_setting=1, frequency_penalty_setting=0, logprobs_setting=False, model_setting="gpt-4o-mini", chain_of_thought=True):
    """
    Categorizes text responses based on the 'Blursday Codebook' using GPT-like models and returns structured output in JSON.
    Parameters:
    - text_messages (list of str): A list of text-based responses to be categorized.
    - base64_codebook_images (list of str): A list of base64-encoded codebook images to serve as reference for the categorization task.
    - temperature_setting (float): Temperature setting for model creativity.
    - model_setting (str): The model to be used for the task.
    - chain_of_thought (bool): Whether to include chain-of-thought reasoning in the response.
    Returns:
    - dict: Categorized data in JSON format.
    """
    # Build the system prompt for the model using the Blursday Codebook guidelines
    system_prompt = f"""You are a categorization assistant tasked with coding responses based on the 'Blursday Codebook'. The codebook (refer to 'Blursday Codebook' images) helps categorize individual reflections about time (past and future) according to a structured taxonomy/classification system. 
    For each response, follow these 6 steps:
    1. Identify all applicable categories for the response from the four categories: Temporal Landmark, Utilitarian Activities, Discretionary Activities, Evaluations.
    2. Rank the applicable categories from the previous step in order of relevance to the response (i.e. ordered from most to least relevant).
    3. Assign up to 3 activity codes/labels from the primary/most relevant category (identified in the previous step) based on the codebook's activity codes and their definitions and examples/keywords (e.g. ['Personal Temporal Landmark', 'Fact of Life Experiences'] under the 'Temporal Landmark' category).
    4. Flag any ambiguous or uncertain responses for further review.
    5. Provide a comment on the justification/reasoning for your choices, particularly for the flagged items.
    6. Ensure the output is in JSON format, with distinct columns for each data element: 'original_response', 'categories_selected', 'ranking', 'activity_labels', 'flagged', and 'justification_comments'."""
    # Dynamic user content based on whether chain_of_thought is enabled
    user_message = "Categorize the following responses based on the 'Blursday Codebook' and provide structured output as JSON."
    # Initialize the messages list for the OpenAI API call
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text",
                 "text": system_prompt}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": user_message}
            ]
        }
    ]
    # Append the base64-encoded codebook images
    for base64_image in base64_images:
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": "high"
            }
        })
    # Chunk the input text messages and add them to the user message
    for chunk in text_messages:
        messages[1]["content"].append({
            "type": "text",
            "text": chunk
        })
    if chain_of_thought:
        messages[1]["content"].append({
            "type": "text",
            "text": "Let's think step-by-step."
        })
    # API call to OpenAI for categorization (using a hypothetical beta API method for structured output)
    response = client.beta.chat.completions.parse(
        model=model_setting,
        messages=messages,
        response_format=QuestionExtraction,
        max_tokens=max_tokens_setting,
        temperature=temperature_setting,
        #stop=stop_setting,
        top_p=top_p_setting,
        presence_penalty=presence_penalty_setting,
        n=n_setting,
        frequency_penalty=frequency_penalty_setting,
        #logit_bias=logit_bias_setting,
        logprobs=logprobs_setting,
    )
    # Process and return the response content in structured format (JSON)
    return response


def extract_from_multiple_pages(text_messages, base64_images, output_filename):
    structured_json = categorize_blursday_responses(text_messages, base64_images, model_setting="gpt-4o-2024-08-06", temperature_setting=0.8)
    # Check if the result is None/empty or if it is NOT an instance of str, bytes, or bytearray
    structured_data = json.loads(structured_json.choices[0].message.content)
    # Convert the structured data into a pandas DataFrame
    structured_df = pd.DataFrame(structured_data['items'])
    structured_df.to_excel(output_filename, index=False)
    return output_filename


def batch_process_responses(df, column_name, base64_images, file_path, write_path, batch_size=32):
    """
    Processes text responses from a DataFrame column in batches using the extract_from_multiple_pages function.
    Parameters:
    - df (pd.DataFrame): DataFrame containing the responses to process.
    - column_name (str): The column name in the DataFrame with the text responses (e.g., 'Response_translated').
    - base64_images (list of str): A list of base64-encoded images to serve as reference for categorization.
    - file_path (str): File path for the input file (if needed by the extraction function).
    - write_path (str): File path where the output should be written to.
    - batch_size (int): The size of each batch to process. Default is 32.
    Returns:
    - None: The function processes the batches and writes the output directly.
    """
    # Ensure the output directory exists
    os.makedirs(write_path, exist_ok=True)
    # Get the list of responses from the specified column in the DataFrame
    responses = df[column_name].tolist()
    # Determine the number of batches based on the batch size
    total_batches = len(responses) // batch_size + (1 if len(responses) % batch_size > 0 else 0)
    # Loop through each batch
    for batch_number in range(total_batches):
        # Construct the output file path
        output_filename = os.path.join(write_path, file_path.replace('.pdf', f'_batch{batch_number}_extracted.xlsx'))
        # Calculate the start and end index for the current batch
        start_index = batch_number * batch_size
        end_index = start_index + batch_size
        # Slice the responses to get the current batch
        batch_text_messages = responses[start_index:end_index]
        # Call the extraction function with the current batch
        extract_from_multiple_pages(
            text_messages=batch_text_messages,
            base64_images=base64_images,
            output_filename=output_filename
        )
        # Optionally, print progress
        print(f"Processed batch {batch_number + 1}/{total_batches}")


##### ------ MAIN CODE ------- ####

# -- Step 1) Read in the original responses
folder_path = '/Users/bickley/stevejbickley/blursday_assistant/' # '/Users/stevenbickley/Library/CloudStorage/Dropbox/Project 2 - Temporal Landmarks/data/'
file_path = 'blursday_PastFluency-FutureFluency-TemporalLandmarks_2023-03-11_translated.csv' #'blursday_PastFluency-FutureFluency_2023-11-30_translated.csv'

# Read in the raw/full dataset
df = pd.read_csv(file_path, dtype=str) # just read the full dataset

# Apply the filter condition
df = df[~((df['Unique_Name'] == 'TemporalLandmark') & (df['Screen_Number'] != '2'))]

# -- Step 2) Convert the codebook pdf into base64 images
file_path = "./Blursday_Codebook_Instructions_Final.pdf"
base64_images = pdf_to_base64_images(file_path)

# -- Step 3) Run the codebook classification in batches
write_path = "./output_data/"
batch_process_responses(df, 'Response_translated', base64_images, file_path, write_path, batch_size=32)

######### ------ END OF NEW CLASSIFICATIONS CODE (No Assistants API - Directly using the Structured Outputs with Chat Completions API) ADDED ON 19 OCTOBER (UNUSED SO FAR) ------ #########

######### ------ START OF OLD CLASSIFICATIONS CODE (Assistants API with Structured Outputs via Chat Completions API added on 20 October) ------ #########

# Read and select the first n = number of rows (default = 10) from the file
def read_and_select_rows(file_path, number_of_rows=10):
    """
    Reads in a CSV file as a pandas dataframe and selects the first 'number_of_rows' rows.
    :param file_path: Path to the Excel file.
    :param number_of_rows: Number of rows to select from the dataframe.
    :return: A pandas dataframe with the specified number of rows.
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, dtype=str)
    # Select the first 'number_of_rows' rows
    selected_df = df.head(number_of_rows)
    return selected_df

# Chunk Messages into Batches of 32
def chunk_messages(messages, chunk_size=32):
    """Yield successive chunk_size chunks from messages."""
    for i in range(0, len(messages), chunk_size):
        yield messages[i:i + chunk_size]

# Function to parse the response and extract values
def parse_assistant_response(response,thread_id,run_id):
    parsed_data = {
        'raw_data': [],
        'categories': [],
        'ranking': [],
        'primary_codes': [],
        'secondary_codes': [],
        'flagged': [],
        'justification_comments': [],
        'dump': []  # Add a 'dump' key to store unparsed responses
    }
    # Store the entire response in 'dump'
    parsed_data['dump'].append(response)
    # Attempt to match patterns for each expected tag
    tags_found = {}
    for tag in ['raw_data', 'categories', 'ranking', 'primary_codes', 'secondary_codes', 'flagged',
                'justification_comments']:
        matches = re.findall(f"<{tag}>(.*?)</{tag}>", response)
        if matches:
            # Flatten the list of matches into the parsed_data dictionary
            parsed_data[tag].extend(matches)
            tags_found[tag] = True
        else:
            tags_found[tag] = False
    # Check if any of the tags were not found; if so, store the response in 'dump'
    #if not all(tags_found.values()):
    #    print('\n\nNot all tags were found:\n\n')
    #    print(tags_found)
    #    print('\n\nIn the following raw data:\n' + response + '\n\nFor thread ID: ' + thread_id + ' and run ID: ' + run_id +'\n')
    return parsed_data

# Function to check if the item is a list of lists (REDUNDANT)
def is_list_of_lists(item):
    return all(isinstance(elem, list) for elem in item) if isinstance(item, list) else False

# Function to expand the lists
def expand_lists_in_df(df, cols_to_expand):
    expanded_data = []
    for index, row in df.iterrows():
        # Determine the max length for expansion in this row, handling non-lists as length 1
        max_len = max((len(item) if isinstance(item, list) else 1 for item in row[cols_to_expand]), default=1)
        # Expand the row based on max_len
        for i in range(max_len):
            new_row = {}
            for col in df.columns:
                if col in cols_to_expand and isinstance(row[col], list):
                    # If it's a list, take the ith element if within range, otherwise None
                    new_row[col] = row[col][i] if i < len(row[col]) else None
                else:
                    new_row[col] = row[col]
            expanded_data.append(new_row)
    return pd.DataFrame(expanded_data)

# Function to parse and clean up the response from the 'dump' column
def clean_up_agent_dump(response_dump, thread_id, run_id, chunked_messages):
    parsed_data = {
        'thread_id': thread_id,
        'run_id': run_id,
        'raw_data': [],
        'categories': [],
        'ranking': [],
        'primary_codes': [],
        'secondary_codes': [],
        'flagged': [],
        'justification_comments': [],
        'dump': [response_dump]
    }
    # Define patterns for extracting different parts of the response
    #raw_data_pattern = r"content': '(.*?)'"
    #category_pattern = r'Categories:\*?\*?\s*(.*?)\\|$'
    #ranking_pattern = r'Ranking:\*?\*?\s*(.*?)\\|$'
    #primary_codes_pattern = r'Primary Codes:\*?\*?\s*(.*?)\\|$'
    #secondary_codes_pattern = r'Secondary Codes:\*?\*?\s*(.*?)\\|$'
    #flagged_pattern = r'Flagged:\*?\*?\s*(.*?)\\|$'
    #justification_pattern = r'Justification/Comments:\*?\*?\s*(.*?)\\|$'
    raw_data_pattern = r"content':\s*'(.*?)'"
    category_pattern = r'Categories:.*?\*?\*?\s*(.*?)\\'
    ranking_pattern = r'Ranking:.*?\*?\*?\s*(.*?)\\'
    primary_codes_pattern = r'Primary Codes:.*?\*?\*?\s*(.*?)\\'
    secondary_codes_pattern = r'Secondary Codes:.*?\*?\*?\s*(.*?)\\'
    flagged_pattern = r'Flagged:.*?\*?\*?\s*(.*?)\\'
    justification_pattern = r'Justification/Comments:.*?\*?\*?\s*(.*?)\\'
    # Define a list of possible tags and their corresponding patterns
    tag_patterns = {
        'raw_data': raw_data_pattern,
        'categories': category_pattern,
        'ranking': ranking_pattern,
        'primary_codes': primary_codes_pattern,
        'secondary_codes': secondary_codes_pattern,
        'flagged': flagged_pattern,
        'justification_comments': justification_pattern,
    }
    # Attempt to match patterns for each expected tag
    for tag, pattern in tag_patterns.items():
        if tag == "raw_data":
            regex_matches = re.findall(pattern, chunked_messages, re.DOTALL)
        else:
            regex_matches = re.findall(pattern, response_dump, re.DOTALL)
        if regex_matches:
            regex_clean = [re.sub(r'[\*\n_—-]', '', reg).strip() for reg in regex_matches]
            regex_clean = [re.findall(r'>\s*(.*?)\s*<', reg)[0] if re.search(r'>\s*(.*?)\s*<', reg) else reg for reg in regex_clean] # Extract text between ">" and "<" if present
            regex_clean = [re.findall(r'>\s*(.*?)', reg)[0] if re.search(r'>\s*(.*?)', reg) else reg for reg in regex_clean]  # Extract text after ">" if present
            parsed_data[tag].extend(regex_clean)
    # Print debug information (OPTIONAL)
    #print(f"Parsed data for thread_id {thread_id}, run_id {run_id}: {parsed_data}")
    return parsed_data # Add the thread and run IDs back into the parsed_data dictionary

# Function to parse the DataFrame
def parse_dataframe(df):
    # Iterate over the rows and parse the 'dump' column
    parsed_rows = []
    for iii in range(0,len(df)):
        parsed_response = clean_up_agent_dump(response_dump=df.iloc[iii,1], thread_id=df.iloc[iii,12], run_id=df.iloc[iii,13], chunked_messages=df.iloc[iii,11])
        parsed_rows.append(parsed_response)
    # Create a DataFrame from parsed rows
    parsed_df = pd.DataFrame(parsed_rows)
    # Expand lists in the DataFrame
    cols_to_expand = ['raw_data', 'categories', 'ranking', 'primary_codes', 'secondary_codes', 'flagged', 'justification_comments']
    expanded_df = expand_lists_in_parsed_df(parsed_df, cols_to_expand)
    return expanded_df

def expand_lists_in_parsed_df(df, cols_to_expand):
    expanded_data = []
    for index, row in df.iterrows():
        max_len = max((len(row[col]) if isinstance(row[col], list) else 1 for col in cols_to_expand), default=1)
        for i in range(max_len):
            new_row = {col: (
                row[col][i] if col in cols_to_expand and isinstance(row[col], list) and i < len(row[col]) else row[col])
                       for col in df.columns}
            expanded_data.append(new_row)
    # Convert expanded data to pandas dataframe
    expanded_df = pd.DataFrame(expanded_data)
    # Print debug information (OPTIONAL)
    #print(f"Expanded dataframe length: {len(expanded_df)}")
    return expanded_df

##### ------ DEFINE FUNCTIONS - END ------- ####

##### ------ MAIN CODE - START ------- ####
# Assuming you have a file path, you can customize the number of rows as needed.
#folder_path = '/Users/stevenbickley/stevejbickley/blursday_assistant/' # '/Users/stevenbickley/Library/CloudStorage/Dropbox/Project 2 - Temporal Landmarks/data/'
folder_path = '/Users/bickley/stevejbickley/blursday_assistant/' # '/Users/stevenbickley/Library/CloudStorage/Dropbox/Project 2 - Temporal Landmarks/data/'
file_path = 'blursday_PastFluency-FutureFluency-TemporalLandmarks_2023-03-11_translated.csv' #'blursday_PastFluency-FutureFluency_2023-11-30_translated.csv'
#df_custom_rows = read_and_select_rows(str(folder_path + file_path), 50) # Will return the first 50 rows of the dataframe.
df = pd.read_csv(file_path, dtype=str) # just read the full dataset

# Apply the filter condition
df = df[~((df['Unique_Name'] == 'TemporalLandmark') & (df['Screen_Number'] != '2'))]

# Filter to Canada
#df_custom_rows = df[ (df['Country_Name'] == 'Canada') | (df['Country_Name'] == 'United Kingdom') | (df['Country_Name'] == 'US')]
#df_custom_rows = df[ (df['Country_Name'] == 'US')]
df_custom_rows = df

# Create a list of messages to be added to the thread
messages_to_create = [
    {
        "role": "user",
        "content": str(response)  # Ensure the content is a string
    }
    for response in df_custom_rows['Response_translated']  # Iterate over each response
]

# Create a list of metainfo to be added to the thread
metainfo_to_create = [
    {
        "Experiment_ID": str(row['Experiment_ID']),
        "PID": str(row['PID']),  # Ensure the content is a string
        "UTC_Date": str(row['UTC_Date'])  # Ensure the content is a string
    }
    for index, row in df_custom_rows.iterrows()  # Iterate over each response
]

# Chunk Messages into Batches of 10
chunked_messages = list(chunk_messages(messages=messages_to_create, chunk_size=5)) # fine-tune/modify/adapt chunk_size depending on number of tokens in both system message and individual user messages
chunked_metainfo = list(chunk_messages(messages=metainfo_to_create,chunk_size=5))

# List to store created thread IDs and the associated chunked messages
created_thread_ids = []
linked_message_chunks = []
linked_metainfo_chunks = []
#created_run_ids = []

# Variable to store output file name
file_path = 'parsed_responses.xlsx'

# If file_name already exists load the existing_data into working memory
if os.path.exists(file_path):  # Check if 'parsed_responses.xlsx' exists
    # If yes, load the existing data from file_path
    existing_data = pd.read_excel(file_path)
    existing_data['thread_id'] = existing_data['thread_id'].astype(str)  # Enforce string data type
    existing_data['chunked_message'] = existing_data['chunked_message'].astype(str)  # Enforce string data type
    # existing_data.dtypes # Get data types of all columns

# Create Threads for Each Chunk
for i in range(0,len(chunked_messages)):
    chunk = chunked_messages[i]
    metainfo = chunked_metainfo[i]
    if os.path.exists(file_path): # Check if 'parsed_responses.xlsx' exists
        existing_data['chunked_message'] = existing_data['chunked_message'].astype(str)  # Enforce string data type
        if str(chunk) in list(existing_data['chunked_message'].unique()):  # Check if the current chunk exists in the 'chunked_message' column
            continue  # If yes, skip to the next iteration
        else:
            pass  # Does nothing, just acts as a placeholder
    thread = client.beta.threads.create(messages=chunk)
    created_thread_ids.append(thread.id)
    linked_message_chunks.append(chunk)
    linked_metainfo_chunks.append(metainfo)
    print(f"Thread Created: {thread.id}")

# Handling Multiple Threads
#for i in range(len(created_thread_ids)-1,0-1,-1): # backwards
for i in range(0,len(created_thread_ids),1): # forwards
    thread_id = created_thread_ids[i]
    chunk = linked_message_chunks[i]
    metainfo = linked_metainfo_chunks[i]
    if os.path.exists(file_path): # Check if 'parsed_responses.xlsx' exists
        try:
            existing_data = pd.read_excel(file_path) # If yes, load the existing data from file_path
        except:
            try:
                time.sleep(3)
                existing_data = pd.read_excel(file_path)
            except:
                try:
                    time.sleep(3)
                    existing_data = pd.read_excel(file_path)
                except:
                    continue
        existing_data['thread_id'] = existing_data['thread_id'].astype(str)  # Enforce string data type
        existing_data['chunked_message'] = existing_data['chunked_message'].astype(str)  # Enforce string data type
        # existing_data.dtypes # Get data types of all columns
        if str(chunk) in list(existing_data['chunked_message'].unique()): # Check if the current chunk exists in the 'chunked_message' column
            continue  # If yes, skip to the next iteration
        else:
            pass  # Does nothing, just acts as a placeholder
    # Submit the thread to the assistant
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=ASSISTANT_ID, model="gpt-3.5-turbo-0125", tools=[{"type": "file_search"}], temperature=float(0.7)) # "gpt-4-turbo"
    #run = client.beta.threads.create_and_run(assistant_id=ASSISTANT_ID, model="gpt-3.5-turbo-0125", tools=[{"type": "retrieval"}], temperature=float(0.7))
    print(f"👉 Run Created: {run.id}")
    time.sleep(2)
    run_failed = False  # A flag to indicate if the run failed
    # Wait for run to complete
    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(thread_id=run.thread_id, run_id=run.id)
        time.sleep(3)
        if run.status == "failed":
            print(f"Run failed on iteration: {i}")
            run_failed = True
            break  # Exit the while loop
        if run.status == "incomplete":
            print(f"Run incomplete on iteration: {i}")
            run_failed = True
            break  # Exit the while loop
        else:
            print(f"🏃 Run Status: {run.status}")
    # If run failed then skip to next thread
    if run_failed:
        continue  # Skip the remaining part of this 'for' loop iteration
    # Process each thread's messages as needed
    message_response = client.beta.threads.messages.list(thread_id=run.thread_id)
    messages = message_response.data
    # Parse the assistant's responses
    parsed_responses = [parse_assistant_response(messages[0].content[0].text.value, run.thread_id, run.id) for message in messages]
    # Filter out empty rows: check if all values in each dictionary are empty
    filtered_responses = [response for response in parsed_responses if not (all(value == [] or value == "" for key, value in response.items() if key != 'dump') and not any("Ranking" in s for s in response.get('dump', [])))]
    # Define a standard set of keys/columns you expect
    expected_keys = {'raw_data', 'categories', 'ranking', 'primary_codes', 'secondary_codes', 'flagged', 'justification_comments','dump'}
    # Ensure each dictionary has all the expected keys, insert None for missing keys
    standardized_responses = [{key: response.get(key, None) for key in expected_keys} for response in filtered_responses]
    # Now pass the standardized list to create a DataFrame
    df_responses = pd.DataFrame(standardized_responses)
    # Columns you want to expand
    cols_to_expand = ['raw_data', 'categories', 'ranking', 'primary_codes', 'secondary_codes', 'flagged', 'justification_comments','dump']
    # Use the function to expand the DataFrame
    df_responses = expand_lists_in_df(df_responses, cols_to_expand)
    # Assuming chunked_metainfo is your list of dictionaries and df_responses is your existing DataFrame
    chunked_metainfo_df = pd.DataFrame(metainfo)
    # Concatenate chunked_metainfo_df with df_responses
    df_responses = pd.concat([df_responses.reset_index(drop=True), chunked_metainfo_df.reset_index(drop=True)], axis=1)
    try:
        df_responses = df_responses.dropna(subset='dump') # Drop rows where any of the cells in 'dump' column have NaN values
    except:
        print('skipping thread_id: ' + str(run.thread_id))
        continue
    # Add the thread_id and run_id's back into it
    df_responses['chunked_message'] = str(chunk)
    df_responses['thread_id'] = str(run.thread_id)
    df_responses['run_id'] = str(run.id)
    df_responses['run_status'] = str(run.status)
    df_responses['run_timestamp_created'] = str(run.created_at) if run.created_at not in [None, ""] else "NaN"
    df_responses['run_timestamp_completed'] = str(run.completed_at) if run.completed_at not in [None, ""] else "NaN"
    df_responses['run_model_used'] = str(run.model) if run.model not in [None, ""] else "NaN"
    df_responses['run_completion_tokens'] = str(run.usage.completion_tokens) if run.usage.completion_tokens not in [None, ""] else "NaN"
    df_responses['run_prompt_tokens'] = str(run.usage.prompt_tokens) if run.usage.prompt_tokens not in [None, ""] else "NaN"
    df_responses['run_model_temperature'] = str(run.temperature) if run.temperature not in [None, ""] else "NaN"
    try: # Final variable to add with some exception handling
        df_responses['thread_timestamp_created'] = str(client.beta.threads.retrieve(thread_id=run.thread_id).created_at)
    except:
        df_responses['thread_timestamp_created'] = "NaN"
    # One final drop of duplicate rows
    df_responses = df_responses.drop_duplicates(subset=['run_completion_tokens', 'run_prompt_tokens', 'thread_id', 'chunked_message'])
    # Check if the file already exists
    if os.path.exists(file_path): # If True, append new data to the existing data
        try:
            existing_data = pd.read_excel(file_path) # If yes, load the existing data from file_path
        except:
            try:
                time.sleep(3)
                existing_data = pd.read_excel(file_path)
            except:
                try:
                    time.sleep(3)
                    existing_data = pd.read_excel(file_path)
                except:
                    continue
        # Open the existing Excel file
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            try:
                updated_df = pd.concat([existing_data, df_responses], ignore_index=True) # Append new data to existing data
                updated_df.to_excel(writer, index=False, sheet_name='Sheet1') # Write the updated DataFrame to the Excel file
            except Exception as e:  # Handle cases where the existing file is empty or has a different structure
                print('skipping thread_id: '+ str(run.thread_id))
                continue
    else:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer: # If False, create a new Excel file
            df_responses.to_excel(writer, index=False)
    time.sleep(2)

######### ------ END OF OLD CLASSIFICATIONS CODE (Assistants API with Structured Outputs via Chat Completions API added on 20 October) ------ #########

######### ------ START OF NEW CLEAN-UP CODE (Structured Outputs via Chat Completions API added on 22 October) ------ #########

##### ------ DEFINE FUNCTIONS - START ------- ####

# Define BASIC item schema for coding responses
class ItemBasic(BaseModel):
    original_response: str
    categories_selected: list[str]
    ranking: list[str]
    primary_codes: list[str]
    secondary_codes: list[str]
    flagged: bool
    justification_comments: str


# List of BASIC item schema for coding multiple responses
class QuestionExtraction(BaseModel):
    items: list[ItemBasic]


def categorize_blursday_responses_standardized(input_df, base64_images, temperature_setting=0.2,
                                               max_tokens_setting=None, top_p_setting=1, presence_penalty_setting=0,
                                               n_setting=1, frequency_penalty_setting=0, logprobs_setting=False,
                                               model_setting="gpt-4o-mini", chain_of_thought=True):
    """
    Categorizes text responses based on the 'Blursday Codebook' using GPT-like models and restructures the output to be standardized.
    Each response (e.g., 'cut my hair') will be broken down into its relevant components ('original_response', 'categories', etc.) into distinct rows.
    Parameters:
    - input_df (dataframe): A pandas dataframe containing the raw responses of text-based responses to be categorized/standardized.
    - base64_codebook_images (list of str): A list of base64-encoded codebook images to serve as reference for the categorization task.
    - temperature_setting (float): Temperature setting for model creativity.
    - model_setting (str): The model to be used for the task.
    - chain_of_thought (bool): Whether to include chain-of-thought reasoning in the response.
    Returns:
    - list of dicts: A list where each dict contains the standardized output for a single 'original_response'.
    """
    # Build the system prompt for the model using the Blursday Codebook guidelines
    system_prompt = f"""You are a categorization assistant tasked with coding responses based on the 'Blursday Codebook'. The codebook (refer to 'Blursday Codebook' images) helps categorize individual reflections about time (past and future) according to a structured taxonomy/classification system. 
    For each response, follow these 6 steps:
    1. Identify all applicable categories for the response from the four categories: Temporal Landmark, Utilitarian Activities, Discretionary Activities, Evaluations.
    2. Rank the applicable categories from the previous step in order of relevance to the response (i.e., ordered from most to least relevant), creating a list that looks something like [1 (Utilitarian Activities), 2 (Evaluations)], for example.
    3. Assign up to 3 activity codes/labels from the primary/most relevant category (identified in the previous step) based on the codebook's activity codes and their definitions and examples/keywords.
    4. Assign up to 3 activity codes/labels from the secondary/second most relevant category (identified in the previous step, if any) based on the codebook's activity codes and their definitions and examples/keywords.
    5. Flag any ambiguous or uncertain responses for further review.
    6. Provide a comment on the justification/reasoning for your choices, particularly for the flagged items.
    7. Ensure the output is in a standardized format where each 'original_response' is placed in its own row, with 'categories_selected', 'ranking', 'primary_codes', 'secondary_codes', 'flagged', and 'justification_comments' provided."""
    # Dynamic user content based on whether chain_of_thought is enabled
    user_message = "Categorize the following responses based on the 'Blursday Codebook' and provide structured output for each response in the format specified."
    # Locate the "dump" column
    text_messages = input_df.iloc[:, 1]
    thread_ids = input_df.iloc[:, 12]
    run_ids = input_df.iloc[:, 13]
    chunked_messages = input_df.iloc[:, 11]
    # Prepare a list to hold results along with the metadata for each response
    results_with_metadata = []
    # Chunk the input text messages and add them to the user message
    for i, chunk in enumerate(text_messages):
        # Initialize the messages list for the OpenAI API call
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message}
                ]
            }
        ]
        # Append the base64-encoded codebook images
        for base64_image in base64_images:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high"
                }
            })
        # Add the actual chunk itself
        messages[1]["content"].append({
            "type": "text", "text": chunk
        })
        # If chain_of_thought is enabled, add the prompt for step-by-step thinking
        if chain_of_thought:
            messages[1]["content"].append({
                "type": "text", "text": "Let's think step-by-step."
            })
        # API call to OpenAI for categorization (hypothetical beta API method for structured output)
        response = client.beta.chat.completions.parse(
            model=model_setting,
            messages=messages,
            response_format=QuestionExtraction,
            max_tokens=max_tokens_setting,
            temperature=temperature_setting,
            top_p=top_p_setting,
            presence_penalty=presence_penalty_setting,
            n=n_setting,
            frequency_penalty=frequency_penalty_setting,
            logprobs=logprobs_setting,
        )
        # Add the metadata along with the response to the final output
        result_entry = {
            "thread_id": thread_ids.iloc[i],
            "run_id": run_ids.iloc[i],
            "chunked_message": chunked_messages.iloc[i],
            "response": response  # Raw or processed response depending on your use case
        }
        # Append to the results list
        results_with_metadata.append(result_entry)
    # Return the results, with each entry containing the response and associated metadata
    return results_with_metadata


def extract_from_standardized(input_df, base64_images, output_filename):
    structured_json = categorize_blursday_responses_standardized(input_df, base64_images, model_setting="gpt-4o-2024-08-06", temperature_setting=0.8)
    # Initialise the output 'rows' list
    rows=[]
    # For each of the returned jsons
    for item in structured_json:
        # Check if the result is None/empty or if it is NOT an instance of str, bytes, or bytearray
        structured_data = json.loads(item['response'].choices[0].message.content)
        # Convert the structured data into a pandas DataFrame
        structured_df = pd.DataFrame(structured_data['items'])
        # Add the linking keys back to the structured_df
        structured_df['thread_id'] = item['thread_id']
        structured_df['run_id'] = item['run_id']
        structured_df['chunked_message'] = item['chunked_message']
        structured_df['chat_completion_id'] = item['response'].id
        structured_df['chat_completion_created'] = item['response'].created
        structured_df['chat_completion_model'] = item['response'].model
        structured_df['chat_completion_system_fingerprint'] = item['response'].system_fingerprint
        structured_df['chat_completion_completion_tokens'] = item['response'].usage.completion_tokens
        structured_df['chat_completion_prompt_tokens'] = item['response'].usage.prompt_tokens
        structured_df['chat_completion_total_tokens'] = item['response'].usage.total_tokens
        rows.append(structured_df)
    #rows = np.squeeze(rows)  # This will remove the extra dimension if it's redundant
    #output_df = pd.DataFrame(rows)
    # Convert the list of dataframes to a single dataframe
    output_df = pd.concat(rows, ignore_index=True)  # Concatenates the list of dataframes into one
    output_df.to_excel(output_filename, index=False)
    return output_df


def add_category_and_ranking_columns(df):
    # Define patterns for categories
    category_patterns = {
        'category_utilitarian': r'(?i)utilitarian',
        'category_discretionary': r'(?i)discretionary',
        'category_landmark': r'(?i)temporal landmark',
        'category_evaluation': r'(?i)evaluation',
    }
    # Add category columns based on category patterns (set to 1 if present, else 0)
    for col, pattern in category_patterns.items():
        df[col] = df['categories_selected'].apply(
            lambda x: 1 if any(re.search(pattern, str(item)) for item in (x if isinstance(x, list) else [x])) else 0
        )
    # Define patterns for ranking, including capturing the number before the category name
    ranking_patterns = {
        'ranking_utilitarian': r'(?i)(\d+)\s*\(utilitarian',
        'ranking_discretionary': r'(?i)(\d+)\s*\(discretionary',
        'ranking_landmark': r'(?i)(\d+)\s*\(temporal landmark',
        'ranking_evaluation': r'(?i)(\d+)\s*\(evaluation',
    }
    # Helper function to get the ranking value
    def get_ranking_value(text, pattern):
        # Find all matches of the pattern in the text
        matches = re.search(pattern, text)
        if matches:
            # If there are multiple matches, return the first (or handle as needed)
            return int(matches.group(1))  # We assume the first match is the correct one to use
        return 0  # Return 0 if no match is found
    # Extract rankings for each category based on the pattern
    for col, pattern in ranking_patterns.items():
        df[col] = df['ranking'].apply(lambda x: get_ranking_value(str(x), pattern))
    # Add missing columns for categories and rankings
    df['category_missing'] = df['categories_selected'].apply(
        lambda x: 1 if not x or (isinstance(x, str) and x.strip() == '') or (isinstance(x, list) and len(x) == 0) else 0
    )
    df['ranking_missing'] = df['ranking'].apply(
        lambda x: 1 if not x or (isinstance(x, str) and x.strip() == '') or (isinstance(x, list) and len(x) == 0) else 0
    )
    # Define a function to capture any other rankings that are not in the main categories
    def get_ranking_other(row):
        if row['ranking_missing'] == 1:
            return 0
        rankings = str(row['ranking']).split(',')
        other_rankings = []
        for i, rank in enumerate(rankings):
            rank = rank.strip()
            if not any(re.search(pattern, rank) for pattern in ranking_patterns.values()):
                other_rankings.append(int(i))
        if other_rankings:
            return other_rankings
        else:
            return 0
    # Add other ranking and category logic
    df['ranking_other'] = df.apply(get_ranking_other, axis=1)
    # NOTE.. below code needs to be updated to include counts of number of "other" categories rather than just binary variable..
    df['category_other'] = df.apply(lambda row: len(row['ranking_other']) if row['ranking_other'] != 0 else 0, axis=1)
    return df


# Define a list of valid codes based on the tables from the PDF (pages 5 to 8)
valid_primary_codes = [
    # Add all relevant primary codes here, as per the codebook in the PDF
    'personal_temporal_landmark', 'calendar_temporal_landmark', 'reference_points',
    'personal_narrative_events', 'facts_of_life_experiences', 'household_obligations',
    'physiological_needs_personal_care', 'work_school_activities', 'services',
    'care_duties', 'return_to_routine', 'career_planning', 'civic_duties',
    'recreation', 'entertainment', 'social', 'family', 'altruistic', 'aspirational',
    'recreation_services', 'shopping', 'introspection', 'home_improvement', 'travel',
    'self_improvement', 'new_connection_seeking', 'spiritual_activities',
    'negative', 'neutral', 'positive', 'lockdown'
]


def add_primary_code_columns(df, primary_or_secondary='primary'):
    # Add primary_XXXX columns based on primary_code patterns (set to 1 if present, else 0)
    for code in valid_primary_codes:
        pattern = re.compile(fr'(?i){code}')  # Pre-compile the regex pattern for efficiency
        df[f'{primary_or_secondary}_{code}'] = df[f'{primary_or_secondary}_codes'].apply(lambda x: 1 if any(pattern.search(str(item)) for item in (x if isinstance(x, list) else [x])) else 0)
    # Add column for missing primary_codes
    df[f'{primary_or_secondary}_missing'] = df[f'{primary_or_secondary}_codes'].apply(lambda x: 1 if not x or (isinstance(x, str) and x.strip() == '') or (isinstance(x, list) and len(x) == 0) else 0)
    # Define a function to capture any other activity codes/labels that are not in the defined set in valid_primary_codes
    def get_codes_other(row, primary_or_secondary):
        if row[f'{primary_or_secondary}_missing'] == 1:
            return 0
        codes = str(row[f'{primary_or_secondary}_codes']).split(',')
        other_codes = 0
        for code in codes:
            code = code.strip()
            if not any(re.search(fr'(?i){pattern}', code) for pattern in valid_primary_codes):
                other_codes += 1
        return other_codes
    # Add other ranking and category logic
    df[f'{primary_or_secondary}_other'] = df.apply(get_codes_other, primary_or_secondary=primary_or_secondary, axis=1)
    return df

##### ------ DEFINE FUNCTIONS - END ------- ####

##### ------ MAIN CODE - START ------- ####
# Read in the "raw" coded data
file_path = 'parsed_responses.xlsx' #'blursday_PastFluency-FutureFluency_2023-11-30_translated.csv'
raw_df = pd.read_excel(file_path, dtype=str) # just read the full dataset

# Use the 'extract_from_standardized' function to standardise the raw dataset
write_path = "./output_data/parsed_responses_clean.xlsx"
clean_df = extract_from_standardized(raw_df, base64_images, write_path)

# Apply the function to add new columns to clean_df
clean_df = add_category_and_ranking_columns(clean_df)

# Now write the clean data out to xlsx file
#with pd.ExcelWriter("./output_data/"+file_path[:-5]+"_clean_wide.xlsx", engine='openpyxl') as writer:
#    clean_df.to_excel(writer, index=False)

# Apply the function to add new columns to clean_df
clean_df = add_primary_code_columns(clean_df,'primary')
clean_df = add_primary_code_columns(clean_df,'secondary')

# Now write the clean data out to xlsx file
with pd.ExcelWriter("./output_data/"+file_path[:-5]+"_clean_wide.xlsx", engine='openpyxl') as writer:
    clean_df.to_excel(writer, index=False)


######### ------ START OF NEW CLEAN-UP CODE (Structured Outputs via Chat Completions API added on 22 October) ------ #########

# This function takes a DataFrame, a list of columns, and an optional parameter top_n (default is 20).
# It calculates the top n unique counts for each specified column and prints them.
def detailed_summary_statistics(df, columns, top_n=20):
    summary_stats = {}
    for column in columns:
        # Summary statistics
        col_summary = df[column].describe(include='all')
        summary_stats[column] = col_summary
        # Top N unique values
        top_counts = df[column].value_counts().head(top_n)
        summary_stats[f'top_{top_n}_{column}'] = top_counts
        # Print summary statistics
        print(f"Summary statistics for {column}:\n", col_summary, "\n")
        print(f"Top {top_n} unique values for {column}:\n", top_counts, "\n")
        # Get the value counts for the column
        #value_counts = df[column].value_counts().head(top_n)
        #summary_stats[column] = value_counts
        # Print the top n counts
        #print(f"Top {top_n} counts for column '{column}':")
        #print(value_counts)
        print("\n")
    return summary_stats

# Assuming clean_df is already loaded
columns_to_summarize = ['categories', 'ranking', 'primary_codes', 'secondary_codes', 'flagged', 'justification_comments']

# Get detailed summary statistics
summary_stats = detailed_summary_statistics(clean_df, columns_to_summarize, top_n=30)

# Function to plot histograms for specified columns
import matplotlib.pyplot as plt
def plot_histograms(df, columns):
    for column in columns:
        plt.figure(figsize=(10, 6))
        df[column].value_counts().plot(kind='bar')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Columns to plot histograms for
columns_to_plot = ['categories', 'ranking', 'primary_codes', 'secondary_codes']

# Plot histograms
plot_histograms(clean_df, columns_to_plot)
