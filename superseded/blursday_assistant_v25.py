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
file_path = 'parsed_responses.xlsx'
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
with pd.ExcelWriter("./output_data/"+file_path[:-5]+"_clean_wide_for_analysis.xlsx", engine='openpyxl') as writer:
    clean_df.to_excel(writer, index=False)


######### ------ START OF NEW CLEAN-UP CODE (Structured Outputs via Chat Completions API added on 22 October) ------ #########

# Read in the "raw" coded data
file_path = './output_data/parsed_responses_clean_wide_for_analysis.xlsx' #'blursday_PastFluency-FutureFluency_2023-11-30_translated.csv'
analysis_df = pd.read_excel(file_path, dtype=str) # just read the full dataset

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
columns_to_summarize = ['categories_selected', 'ranking', 'primary_codes', 'secondary_codes', 'flagged', 'justification_comments']

# Get detailed summary statistics
summary_stats = detailed_summary_statistics(analysis_df, columns_to_summarize, top_n=30)

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
#columns_to_plot = ['categories_selected', 'ranking', 'primary_codes', 'secondary_codes']
columns_to_plot = ['category_utilitarian','category_discretionary', 'category_landmark', 'category_evaluation']

# Plot histograms
plot_histograms(analysis_df, columns_to_plot)

#### NEW ANALYSIS CODES (translating the R code into python code)

# Import packages required for analysis and visualisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score  # For reliability calculations
from statsmodels.stats.inter_rater import fleiss_kappa  # Alternative library for Fleiss' Kappa
from openpyxl import load_workbook  # For working with Excel files
import re
import gc

# Load the analysis dataset
file_path = './output_data/parsed_responses_clean_wide_for_analysis.xlsx' #'blursday_PastFluency-FutureFluency_2023-11-30_translated.csv'
analysis_df = pd.read_excel(file_path, dtype=str) # just read the full dataset
analysis_df["original_response_lower"] = analysis_df["original_response"].str.lower() # convert the original (translated) responses to all lower case

# Load datasets
currwd = './'  # Set your working directory path here

# Load main CSV files
parsed_responses = pd.read_excel(f"{currwd}parsed_responses.xlsx", dtype=str)
blursday = pd.read_csv(f"{currwd}blursday_PastFluency-FutureFluency-TemporalLandmarks_2023-03-11_translated.csv", dtype=str)

# Convert the UTC_Date column to datetime format, handling errors by setting invalid formats to NaT
parsed_responses['UTC_Date'] = pd.to_datetime(parsed_responses['UTC_Date'], errors='coerce')

# Create a new column UTC_yyyymmdd in the format yyyymmdd
parsed_responses['UTC_yyyymmdd'] = parsed_responses['UTC_Date'].dt.strftime('%Y%m%d')

# Apply the filter condition
blursday = blursday[~((blursday['Unique_Name'] == 'TemporalLandmark') & (blursday['Screen_Number'] != '2'))]
blursday["Response_translated_lower"] = blursday["Response_translated"].str.lower() # convert the original (translated) responses to all lower case

# Convert the UTC_Date column to datetime format, handling errors by setting invalid formats to NaT
blursday['UTC_Date'] = pd.to_datetime(blursday['UTC_Date'], errors='coerce')

# Create a new column UTC_yyyymmdd in the format yyyymmdd
blursday['UTC_yyyymmdd'] = blursday['UTC_Date'].dt.strftime('%Y%m%d')

# Fill NaN values with a placeholder in both dataframes
parsed_responses['UTC_yyyymmdd'].fillna("unknown", inplace=True)
blursday['UTC_yyyymmdd'].fillna("unknown", inplace=True)

# Temporary placeholder columns for 'PID','UTC_Date'
blursday['PID_tempholder'] = blursday['PID']
blursday['UTC_Date_tempholder'] = blursday['UTC_Date']
blursday['UTC_yyyymmdd_tempholder'] = blursday['UTC_yyyymmdd']
blursday['Experiment_ID_tempholder'] = blursday['Experiment_ID']

# Merge datasets
analysis_df = pd.merge(analysis_df, parsed_responses[['Experiment_ID','PID','UTC_Date','UTC_yyyymmdd','thread_id','run_id']], on=['thread_id', 'run_id'], how='left') # Columns not merged from parsed_responses: 'run_status', 'run_timestamp_created', 'run_timestamp_completed', 'run_model_used', 'run_completion_tokens', 'run_prompt_tokens', 'run_model_temperature', 'thread_timestamp_created'
analysis_df = pd.merge(analysis_df, blursday[['Experiment_ID','Experiment_ID_tempholder','PID_tempholder','UTC_Date_tempholder','UTC_yyyymmdd_tempholder','UTC_yyyymmdd','Country','Session','Task_Name','Task_Version','Screen_Number','Event_Index','Attempt','Reaction_Time','Reaction_Onset','Participant_OS','Participant_Browser','Handedness','Sex','Age','Stringency_Index','Mobility_Transit','Mobility_Retail','Mobility_Parks','Mobility_WorkPlaces','Mobility_Residential','Reported_Loneliness','Felt_Loneliness','Subjective_Confinement','ConfDuration','Response_translated','Response_translated_lower']], left_on=['Experiment_ID','UTC_yyyymmdd','original_response_lower'], right_on=['Experiment_ID','UTC_yyyymmdd','Response_translated_lower'], how='outer', indicator=True)  # Add the merge indicator column

# Extract rows from blursday that did not match
blursday_nomatch = analysis_df[analysis_df['_merge'] == 'right_only']
blursday_nomatch = blursday_nomatch.drop(columns=['_merge']) # Optional: Drop the indicator column since we don't need it
blursday_nomatch = blursday_nomatch.drop_duplicates(subset=['Experiment_ID', 'PID', 'UTC_yyyymmdd', 'Response_translated_lower', 'Session','Task_Name','Task_Version','Screen_Number'])
blursday_nomatch = blursday_nomatch.drop(columns=['Experiment_ID', 'PID', 'UTC_Date', 'UTC_yyyymmdd','original_response', 'categories_selected', 'ranking', 'primary_codes', 'secondary_codes', 'flagged', 'justification_comments', 'thread_id', 'run_id', 'chunked_message', 'chat_completion_id', 'chat_completion_created', 'chat_completion_model', 'chat_completion_system_fingerprint', 'chat_completion_completion_tokens', 'chat_completion_prompt_tokens', 'chat_completion_total_tokens', 'category_utilitarian', 'category_discretionary', 'category_landmark', 'category_evaluation', 'ranking_utilitarian', 'ranking_discretionary', 'ranking_landmark', 'ranking_evaluation', 'category_missing', 'ranking_missing', 'ranking_other', 'category_other', 'primary_personal_temporal_landmark', 'primary_calendar_temporal_landmark', 'primary_reference_points', 'primary_personal_narrative_events', 'primary_facts_of_life_experiences', 'primary_household_obligations', 'primary_physiological_needs_personal_care', 'primary_work_school_activities', 'primary_services', 'primary_care_duties', 'primary_return_to_routine', 'primary_career_planning', 'primary_civic_duties', 'primary_recreation', 'primary_entertainment', 'primary_social', 'primary_family', 'primary_altruistic', 'primary_aspirational', 'primary_recreation_services', 'primary_shopping', 'primary_introspection', 'primary_home_improvement', 'primary_travel', 'primary_self_improvement', 'primary_new_connection_seeking', 'primary_spiritual_activities', 'primary_negative', 'primary_neutral', 'primary_positive', 'primary_lockdown', 'primary_missing', 'primary_other', 'secondary_personal_temporal_landmark', 'secondary_calendar_temporal_landmark', 'secondary_reference_points', 'secondary_personal_narrative_events', 'secondary_facts_of_life_experiences', 'secondary_household_obligations', 'secondary_physiological_needs_personal_care', 'secondary_work_school_activities', 'secondary_services', 'secondary_care_duties', 'secondary_return_to_routine', 'secondary_career_planning', 'secondary_civic_duties', 'secondary_recreation', 'secondary_entertainment', 'secondary_social', 'secondary_family', 'secondary_altruistic', 'secondary_aspirational', 'secondary_recreation_services', 'secondary_shopping', 'secondary_introspection', 'secondary_home_improvement', 'secondary_travel', 'secondary_self_improvement', 'secondary_new_connection_seeking', 'secondary_spiritual_activities', 'secondary_negative', 'secondary_neutral', 'secondary_positive', 'secondary_lockdown', 'secondary_missing', 'secondary_other', 'original_response_lower'])

# Drop duplicates in the specified columns
analysis_df = analysis_df[(analysis_df['_merge'] == 'left_only') | (analysis_df['_merge'] == 'both')] # Extract rows from analysis_df so it is like a left join
analysis_df = analysis_df.drop(columns=['_merge']) # Optional: Drop the indicator column since we don't need it
analysis_df = analysis_df.drop_duplicates(subset=['Experiment_ID', 'PID', 'UTC_yyyymmdd', 'Response_translated_lower', 'Session','Task_Name','Task_Version','Screen_Number'])

# Fix up the temporary placeholder columns
#analysis_df['UTC_yyyymmdd_temp'] = analysis_df['UTC_yyyymmdd'] # Reassign the dropped columns
#analysis_df['PID_temp'] = analysis_df['PID'] # Reassign the dropped columns
#analysis_df['Experiment_ID_temp'] = analysis_df['Experiment_ID'] # Reassign the dropped columns
analysis_df = analysis_df.drop(columns=['PID','UTC_Date','UTC_yyyymmdd','Experiment_ID']) # Drop the old/faulty columns
analysis_df['PID'] = analysis_df['PID_tempholder'] # Reassign the dropped columns
analysis_df['UTC_Date'] = analysis_df['UTC_Date_tempholder']
analysis_df['UTC_yyyymmdd'] = analysis_df['UTC_yyyymmdd_tempholder']
analysis_df['Experiment_ID'] = analysis_df['Experiment_ID_tempholder']
analysis_df = analysis_df.drop(columns=['PID_tempholder','UTC_Date_tempholder','UTC_yyyymmdd_tempholder','Experiment_ID_tempholder']) # no longer need the temporary columns after this so we drop them

# (Optional) Write the resulting dataframe out to csv
#analysis_df.to_csv("main_data_for_analysis.csv", index=False)

# Remove specific variables or DataFrames
del blursday, blursday_nomatch, file_path, parsed_responses
gc.collect()

# Load supplementary CSV files
ffa_participant = pd.read_csv(f"{currwd}data-FFA-ParticipantTrack-ParticipantTracker_France-Argentina-Canada-Colombia-Germany-Greece-India-Italy-Japan-Turkey-United Kingdom-US_2024-02-26.csv", dtype=str) # Walach, H., Buchheld, N., Buttenmüller, V., Kleinknecht, N., & Schmidt, S. (2006). Measuring mindfulness—The Freiburg mindfulness inventory (FMI). Personality and Individual Differences, 40(8), 1543–1555.
bfi_delay = pd.read_csv(f"{currwd}data-BFI-DelayDiscount_France-Argentina-Canada-Colombia-Germany-Greece-India-Italy-Japan-Turkey-United Kingdom-US_2024-02-26.csv", dtype=str) # Rammstedt, B., & John, O. P. (2007). Measuring personality in one minute or less: A 10-item short version of the Big Five Inventory in English and German. Journal of Research in Personality, 41(1), 203–212.
retro_duration = pd.read_csv(f"{currwd}data-RetroDuration-ZTPI_France-Argentina-Canada-Colombia-Germany-Greece-India-Italy-Japan-Turkey-United Kingdom-US_2024-02-26.csv", dtype=str) # Assesses a participant retrospective duration estimation (hours: minutes: seconds) and passage of time (over days). This test was randomly displayed to the participant during he first round of questionnaires.

# Select columns from each dataset
ffa_participant_selected = ffa_participant[['Country', 'Session', 'PID', 'Task_Name', 'Task_Version', 'Question_Key', 'Response']] # Noting, there is no Screen_Number in this dataset
bfi_delay_selected = bfi_delay[['Country', 'Session', 'PID', 'Task_Name', 'Task_Version', 'Question_Key', 'Screen_Number', 'Response']]
retro_duration_selected = retro_duration[['Country', 'Session', 'PID', 'Task_Name', 'Task_Version', 'Question_Key', 'Response']] # Noting, there is no Screen_Number in this dataset

# Define a function to transform a dataframe into wide format
def transform_to_wide(df, index_columns=['Country', 'Session', 'PID'], columns=['Question_Key'], value_column='Response', aggfunc='first'):
    return df.pivot_table(
        index=index_columns,
        columns=columns,
        values=value_column,
        aggfunc=aggfunc
    ).reset_index()

# Transform each dataframe to wide format
ffa_participant_wide = transform_to_wide(ffa_participant_selected)
bfi_delay_wide = transform_to_wide(bfi_delay_selected)
retro_duration_wide = transform_to_wide(retro_duration_selected)

# Now merge in the other datasets ffa_participant, bfi_delay, retro_duration, ...
analysis_df = pd.merge(analysis_df, ffa_participant_wide, on=['Country', 'Session', 'PID'], how='left', suffixes=('', '_ffa'))
analysis_df = pd.merge(analysis_df, bfi_delay_wide, on=['Country', 'Session', 'PID'], how='left', suffixes=('', '_bfi'))
analysis_df = pd.merge(analysis_df, retro_duration_wide, on=['Country', 'Session', 'PID'], how='left', suffixes=('', '_retro_duration'))

# Remove specific variables or DataFrames
del ffa_participant, bfi_delay, retro_duration, ffa_participant_selected, bfi_delay_selected, retro_duration_selected, ffa_participant_wide, bfi_delay_wide, retro_duration_wide, transform_to_wide
gc.collect()

# Check length of analysis_df
len(analysis_df) # 68799 rows

# Drop rows where either the 'Screen_Number' or 'Task_Name' columns are NaN
analysis_df = analysis_df.dropna(
    subset=["Screen_Number", "Task_Name"]) # 66762 rows after this line

# Create a new column 'Task_Name_clean' based on specific keywords in 'Task_Name'
analysis_df['Task_Name_clean'] = analysis_df['Task_Name'].apply(
    lambda x: 'Future Fluency' if 'Future Fluency' in x else
              'Past Fluency' if 'Past Fluency' in x else
              'Temporal Landmark' if 'Temporal Landmark' in x else x
)




#### --- Fig. 1 - Total frequency/counts of activity categories across past/future fluency and temporal landmark tasks in Task_Name and projection timeline in Screen_Number

# Drop rows where 'Screen_Number' or other important columns are NaN
analysis_data = analysis_df.dropna(
    subset=["Screen_Number", "Task_Name_clean", "category_utilitarian", "category_discretionary", "category_landmark",
            "category_evaluation", "category_other"])

# Check length of analysis_data
len(analysis_data) # 66762 rows

# Filter out rows where 'Screen_Number' is not 1, 2, or 3
analysis_data = analysis_data[analysis_data["Screen_Number"].isin(["1", "2", "3"])] # still 66762 rows

# Filter out the "Temporal Landmark" tasks
analysis_data = analysis_data[analysis_data['Task_Name_clean'] != "Temporal Landmark"] # 63750 rows after this line

# Filter columns that end with '_max' and find rows where any '_max' value is above 40
max_columns = [col for col in analysis_data.columns if col.endswith('_max')]

# Exclude rows where any of the '_max' columns have values above 40
analysis_data = analysis_data[~(analysis_data[max_columns] > 40).any(axis=1)]

# Select the columns to sum
columns_to_sum = ["category_utilitarian", "category_discretionary", "category_landmark", "category_evaluation", "category_other"]

# Convert the selected columns to numeric, coercing any errors to NaN
for column in columns_to_sum:
    analysis_data[column] = pd.to_numeric(analysis_data[column], errors='coerce')

# Group by 'Screen_Number', 'Task_Name_clean', and 'PID' to get counts for each 'PID'
pid_counts = analysis_data.groupby(["Screen_Number", "Task_Name_clean", "PID"])[columns_to_sum].sum().reset_index()

# Calculate the minimum, maximum, mean, and standard deviation for each Screen_Number and Task_Name_clean
summary_stats = pid_counts.groupby(["Screen_Number", "Task_Name_clean"])[columns_to_sum].agg(['min', 'max', 'mean', 'std']).reset_index()

# Flatten the MultiIndex columns created by aggregation for easier readability
summary_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary_stats.columns.values]

# Group by 'Screen_Number' and 'Task_Name_clean' again to get the total sums
grouped_df = analysis_data.groupby(["Screen_Number", "Task_Name_clean"])[columns_to_sum].sum().reset_index()

# Merge the summary statistics into grouped_df
grouped_df = pd.merge(grouped_df, summary_stats, on=["Screen_Number", "Task_Name_clean"], how="left")

grouped_df['Screen_Number'] = pd.to_numeric(grouped_df['Screen_Number'], errors='coerce')

for i in range(0,len(grouped_df)):
    if grouped_df.iloc[i,1] == 'Past Fluency':
        grouped_df.iloc[i, 0] = -grouped_df.iloc[i, 0]

#grouped_df['Screen_Number'] = grouped_df['Screen_Number'].astype(str)

# Separate the data for Past Fluency and Future Fluency
#past_fluency_data = grouped_df[grouped_df["Task_Name_clean"] == "Past Fluency"].set_index("Screen_Number")[columns_to_sum]
#future_fluency_data = grouped_df[grouped_df["Task_Name_clean"] == "Future Fluency"].set_index("Screen_Number")[columns_to_sum]

# Create a dictionary to map old column names to new names with "_count" suffix
rename_dict = {col: f"{col}_count" for col in columns_to_sum}

# Rename the columns in the DataFrame
grouped_df = grouped_df.rename(columns=rename_dict)
#past_fluency_data = past_fluency_data.rename(columns=rename_dict)
#future_fluency_data = future_fluency_data.rename(columns=rename_dict)



# Set up a grid of subplots by Task_Name
task_names = grouped_df["Task_Name_clean"].unique()
num_tasks = len(task_names)
fig, axes = plt.subplots(num_tasks, 1, figsize=(10, num_tasks * 5), sharex=True)

# Plot each 'Task_Name' in a separate subplot
for i, task in enumerate(task_names):
    task_data = grouped_df[grouped_df["Task_Name_clean"] == task]
    task_data = task_data.set_index("Screen_Number")[
        ["category_utilitarian_count", "category_discretionary_count", "category_landmark_count", "category_evaluation_count",
         "category_other_count"]]
    # Plot grouped bar plot
    task_data.plot(kind="bar", ax=axes[i], width=0.7)
    axes[i].set_title(f"Frequency Count by Category for Task: {task}")
    axes[i].set_ylabel("Count")
    axes[i].legend(title="Category")

# Set common x-label
axes[-1].set_xlabel("Screen Number")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()



# Merge the task data into a single DataFrame and set the 'Screen_Number' as the x-axis
task_data_combined = grouped_df.pivot_table(
    index="Screen_Number",
    columns="Task_Name_clean",
    values=["category_utilitarian_count", "category_discretionary_count", "category_landmark_count", "category_evaluation_count", "category_other_count"],
    aggfunc='sum'
)

# Flatten the multi-level column names for easier handling in plotting
task_data_combined.columns = ["_".join(col).strip() for col in task_data_combined.columns.values]

# Plot combined data in a grouped bar plot
fig, ax = plt.subplots(figsize=(12, 6))
task_data_combined.plot(kind="bar", ax=ax, width=0.8)

# Set plot title and labels
ax.set_title("Frequency Count by Category for All Tasks")
ax.set_xlabel("Screen Number")
ax.set_ylabel("Count")

# Add legend
ax.legend(title="Task & Category")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()





# Reverse the index for past_fluency_data to start with Screen 1 near the center
past_fluency_data = past_fluency_data.sort_index(ascending=False)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Plot Past Fluency on the left with negative values to create a mirrored effect
past_fluency_data.plot(kind="bar", stacked=True, ax=ax, position=1, width=0.4, legend=False, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
for container in ax.containers:
    container.set_offsets(container.get_offsets() * -1)  # Flip bars to the left side

# Plot Future Fluency on the right with positive values
future_fluency_data.plot(kind="bar", stacked=True, ax=ax, position=0, width=0.4, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

# Customize axis and labels
ax.set_xlabel("Screen Number")
ax.set_ylabel("Count")
ax.set_title("Frequency Count by Category for Task: Past and Future Fluency")

# Add the legend
plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')

# Set custom x-tick labels for each side
ax.set_xticks(range(len(past_fluency_data) + len(future_fluency_data)))
ax.set_xticklabels(
    [f"Screen {i}" for i in reversed(past_fluency_data.index)] + [f"Screen {i}" for i in future_fluency_data.index],
    rotation=0
)

# Add a line at the origin for the mirrored y-axis
ax.axvline(x=len(past_fluency_data) - 0.5, color='black', linewidth=0.5)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()



# (Optional) Remove specific variables or DataFrames
#del pid_counts, summary_stats
#gc.collect()





#### ----- Coherency check with human RAs and blursday codebot

# Load Excel files and select specific columns
file_paths = [
    '/Users/stevenbickley/Library/CloudStorage/Dropbox/Project 2 - Temporal Landmarks/data/working_files/RAs_round_1/LUCAS_blursday_PastFluency-FutureFluency-TemporalLandmarks_2023-03-11_translated.xlsx',
    '/Users/stevenbickley/Library/CloudStorage/Dropbox/Project 2 - Temporal Landmarks/data/working_files/RAs_round_1/parsed_responses_chunksize_10_2apr2024.xlsx',
    '/Users/stevenbickley/Library/CloudStorage/Dropbox/Project 2 - Temporal Landmarks/data/working_files/RAs_round_1/SARAH_Blursday Codebook Current Version.xlsx',
    '/Users/stevenbickley/Library/CloudStorage/Dropbox/Project 2 - Temporal Landmarks/data/working_files/RAs_round_1/Sean Blursday Coded Workbook.xlsx'
]
columns_of_interest = ["Experiment_ID", "PID", "UTC_Date", "Response_translated", "categories_selected", "ranking", "chosen_labels"]

datasets = [pd.read_excel(file, usecols=columns_of_interest, dtype=str) for file in file_paths]

# Merge datasets
merged_1234 = datasets[0]
for i, dataset in enumerate(datasets[1:], start=1):
    suffix = f"_{chr(120+i)}"  # To replicate suffixes _x, _y, _z
    merged_1234 = pd.merge(merged_1234, dataset, on=["Experiment_ID", "PID", "Response_translated"], suffixes=('', suffix))

# Function for calculating inter-rater reliability using Fleiss' Kappa
def calculate_fleiss_kappa(data, rating_columns):
    ratings = data[rating_columns].apply(pd.Series.value_counts, axis=1).fillna(0)
    return fleiss_kappa(ratings)

# Define columns for reliability calculation
rating_columns = ['categories_selected_x', 'categories_selected_y', 'categories_selected_z']  # Adjust based on suffixes

# Calculate Fleiss' Kappa
fleiss_kappa_result = calculate_fleiss_kappa(merged_1234, rating_columns)
print("Fleiss' Kappa for categories_selected:", fleiss_kappa_result)

# Function to calculate summary statistics and plot histograms
def detailed_summary_statistics(df, columns, top_n=20):
    summary_stats = {}
    for column in columns:
        col_summary = df[column].describe(include='all')
        summary_stats[column] = col_summary
        top_counts = df[column].value_counts().head(top_n)
        summary_stats[f'top_{top_n}_{column}'] = top_counts
        print(f"Summary statistics for {column}:\n", col_summary, "\n")
        print(f"Top {top_n} unique values for {column}:\n", top_counts, "\n")
    return summary_stats

# Plot histograms
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

# Columns to summarize and plot
columns_to_summarize = ['categories_selected', 'ranking', 'chosen_labels']
columns_to_plot = columns_to_summarize

# Generate detailed summary statistics
summary_stats = detailed_summary_statistics(merged_1234, columns_to_summarize)

# Plot histograms
plot_histograms(merged_1234, columns_to_plot)