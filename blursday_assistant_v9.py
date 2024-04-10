##### ------ IMPORT FUNCTIONS + SETUP CODE - START ------- ####

import os
import re
import time
import pandas as pd
from openai import OpenAI
import openpyxl

# See reference github repo:  https://github.com/pixegami/openai-assistants-api-demo
# See also OpenAI reference documentation:  ttps://platform.openai.com/docs/assistants/how-it-works

# Enter your Assistant ID here.
ASSISTANT_ID = "asst_DmcouHcpU3rAZ4MV2FbwRrkR" # Old (superseded on 10 April - accidental delete): "asst_wWt15CA9kKqTI79SLQDPlGWm"

# Make sure your API key is set as an environment variable.
client = OpenAI()

##### ------ IMPORT FUNCTIONS + SETUP CODE - END ------- ####

##### ------ DEFINE FUNCTIONS - START ------- ####

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
def parse_assistant_response(response):
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
    if not all(tags_found.values()):
        parsed_data['dump'].append(response)
    else:
        parsed_data['dump'].append(response)
    return parsed_data

# Custom function to expand the lists
def expand_lists_in_df(df, cols_to_expand):
    expanded_data = []
    # Function to check if the item is a list of lists
    def is_list_of_lists(item):
        return all(isinstance(elem, list) for elem in item) if isinstance(item, list) else False
    for index, row in df.iterrows():
        # Determine the max length for expansion in this row, handling non-lists as length 1
        max_len = 0
        for col in cols_to_expand:
            item = row[col]
            if isinstance(item, list):
                if is_list_of_lists(item):
                    max_len = max(max_len, len(item))
                else:
                    max_len = max(max_len, 1)
            else:
                max_len = max(max_len, 1)
        # Expand the row based on max_len
        for i in range(max_len):
            new_row = {}
            for col in df.columns:
                if col in cols_to_expand and isinstance(row[col], list):
                    # Handle list of lists vs single list differently
                    if is_list_of_lists(row[col]):
                        new_row[col] = row[col][i] if i < len(row[col]) else None
                    else:
                        # If it's a single list but max_len > 1, duplicate across new rows; else, use as is
                        new_row[col] = row[col][i] if max_len > 1 and i < len(row[col]) else row[col]
                else:
                    new_row[col] = row[col]
            expanded_data.append(new_row)
    return pd.DataFrame(expanded_data)

##### ------ DEFINE FUNCTIONS - END ------- ####

##### ------ MAIN CODE - START ------- ####
# Assuming you have a file path, you can customize the number of rows as needed.
folder_path = '/Users/stevenbickley/stevejbickley/blursday_assistant/' # '/Users/stevenbickley/Library/CloudStorage/Dropbox/Project 2 - Temporal Landmarks/data/'
file_path = 'blursday_PastFluency-FutureFluency-TemporalLandmarks_2023-03-11_translated.csv' #'blursday_PastFluency-FutureFluency_2023-11-30_translated.csv'
#df_custom_rows = read_and_select_rows(str(folder_path + file_path), 50) # Will return the first 50 rows of the dataframe.
df = pd.read_csv(file_path, dtype=str) # just read the full dataset

# Filter to Canada
#df_custom_rows = df[ (df['Country_Name'] == 'Canada') | (df['Country_Name'] == 'United Kingdom') | (df['Country_Name'] == 'US')]
df_custom_rows = df[ (df['Country_Name'] == 'US')]

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

# List to store created thread IDs
created_thread_ids = []
#created_run_ids = []

# Create Threads for Each Chunk
for chunk in chunked_messages:
    thread = client.beta.threads.create(messages=chunk)
    created_thread_ids.append(thread.id)
    print(f"Thread Created: {thread.id}")
    #created_run_ids.append(run.id)
    #print(f"Run id: {run.id}")

# Handling Multiple Threads
for i in range(0,len(created_thread_ids)):
    thread_id = created_thread_ids[i]
    #run_id = created_run_ids[i]
    # Submit the thread to the assistant
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=ASSISTANT_ID)
    print(f"👉 Run Created: {run.id}")
    # Wait for run to complete
    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        print(f"🏃 Run Status: {run.status}")
        time.sleep(1)
    # Process each thread's messages as needed
    message_response = client.beta.threads.messages.list(thread_id=thread_id)
    messages = message_response.data
    # Parse the assistant's responses
    parsed_responses = [parse_assistant_response(message.content[0].text.value) for message in messages]
    # Filter out empty rows: check if all values in each dictionary are empty
    filtered_responses = [
        response for response in parsed_responses
        if not (
                all(value == [] or value == "" for key, value in response.items() if key != 'dump') and
                not any("Ranking" in s for s in response.get('dump', [])))]
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
    chunked_metainfo_df = pd.DataFrame(chunked_metainfo[i])
    # Concatenate chunked_metainfo_df with df_responses
    df_responses = pd.concat([df_responses.reset_index(drop=True), chunked_metainfo_df.reset_index(drop=True)], axis=1)
    # Drop rows where any of the cells have NaN values
    try:
        df_responses = df_responses.dropna(subset='dump')
    except:
        next
    # Drop rows where any of the cells have NaN values
    df_responses['thread_id'] = thread_id
    # Create a new Excel writer object and save the DataFrame to an xlsx file
    file_path = 'parsed_responses.xlsx'
    # Check if the file already exists
    if os.path.exists(file_path):
        # Open the existing Excel file
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Read existing data
            try:
                existing_data = pd.read_excel(file_path)
                # Append new data to existing data
                updated_df = pd.concat([existing_data, df_responses], ignore_index=True)
                updated_df.to_excel(writer, index=False, sheet_name='Sheet1') # Write/Append the updated DataFrame to the Excel file
            except Exception as e:  # Handle cases where the existing file is empty or has a different structure
                print('skipping thread_id: '+ str(thread_id))
                next
    else:
        # Create a new Excel file
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df_responses.to_excel(writer, index=False)
    time.sleep(1)


##### ------ MAIN CODE - END ------- ####

