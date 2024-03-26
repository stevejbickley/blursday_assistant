##### ------ IMPORT FUNCTIONS + SETUP CODE - START ------- ####

import os
import re
import time
import pandas as pd
from openai import OpenAI

# See reference github repo:  https://github.com/pixegami/openai-assistants-api-demo
# See also OpenAI reference documentation:  ttps://platform.openai.com/docs/assistants/how-it-works

# Enter your Assistant ID here.
ASSISTANT_ID = "asst_wWt15CA9kKqTI79SLQDPlGWm"

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
        'original_data': "",
        'categories_selected': [],
        'ranking': [],
        'chosen_labels': []
    }
    # Use re.search and check if the result is not None before accessing .group()
    original_data_match = re.search(r"<original_data>(.*?)</original_data>", response)
    if original_data_match:
        parsed_data['original_data'] = original_data_match.group(1)
    categories_selected_match = re.search(r"<categories_selected>(.*?)</categories_selected>", response)
    if categories_selected_match:
        parsed_data['categories_selected'] = categories_selected_match.group(1).split(', ')
    # Assuming ranking and chosen_labels should be extracted similarly
    ranking_match = re.search(r"<ranking>(.*?)</ranking>", response)
    if ranking_match:
        parsed_data['ranking'] = [ranking_match.group(1)]
    chosen_labels_match = re.search(r"<chosen_labels>(.*?)</chosen_labels>", response)
    if chosen_labels_match:
        parsed_data['chosen_labels'] = [chosen_labels_match.group(1)]
    return parsed_data

##### ------ DEFINE FUNCTIONS - END ------- ####

##### ------ MAIN CODE - START ------- ####
# Assuming you have a file path, you can customize the number of rows as needed.
folder_path = '/Users/stevenbickley/stevejbickley/assistant_API/' # '/Users/stevenbickley/Library/CloudStorage/Dropbox/Project 2 - Temporal Landmarks/data/'
file_path = 'blursday_PastFluency-FutureFluency_2023-11-30_translated.csv'
df_custom_rows = read_and_select_rows(str(folder_path + file_path), 50) # Will return the first 50 rows of the dataframe.
#df = pd.read_csv(file_path, dtype=str) # just read the full dataset

# Create a list of messages to be added to the thread
messages_to_create = [
    {
        "role": "user",
        "content": str(response)  # Ensure the content is a string
    }
    for response in df_custom_rows['Response_translated']  # Iterate over each response
]

# Chunk Messages into Batches of 32
chunked_messages = list(chunk_messages(messages_to_create))

# List to store created thread IDs
created_thread_ids = []

# Create Threads for Each Chunk
for chunk in chunked_messages:
    thread = client.beta.threads.create(messages=chunk)
    created_thread_ids.append(thread.id)
    print(f"Thread Created: {thread.id}")

# Handling Multiple Threads
for thread_id in created_thread_ids:
    # Submit the thread to the assistant
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=ASSISTANT_ID)
    print(f"👉 Run Created: {run.id}")
    # Wait for run to complete
    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        print(f"🏃 Run Status: {run.status}")
        time.sleep(1)
    print(f"🏁 Run Completed for Thread ID: {thread_id}")
    # Process each thread's messages as needed
    message_response = client.beta.threads.messages.list(thread_id=thread_id)
    messages = message_response.data
    # Extract and print the latest message.
    #latest_message = messages[0]
    #print(f"💬 Response: {latest_message.content[0].text.value}")
    # Parse the assistant's responses
    parsed_responses = [parse_assistant_response(message.content[0].text.value) for message in messages]
    parsed_responses
    # Convert the list of parsed responses to a pandas DataFrame
    df_responses = pd.DataFrame(parsed_responses)
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
            except Exception as e:  # Handle cases where the existing file is empty or has a different structure
                updated_df = df_responses
            # Write/Append the updated DataFrame to the Excel file
            updated_df.to_excel(writer, index=False, sheet_name='Sheet1')
    else:
        # Create a new Excel file
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            df_responses.to_excel(writer, index=False)


##### ------ MAIN CODE - END ------- ####

