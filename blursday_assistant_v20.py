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


##### ------ MAIN CODE - END ------- ####

## Set folder and file paths
folder_path = '/Users/bickley/stevejbickley/blursday_assistant/' # '/Users/stevenbickley/Library/CloudStorage/Dropbox/Project 2 - Temporal Landmarks/data/'
file_path = 'parsed_responses.xlsx' #'blursday_PastFluency-FutureFluency_2023-11-30_translated.csv'

# Read in the "raw" coded data
raw_df = pd.read_excel(file_path, dtype=str) # just read the full dataset

## Now clean up the coded data
clean_df = parse_dataframe(raw_df)
len(clean_df) # Expecting approx. 89,000 messages... where 89,000 = 17,800 chunked messages x 5 messages per chunk
# Final output at blursday_assistant_v20.py is n=87,792... close enough!
# Need to check which ones are missing and re-run the coding at some point in future..?

# Function to create new columns based on substrings in categories and ranking
def add_category_and_ranking_columns(df):
    # Define patterns for categories
    category_patterns = {
        'category_utilitarian': r'(?i)utilitarian',
        'category_discretionary': r'(?i)discretionary',
        'category_landmark': r'(?i)temporal landmark',
        'category_evaluation': r'(?i)evaluation',
    }
    # Add new columns based on category patterns
    for col, pattern in category_patterns.items():
        df[col] = df['categories'].apply(lambda x: 1 if any(re.search(pattern, str(item)) for item in (x if isinstance(x, list) else [x])) else 0)
    # Define patterns for ranking
    ranking_patterns = {
        'ranking_utilitarian': r'(?i)(\d+) \((?=.*utilitarian)',
        'ranking_discretionary': r'(?i)(\d+) \((?=.*discretionary)',
        'ranking_landmark': r'(?i)(\d+) \((?=.*temporal landmark)',
        'ranking_evaluation': r'(?i)(\d+) \((?=.*evaluation)',
    }
    # Add new columns based on ranking patterns
    #for col, pattern in ranking_patterns.items():
    #    df[col] = df['ranking'].apply(lambda x: 1 if re.search(pattern, str(x)) else 0)
    # Adjust columns to reflect "1" if found, and "0" otherwise
    #for col, pattern in ranking_patterns.items():
    #    df[col] = df['ranking'].apply(lambda x: max([int(m.group(1)) for m in re.finditer(pattern, str(x))], default=0))
    for col, pattern in ranking_patterns.items():
        df[col] = df['ranking'].apply(
            lambda x: max(map(int, re.findall(pattern, str(x))), default=0) if re.search(pattern, str(x)) else 0)
    # Adjust columns to reflect the highest ranking number if found
    for col in ranking_patterns.keys():
        df[col] = df[col].apply(lambda x: x if x > 0 else 0)
    # Add missing columns for categories and rankings
    df['category_missing'] = df['categories'].apply(lambda x: 1 if not x or (isinstance(x, str) and x.strip() == '') or (isinstance(x, list) and len(x) == 0) else 0)
    df['ranking_missing'] = df['ranking'].apply(lambda x: 1 if not x or (isinstance(x, str) and x.strip() == '') or (isinstance(x, list) and len(x) == 0) else 0)
    def get_ranking_other(row):
        if row['ranking_missing'] == 1:
            return 0
        rankings = str(row['ranking']).split(',')
        other_rankings = []
        for rank in rankings:
            rank = rank.strip()
            if not any(re.search(pattern, rank) for pattern in ranking_patterns.values()):
                try:
                    other_rankings.append(int(rank.split('(')[0].strip()))
                except ValueError:
                    continue
        return max(other_rankings, default=0)
    df['ranking_other'] = df.apply(get_ranking_other, axis=1)
    # Add other columns for categories and rankings
    df['category_other'] = df.apply(lambda row: 1 if row['ranking_other'] != 0 else 0, axis=1)
    #df['ranking_other'] = df.apply(lambda row: 1 if row[['ranking_utilitarian', 'ranking_discretionary','ranking_landmark', 'ranking_evaluation']].sum() == 0 and row['ranking_missing'] == 0 else 0, axis=1)
    return df


# df['category_other'] = df.apply(lambda row: 1 if row[['category_utilitarian', 'category_discretionary', 'category_landmark', 'category_evaluation', 'category_missing']].sum() == 0 else 0, axis=1)
# df['ranking_other'] = df.apply(lambda row: 1 if row[['ranking_utilitarian', 'ranking_discretionary', 'ranking_landmark', 'ranking_evaluation', 'ranking_missing']].sum() == 0 else 0, axis=1)


# Apply the function to add new columns to clean_df
clean_df = add_category_and_ranking_columns(clean_df)

clean_df[['categories','ranking','category_utilitarian','category_utilitarian','category_discretionary','category_landmark',
         'category_evaluation','category_missing','category_other','ranking_utilitarian','ranking_discretionary',
         'ranking_landmark','ranking_evaluation', 'ranking_missing', 'ranking_other']].head(n=50)

#clean_df.iloc[45,:]
#clean_df.iloc[25,:]
#clean_df[clean_df['ranking']=='1 (Discretionary Activities), 2 (Social)'].iloc[0,:]

# Display the updated DataFrame
print(clean_df.head())

# Write the clean data out to xlsx file
with pd.ExcelWriter(file_path[:-5]+"_clean.xlsx", engine='openpyxl') as writer:
    clean_df.to_excel(writer, index=False)

# Read in the "raw" coded data
clean_df = pd.read_excel(file_path[:-5]+"_clean.xlsx") # just read the full dataset

# Assuming clean_df is already loaded
# clean_df = pd.read_excel("your_file_path_here")

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
