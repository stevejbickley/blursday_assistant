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

##### ------ DEFINE FUNCTIONS - START ------- ####
def read_and_select_rows(file_path, number_of_rows=10):
    """
    Reads in an Excel file as a pandas dataframe and selects the first 'number_of_rows' rows.

    :param file_path: Path to the Excel file.
    :param number_of_rows: Number of rows to select from the dataframe.
    :return: A pandas dataframe with the specified number of rows.
    """
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(file_path)

    # Select the first 'number_of_rows' rows
    selected_df = df.head(number_of_rows)

    return selected_df


# Example usage:
# Assuming you have a file path, you can customize the number of rows as needed.
# file_path = 'path_to_your_excel_file.xlsx'
# df_custom_rows = read_and_select_rows(file_path, 50)
# The above example will return the first 50 rows of the dataframe.

# Assuming 'client' is already defined and authenticated
# and 'df_all_rows' is the DataFrame obtained from the read_all_rows function

# Create a list of messages to be added to the thread
messages_to_create = [
    {
        "role": "user",
        "content": str(response)  # Ensure the content is a string
    }
    for response in df_all_rows['Response_translated']  # Iterate over each response
]

# Create a thread with the messages
thread = client.beta.threads.create(messages=messages_to_create)

# Create a thread with a message.
thread = client.beta.threads.create(
    messages=[
        {
            "role": "user",
            # Update this with the query you want to use.
            "content": "Taking care of my children by taking them to the movies",
        }
    ]
)

# Submit the thread to the assistant (as a new run).
run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=ASSISTANT_ID)
print(f"👉 Run Created: {run.id}")

# Wait for run to complete.
while run.status != "completed":
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    print(f"🏃 Run Status: {run.status}")
    time.sleep(1)
else:
    print(f"🏁 Run Completed!")

# Get the latest message from the thread.
message_response = client.beta.threads.messages.list(thread_id=thread.id)
messages = message_response.data

# Print the latest message.
latest_message = messages[0]
print(f"💬 Response: {latest_message.content[0].text.value}")

# Example response from the assistant
assistant_response = ("Blursday Thematic Coding Assistant "
                      "<original_data>Taking care of my children by taking them to the movies</original_data> "
                      "<categories_selected>Discretionary Activities, Utilitarian Activities</categories_selected> "
                      "<ranking>1 - Utilitarian Activities, 2 - Discretionary Activities</ranking> "
                      "<chosen_labels>For Utilitarian Activities: Childcare "
                      "For Discretionary Activities: Leisure, Entertainment, Family Activities</chosen_labels>")

# Function to parse the response and extract values
def parse_assistant_response(response):
    parsed_data = {
        'original_data': re.search(r"<original_data>(.*?)</original_data>", response).group(1),
        'categories_selected': re.search(r"<categories_selected>(.*?)</categories_selected>", response).group(1).split(', '),
        'ranking': re.findall(r"<ranking>(.*?)</ranking>", response),
        'chosen_labels': re.findall(r"<chosen_labels>(.*?)</chosen_labels>", response)
    }
    return parsed_data

# Parse the assistant's response
#parsed_response = parse_assistant_response(assistant_response)
#parsed_response

# Parse the assistant's responses
parsed_responses = [parse_assistant_response(message) for message in messages]
parsed_responses

# Convert the list of parsed responses to a pandas DataFrame
df_responses = pd.DataFrame(parsed_responses)

# Create a new Excel writer object and save the DataFrame to an xlsx file
file_path = '/mnt/data/parsed_responses.xlsx'
with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
    df_responses.to_excel(writer, index=False)
