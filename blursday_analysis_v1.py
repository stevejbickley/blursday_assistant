#### NEW ANALYSIS CODES (translating the R code into python code)

# Import packages required for analysis and visualisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score  # For reliability calculations
from statsmodels.stats.inter_rater import fleiss_kappa  # Alternative library for Fleiss' Kappa
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.formula.api import ols
from openpyxl import load_workbook  # For working with Excel files
import re
import gc
import statsmodels.formula.api as smf

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
blursday = blursday.dropna(subset=["Screen_Number", "Response_translated"])

# Merge datasets
analysis_df = pd.merge(analysis_df, parsed_responses[['Experiment_ID','PID','UTC_Date','UTC_yyyymmdd','thread_id','run_id']], on=['thread_id', 'run_id'], how='left') # Columns not merged from parsed_responses: 'run_status', 'run_timestamp_created', 'run_timestamp_completed', 'run_model_used', 'run_completion_tokens', 'run_prompt_tokens', 'run_model_temperature', 'thread_timestamp_created'
#analysis_df2 = pd.merge(analysis_df.copy(), blursday[['Experiment_ID','Experiment_ID_tempholder','PID_tempholder','UTC_Date_tempholder','UTC_yyyymmdd_tempholder','UTC_yyyymmdd','Country','Session','Task_Name','Task_Version','Screen_Number','Event_Index','Attempt','Reaction_Time','Reaction_Onset','Participant_OS','Participant_Browser','Handedness','Sex','Age','Stringency_Index','Mobility_Transit','Mobility_Retail','Mobility_Parks','Mobility_WorkPlaces','Mobility_Residential','Reported_Loneliness','Felt_Loneliness','Subjective_Confinement','ConfDuration','Response_translated','Response_translated_lower']], left_on=['Experiment_ID','PID','UTC_yyyymmdd','original_response_lower'], right_on=['Experiment_ID','PID_tempholder','UTC_yyyymmdd','Response_translated_lower'], how='outer', indicator=True)  # Add the merge indicator column
analysis_df = pd.merge(analysis_df, blursday[['Experiment_ID','Experiment_ID_tempholder','PID_tempholder','UTC_Date_tempholder','UTC_yyyymmdd_tempholder','UTC_yyyymmdd','Country','Session','Task_Name','Task_Version','Screen_Number','Event_Index','Attempt','Reaction_Time','Reaction_Onset','Participant_OS','Participant_Browser','Handedness','Sex','Age','Stringency_Index','Mobility_Transit','Mobility_Retail','Mobility_Parks','Mobility_WorkPlaces','Mobility_Residential','Reported_Loneliness','Felt_Loneliness','Subjective_Confinement','ConfDuration','Response_translated','Response_translated_lower']], left_on=['Experiment_ID','UTC_yyyymmdd','original_response_lower'], right_on=['Experiment_ID','UTC_yyyymmdd','Response_translated_lower'], how='outer', indicator=True)  # Add the merge indicator column

# Extract rows from blursday that did not match
blursday_nomatch = analysis_df[analysis_df['_merge'] == 'right_only']
blursday_nomatch = blursday_nomatch.drop(columns=['_merge']) # Optional: Drop the indicator column since we don't need it
blursday_nomatch = blursday_nomatch.drop_duplicates(subset=['Experiment_ID', 'PID', 'UTC_yyyymmdd', 'Response_translated_lower', 'Session','Task_Name','Task_Version','Screen_Number'])
blursday_nomatch = blursday_nomatch.drop(columns=['Experiment_ID', 'PID', 'UTC_Date', 'UTC_yyyymmdd','original_response', 'categories_selected', 'ranking', 'primary_codes', 'secondary_codes', 'flagged', 'justification_comments', 'thread_id', 'run_id', 'chunked_message', 'chat_completion_id', 'chat_completion_created', 'chat_completion_model', 'chat_completion_system_fingerprint', 'chat_completion_completion_tokens', 'chat_completion_prompt_tokens', 'chat_completion_total_tokens', 'category_utilitarian', 'category_discretionary', 'category_landmark', 'category_evaluation', 'ranking_utilitarian', 'ranking_discretionary', 'ranking_landmark', 'ranking_evaluation', 'category_missing', 'ranking_missing', 'ranking_other', 'category_other', 'primary_personal_temporal_landmark', 'primary_calendar_temporal_landmark', 'primary_reference_points', 'primary_personal_narrative_events', 'primary_facts_of_life_experiences', 'primary_household_obligations', 'primary_physiological_needs_personal_care', 'primary_work_school_activities', 'primary_services', 'primary_care_duties', 'primary_return_to_routine', 'primary_career_planning', 'primary_civic_duties', 'primary_recreation', 'primary_entertainment', 'primary_social', 'primary_family', 'primary_altruistic', 'primary_aspirational', 'primary_recreation_services', 'primary_shopping', 'primary_introspection', 'primary_home_improvement', 'primary_travel', 'primary_self_improvement', 'primary_new_connection_seeking', 'primary_spiritual_activities', 'primary_negative', 'primary_neutral', 'primary_positive', 'primary_lockdown', 'primary_missing', 'primary_other', 'secondary_personal_temporal_landmark', 'secondary_calendar_temporal_landmark', 'secondary_reference_points', 'secondary_personal_narrative_events', 'secondary_facts_of_life_experiences', 'secondary_household_obligations', 'secondary_physiological_needs_personal_care', 'secondary_work_school_activities', 'secondary_services', 'secondary_care_duties', 'secondary_return_to_routine', 'secondary_career_planning', 'secondary_civic_duties', 'secondary_recreation', 'secondary_entertainment', 'secondary_social', 'secondary_family', 'secondary_altruistic', 'secondary_aspirational', 'secondary_recreation_services', 'secondary_shopping', 'secondary_introspection', 'secondary_home_improvement', 'secondary_travel', 'secondary_self_improvement', 'secondary_new_connection_seeking', 'secondary_spiritual_activities', 'secondary_negative', 'secondary_neutral', 'secondary_positive', 'secondary_lockdown', 'secondary_missing', 'secondary_other', 'original_response_lower'])
blursday_nomatch = blursday_nomatch.dropna(subset=["Screen_Number", "Response_translated_lower"]) # 25903 rows down to 25903 rows
#blursday_nomatch2 = analysis_df2[analysis_df2['_merge'] == 'right_only']
#blursday_nomatch2 = blursday_nomatch2.drop(columns=['_merge']) # Optional: Drop the indicator column since we don't need it
#blursday_nomatch2 = blursday_nomatch2.drop_duplicates(subset=['Experiment_ID', 'PID', 'UTC_yyyymmdd', 'Response_translated_lower', 'Session','Task_Name','Task_Version','Screen_Number'])
#blursday_nomatch2 = blursday_nomatch2.drop(columns=['Experiment_ID', 'PID', 'UTC_Date', 'UTC_yyyymmdd','original_response', 'categories_selected', 'ranking', 'primary_codes', 'secondary_codes', 'flagged', 'justification_comments', 'thread_id', 'run_id', 'chunked_message', 'chat_completion_id', 'chat_completion_created', 'chat_completion_model', 'chat_completion_system_fingerprint', 'chat_completion_completion_tokens', 'chat_completion_prompt_tokens', 'chat_completion_total_tokens', 'category_utilitarian', 'category_discretionary', 'category_landmark', 'category_evaluation', 'ranking_utilitarian', 'ranking_discretionary', 'ranking_landmark', 'ranking_evaluation', 'category_missing', 'ranking_missing', 'ranking_other', 'category_other', 'primary_personal_temporal_landmark', 'primary_calendar_temporal_landmark', 'primary_reference_points', 'primary_personal_narrative_events', 'primary_facts_of_life_experiences', 'primary_household_obligations', 'primary_physiological_needs_personal_care', 'primary_work_school_activities', 'primary_services', 'primary_care_duties', 'primary_return_to_routine', 'primary_career_planning', 'primary_civic_duties', 'primary_recreation', 'primary_entertainment', 'primary_social', 'primary_family', 'primary_altruistic', 'primary_aspirational', 'primary_recreation_services', 'primary_shopping', 'primary_introspection', 'primary_home_improvement', 'primary_travel', 'primary_self_improvement', 'primary_new_connection_seeking', 'primary_spiritual_activities', 'primary_negative', 'primary_neutral', 'primary_positive', 'primary_lockdown', 'primary_missing', 'primary_other', 'secondary_personal_temporal_landmark', 'secondary_calendar_temporal_landmark', 'secondary_reference_points', 'secondary_personal_narrative_events', 'secondary_facts_of_life_experiences', 'secondary_household_obligations', 'secondary_physiological_needs_personal_care', 'secondary_work_school_activities', 'secondary_services', 'secondary_care_duties', 'secondary_return_to_routine', 'secondary_career_planning', 'secondary_civic_duties', 'secondary_recreation', 'secondary_entertainment', 'secondary_social', 'secondary_family', 'secondary_altruistic', 'secondary_aspirational', 'secondary_recreation_services', 'secondary_shopping', 'secondary_introspection', 'secondary_home_improvement', 'secondary_travel', 'secondary_self_improvement', 'secondary_new_connection_seeking', 'secondary_spiritual_activities', 'secondary_negative', 'secondary_neutral', 'secondary_positive', 'secondary_lockdown', 'secondary_missing', 'secondary_other', 'original_response_lower'])

# Drop duplicates in the specified columns
analysis_df = analysis_df[(analysis_df['_merge'] == 'left_only') | (analysis_df['_merge'] == 'both')] # Extract rows from analysis_df so it is like a left join
analysis_df = analysis_df.drop(columns=['_merge']) # Optional: Drop the indicator column since we don't need it
analysis_df = analysis_df.drop_duplicates(subset=['Experiment_ID', 'PID', 'UTC_yyyymmdd', 'Response_translated_lower', 'Session','Task_Name','Task_Version','Screen_Number'])
#analysis_df2 = analysis_df2[(analysis_df2['_merge'] == 'left_only') | (analysis_df2['_merge'] == 'both')] # Extract rows from analysis_df so it is like a left join
#analysis_df2 = analysis_df2.drop(columns=['_merge']) # Optional: Drop the indicator column since we don't need it
#analysis_df2 = analysis_df2.drop_duplicates(subset=['Experiment_ID', 'PID', 'UTC_yyyymmdd', 'Response_translated_lower', 'Session','Task_Name','Task_Version','Screen_Number'])

# Fix up the temporary placeholder columns
analysis_df = analysis_df.drop(columns=['PID','UTC_Date','UTC_yyyymmdd','Experiment_ID']) # Drop the old/faulty columns
analysis_df['PID'] = analysis_df['PID_tempholder'] # Reassign the dropped columns
analysis_df['UTC_Date'] = analysis_df['UTC_Date_tempholder']
analysis_df['UTC_yyyymmdd'] = analysis_df['UTC_yyyymmdd_tempholder']
analysis_df['Experiment_ID'] = analysis_df['Experiment_ID_tempholder']
analysis_df = analysis_df.drop(columns=['PID_tempholder','UTC_Date_tempholder','UTC_yyyymmdd_tempholder','Experiment_ID_tempholder']) # no longer need the temporary columns after this so we drop them
analysis_df = analysis_df.dropna(subset=["Screen_Number", "original_response_lower"]) # 68792 rows down to 66747 rows
#analysis_df2 = analysis_df2.drop(columns=['PID','UTC_Date','UTC_yyyymmdd','Experiment_ID']) # Drop the old/faulty columns
#analysis_df2['PID'] = analysis_df2['PID_tempholder'] # Reassign the dropped columns
#analysis_df2['UTC_Date'] = analysis_df2['UTC_Date_tempholder']
#analysis_df2['UTC_yyyymmdd'] = analysis_df2['UTC_yyyymmdd_tempholder']
#analysis_df2['Experiment_ID'] = analysis_df2['Experiment_ID_tempholder']
#analysis_df2 = analysis_df2.drop(columns=['PID_tempholder','UTC_Date_tempholder','UTC_yyyymmdd_tempholder','Experiment_ID_tempholder']) # no longer need the temporary columns after this so we drop them

# Read in the other analysis_df and join it to the bigger one
file_path = './output_data/parsed_responses_extracted_clean_wide_for_analysis.xlsx'
analysis_df2 = pd.read_excel(file_path, dtype=str) # just read the full dataset
file_path = './output_data/parsed_responses_extracted2_clean_wide_for_analysis.xlsx'
analysis_df3 = pd.read_excel(file_path, dtype=str) # just read the full dataset
file_path = './output_data/parsed_responses_extracted3_clean_wide_for_analysis.xlsx'
analysis_df4 = pd.read_excel(file_path, dtype=str) # just read the full dataset
file_path = './output_data/parsed_responses_extracted4_clean_wide_for_analysis.xlsx'
analysis_df5 = pd.read_excel(file_path, dtype=str) # just read the full dataset

# Transform some of the columns
analysis_df2["original_response_lower"] = analysis_df2["original_response"].str.lower() # convert the original (translated) responses to all lower case
analysis_df3["original_response_lower"] = analysis_df3["original_response"].str.lower() # convert the original (translated) responses to all lower case
analysis_df4["original_response_lower"] = analysis_df4["original_response"].str.lower() # convert the original (translated) responses to all lower case
analysis_df5["original_response_lower"] = analysis_df5["original_response"].str.lower() # convert the original (translated) responses to all lower case

# Merge analysis_df2 dataset
analysis_df2 = pd.merge(analysis_df2, blursday[['Experiment_ID','PID','UTC_Date','UTC_yyyymmdd','Country','Session','Task_Name','Task_Version','Screen_Number','Event_Index','Attempt','Reaction_Time','Reaction_Onset','Participant_OS','Participant_Browser','Handedness','Sex','Age','Stringency_Index','Mobility_Transit','Mobility_Retail','Mobility_Parks','Mobility_WorkPlaces','Mobility_Residential','Reported_Loneliness','Felt_Loneliness','Subjective_Confinement','ConfDuration','Response_translated','Response_translated_lower']], left_on=['Experiment_ID', 'Session', 'PID','original_response_lower'], right_on=['Experiment_ID','Session','PID','Response_translated_lower'], how='left')  # Add the merge indicator column
analysis_df2 = analysis_df2.drop_duplicates(subset=['Experiment_ID', 'PID', 'Response_translated_lower','Task_Name','Screen_Number','chat_completion_id'])
analysis_df3 = pd.merge(analysis_df3, blursday[['Experiment_ID','PID','UTC_Date','UTC_yyyymmdd','Session','Country','Screen_Number','Task_Name','Task_Version','Event_Index','Attempt','Reaction_Time','Reaction_Onset','Participant_OS','Participant_Browser','Handedness','Sex','Age','Stringency_Index','Mobility_Transit','Mobility_Retail','Mobility_Parks','Mobility_WorkPlaces','Mobility_Residential','Reported_Loneliness','Felt_Loneliness','Subjective_Confinement','ConfDuration','Response_translated','Response_translated_lower']], left_on=['Experiment_ID', 'Session', 'PID', 'Country','Screen_Number', 'original_response_lower'], right_on=['Experiment_ID','Session','PID', 'Country','Screen_Number', 'Response_translated_lower'], how='left')  # Add the merge indicator column
analysis_df3 = analysis_df3.drop_duplicates(subset=['Experiment_ID', 'PID', 'Response_translated_lower','Task_Name','Screen_Number','chat_completion_id'])
analysis_df4 = pd.merge(analysis_df4, blursday[['Experiment_ID','PID','UTC_Date','UTC_yyyymmdd','Session','Country','Screen_Number','Task_Name','Task_Version','Event_Index','Attempt','Reaction_Time','Reaction_Onset','Participant_OS','Participant_Browser','Handedness','Sex','Age','Stringency_Index','Mobility_Transit','Mobility_Retail','Mobility_Parks','Mobility_WorkPlaces','Mobility_Residential','Reported_Loneliness','Felt_Loneliness','Subjective_Confinement','ConfDuration','Response_translated','Response_translated_lower']], left_on=['Experiment_ID', 'Session', 'PID', 'Country','Screen_Number', 'original_response_lower'], right_on=['Experiment_ID','Session','PID', 'Country','Screen_Number', 'Response_translated_lower'], how='left')  # Add the merge indicator column
analysis_df4 = analysis_df4.drop_duplicates(subset=['Experiment_ID', 'PID', 'Response_translated_lower','Task_Name','Screen_Number','chat_completion_id'])
analysis_df5 = pd.merge(analysis_df5, blursday[['Experiment_ID','PID','UTC_Date','UTC_yyyymmdd','Session','Country','Screen_Number','Task_Name','Task_Version','Event_Index','Attempt','Reaction_Time','Reaction_Onset','Participant_OS','Participant_Browser','Handedness','Sex','Age','Stringency_Index','Mobility_Transit','Mobility_Retail','Mobility_Parks','Mobility_WorkPlaces','Mobility_Residential','Reported_Loneliness','Felt_Loneliness','Subjective_Confinement','ConfDuration','Response_translated','Response_translated_lower']], left_on=['Experiment_ID', 'Session', 'PID', 'Country','Screen_Number', 'original_response_lower'], right_on=['Experiment_ID','Session','PID', 'Country','Screen_Number', 'Response_translated_lower'], how='left')  # Add the merge indicator column
analysis_df5 = analysis_df5.drop_duplicates(subset=['Experiment_ID', 'PID', 'Response_translated_lower','Task_Name','Screen_Number','chat_completion_id'])

# Concatenate the dataframes (rows)
analysis_df = pd.concat([analysis_df, analysis_df2], ignore_index=True) # 69603 rows
analysis_df = pd.concat([analysis_df, analysis_df3], ignore_index=True) # 100443 rows
analysis_df = pd.concat([analysis_df, analysis_df4], ignore_index=True) # 102078 rows
analysis_df = pd.concat([analysis_df, analysis_df5], ignore_index=True) # 102078 rows
analysis_df = analysis_df.dropna(subset=["Screen_Number", "original_response_lower"]) # 102078 rows down to 101893 rows
analysis_df = analysis_df.drop_duplicates(subset=['Experiment_ID', 'PID', 'UTC_yyyymmdd', 'Response_translated_lower', 'Screen_Number']) # 76173 rows

# Find the common columns between the two dataframes
common_columns = list(set(blursday.columns).intersection(analysis_df.columns))

# Identify rows in blursday that are not present in analysis_df based on common columns
not_in_analysis_df = blursday.merge(
    analysis_df[common_columns],
    on=common_columns,
    how='left',
    indicator=True
).query('_merge == "left_only"').drop(columns=['_merge'])

# Drop any na rows
not_in_analysis_df = not_in_analysis_df.dropna(subset=["Screen_Number", "Task_Name", "Response_translated_lower"])
#not_in_analysis_df[not_in_analysis_df['Unique_Name'] != "TemporalLandmark"]

# Drop duplicates
not_in_analysis_df = not_in_analysis_df.drop_duplicates(subset=['Experiment_ID', 'PID', 'Country', 'Response_translated', 'Session','Task_Name','Task_Version','Screen_Number']) # len(df) == 81.777 after this line

# (Optional) Write the resulting dataframe out to csv
#analysis_df.to_csv("main_data_for_analysis.csv", index=False)

# Remove specific variables or DataFrames
del blursday, blursday_nomatch, file_path, parsed_responses, analysis_df2, analysis_df3, common_columns, not_in_analysis_df
gc.collect()
dir() # view all defined variables

# Load supplementary CSV files
ffa_participant = pd.read_csv(f"{currwd}../data_assorted/blursday_assistant/data-FFA-ParticipantTrack-ParticipantTracker_France-Argentina-Canada-Colombia-Germany-Greece-India-Italy-Japan-Turkey-United Kingdom-US_2024-02-26.csv", dtype=str) # Walach, H., Buchheld, N., Buttenmüller, V., Kleinknecht, N., & Schmidt, S. (2006). Measuring mindfulness—The Freiburg mindfulness inventory (FMI). Personality and Individual Differences, 40(8), 1543–1555.
bfi_delay = pd.read_csv(f"{currwd}../data_assorted/blursday_assistant/data-BFI-DelayDiscount_France-Argentina-Canada-Colombia-Germany-Greece-India-Italy-Japan-Turkey-United Kingdom-US_2024-02-26.csv", dtype=str) # Rammstedt, B., & John, O. P. (2007). Measuring personality in one minute or less: A 10-item short version of the Big Five Inventory in English and German. Journal of Research in Personality, 41(1), 203–212.
retro_duration = pd.read_csv(f"{currwd}../data_assorted/blursday_assistant/data-RetroDuration-ZTPI_France-Argentina-Canada-Colombia-Germany-Greece-India-Italy-Japan-Turkey-United Kingdom-US_2024-02-26.csv", dtype=str) # Assesses a participant retrospective duration estimation (hours: minutes: seconds) and passage of time (over days). This test was randomly displayed to the participant during he first round of questionnaires.

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
dir() # view all defined variables

# Check length of analysis_df
len(analysis_df) # 75353 rows

# Drop rows where either the 'Screen_Number' or 'Task_Name' columns are NaN
analysis_df = analysis_df.dropna(
    subset=["Screen_Number", "Task_Name"]) # 73886 rows

# Create a new column 'Task_Name_clean' based on specific keywords in 'Task_Name'
analysis_df['Task_Name_clean'] = analysis_df['Task_Name'].apply(
    lambda x: 'Future Fluency' if 'Future Fluency' in x else
              'Past Fluency' if 'Past Fluency' in x else
              'Temporal Landmark' if 'Temporal Landmark' in x else x)

# Drop rows where 'Screen_Number' or other important columns are NaN
analysis_data = analysis_df.dropna(
    subset=["Screen_Number", "Task_Name_clean", "category_utilitarian", "category_discretionary", "category_landmark",
            "category_evaluation", "category_other"])

# Check length of analysis_data
len(analysis_data) # 73886 rows

# Filter out rows where 'Screen_Number' is not 1, 2, or 3
analysis_data = analysis_data[analysis_data["Screen_Number"].isin(["1", "2", "3"])] # still 66762 rows

# Filter out the "Temporal Landmark" tasks
analysis_data = analysis_data[analysis_data['Task_Name_clean'] != "Temporal Landmark"]

# Check length of analysis_data
len(analysis_data) # 69882 rows

# Define main categories and subcategories for each one, as shown in the provided images
categories = {
    "Temporal Landmarks": [
        "Personal Temporal Landmark", "Calendar Temporal Landmark", "Reference Points",
        "Personal Narrative Events", "Facts of Life Experiences", "Absence of activity"
    ],
    "Utilitarian Activities": [
        "Household Obligations", "Physiological Needs and Personal Care", "Work/School Activities",
        "Services", "Care Duties", "Return to Routine", "Career Planning", "Civic Duties"
    ],
    "Discretionary Activities": [
        "Recreation", "Entertainment", "Social", "Family", "Altruistic", "Aspirational",
        "Recreation (Services)", "Shopping", "Introspection", "Home Improvement", "Travel",
        "Self-Improvement", "New Connection Seeking", "Spiritual Activities"
    ],
    "Evaluations": ["Negative", "Neutral", "Positive", "Lockdown"]
}

# Create a dictionary to map the primary and secondary columns to the categories
primary_columns = [f"primary_{cat.lower().replace('-', '_').replace('/', '_').replace(' and ', '_').replace(' ', '_').replace('(', '').replace(')', '')}" for category in categories.values() for cat in category]
secondary_columns = [f"secondary_{cat.lower().replace('-', '_').replace('/', '_').replace(' and ', '_').replace(' ', '_').replace('(', '').replace(')', '')}" for category in categories.values() for cat in category]

# Convert columns to numeric, coercing errors to NaN
analysis_data[primary_columns] = analysis_data[primary_columns].apply(pd.to_numeric, errors='coerce')
analysis_data[secondary_columns] = analysis_data[secondary_columns].apply(pd.to_numeric, errors='coerce')

# Initialize new columns for counts
for top_level_category in categories:
    analysis_data[f"{top_level_category.lower().replace(' ', '_')}_count_primary"] = 0
    analysis_data[f"{top_level_category.lower().replace(' ', '_')}_count_secondary"] = 0

# Function to sum counts for each top-level category based on the specified primary/secondary columns
def count_activities(row, columns, activities):
    return row[columns].isin(activities).sum()

# Update counts for each category and each row
for index, row in analysis_data.iterrows():
    for top_level_category, activity_list in categories.items():
        # Primary count
        primary_columns_for_category = [f"primary_{act.lower().replace('-', '_').replace('/', '_').replace(' and ', '_').replace(' ', '_').replace('(', '').replace(')', '')}" for act in activity_list]
        primary_count = row[primary_columns_for_category].sum()
        analysis_data.at[index, f"{top_level_category.lower().replace(' ', '_')}_count_primary"] = primary_count
        # Secondary count
        secondary_columns_for_category = [f"secondary_{act.lower().replace('-', '_').replace('/', '_').replace(' and ', '_').replace(' ', '_').replace('(', '').replace(')', '')}" for act in activity_list]
        secondary_count = row[secondary_columns_for_category].sum()
        analysis_data.at[index, f"{top_level_category.lower().replace(' ', '_')}_count_secondary"] = secondary_count

# Columns to sum for total counts
columns_to_sum = [
    'temporal_landmarks_count_primary', 'temporal_landmarks_count_secondary',
    'utilitarian_activities_count_primary', 'utilitarian_activities_count_secondary',
    'discretionary_activities_count_primary', 'discretionary_activities_count_secondary',
    'evaluations_count_primary', 'evaluations_count_secondary'
]

# Add "total_count_primary" by summing all primary counts
analysis_data['total_count_primary'] = analysis_data[[col for col in columns_to_sum if 'primary' in col]].sum(axis=1)

# Add "total_count_secondary" by summing all secondary counts
analysis_data['total_count_secondary'] = analysis_data[[col for col in columns_to_sum if 'secondary' in col]].sum(axis=1)

# Add "total_count" by summing both primary and secondary counts
analysis_data['total_count'] = analysis_data[columns_to_sum].sum(axis=1)

# Select the columns to sum
columns_to_sum = ["category_utilitarian", "category_discretionary", "category_landmark", "category_evaluation", "category_other"]

# Convert columns to numeric, coercing errors to NaN
analysis_data[columns_to_sum] = analysis_data[columns_to_sum].apply(pd.to_numeric, errors='coerce')

# Add "total_count" by summing category counts
analysis_data['total_category'] = analysis_data[['category_utilitarian','category_discretionary', 'category_landmark', 'category_evaluation', 'category_other']].sum(axis=1)

# Filter columns that in focus for the exclusion of individual rows counting above 40 activities
#max_columns = [col for col in analysis_data.columns if col.endswith('_max')]
max_columns = ['total_category', 'total_count']

# Exclude rows where any of the 'max_columns' have values above 40
analysis_data = analysis_data[~(analysis_data[max_columns] > 40).any(axis=1)]

# Mapping of BFI-10 items to Big Five dimensions
# (Reverse coded items are marked with an asterisk *)
big_five_mapping = {
    "Extraversion": ["BFI-10_1", "BFI-10_6"],  # Reverse code BFI-10_6
    "Agreeableness": ["BFI-10_2", "BFI-10_7"],  # Reverse code BFI-10_2
    "Conscientiousness": ["BFI-10_3", "BFI-10_8"],  # Reverse code BFI-10_8
    "Neuroticism": ["BFI-10_4", "BFI-10_9"],  # Reverse code BFI-10_9
    "Openness": ["BFI-10_5", "BFI-10_10"],  # Reverse code BFI-10_5
}

# Function to reverse code a column
def reverse_code(series, scale_max=5):
    return scale_max + 1 - series

# Ensure the BFI-10 quantised columns are numeric
for column in analysis_data.columns:
    if "BFI-10_" in column and "-quantised" in column:
        analysis_data[column] = pd.to_numeric(analysis_data[column], errors="coerce")

# Calculate Big Five scores
for dimension, items in big_five_mapping.items():
    scores = []
    for item in items:
        quantised_col = f"{item}-quantised"
        if item in ["BFI-10_2", "BFI-10_6", "BFI-10_8", "BFI-10_9", "BFI-10_5"]:  # Reverse coded items
            scores.append(reverse_code(analysis_data[quantised_col]))
        else:
            scores.append(analysis_data[quantised_col])
    # Calculate the average score for the dimension
    analysis_data[dimension] = pd.concat(scores, axis=1).mean(axis=1)

# Verify the results
#print(analysis_data[["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]])

# Mapping of ZTPI items to dimensions
# Reverse-coded items are marked with an asterisk (*)
ztpi_mapping = {
    "PastNegative": ["q02", "q03", "q04", "q05", "q07", "q08", "q09", "q10", "q12", "q14"],
    "PastPositive": ["q_01", "q06", "q11", "q13", "q15"],
    "PresentHedonistic": ["q16", "q18", "q19", "q20", "q23", "q24", "q26", "q28", "q29", "q30"],
    "PresentFatalistic": ["q17", "q21", "q22", "q25", "q27"],
    "Future": ["q31", "q32", "q33", "q34", "q35", "q36", "q37", "q38", "q39", "q40"]
}

# List of reverse-coded items (these should be quantified versions)
reverse_coded_items = [
    "q02", "q05", "q07", "q08", "q17", "q21", "q22", "q25", "q27", "q35"
]

# Ensure quantised columns are numeric
for column in analysis_data.columns:
    if "-quantised" in column:
        analysis_data[column] = pd.to_numeric(analysis_data[column], errors="coerce")

# Function to reverse code a column
def reverse_code(series, scale_max=5):
    return scale_max + 1 - series

# Calculate ZTPI scores
for dimension, items in ztpi_mapping.items():
    scores = []
    for item in items:
        quantised_col = f"{item}-quantised"
        if item in reverse_coded_items:
            scores.append(reverse_code(analysis_data[quantised_col]))
        else:
            scores.append(analysis_data[quantised_col])
    # Calculate the average score for the dimension
    analysis_data[dimension] = pd.concat(scores, axis=1).mean(axis=1)

# Verify the results
#print(analysis_data[["PastNegative", "PastPositive", "PresentHedonistic", "PresentFatalistic", "Future"]])

# List of FFA items
ffa_items = [
    "FFA_1", "FFA_2", "FFA_3", "FFA_4", "FFA_5", "FFA_6", "FFA_7", "FFA_8", "FFA_9",
    "FFA_10", "FFA_11", "FFA_12", "FFA_13", "FFA_14"
]

# Reverse-coded items in FFA (if applicable, based on the inventory guidelines)
reverse_coded_items = ["FFA_3", "FFA_8", "FFA_11"]  # Example items; adjust based on the actual inventory scoring.

# Ensure the FFA quantised columns are numeric
for item in ffa_items:
    quantised_col = f"{item}-quantised"
    if quantised_col in analysis_data.columns:
        analysis_data[quantised_col] = pd.to_numeric(analysis_data[quantised_col], errors="coerce")

# Function to reverse code a column
def reverse_code(series, scale_max=5):
    return scale_max + 1 - series

# Compute the composite mindfulness score
mindfulness_scores = []
for item in ffa_items:
    quantised_col = f"{item}-quantised"
    if item in reverse_coded_items:
        mindfulness_scores.append(reverse_code(analysis_data[quantised_col]))
    else:
        mindfulness_scores.append(analysis_data[quantised_col])

# Calculate the average mindfulness score
analysis_data["Mindfulness_Score"] = pd.concat(mindfulness_scores, axis=1).mean(axis=1)

# Verify the results
#print(analysis_data[["Mindfulness_Score"]])

# Define session dummy variables
session_dummies = pd.get_dummies(analysis_data['Session'], prefix='Session', drop_first=False)
analysis_data = pd.concat([analysis_data, session_dummies], axis=1)
#analysis_data = analysis_data.iloc[:, :-4]

# Select the columns to convert to numeric and categorical variables
columns_to_numeric = ["Age", "Stringency_Index", "Reported_Loneliness", "Felt_Loneliness", "Subjective_Confinement", "ConfDuration"]
columns_to_category = ["Sex", "Country", "Screen_Number", "Session"]
#columns_to_standardize = ["ConfDuration"]

# Convert columns to numeric, coercing errors to NaN
analysis_data[columns_to_numeric] = analysis_data[columns_to_numeric].apply(pd.to_numeric, errors='coerce')
analysis_data[columns_to_category] = analysis_data[columns_to_category].astype('category')

#

# Normalize the Stringency_Index (range 0 to 100)
analysis_data['Stringency_Index_normalized'] = analysis_data['Stringency_Index'] / 100

# Normalize the Subjective_Confinement (range 5 to 20)
analysis_data['Subjective_Confinement_normalized'] = (analysis_data['Subjective_Confinement'] - 5) / (20 - 5)

# Calculate the difference between the two normalized values
analysis_data['Stringency_Confinement_Difference'] = (analysis_data['Stringency_Index_normalized'] - analysis_data['Subjective_Confinement_normalized'])

# (Optional) Remove specific variables or DataFrames
del primary_columns, secondary_columns, categories, session_dummies, columns_to_numeric, columns_to_category, columns_to_sum, top_level_category, activity_list, count_activities, primary_columns_for_category, secondary_columns_for_category, primary_count, secondary_count, index, row, max_columns
gc.collect()
dir() # view all defined variables

# (Optional) Write the resulting dataframe out to csv
#analysis_data.to_csv("main_data_for_analysis_final.csv", index=False)

#### --- Exploratory analysis... summary stats and histograms

# Read in any required packages
import os

# This function takes a DataFrame, a list of columns, and an optional parameter top_n (default is 20).
# It calculates the top n unique counts for each specified column and prints them.
def detailed_summary_statistics_print(df, columns, top_n=20):
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

# Updated function to generate structured table outputs
def detailed_summary_statistics(df, columns, top_n=20):
    summary_rows = []
    for column in columns:
        # Get summary statistics
        col_summary = df[column].describe(include='all')
        # Add data type to summary
        col_summary = pd.concat([col_summary, pd.Series({"dtype": df[column].dtype})])
        # Top N unique values
        top_counts = df[column].value_counts().head(top_n)
        # Add summary statistics
        summary_rows.append({
            "Column": column,
            "Count": col_summary.get("count", None),
            "Unique": col_summary.get("unique", None),
            "Mean": col_summary.get("mean", None),
            "Std": col_summary.get("std", None),
            "Min": col_summary.get("min", None),
            "25%": col_summary.get("25%", None),
            "50% (Median)": col_summary.get("50%", None),
            "75%": col_summary.get("75%", None),
            "Max": col_summary.get("max", None),
            "Dtype": col_summary.get("dtype", None),
            "Top N Unique Values": "; ".join([f"{val} ({count})" for val, count in top_counts.items()])
        })
    # Convert to a DataFrame
    summary_df = pd.DataFrame(summary_rows)
    # Format the table for better readability
    summary_df = summary_df[[
        "Column", "Dtype", "Count", "Unique", "Mean", "Std", "Min",
        "25%", "50% (Median)", "75%", "Max", "Top N Unique Values"
    ]]
    return summary_df


# Assuming clean_df is already loaded
#columns_to_summarize = ['categories_selected', 'ranking', 'primary_codes', 'secondary_codes', 'flagged', 'justification_comments']
#columns_to_summarize = ["Sex", "Age", "Stringency_Index", "Reported_Loneliness", "Felt_Loneliness", "Subjective_Confinement", "ConfDuration"]
columns_to_summarize = ['category_utilitarian','category_discretionary', 'category_landmark', 'category_evaluation', 'Country', "Sex", "Age", "Stringency_Index", "Reported_Loneliness", "Felt_Loneliness", "Subjective_Confinement", "ConfDuration"]

# Get detailed summary statistics
#summary_stats = detailed_summary_statistics_print(analysis_data, columns_to_summarize, top_n=10)
summary_stats = detailed_summary_statistics(analysis_data, columns_to_summarize, top_n=10)

# Display the table
print(summary_stats)

# Save to Excel for better visualization
summary_stats.to_excel("summary_statistics.xlsx", index=False, sheet_name="Summary Stats")

# Function to plot histograms for specified columns
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

# Function to plot histograms for specified columns as subplots
def plot_histograms_subplots(df, columns, rows=2, cols=3, save_path=None):
    """
    Plots histograms for specified columns in a single figure with subplots.
    Optionally saves the plot as a PNG file and warns if too many plots are requested.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list): List of column names to plot.
        rows (int): Number of rows for the subplot grid.
        cols (int): Number of columns for the subplot grid.
        save_path (str): Path to save the plot as a PNG file (optional).
    """
    # Error handling
    num_plots = len(columns)
    max_slots = rows * cols
    if num_plots > max_slots:
        print(f"Warning: Number of plots ({num_plots}) exceeds available subplot slots ({max_slots}).")
        print("Some columns will not be plotted. Consider increasing rows and/or cols.")
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    for i, column in enumerate(columns):
        if i < len(axes):  # Ensure we don't exceed available subplot slots
            ax = axes[i]
            # Plot histogram
            df[column].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f'Histogram of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax.tick_params(axis='x', rotation=45)
        else:
            break  # Stop if there are more columns than subplot slots
    # Hide any unused subplots
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    # Save to PNG if save_path is provided
    if save_path:
        # Ensure the directory exists
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Plot saved as {save_path}")
    else:
        plt.show()

# Columns to plot histograms for
#columns_to_plot = ['categories_selected', 'ranking', 'primary_codes', 'secondary_codes']
#columns_to_plot = ['category_utilitarian','category_discretionary', 'category_landmark', 'category_evaluation']
#columns_to_plot = ["Sex", "Age", "Stringency_Index", "Reported_Loneliness", "Felt_Loneliness", "Subjective_Confinement", "ConfDuration"]
columns_to_plot = ['category_utilitarian','category_discretionary', 'category_landmark', 'category_evaluation', 'Country', "Sex", "Age", "Stringency_Index", "Reported_Loneliness", "Felt_Loneliness", "Subjective_Confinement", "ConfDuration"]

# Plot histograms
#plot_histograms(analysis_df, columns_to_plot)
plot_histograms_subplots(analysis_data, columns_to_plot, rows=4, cols=3, save_path="histograms.png")

# (Optional) Remove specific variables or DataFrames
del detailed_summary_statistics, plot_histograms, columns_to_summarize, summary_stats, columns_to_plot
gc.collect()


#### --- Fig. 1 - Total frequency/counts of activity categories across past/future fluency and temporal landmark tasks in Task_Name and projection timeline in Screen_Number

# Select the columns to sum
columns_to_sum = ["category_utilitarian", "category_discretionary", "category_landmark", "category_evaluation", "category_other"]

# Group by 'Screen_Number', 'Task_Name_clean', and 'PID' to get counts for each 'PID'
pid_counts = analysis_data.groupby(["Screen_Number", "Task_Name_clean", "PID"])[columns_to_sum].sum().reset_index()

# Exclude rows where any of the 'max_columns' have values above 40
#max_columns = [col for col in pid_counts.columns if col.endswith('_max')]
#pid_counts = pid_counts[~(pid_counts[max_columns] > 40).any(axis=1)]

# Calculate the minimum, maximum, mean, and standard deviation for each Screen_Number and Task_Name_clean
summary_stats = pid_counts.groupby(["Screen_Number", "Task_Name_clean"])[columns_to_sum].agg(['min', 'max', 'mean', 'std']).reset_index()

# Flatten the MultiIndex columns created by aggregation for easier readability
summary_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary_stats.columns.values]

# Group by 'Screen_Number' and 'Task_Name_clean' again to get the total sums
grouped_df = analysis_data.groupby(["Screen_Number", "Task_Name_clean"])[columns_to_sum].sum().reset_index()

# Merge the summary statistics into grouped_df
grouped_df = pd.merge(grouped_df, summary_stats, on=["Screen_Number", "Task_Name_clean"], how="left")

# Convert the 'Screen_Number' column to numeric instead of string value
grouped_df['Screen_Number'] = pd.to_numeric(grouped_df['Screen_Number'], errors='coerce')

# Create the mirror effect
for i in range(0,len(grouped_df)):
    if grouped_df.iloc[i,1] == 'Past Fluency':
        grouped_df.iloc[i, 0] = -grouped_df.iloc[i, 0]

# Create a dictionary to map old column names to new names with "_count" suffix
rename_dict = {col: f"{col}_count" for col in columns_to_sum}

# Rename the columns in the DataFrame
grouped_df = grouped_df.rename(columns=rename_dict)

# Combine task data for counts (subplot A) and means (subplot B)
task_data_combined_counts = grouped_df.pivot_table(
    index="Screen_Number",
    columns="Task_Name_clean",
    values=["category_utilitarian_count", "category_discretionary_count", "category_landmark_count", "category_evaluation_count", "category_other_count"],
    aggfunc='sum'
)
task_data_combined_means = grouped_df.pivot_table(
    index="Screen_Number",
    columns="Task_Name_clean",
    values=["category_utilitarian_mean", "category_discretionary_mean", "category_landmark_mean", "category_evaluation_mean", "category_other_mean"],
    aggfunc='sum'
)

# Flatten multi-level column names for easier handling in plotting
task_data_combined_counts.columns = ["_".join(col).strip() for col in task_data_combined_counts.columns.values]
task_data_combined_means.columns = ["_".join(col).strip() for col in task_data_combined_means.columns.values]

# Define the width of each bar and calculate positions for Future and Past Fluency
bar_width = 0.8  # Total width for both Future and Past Fluency within each category

# Define key categories
key_categories = [
    "category_utilitarian",
    "category_discretionary",
    "category_landmark",
    "category_evaluation",
    "category_other"
]

# Define a professional color palette using Seaborn
# Use a colorblind-friendly palette (e.g., "muted" or "colorblind")
palette = sns.color_palette("colorblind", n_colors=len(key_categories))

# Create a dynamic color map for counts and means
color_map_count = {f"{category}_count": palette[i] for i, category in enumerate(key_categories)}
color_map_mean = {f"{category}_mean": palette[i] for i, category in enumerate(key_categories)}

# Create the figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(20, 12), sharex=True) # (20, 12) works well for most standard plots. (24, 16) or (30, 18) can be used for larger, complex plots with multiple subplots.

for measure in ['count','mean']:
    # Define which color_map to focus on
    if measure == 'mean':
        ax = axs[1]
        title = "B: Mean Per Person by Category for Fluency Tasks"
        title_label = "B"  # Label for the second subplot
        ylabel = "Mean"
        color_map = color_map_mean  # color_map_count, color_map_mean
        task_data_combined = task_data_combined_means
        x = np.arange(len(task_data_combined_means.index))  # screen numbers
    if measure == 'count':
        ax = axs[0]
        title = "A: Total Counts by Category for Fluency Tasks"
        title_label = "A"  # Label for the first subplot
        ylabel = "Count"
        color_map = color_map_count  # color_map_count, color_map_mean
        task_data_combined = task_data_combined_counts
        x = np.arange(len(task_data_combined_counts.index))  # screen numbers
    # Store the bar tops for both Future and Past Fluency for each category color to plot the lines
    bar_tops_future = {category: [] for category in color_map.keys()}
    bar_tops_past = {category: [] for category in color_map.keys()}
    # Plot each category with grouped bars for Future and Past Fluency
    for i, (category, color) in enumerate(color_map.items()):
        # Get the second part of the category name after splitting by '_'
        category_name = category.split('_')[1]
        future_fluency_col = f"{category}_Future Fluency"
        past_fluency_col = f"{category}_Past Fluency"
        if future_fluency_col in task_data_combined.columns and past_fluency_col in task_data_combined.columns:
            # Calculate offsets for each category so that bars are centered on screen numbers
            offset = (i - (len(color_map) - 1) / 2) * (bar_width / len(color_map))
            # Plot Future Fluency bars and store heights
            future_bars = ax.bar(
                x + offset + bar_width / 24,  # Adjusted for center alignment
                task_data_combined[future_fluency_col],
                bar_width / len(color_map),
                color=color,
                label=category_name.capitalize()
            )
            bar_tops_future[category].extend([bar.get_height() for bar in future_bars])
            # Plot Past Fluency bars and store heights (shifted slightly to the right)
            past_bars = ax.bar(
                x + offset + bar_width / 24,  # Adjusted for center alignment
                task_data_combined[past_fluency_col],
                bar_width / len(color_map),
                color=color
            )
            # Store the top positions of both future and past bars for connecting lines
            bar_tops_past[category].extend([bar.get_height() for bar in past_bars])
    # Plot lines connecting the tops of both Future and Past bars for each color group, including across the boundary
    for i, (category, color) in enumerate(color_map.items()):
        # Calculate offsets for each category so that bars are centered on screen numbers
        offset = (i - (len(color_map) - 1) / 2) * (bar_width / len(color_map))
        # Calculate the center positions of each bar for the dotted lines
        past_positions = x[:3] + offset + bar_width / 24  # x-coordinates for the Past Fluency bars
        future_positions = x[3:] + offset + bar_width / 24  # x-coordinates for the Future Fluency bars
        # Plot the dotted lines for Past Fluency section
        ax.plot(past_positions, bar_tops_past[category][:3], color=color, linewidth=1, linestyle='--', alpha=0.7)
        # Plot the dotted lines for Future Fluency section
        ax.plot(future_positions, bar_tops_future[category][3:], color=color, linewidth=1, linestyle='--', alpha=0.7)
        # Connect the last Past Fluency bar to the first Future Fluency bar across the boundary
        ax.plot(
            [past_positions[-1], future_positions[0]],
            [bar_tops_past[category][2], bar_tops_future[category][3]],
            color=color, linewidth=1, linestyle='--', alpha=0.7)
    # Add a vertical dotted line at x=0
    ax.axvline(x=2.5, color='black', linestyle='--', linewidth=1)
    # Set plot title and labels
    #ax.set_title("Total Counts by Category for Future Fluency and Past Fluency Tasks")
    #ax.set_title("Mean Per Person by Category for Future Fluency and Past Fluency Tasks")
    #ax.set_title(title)
    # Add subplot label (e.g., A or B) to the top left corner
    ax.text(-0.05, 1.1, f"{title_label}", transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=16) # "Count", "Mean"
    ax.set_xticks(x)
    ax.set_xticklabels(["1 year ago", "1 month ago", "1 week ago", "1 week ahead", "1 month ahead", "1 year ahead"],fontsize=14)
    ax.tick_params(axis='y', labelsize=14)  # Increase y-axis tick size
    # Add legend
    if measure == 'count':
        #ax.legend(title="Category", loc="upper left", bbox_to_anchor=(0, 1), fontsize=12, title_fontsize=14)  # Move legend to top left
        ax.legend(title="Category", loc="upper right", bbox_to_anchor=(1, 1), fontsize=14, title_fontsize=16)  # Move legend to top right


# Adjust layout and show the plot
plt.tight_layout()
plt.savefig("Fig1.png", format="png", dpi=300, bbox_inches='tight')  # Save with high resolution (300 dpi)
plt.show()

# (Optional) Remove specific variables or DataFrames
del category, color, currwd, color_map, color_map_count, columns_to_sum, bar_width, color_map_mean, pid_counts, summary_stats, i, rename_dict, task_data_combined, x, fig, ax, bar_tops_future, bar_tops_past, category_name, future_fluency_col, past_fluency_col, offset, past_bars, future_bars, past_positions, future_positions, max_columns
gc.collect()
dir()




#### --- Fig. 1.5 (Sex-Differences) - Total frequency/counts of activity categories across past/future fluency and temporal landmark tasks in Task_Name and projection timeline in Screen_Number

# Select the columns to sum
columns_to_sum = ["category_utilitarian", "category_discretionary", "category_landmark", "category_evaluation", "category_other"]

# Group by 'Screen_Number', 'Task_Name_clean', "Sex", and 'PID' to get counts for each 'PID'
pid_counts = analysis_data.groupby(["Screen_Number", "Task_Name_clean", "Sex", "PID"])[columns_to_sum].sum().reset_index()

# Exclude rows where any of the 'max_columns' have values above 40
#max_columns = [col for col in pid_counts.columns if col.endswith('_max')]
#pid_counts = pid_counts[~(pid_counts[max_columns] > 40).any(axis=1)]

# Calculate the minimum, maximum, mean, and standard deviation for each Screen_Number and Task_Name_clean
summary_stats = pid_counts.groupby(["Screen_Number", "Task_Name_clean", "Sex"])[columns_to_sum].agg(['min', 'max', 'mean', 'std']).reset_index()

# Flatten the MultiIndex columns created by aggregation for easier readability
summary_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary_stats.columns.values]

# Group by 'Screen_Number' and 'Task_Name_clean' again to get the total sums
grouped_df = analysis_data.groupby(["Screen_Number", "Task_Name_clean", "Sex"])[columns_to_sum].sum().reset_index()

# Merge the summary statistics into grouped_df
grouped_df = pd.merge(grouped_df, summary_stats, on=["Screen_Number", "Task_Name_clean", "Sex"], how="left")

# Convert the 'Screen_Number' column to numeric instead of string value
grouped_df['Screen_Number'] = pd.to_numeric(grouped_df['Screen_Number'], errors='coerce')

# Create the mirror effect
for i in range(0,len(grouped_df)):
    if grouped_df.iloc[i,1] == 'Past Fluency':
        grouped_df.iloc[i, 0] = -grouped_df.iloc[i, 0]

# Create a dictionary to map old column names to new names with "_count" suffix
rename_dict = {col: f"{col}_count" for col in columns_to_sum}

# Rename the columns in the DataFrame
grouped_df = grouped_df.rename(columns=rename_dict)

# Combine task data for counts (subplot A) and means (subplot B)
task_data_combined_counts = grouped_df.pivot_table(
    index="Screen_Number",
    columns=["Task_Name_clean", "Sex"],
    values=["category_utilitarian_count", "category_discretionary_count", "category_landmark_count", "category_evaluation_count", "category_other_count"],
    aggfunc='sum'
)
task_data_combined_means = grouped_df.pivot_table(
    index="Screen_Number",
    columns=["Task_Name_clean", "Sex"],
    values=["category_utilitarian_mean", "category_discretionary_mean", "category_landmark_mean", "category_evaluation_mean", "category_other_mean"],
    aggfunc='sum'
)

# Flatten multi-level column names for easier handling in plotting
task_data_combined_counts.columns = ["_".join(col).strip() for col in task_data_combined_counts.columns.values]
task_data_combined_means.columns = ["_".join(col).strip() for col in task_data_combined_means.columns.values]

# Define the width of each bar and calculate positions for Future and Past Fluency
bar_width = 0.8  # Total width for both Future and Past Fluency within each category

# Define key categories
key_categories = [
    "category_utilitarian",
    "category_discretionary",
    "category_landmark",
    "category_evaluation",
    "category_other"
]

# Define a professional color palette using Seaborn
# Use a colorblind-friendly palette (e.g., "muted" or "colorblind")
palette = sns.color_palette("colorblind", n_colors=len(key_categories))

# Create a dynamic color map for counts and means
color_map_count = {f"{category}_count": palette[i] for i, category in enumerate(key_categories)}
color_map_mean = {f"{category}_mean": palette[i] for i, category in enumerate(key_categories)}

# Create the figure with two subplots
fig, axs = plt.subplots(2, 2, figsize=(30, 18), sharex=True)

for i, (measure, sex) in enumerate([('count', 'Male'), ('mean', 'Male'), ('count', 'Female'), ('mean', 'Female')]):
    # Determine subplot indices
    row = 0 if measure == 'count' else 1
    col = 0 if sex == 'Male' else 1
    ax = axs[row, col]
    # Define which color_map to focus on
    if measure == 'count' and sex == 'Male':
        title_label = "A"  # Label for the first subplot - "A: Total Counts by Category for Fluency Tasks"
        ylabel = "Count"
    elif measure == 'mean' and sex == 'Male':
        title_label = "B"  # Label for the second subplot - "B: Mean Per Person by Category for Fluency Tasks"
        ylabel = "Mean"
    elif measure == 'count' and sex == 'Female':
        title_label = "C"  # Label for the first subplot
        ylabel = "Count"
    elif measure == 'mean' and sex == 'Female':
        title_label = "D"  # Label for the second subplot
        ylabel = "Mean"
    title = f"{title_label}: {sex} {'Total Counts' if measure == 'count' else 'Mean Per Person'} by Category for Fluency Tasks"
    sex_blursday = 'M' if sex == 'Male' else 'F'
    if measure == 'count':
        task_data_combined = task_data_combined_counts.filter(regex=f"_{sex_blursday}$") #[task_data_combined_counts["Sex"==f"{sex_blursday}$"]]
        color_map = color_map_count
        x = np.arange(len(task_data_combined_counts.index))  # screen numbers
    elif measure == 'mean':
        color_map = color_map_mean  # color_map_count, color_map_mean
        task_data_combined = task_data_combined_means
        x = np.arange(len(task_data_combined_means.index))  # screen numbers
    else:
        continue
    # Store the bar tops for both Future and Past Fluency for each category color to plot the lines
    bar_tops_future = {category: [] for category in color_map.keys()}
    bar_tops_past = {category: [] for category in color_map.keys()}
    # Plot each category with grouped bars for Future and Past Fluency
    for ii, (category, color) in enumerate(color_map.items()):
        # Get the second part of the category name after splitting by '_'
        category_name = category.split('_')[1]
        future_fluency_col = f"{category}_Future Fluency_{sex_blursday}"
        past_fluency_col = f"{category}_Past Fluency_{sex_blursday}"
        if future_fluency_col in task_data_combined.columns and past_fluency_col in task_data_combined.columns:
            # Calculate offsets for each category so that bars are centered on screen numbers
            offset = (ii - (len(color_map) - 1) / 2) * (bar_width / len(color_map))
            # Plot Future Fluency bars and store heights
            future_bars = ax.bar(
                x + offset + bar_width / 24,  # Adjusted for center alignment
                task_data_combined[future_fluency_col],
                bar_width / len(color_map),
                color=color,
                label=category_name.capitalize()
            )
            bar_tops_future[category].extend([bar.get_height() for bar in future_bars])
            # Plot Past Fluency bars and store heights (shifted slightly to the right)
            past_bars = ax.bar(
                x + offset + bar_width / 24,  # Adjusted for center alignment
                task_data_combined[past_fluency_col],
                bar_width / len(color_map),
                color=color
            )
            # Store the top positions of both future and past bars for connecting lines
            bar_tops_past[category].extend([bar.get_height() for bar in past_bars])
        else:
            continue
    # Plot lines connecting the tops of both Future and Past bars for each color group, including across the boundary
    for ii, (category, color) in enumerate(color_map.items()):
        # Calculate offsets for each category so that bars are centered on screen numbers
        offset = (ii - (len(color_map) - 1) / 2) * (bar_width / len(color_map))
        # Calculate the center positions of each bar for the dotted lines
        past_positions = x[:3] + offset + bar_width / 24  # x-coordinates for the Past Fluency bars
        future_positions = x[3:] + offset + bar_width / 24  # x-coordinates for the Future Fluency bars
        # Plot the dotted lines for Past Fluency section
        ax.plot(past_positions, bar_tops_past[category][:3], color=color, linewidth=1, linestyle='--', alpha=0.7)
        # Plot the dotted lines for Future Fluency section
        ax.plot(future_positions, bar_tops_future[category][3:], color=color, linewidth=1, linestyle='--', alpha=0.7)
        # Connect the last Past Fluency bar to the first Future Fluency bar across the boundary
        ax.plot(
            [past_positions[-1], future_positions[0]],
            [bar_tops_past[category][2], bar_tops_future[category][3]],
            color=color, linewidth=1, linestyle='--', alpha=0.7)
    # Add a vertical dotted line at x=0
    ax.axvline(x=2.5, color='black', linestyle='--', linewidth=1)
    # Set plot title and labels
    #ax.set_title("Total Counts by Category for Future Fluency and Past Fluency Tasks")
    #ax.set_title("Mean Per Person by Category for Future Fluency and Past Fluency Tasks")
    #ax.set_title(title)
    # Add subplot label (e.g., A or B) to the top left corner
    ax.text(-0.015, 1.025, f"{title_label}", transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=16) # "Count", "Mean"
    ax.set_xticks(x)
    ax.set_xticklabels(["1 year ago", "1 month ago", "1 week ago", "1 week ahead", "1 month ahead", "1 year ahead"],fontsize=14)
    ax.tick_params(axis='y', labelsize=14)  # Increase y-axis tick size
    # Add legend
    if measure == 'count' and sex == "Female":
        #ax.legend(title="Category", loc="upper left", bbox_to_anchor=(0, 1), fontsize=12, title_fontsize=14)  # Move legend to top left
        ax.legend(title="Category", loc="upper right", bbox_to_anchor=(1, 1), fontsize=14, title_fontsize=16)  # Move legend to top right

# Add subplot labels manually
#axs[0, 0].text(-0.03, 1.175, "A", transform=axs[0, 0].transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
#axs[0, 1].text(-0.03, 1.175, "B", transform=axs[0, 1].transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
#axs[1, 0].text(-0.03, 1.175, "C", transform=axs[1, 0].transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
#axs[1, 1].text(-0.03, 1.175, "D", transform=axs[1, 1].transAxes, fontsize=18, fontweight='bold', ha='left', va='top')


# Add titles for Male and Female comparisons
fig.text(0.25, 1.009, "Males", fontsize=20, fontweight='bold', ha='center')
fig.text(0.75, 1.009, "Females", fontsize=20, fontweight='bold', ha='center')

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig("Fig1_5.png", format="png", dpi=300, bbox_inches='tight')  # Save with high resolution (300 dpi)
plt.show()

# (Optional) Remove specific variables or DataFrames
del category, color, currwd, color_map, color_map_count, columns_to_sum, bar_width, color_map_mean, pid_counts, summary_stats, i, rename_dict, task_data_combined, x, fig, ax, bar_tops_future, bar_tops_past, category_name, future_fluency_col, past_fluency_col, offset, past_bars, future_bars, past_positions, future_positions, max_columns
gc.collect()
dir()



#### --- Let's check the significance of the means...
from scipy.stats import ttest_ind, mannwhitneyu

# Select the columns to sum
columns_to_sum = ["category_utilitarian", "category_discretionary", "category_landmark", "category_evaluation", "category_other"]

# Step 1: Calculate counts and participant-level means
pid_counts = analysis_data.copy().groupby(["Screen_Number", "Task_Name_clean", "PID"])[columns_to_sum].sum().reset_index()

# Step 2: Aggregate per task
task_counts = pid_counts.groupby(["Screen_Number", "Task_Name_clean"])[columns_to_sum].sum().reset_index()
task_means = pid_counts.groupby(["Screen_Number", "Task_Name_clean"])[columns_to_sum].mean().reset_index()

# Calculate the number of participants and tasks
task_participants = pid_counts.groupby(["Screen_Number", "Task_Name_clean"])["PID"].nunique().reset_index(name="participant_count")
task_task_counts = analysis_data.groupby(["Screen_Number", "Task_Name_clean"]).size().reset_index(name="task_count")

# Merge participant and task counts into `task_counts`
task_counts = task_counts.merge(task_participants, on=["Screen_Number", "Task_Name_clean"], how="left")
task_counts = task_counts.merge(task_task_counts, on=["Screen_Number", "Task_Name_clean"], how="left")

# Step 3: Normalize counts
for column in columns_to_sum:
    task_counts[f"{column}_count_person"] = task_counts[column] / task_counts["participant_count"]  # Normalize by participants
    task_counts[f"{column}_count_task"] = task_counts[column] / task_counts["task_count"]  # Normalize by tasks

# Update column names
task_counts = task_counts.rename(columns={
    "category_utilitarian": "category_utilitarian_count",
    "category_discretionary": "category_discretionary_count",
    "category_landmark": "category_landmark_count",
    "category_evaluation": "category_evaluation_count",
    "category_other": "category_other_count"
})

# Merge task_means for `_mean` columns
grouped_df = pd.merge(task_counts, task_means, on=["Screen_Number", "Task_Name_clean"], suffixes=("", "_mean"))

# Update column names
grouped_df = grouped_df.rename(columns={
    "category_utilitarian": "category_utilitarian_mean",
    "category_discretionary": "category_discretionary_mean",
    "category_landmark": "category_landmark_mean",
    "category_evaluation": "category_evaluation_mean",
    "category_other": "category_other_mean"
})

# Include `_count_person` and `_count_task` columns explicitly
grouped_df = grouped_df.assign(
    **{f"{col}_count_person": task_counts[f"{col}_count_person"] for col in columns_to_sum},
    **{f"{col}_count_task": task_counts[f"{col}_count_task"] for col in columns_to_sum}
)

# Step 4: Separate data into Past and Future Fluency
past_fluency = grouped_df[grouped_df["Task_Name_clean"] == "Past Fluency"]
future_fluency = grouped_df[grouped_df["Task_Name_clean"] == "Future Fluency"]

# Step 5: Perform t-tests and Mann-Whitney U tests for `_mean`, `_count_person`, and `_count_task`

# Initialize the categories list
categories = [f"{col}_count" for col in columns_to_sum] + \
             [f"{col}_mean" for col in columns_to_sum] + \
             [f"{col}_count_person" for col in columns_to_sum] + \
             [f"{col}_count_task" for col in columns_to_sum]

# Initialize lists to store results
t_test_results = []
rank_sum_results = []

# Perform t-tests and rank-sum tests for each category
for category in categories:
    # T-test for means
    t_stat, t_p_value = ttest_ind(past_fluency[category], future_fluency[category], nan_policy='omit')
    t_test_results.append({
        "Category": category,
        "Shapiro Normality (Past Fluency)": stats.shapiro(past_fluency[category]),
        "Shapiro Normality (Future Fluency)": stats.shapiro(future_fluency[category]),
        "T-Statistic": t_stat,
        "T-Test P-Value": t_p_value
    })
    # Rank-sum test for distributions - Note: suitable for non-normally distributed data
    rank_stat, rank_p_value = mannwhitneyu(past_fluency[category], future_fluency[category], alternative='two-sided')
    rank_sum_results.append({
        "Category": category,
        "Rank-Sum Statistic": rank_stat,
        "Rank-Sum P-Value": rank_p_value
    })

# Convert results to DataFrames for easy viewing
t_test_results_df = pd.DataFrame(t_test_results)
rank_sum_results_df = pd.DataFrame(rank_sum_results)

# Display the results
print("T-Test Results:")
print(t_test_results_df)

print("\nRank-Sum Test Results:")
print(rank_sum_results_df)

# Initialize a list to store results
mean_difference_results = []

# Perform mean calculations and t-tests for each category
for category in categories:
    if ("_mean" in category) or ('_count_' in category): # Weighted means for pre-aggregated data
        past_mean = (past_fluency[category] * past_fluency['participant_count']).sum() / past_fluency['participant_count'].sum()
        future_mean = (future_fluency[category] * future_fluency['participant_count']).sum() / future_fluency['participant_count'].sum()
    else: # Calculate the mean for each group
        past_mean = past_fluency[category].mean()
        future_mean = future_fluency[category].mean()
    # Calculate the means for each group
    past_mean = past_fluency[category].mean()
    future_mean = future_fluency[category].mean()
    # Calculate the means difference
    mean_difference = future_mean - past_mean
    # Perform the t-test
    t_stat, p_value = ttest_ind(past_fluency[category], future_fluency[category], nan_policy='omit')
    # Determine significance based on p-value
    significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
    # Append the results
    mean_difference_results.append({
        "Category": category,
        "Past Fluency Mean": round(past_mean, 3),
        "Future Fluency Mean": round(future_mean, 3),
        "Mean Difference (Future - Past)": round(mean_difference, 3),
        "T-Test P-Value": round(p_value, 4),
        "Significance": significance
    })

# Convert results to a DataFrame for easy viewing
mean_difference_df = pd.DataFrame(mean_difference_results)

# Display the results
print(mean_difference_df)
mean_difference_df.to_excel(f"mean_difference_results.xlsx", index=False, sheet_name="Regression Results")


# (Optional) Remove specific variables or DataFrames
del past_mean, future_mean, mean_difference, t_stat, p_value, significance, mean_difference_df, mean_difference_results, rank_sum_results_df, t_test_results_df, category, categories, ttest_ind, mannwhitneyu, past_fluency, future_fluency, t_test_results, rank_sum_results, t_stat, t_p_value
gc.collect()


#### --- Fig. 2 - Total frequency/counts of activity categories across past/future fluency and temporal landmark tasks in Task_Name and projection timeline in Screen_Number
# NOTE: this still needs work/refinement

# Define main categories and subcategories for each one, as shown in the provided images
categories = {
    "Temporal Landmarks": [
        "Personal Temporal Landmark", "Calendar Temporal Landmark", "Reference Points",
        "Personal Narrative Events", "Facts of Life Experiences", "Absence of activity"
    ],
    "Utilitarian Activities": [
        "Household Obligations", "Physiological Needs and Personal Care", "Work/School Activities",
        "Services", "Care Duties", "Return to Routine", "Career Planning", "Civic Duties"
    ],
    "Discretionary Activities": [
        "Recreation", "Entertainment", "Social", "Family", "Altruistic", "Aspirational",
        "Recreation (Services)", "Shopping", "Introspection", "Home Improvement", "Travel",
        "Self-Improvement", "New Connection Seeking", "Spiritual Activities"
    ],
    "Evaluations": ["Negative", "Neutral", "Positive", "Lockdown"]
}

# Define a palette for each main category using seaborn
main_category_palettes = { # Blues, Oranges, Greens, Reds
    "Temporal Landmarks": sns.color_palette("Greens", n_colors=len(categories["Temporal Landmarks"])),
    "Utilitarian Activities": sns.color_palette("Blues", n_colors=len(categories["Utilitarian Activities"])),
    "Discretionary Activities": sns.color_palette("Oranges", n_colors=len(categories["Discretionary Activities"])),
    "Evaluations": sns.color_palette("Reds", n_colors=len(categories["Evaluations"]))
}

# Create the color map for all subcategories
color_map = {}
for main_category, subcategories in categories.items():
    for i, subcategory in enumerate(subcategories):
        color_map[subcategory] = main_category_palettes[main_category][i]


# Loop over each main category and create a plot for each
for main_category, subcategories in categories.items():
    # Create the figure with two subplots
    if main_category in ['Utilitarian Activities','Temporal Landmarks']:
        fig, axs = plt.subplots(2, 1, figsize=(24, 16), sharex=True)
    elif main_category in ['Discretionary Activities']:
        fig, axs = plt.subplots(2, 1, figsize=(30, 16), sharex=True)
    else:
        fig, axs = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
    # Filter columns specific to the current main category's subcategories
    columns_to_sum = [f"primary_{subcat.lower().replace('-', '_').replace('/', '_').replace(' and ', '_').replace(' ', '_').replace('(', '').replace(')', '')}" for subcat in subcategories]
    # Prepare data specifically for the current main category
    main_category_data = analysis_data[columns_to_sum + ["Screen_Number", "Task_Name_clean", "PID"]].copy()
    # Group by 'Screen_Number', 'Task_Name_clean', and 'PID' to get counts for each 'PID'
    pid_counts = main_category_data.groupby(["Screen_Number", "Task_Name_clean", "PID"])[columns_to_sum].sum().reset_index()
    # Exclude rows where any of the 'max_columns' have values above 40
    #max_columns = [col for col in pid_counts.columns if col.endswith('_max')]
    #pid_counts = pid_counts[~(pid_counts[max_columns] > 40).any(axis=1)]
    # Calculate the minimum, maximum, mean, and standard deviation for each Screen_Number and Task_Name_clean
    summary_stats = pid_counts.groupby(["Screen_Number", "Task_Name_clean"])[columns_to_sum].agg(['min', 'max', 'mean', 'std']).reset_index()
    # Flatten the MultiIndex columns created by aggregation for easier readability
    summary_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary_stats.columns.values]
    # Group by 'Screen_Number' and 'Task_Name_clean' again to get the total sums
    grouped_df = main_category_data.groupby(["Screen_Number", "Task_Name_clean"])[columns_to_sum].sum().reset_index()
    # Merge the summary statistics into grouped_df
    grouped_df = pd.merge(grouped_df, summary_stats, on=["Screen_Number", "Task_Name_clean"], how="left")
    # Convert the 'Screen_Number' column to numeric instead of string value
    grouped_df['Screen_Number'] = pd.to_numeric(grouped_df['Screen_Number'], errors='coerce')
    # Create the mirror effect
    for i in range(0, len(grouped_df)):
        if grouped_df.iloc[i, 1] == 'Past Fluency':
            grouped_df.iloc[i, 0] = -grouped_df.iloc[i, 0]
    # Create a dictionary to map old column names to new names with "_count" suffix
    rename_dict = {col: f"{col}_count" for col in columns_to_sum}
    # Rename the columns in the DataFrame
    grouped_df = grouped_df.rename(columns=rename_dict)
    # Combine task data for counts (subplot A) and means (subplot B)
    task_data_combined_counts = grouped_df.pivot_table(
        index="Screen_Number",
        columns="Task_Name_clean",
        values=[f"{col}_count" for col in columns_to_sum],
        aggfunc='sum'
    )
    task_data_combined_means = grouped_df.pivot_table(
        index="Screen_Number",
        columns="Task_Name_clean",
        values=[f"{col}_mean" for col in columns_to_sum],
        aggfunc='sum'
    )
    # Flatten multi-level column names for easier handling in plotting
    task_data_combined_counts.columns = ["_".join(col).strip() for col in task_data_combined_counts.columns.values]
    task_data_combined_means.columns = ["_".join(col).strip() for col in task_data_combined_means.columns.values]
    # Set bar width and then plot for both measures
    bar_width = 0.8  # Width for both Future and Past Fluency within each subcategory
    for measure in ['count', 'mean']:
        # Define which color_map to focus on
        if measure == 'mean':
            ax = axs[1]
            #title = "B: Mean Per Person by Category for Fluency Tasks"
            title_label = "B"  # Label for the second subplot
            ylabel = "Mean"
            task_data_combined = task_data_combined_means
            x = np.arange(len(task_data_combined.index))  # screen numbers
        if measure == 'count':
            ax = axs[0]
            #title = "A: Total Counts by Category for Fluency Tasks"
            title_label = "A"  # Label for the first subplot
            ylabel = "Count"
            task_data_combined = task_data_combined_counts
            x = np.arange(len(task_data_combined.index))  # screen numbers
        bar_tops_future = {subcategory: [] for subcategory in subcategories} #bar_tops_future = {subcategory: [] for f"{subcategory}_count" in columns_to_sum}
        bar_tops_past = {subcategory: [] for subcategory in subcategories} #bar_tops_past = {subcategory: [] for f"{subcategory}_count" in columns_to_sum}
        # Plot each subcategory with grouped bars for Future and Past Fluency
        for i, subcategory in enumerate(subcategories):
            color = color_map[subcategory]
            future_fluency_col = f"primary_{subcategory.lower().replace('-', '_').replace('/', '_').replace(' and ', '_').replace(' ', '_').replace('(', '').replace(')', '')}_{measure}_Future Fluency"
            past_fluency_col = f"primary_{subcategory.lower().replace('-', '_').replace('/', '_').replace(' and ', '_').replace(' ', '_').replace('(', '').replace(')', '')}_{measure}_Past Fluency"
            if future_fluency_col in task_data_combined.columns and past_fluency_col in task_data_combined.columns:
                offset = (i - (len(subcategories) - 1) / 2) * (bar_width / len(subcategories))
                # Plot Future Fluency bars and store heights
                future_bars = ax.bar(x + offset + bar_width / 24, task_data_combined[future_fluency_col], bar_width / len(subcategories), color=color, label=subcategory)
                bar_tops_future[subcategory].extend([bar.get_height() for bar in future_bars])
                # Plot Past Fluency bars and store heights
                past_bars = ax.bar(x + offset + bar_width / 24, task_data_combined[past_fluency_col], bar_width / len(subcategories), color=color)
                bar_tops_past[subcategory].extend([bar.get_height() for bar in past_bars])
        # Plot connecting lines for Future and Past Fluency
        for i, subcategory in enumerate(subcategories):
            # Calculate offsets for each category so that bars are centered on screen numbers
            offset = (i - (len(subcategories) - 1) / 2) * (bar_width / len(subcategories))
            color = color_map[subcategory]
            past_positions = x[:3] + offset + bar_width / 24
            future_positions = x[3:] + offset + bar_width / 24
            ax.plot(past_positions, bar_tops_past[subcategory][:3], color=color, linestyle='--', alpha=0.7, linewidth=1)
            ax.plot(future_positions, bar_tops_future[subcategory][3:], color=color, linestyle='--', alpha=0.7, linewidth=1)
            ax.plot([past_positions[-1], future_positions[0]], [bar_tops_past[subcategory][2], bar_tops_future[subcategory][3]], color=color, linestyle='--', alpha=0.7, linewidth=1)
        # Add a vertical dotted line at x=0
        ax.axvline(x=2.5, color='black', linestyle='--', linewidth=1)
        # Title, labels, and customization
        # Add subplot label (e.g., A or B) to the top left corner
        ax.text(-0.0125, 1.05, f"{title_label}", transform=ax.transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=16)  # "Count", "Mean"
        ax.set_xticks(x)
        ax.set_xticklabels(["1 year ago", "1 month ago", "1 week ago", "1 week ahead", "1 month ahead", "1 year ahead"], fontsize=14)
        ax.tick_params(axis='y', labelsize=14)  # Increase y-axis tick size
        # Add legend
        if measure == 'count':
            # ax.legend(title="Category", loc="upper left", bbox_to_anchor=(0, 1), fontsize=12, title_fontsize=14)  # Move legend to top left
            ax.legend(title="Category", loc="upper right", bbox_to_anchor=(1, 1), fontsize=14, title_fontsize=16)  # Move legend to top right
    # Then save/show the final result
    plt.tight_layout()
    plt.savefig(f"Fig2_{main_category}.png", format="png", dpi=300, bbox_inches='tight')  # Save with high resolution (300 dpi)
    plt.show()

# (Optional) Remove specific variables or DataFrames
del categories, color_map, main_category, subcategories, columns_to_sum, rename_dict, main_category_data, pid_counts, summary_stats, grouped_df, i, task_data_combined, x, bar_width, fig, ax, bar_tops_future, bar_tops_past, subcategory, color, future_fluency_col, past_fluency_col, offset, future_bars, past_bars, past_positions, future_positions, max_columns
gc.collect()

#### --- Main Regression Table

# Function to identify any instances of near-zero variance and/or categorical combinations in the dataset
def identify_low_variance_and_suggest_combinations(subset, threshold=0.01):
    """
    Identifies columns with near-zero variance and suggests combining categories if they are categorical.
    Parameters:
    - subset: pd.DataFrame, the subset of data to analyze.
    - threshold: float, the variance threshold below which a column is considered near-zero variance.
    Returns:
    - A dictionary with two keys:
        - "low_variance": List of columns with near-zero variance.
        - "categorical_suggestions": Suggestions for categorical columns with low unique values.
    """
    results = {
        "low_variance": [],
        "categorical_suggestions": {}
    }
    # Describe the dataset
    desc = subset.describe(include='all')
    for column in subset.columns:
        # Check numeric columns for near-zero variance
        if pd.api.types.is_numeric_dtype(subset[column]):
            variance = subset[column].var()
            if variance < threshold:
                results["low_variance"].append(column)
        # Check categorical columns for suggestions to combine categories
        if pd.api.types.is_categorical_dtype(subset[column]) or subset[column].dtype == 'object':
            unique_count = subset[column].nunique()
            if unique_count <= 3:  # Arbitrary threshold for low unique values
                results["categorical_suggestions"][column] = {
                    "unique_values": subset[column].unique().tolist(),
                    "suggestion": "Consider combining categories to reduce dimensionality."
                }
    return results

# Example usage
results = identify_low_variance_and_suggest_combinations(analysis_data)
print(results)

# (Optional) Remove specific variables or DataFrames
del identify_low_variance_and_suggest_combinations, results
gc.collect()

## ----- Diagnose for Variables that May Need Regularization

# Select the columns to convert to numeric and categorical variables
columns_to_numeric = ["Age", "Stringency_Index", "Reported_Loneliness", "Felt_Loneliness", "Subjective_Confinement", "ConfDuration"]
columns_to_category = ["Sex", "Country", "Screen_Number"]
#columns_to_standardize = ["ConfDuration"]

# Convert columns to numeric, coercing errors to NaN
analysis_data[columns_to_numeric] = analysis_data[columns_to_numeric].apply(pd.to_numeric, errors='coerce')
analysis_data[columns_to_category] = analysis_data[columns_to_category].astype('category')

# --- Step 1) Check Correlations Among Independent Variables:
# Note 1: Look for high correlations (e.g., >0.8 or <-0.8) that may indicate collinearity
# Note 2: For categorical variables, check if specific categories are strongly overrepresented or sparse
numeric_vars = ["Age", "Stringency_Index", "Reported_Loneliness","Felt_Loneliness", "Subjective_Confinement", "ConfDuration"]
corr_matrix = analysis_data[numeric_vars].corr()
print(corr_matrix)

# (Optional) Remove specific variables or DataFrames
del numeric_vars, corr_matrix
gc.collect()

# --- Step 2) Use VIF to check for multicollinearity issues across independent variables
# Note: A VIF above 5 or 10 indicates a potential problem

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Add a constant for the intercept term
X = analysis_data[["Sex", "Age", "Country", "Screen_Number", "Stringency_Index",
            "Reported_Loneliness", "Felt_Loneliness", "Subjective_Confinement", "ConfDuration"]]
X = pd.get_dummies(X, drop_first=True)  # Handle categorical variables if any

# Check for any unexpected non-numeric columns
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    print(f"Non-numeric columns detected: {non_numeric_cols}")
    # Convert problematic columns to numeric (optional) or drop
    X = X.drop(columns=non_numeric_cols)

# Drop rows with NaN or inf values
X_clean = X.replace([np.inf, -np.inf], np.nan).dropna()

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X_clean.columns
vif_data["VIF"] = [variance_inflation_factor(X_clean.values, i) for i in range(X_clean.shape[1])]
print(vif_data)

# (Optional) Remove specific variables or DataFrames
del vif_data, X, X_clean, variance_inflation_factor, non_numeric_cols
gc.collect()

# Use Negative Binomial Regression if your categories represent counts and exhibit overdispersion, especially if each category is analyzed independently or if you want to model a category with its sub-labels.
# Use OLS Regression if the categories are continuous or aggregated values rather than raw counts.

# Define the dataset for regression
dataset = analysis_data.copy()  # Toggle this to grouped_df for aggregated analysis, or analysis_data for de-aggregated analysis

# Define ONCE ONLY
session_dummies = pd.get_dummies(dataset['Session'], prefix='Session', drop_first=False)

# Add session dummy variables to the independent_vars formula
session_fixed_effects = " + ".join(session_dummies.columns)

# Sex and Sex/Loneliness/Confinement/etc Interactions
independent_vars1 = "Screen_Number + Stringency_Index + Reported_Loneliness + Felt_Loneliness + Subjective_Confinement + ConfDuration" #+ ConfDuration
#independent_vars1 += f" + {session_fixed_effects}"
independent_vars2 = independent_vars1 + " + Sex + Age"
independent_vars2 += f" + {session_fixed_effects}"
independent_vars3 = independent_vars2 + " + Sex:Stringency_Index + Sex:Reported_Loneliness + Sex:Felt_Loneliness + Sex:Subjective_Confinement + Sex:ConfDuration"
independent_vars_list = [independent_vars2,independent_vars3]

# BFI-10 and Sex/BFI-10 Interactions
independent_vars1 = "Sex + Age + Screen_Number + Stringency_Index + Reported_Loneliness + Felt_Loneliness + Subjective_Confinement + ConfDuration + Extraversion + Agreeableness + Conscientiousness + Neuroticism + Openness"
independent_vars1 += f" + {session_fixed_effects}"
independent_vars2 = independent_vars1 + " + Sex:Extraversion + Sex:Agreeableness + Sex:Conscientiousness + Sex:Neuroticism + Sex:Openness"
independent_vars_list = [independent_vars1,independent_vars2]

# ZTPI and Sex/ZTPI Interactions
independent_vars1 = "Sex + Age + Screen_Number + Stringency_Index + Reported_Loneliness + Felt_Loneliness + Subjective_Confinement + ConfDuration + PastNegative + PastPositive + PresentHedonistic + PresentFatalistic + Future"
independent_vars1 += f" + {session_fixed_effects}"
independent_vars2 = independent_vars1 + " + Sex:PastNegative + Sex:PastPositive + Sex:PresentHedonistic + Sex:PresentFatalistic + Sex:Future"
independent_vars_list = [independent_vars1,independent_vars2]

# FFA (Mindfulness_Score) and Sex/FFA Interactions
independent_vars1 = "Sex + Age + Screen_Number + Stringency_Index + Reported_Loneliness + Felt_Loneliness + Subjective_Confinement + ConfDuration + Mindfulness_Score"
independent_vars1 += f" + {session_fixed_effects}"
independent_vars2 = independent_vars1 + " + Sex:Mindfulness_Score"
independent_vars_list = [independent_vars1,independent_vars2]

# Define whether to use negative binomial regression or OLS regression
regression_type = 'negative_binomial' # 'negative_binomial', 'OLS'

# Placeholder for regression results
results = []
# Run negative binomial regressions for each dependent variable and condition
for model_no, independent_vars in enumerate(independent_vars_list):
    for dependent_var in ["utilitarian_activities_count_primary", "discretionary_activities_count_primary", "temporal_landmarks_count_primary"]: # "evaluations_count_primary", "total_count_primary", "total_count" - ["utilitarian_activities_count_secondary", "discretionary_activities_count_secondary", "temporal_landmarks_count_secondary"]: # "evaluations_count_secondary", "total_count_secondary", "total_count" - ["category_utilitarian", "category_discretionary", "category_landmark"]: # "category_evaluation", "category_other", "total_category"
        for condition in ["Past Fluency", "Future Fluency"]:
            # Filter data based on condition and screen
            subset = dataset[(dataset["Task_Name_clean"] == condition)]
            #subset[columns_to_standardize] = (subset[columns_to_standardize] - subset[columns_to_standardize].mean()) / subset[columns_to_standardize].std()
            # (OPTIONAL) Include PID as fixed effects by converting to categorical dummy variables if needed
            #subset = pd.get_dummies(subset, columns=["PID"], drop_first=True)
            if regression_type == "OLS":
                # (OPTIONAL) Define the OLS regression model
                #model = ols(f"{dependent_var} ~ {independent_vars}", data=subset).fit()
                # Define the OLS regression model
                model = smf.ols(f"{dependent_var} ~ {independent_vars}", data=subset).fit()
                # Get confidence intervals
                conf_int = model.conf_int(alpha=0.05) # Returns a 95% CI
                conf_int.columns = ["CI Low", "CI High"]  # Name the columns for clarity
                # Fit the null model (only intercept)
                null_model = smf.ols(f"{dependent_var} ~ 1", data=subset).fit()
                # Display model summary
                print(model.summary())
                # Calculate mcfadden r-squared
                mcfadden_r_squared = 1 - (model.llf / null_model.llf)
                # Store the results
                results.append({
                    "Model": model_no,
                    "Dependent Variable": dependent_var,
                    "Condition": condition,
                    "R-squared": model.rsquared, # OLS only...
                    "Pseudo R-squared (McFadden)": mcfadden_r_squared,
                    "Coefficient": model.params,
                    "P-Value": model.pvalues,
                    "CI Low": conf_int["CI Low"],  # Confidence interval lower bounds
                    "CI High": conf_int["CI High"],  # Confidence interval upper bounds
                    "nobs": model.nobs  # Add number of observations
                })
            elif regression_type == "negative_binomial":
                # (OPTIONAL) Define the Negative Binomial model (i.e., without setting alpha)
                #model = smf.glm(f"{dependent_var} ~ {independent_vars}", data=subset, family=sm.families.NegativeBinomial()).fit()
                # (OPTIONAL) Fit the null model (only intercept)
                #null_model = smf.glm(f"{dependent_var} ~ 1", data=subset, family=sm.families.NegativeBinomial()).fit()
                # Step 1: Fit a preliminary Poisson model
                poisson_model = smf.glm(f"{dependent_var} ~ {independent_vars1}", data=subset, family=sm.families.Poisson()).fit()
                # Step 2: Use the Poisson model's deviance to estimate alpha
                alpha_est = poisson_model.deviance / poisson_model.df_resid
                # Step 3: Re-fit with the Negative Binomial model using the estimated alpha
                model = smf.glm(f"{dependent_var} ~ {independent_vars}", data=subset,family=sm.families.NegativeBinomial(alpha=alpha_est)).fit()
                # Get confidence intervals
                conf_int = model.conf_int()
                conf_int.columns = ["CI Low", "CI High"]  # Name the columns for clarity
                # Step 1: Fit a preliminary Poisson model
                null_poisson_model = smf.glm(f"{dependent_var} ~ 1", data=subset, family=sm.families.Poisson()).fit()
                # Step 2: Use the Poisson model's deviance to estimate alpha
                null_alpha_est = null_poisson_model.deviance / null_poisson_model.df_resid
                # Step 3: Re-fit with the Negative Binomial model using the estimated alpha
                null_model = smf.glm(f"{dependent_var} ~ 1", data=subset, family=sm.families.NegativeBinomial(alpha=null_alpha_est)).fit()
                # Calculate Cox-Snell Pseudo R-squared
                #cox_snell_r_squared = 1 - (model.llf / null_model.llf) ** (2 / model.nobs)
                mcfadden_r_squared = 1 - (model.llf / null_model.llf) # maybe replace the cox-snell with alternative pseudo r-squared value (McFadden's)
                # Display model summary
                print(model.summary())
                # Store the results
                results.append({
                    "Model": model_no,
                    "Dependent Variable": dependent_var,
                    "Condition": condition,
                    #"Pseudo R-squared": cox_snell_r_squared, # Negative Binomial only...
                    "Pseudo R-squared (McFadden)": mcfadden_r_squared,  # Negative Binomial only...
                    "Coefficient": model.params,
                    "P-Value": model.pvalues,
                    "CI Low": conf_int["CI Low"],  # Confidence interval lower bounds
                    "CI High": conf_int["CI High"],  # Confidence interval upper bounds
                    "nobs": model.nobs  # Add number of observations
                })
                # Inspect the Log-Likelihood and Est. Alpha Values
                print(f"Model Log-Likelihood: {model.llf}")
                print(f"Estimated Alpha for Model: {alpha_est}")
                print(f"Null Model Log-Likelihood: {null_model.llf}")
                print(f"Estimated Alpha for Null Model: {null_alpha_est}")


# Convert results into a DataFrame for easier display
results_df = pd.DataFrame(results)

# Format the results DataFrame for presentation (showing coefficient and p-value in a simplified way)
formatted_results = []
for _, row in results_df.iterrows():
    dep_var = row["Dependent Variable"]
    condition = row["Condition"]
    model_no = row["Model"]
    coeffs = row["Coefficient"]
    pvals = row["P-Value"]
    ci_low = row["CI Low"]
    ci_high = row["CI High"]
    if regression_type == "OLS":
        rsquared = row["R-squared"] # OLS only
    prsquared = row["Pseudo R-squared (McFadden)"] # Negative Binomial only...
    nobs = row["nobs"]
    # Format coefficients and p-values into strings
    coeff_str = ', '.join([f"{var}: {coef:.3f} (p={pval:.3f})" for var, coef, pval in zip(coeffs.index, coeffs.values, pvals.values)])
    if regression_type == "OLS":
        formatted_results.append({
            "Dependent Variable": dep_var,
            "Condition": condition,
            "Model": model_no,
            "Coefficients (p-values)": coeff_str,
            "CI Low": ci_low,  # Confidence interval lower bounds
            "CI High": ci_high,  # Confidence interval upper bounds
            "R-squared": f"{rsquared:.3f}", # OLS only
            "Pseudo R-squared (McFadden)": f"{prsquared:.3f}",
            "No. Observations": nobs  # Add number of observations
        })
    if regression_type == "negative_binomial":
        formatted_results.append({
            "Dependent Variable": dep_var,
            "Condition": condition,
            "Model": model_no,
            "Coefficients (p-values)": coeff_str,
            "CI Low": ci_low,  # Confidence interval lower bounds
            "CI High": ci_high,  # Confidence interval upper bounds
            "Pseudo R-squared (McFadden)": f"{prsquared:.3f}", # Negative Binomial only...
            "No. Observations": nobs  # Add number of observations
        })

# Convert the formatted results into a DataFrame for display
formatted_results_df = pd.DataFrame(formatted_results)

# Display the formatted results
print(formatted_results_df)

# Export detailed results to a separate sheet (optional)
#with pd.ExcelWriter(f"detailed_and_formatted_results_{regression_type}.xlsx") as writer:
#    formatted_results_df.to_excel(writer, sheet_name="Formatted Results", index=False)
#    results_df.to_excel(writer, sheet_name="Detailed Results", index=False)

# Flatten data into long format
rows = []
for _, entry in results_df.iterrows():  # Iterate over rows in results_df
    dependent_var = entry["Dependent Variable"]
    condition = entry["Condition"]
    model_no = entry["Model"]
    ci_low = entry["CI Low"]
    ci_high = entry["CI High"]
    if regression_type == "OLS":
        rsquared = entry["R-squared"] # OLS only
    prsquared = entry["Pseudo R-squared (McFadden)"] # Negative Binomial only...
    nobs = entry["nobs"]
    for predictor, coef in entry["Coefficient"].items():
        pval = entry["P-Value"][predictor]
        if regression_type == "OLS":
            rows.append({
                "Predictor": predictor,
                "Dependent Variable": dependent_var,
                "Condition": condition,
                "Model": model_no,
                "Coefficient (p-value)": f"{coef:.3f} (p={pval:.3f})",
                "CI Low": ci_low,  # Confidence interval lower bounds
                "CI High": ci_high,  # Confidence interval upper bounds
                "R-squared": f"{rsquared:.3f}", # OLS only
                "Pseudo R-squared (McFadden)": f"{prsquared:.3f}",
                "No. Observations": int(nobs)
            })
        if regression_type == "negative_binomial":
            rows.append({
                "Predictor": predictor,
                "Dependent Variable": dependent_var,
                "Condition": condition,
                "Model": model_no,
                "Coefficient (p-value)": f"{coef:.3f} (p={pval:.3f})",
                "CI Low": ci_low,  # Confidence interval lower bounds
                "CI High": ci_high,  # Confidence interval upper bounds
                "Pseudo R-squared (McFadden)": f"{prsquared:.3f}", # Negative Binomial only...
                "No. Observations": int(nobs)
            })


# Create a DataFrame from the flattened data
long_df = pd.DataFrame(rows)

# Pivot the table
pivot_df = long_df.pivot(index="Predictor", columns=["Dependent Variable", "Condition","Model"], values="Coefficient (p-value)")

# Add Pseudo R-squared and No. Observations as separate rows
metrics = []
for _, entry in results_df.iterrows():
    dependent_var = entry["Dependent Variable"]
    condition = entry["Condition"]
    model_no = entry["Model"]
    if regression_type == "OLS":
        rsquared = f"{entry['R-squared']:.3f}" # OLS only
    prsquared = f"{entry['Pseudo R-squared (McFadden)']:.3f}" # Negative Binomial only...
    nobs = int(entry["nobs"])
    if regression_type == "OLS":
        metrics.append({
            "Metric": "R-squared",
            "Dependent Variable": dependent_var,
            "Condition": condition,
            "Model": model_no,
            "Value": rsquared
        })
        metrics.append({
            "Metric": "Pseudo R-squared (McFadden)",
            "Dependent Variable": dependent_var,
            "Condition": condition,
            "Model": model_no,
            "Value": prsquared
        })
    if regression_type == "negative_binomial":
        metrics.append({
            "Metric": "Pseudo R-squared (McFadden)",
            "Dependent Variable": dependent_var,
            "Condition": condition,
            "Model": model_no,
            "Value": prsquared
        })
    metrics.append({
        "Metric": "No. Observations",
        "Dependent Variable": dependent_var,
        "Condition": condition,
        "Model": model_no,
        "Value": nobs
    })

# Create a metrics DataFrame
metrics_df = pd.DataFrame(metrics)
metrics_df = metrics_df.drop_duplicates(subset=["Metric","Dependent Variable", "Condition", "Model", "Value"]) # "Value"
metrics_df = metrics_df.dropna(subset=["Metric","Dependent Variable", "Condition", "Model", "Value"])

# Pivot metrics DataFrame
metrics_pivot = metrics_df.pivot(index="Metric", columns=["Dependent Variable", "Condition", "Model"], values="Value")

# Combine coefficients and metrics into a single table
combined_df = pd.concat([pivot_df, metrics_pivot])

# Sort for readability and reset index
combined_df = combined_df.sort_index()
combined_df.columns = [f"{col[0]} (Model {col[2]}) ({col[1]})" for col in combined_df.columns]
combined_df.reset_index(inplace=True)
combined_df.rename(columns={"index": "Independent Variables"}, inplace=True)

# Order the columns
column_order = [
    "Independent Variables",
    "utilitarian_activities_count_primary (Model 0) (Past Fluency)",
    "utilitarian_activities_count_primary (Model 1) (Past Fluency)",
    "utilitarian_activities_count_primary (Model 0) (Future Fluency)",
    "utilitarian_activities_count_primary (Model 1) (Future Fluency)",
    'discretionary_activities_count_primary (Model 0) (Past Fluency)',
    'discretionary_activities_count_primary (Model 1) (Past Fluency)',
    'discretionary_activities_count_primary (Model 0) (Future Fluency)',
    'discretionary_activities_count_primary (Model 1) (Future Fluency)',
    'temporal_landmarks_count_primary (Model 0) (Past Fluency)',
    'temporal_landmarks_count_primary (Model 1) (Past Fluency)',
    'temporal_landmarks_count_primary (Model 0) (Future Fluency)',
    'temporal_landmarks_count_primary (Model 1) (Future Fluency)'
]

# Define the desired row order
desired_row_order = [
    "Intercept",
    "Felt_Loneliness",
    "Reported_Loneliness",
    "Stringency_Index",
    "Subjective_Confinement",
    "ConfDuration",
    "Screen_Number[T.2]",
    "Screen_Number[T.3]",
    "Session_S1[T.True]",
    "Session_S2[T.True]",
    "Session_S3[T.True]",
    "Session_SC[T.True]",
    "Age",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
    "PastNegative",
    "PastPositive",
    "PresentHedonistic",
    "PresentFatalistic",
    "Future",
    "Mindfulness_Score",
    "Sex[T.M]",
    "Sex[T.M]:Felt_Loneliness",
    "Sex[T.M]:Reported_Loneliness",
    "Sex[T.M]:Stringency_Index",
    "Sex[T.M]:Subjective_Confinement",
    "Sex[T.M]:ConfDuration",
    "Sex[T.M]:Extraversion",
    "Sex[T.M]:Agreeableness",
    "Sex[T.M]:Conscientiousness",
    "Sex[T.M]:Neuroticism",
    "Sex[T.M]:Openness",
    "Sex[T.M]:PastNegative",
    "Sex[T.M]:PastPositive",
    "Sex[T.M]:PresentHedonistic",
    "Sex[T.M]:PresentFatalistic",
    "Sex[T.M]:Future",
    "Sex[T.M]:Mindfulness_Score",
    "No. Observations",
    "Pseudo R-squared (McFadden)",
]

# Reorder the rows and columns
combined_df = combined_df.set_index('Independent Variables')  # Set index for reordering rows
combined_df = combined_df.reindex(index=desired_row_order, fill_value="N/A")    # Reorder rows + handle missing or additional rows
combined_df = combined_df.reset_index()                      # Reset index
combined_df = combined_df[column_order]                   # Reorder columns

# Function to add stars based on p-values
def add_stars(value):
    import re
    # Check if the value is a string
    if isinstance(value, str):
        match = re.search(r'p=([\d\.]+)', value)
        if match:
            p_value = float(match.group(1))
            if p_value < 0.01:
                return value.split(')')[0] + ')***'
            elif p_value < 0.05:
                return value.split(')')[0] + ')**'
            elif p_value < 0.1:
                return value.split(')')[0] + ')*'
        return value  # Return just the coefficient if no stars
    # If not a string, return the value as-is
    return value


# Define relevant columns and rows
columns_with_stars = [
    "utilitarian_activities_count_primary (Model 0) (Past Fluency)",
    "utilitarian_activities_count_primary (Model 1) (Past Fluency)",
    "utilitarian_activities_count_primary (Model 0) (Future Fluency)",
    "utilitarian_activities_count_primary (Model 1) (Future Fluency)",
    'discretionary_activities_count_primary (Model 0) (Past Fluency)',
    'discretionary_activities_count_primary (Model 1) (Past Fluency)',
    'discretionary_activities_count_primary (Model 0) (Future Fluency)',
    'discretionary_activities_count_primary (Model 1) (Future Fluency)',
    'temporal_landmarks_count_primary (Model 0) (Past Fluency)',
    'temporal_landmarks_count_primary (Model 1) (Past Fluency)',
    'temporal_landmarks_count_primary (Model 0) (Future Fluency)',
    'temporal_landmarks_count_primary (Model 1) (Future Fluency)',
]

rows_with_stars = desired_row_order[:-2]

# Apply the function to the relevant rows and columns
for col in columns_with_stars:
    combined_df.loc[combined_df['Independent Variables'].isin(rows_with_stars), col] = (
        combined_df.loc[combined_df['Independent Variables'].isin(rows_with_stars), col]
        .apply(add_stars)
    )


# Display the resulting DataFrame
print(combined_df)

# Export the combined table
combined_df.to_excel(f"final_regression_results_{regression_type}.xlsx", index=False, sheet_name="Regression Results")


## --- Figures

# Step 1: Filter for Screen_Number variables
screen_vars = ["Screen_Number[T.2]", "Screen_Number[T.3]"]
filtered_data = formatted_results_df.copy()[formatted_results_df["Dependent Variable"].str.contains("activities|landmarks", case=False)]

# Step 2: Check if relevant coefficients are available
# We'll assume the Screen_Number coefficients are stored under "Coefficients (p-values)"
filtered_data = filtered_data[filtered_data["Condition"].isin(["Past Fluency", "Future Fluency"])]

# Define a function to extract coefficients and p-values for all rows
def extract_coefficients_and_pvalues(column):
    pattern = r'([a-zA-Z0-9_\[\].:]+):\s*(-?\d+\.\d+)\s+\(p=([\d\.]+)\)'
    all_extracted = []
    # Iterate through each row in the column
    for index, entry in enumerate(column):
        matches = re.findall(pattern, entry)  # Extract matches for each row
        extracted_data = [
            {"index": index, "Variable": match[0], "Coefficient": float(match[1]), "P-Value": float(match[2])}
            for match in matches
        ]
        all_extracted.extend(extracted_data)  # Accumulate extracted data
    return pd.DataFrame(all_extracted)  # Convert the accumulated data into a DataFrame

# Function to flatten list of lists
def flatten(xss):
    return [x for xs in xss for x in xs]

# Apply the extraction function to the "Coefficients (p-values)" column of filtered_data
extracted_df = extract_coefficients_and_pvalues(filtered_data["Coefficients (p-values)"])

# Expand CI Low and CI High into separate rows
extracted_df['ci_low'] = flatten([CI_low.tolist() for CI_low in filtered_data['CI Low']])
extracted_df['ci_high'] = flatten([CI_low.tolist() for CI_low in filtered_data['CI High']])

# Pivot extracted_df into a wide format where each variable has separate columns for Coefficient and P-Value
extracted_wide = extracted_df.pivot(index='index', columns='Variable', values=['Coefficient', 'P-Value', 'ci_low', 'ci_high'])

# Flatten the multi-level columns
extracted_wide.columns = [f"{value}_{var}" for value, var in extracted_wide.columns]

# Reset index and concatenate
filtered_data.reset_index(drop=True, inplace=True)
extracted_wide.reset_index(drop=True, inplace=True)
filtered_data = pd.concat([filtered_data, extracted_wide], axis=1)


# Step 3: Pivot the data for heatmap
heatmap_pivot = filtered_data.pivot_table(
    index="Dependent Variable",
    columns=["Condition"],
    #values="Coefficients (p-values)"
    values=["Coefficient_Screen_Number[T.2]","Coefficient_Screen_Number[T.3]"]
)

# Flatten multi-level column names for easier plotting
heatmap_pivot.columns = [f"{col[0]}_{col[1]}" for col in heatmap_pivot.columns.to_flat_index()]

# Step 4: Replace missing values with zeros (or any placeholder)
heatmap_pivot = heatmap_pivot.fillna(0)

# Reorder the columns to match the desired order
desired_order = [
    "Coefficient_Screen_Number[T.3]_Past Fluency",
    "Coefficient_Screen_Number[T.2]_Past Fluency",
    "Coefficient_Screen_Number[T.2]_Future Fluency",
    "Coefficient_Screen_Number[T.3]_Future Fluency"
]

# Reindex the heatmap_pivot DataFrame to match the desired column order
heatmap_pivot = heatmap_pivot[desired_order]

# Step 5: Generate the heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    heatmap_pivot,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    ax=ax  # Pass the Axes object to sns.heatmap
)
ax.title("Heatmap of Coefficients: Screen Effects Across Dependent Variables and Conditions", fontsize=14)
ax.set_xticklabels(["1 year ago", "1 month ago" , "1 month ahead", "1 year ahead"],fontsize=10)
plt.xlabel("Condition (Past vs. Future Fluency)", fontsize=14)
ax.set_yticklabels(["Discretionary Activities", "Temporal Landmarks" , "Utilitarian Activities"],fontsize=10)
plt.ylabel("Dependent Variable", fontsize=14)
plt.tight_layout()
plt.show()


# (Optional) Remove specific variables or DataFrames
del independent_vars, columns_to_numeric, columns_to_category, columns_to_standardize, results, dependent_var, condition, subset, model, null_model, poisson_model, alpha_est, null_poisson_model, null_alpha_est, cox_snell_r_squared, results_df, formatted_results_df, formatted_results, _, row, dep_var, condition, coeffs, pvals, rsquared, prsquared, coeff_str
gc.collect()



#### ----- For the Descriptive Statistics section

def generate_descriptive_stats(analysis_data):
    # 1. Descriptive stats for the 5 key categories
    key_categories = ['category_utilitarian', 'category_discretionary', 'category_landmark',
                      'category_evaluation', 'category_other']  # Replace or extend as needed
    #content_analysis_stats = analysis_data[key_categories].describe()
    # Calculate counts and percentages for key categories
    content_analysis_stats = pd.DataFrame({
        "Count": (analysis_data[key_categories] == 1).sum(),  # Count where value is 1
        "Percentage": (analysis_data[key_categories] == 1).mean() * 100  # Percentage where value is 1
    })
    # 2. Descriptive stats for participant details
    participant_details = ['Age', 'Stringency_Index', 'Reported_Loneliness',
                           'Felt_Loneliness', 'Subjective_Confinement', 'ConfDuration']
    # 2a. Sex distribution (categorical example)
    sex_distribution = analysis_data['Sex'].value_counts(normalize=True).multiply(100).round(1)
    sex_distribution_counts = analysis_data['Sex'].value_counts()
    sex_stats = pd.DataFrame({
        "Percentage": sex_distribution,
        "Count": sex_distribution_counts
    })
    # 2b. Country distribution (categorical example)
    country_distribution = analysis_data['Country'].value_counts(normalize=True).multiply(100).round(1)
    country_distribution_counts = analysis_data['Country'].value_counts()
    country_stats = pd.DataFrame({
        "Percentage": country_distribution,
        "Count": country_distribution_counts
    })
    # 2c. Summary stats for numeric variables
    analysis_data[participant_details] = analysis_data[participant_details].apply(pd.to_numeric, errors='coerce')
    participant_numeric_stats = analysis_data[participant_details].describe()
    #
    # 3. Task/Screen summary stats (categorical example)
    task_screen_distribution = analysis_data[["Task_Name_clean","Screen_Number"]].value_counts(normalize=True).multiply(100).round(1)
    task_screen_distribution_counts = analysis_data[["Task_Name_clean","Screen_Number"]].value_counts()
    task_screen_stats = pd.DataFrame({
        "Percentage": task_screen_distribution,
        "Count": task_screen_distribution_counts
    })
    # Return both tables
    return task_screen_stats, content_analysis_stats, sex_stats, country_stats, participant_numeric_stats


# Generate the descriptive stats
task_screen_stats, content_analysis_stats, sex_stats, country_stats, participant_numeric_stats = generate_descriptive_stats(analysis_data)

## -- Table 1 & 2 -- ##
# Print results for review
print(content_analysis_stats)
print(task_screen_stats)

## -- Table 2 -- ##
def combine_stats_into_table(sex_stats, country_stats, participant_numeric_stats):
    """
    Combine stats into one cohesive table for easier interpretation.
    Args:
        sex_stats (pd.DataFrame): Categorical stats for Sex.
        country_stats (pd.DataFrame): Categorical stats for Country.
        participant_numeric_stats (pd.DataFrame): Numeric stats for participants.
    Returns:
        pd.DataFrame: Combined table with formatted stats.
    """
    # Format Sex Stats
    sex_table = sex_stats.copy()
    sex_table["Variable"] = "Sex"
    sex_table["Category"] = sex_table.index
    sex_table.reset_index(drop=True, inplace=True)
    # Format Country Stats
    country_table = country_stats.copy()
    country_table["Variable"] = "Country"
    country_table["Category"] = country_table.index
    country_table.reset_index(drop=True, inplace=True)
    # Combine Sex and Country
    categorical_table = pd.concat([sex_table, country_table], ignore_index=True)
    categorical_table = categorical_table[["Variable", "Category", "Count", "Percentage"]]
    # Format Numeric Stats
    numeric_table = participant_numeric_stats.T.reset_index()
    numeric_table.columns = ["Variable", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    numeric_table.insert(1, "Category", "N/A")  # Add placeholder for category
    # Combine Categorical and Numeric Tables
    combined_table = pd.concat([categorical_table, numeric_table], ignore_index=True)
    # Fill missing columns for consistency
    combined_table["Count"] = combined_table["Count"].fillna("-")
    combined_table["Percentage"] = combined_table["Percentage"].fillna("-")
    return combined_table


# Example usage
combined_table = combine_stats_into_table(sex_stats, country_stats, participant_numeric_stats)

# Save to Excel (optional)
#combined_table.to_excel("combined_descriptive_stats.xlsx", index=False)

# Print results for review
print(combined_table)

## -- Table S1 -- ##
def generate_detailed_descriptive_stats(analysis_data, key_categories, categories):
    """
    Generate count and percentage stats for main categories and detailed breakdowns of subcategories.
    Args:
        analysis_data (pd.DataFrame): The data to analyze.
        key_categories (list): Main categories (e.g., 'category_utilitarian').
        categories (dict): Mapping of 1st-level to 2nd-level categories.
    Returns:
        pd.DataFrame: Main categories stats.
        pd.DataFrame: Combined breakdown of subcategories grouped by their 1st-level codes.
    """
    # Main categories: Count and Percentage
    main_category_stats = pd.DataFrame({
        "Count": (analysis_data[key_categories] == 1).sum(),
        "Percentage": (analysis_data[key_categories] == 1).mean() * 100
    })
    main_category_stats.index.name = "Main Category"
    # Prepare subcategory breakdown
    subcategory_stats = []
    for main_category, subcategories in categories.items():
        # Generate column names for primary and secondary subcategories
        primary_cols = [
            f"primary_{sub.lower().replace('-', '_').replace('/', '_').replace(' and ', '_').replace(' ', '_').replace('(', '').replace(')', '')}"
            for sub in subcategories]
        secondary_cols = [
            f"secondary_{sub.lower().replace('-', '_').replace('/', '_').replace(' and ', '_').replace(' ', '_').replace('(', '').replace(')', '')}"
            for sub in subcategories]
        # Calculate stats for primary and secondary subcategories
        primary_stats = pd.DataFrame({
            "Count": (analysis_data[primary_cols] == 1).sum(),
            "Percentage": (analysis_data[primary_cols] == 1).mean() * 100
        })
        secondary_stats = pd.DataFrame({
            "Count": (analysis_data[secondary_cols] == 1).sum(),
            "Percentage": (analysis_data[secondary_cols] == 1).mean() * 100
        })
        # Add labels for grouping
        primary_stats["Category Type"] = "Primary"
        secondary_stats["Category Type"] = "Secondary"
        # Combine stats for the current main category
        combined_stats = pd.concat([primary_stats, secondary_stats])
        combined_stats["Main Category"] = main_category
        combined_stats.index.name = "Subcategory"
        subcategory_stats.append(combined_stats)
    # Combine all subcategory stats into a single table
    combined_subcategory_stats = pd.concat(subcategory_stats)
    # Reorder columns for better readability
    combined_subcategory_stats = combined_subcategory_stats.reset_index()
    combined_subcategory_stats = combined_subcategory_stats[
        ["Main Category", "Category Type", "Subcategory", "Count", "Percentage"]]
    return main_category_stats, combined_subcategory_stats


# Main categories and subcategories setup
key_categories = ['category_utilitarian', 'category_discretionary', 'category_landmark',
                  'category_evaluation', 'category_other']

# Define mapping of main to subcategories
categories = {
    "Temporal Landmarks": [
        "Personal Temporal Landmark", "Calendar Temporal Landmark", "Reference Points",
        "Personal Narrative Events", "Facts of Life Experiences", "Absence of activity"
    ],
    "Utilitarian Activities": [
        "Household Obligations", "Physiological Needs and Personal Care", "Work/School Activities",
        "Services", "Care Duties", "Return to Routine", "Career Planning", "Civic Duties"
    ],
    "Discretionary Activities": [
        "Recreation", "Entertainment", "Social", "Family", "Altruistic", "Aspirational",
        "Recreation (Services)", "Shopping", "Introspection", "Home Improvement", "Travel",
        "Self-Improvement", "New Connection Seeking", "Spiritual Activities"
    ],
    "Evaluations": ["Negative", "Neutral", "Positive", "Lockdown"]
}

# Generate descriptive statistics
main_category_stats, combined_subcategory_stats = generate_detailed_descriptive_stats(analysis_data, key_categories, categories)

# Save outputs to Excel (optional)
with pd.ExcelWriter("detailed_descriptive_stats.xlsx") as writer:
    main_category_stats.to_excel(writer, sheet_name="Main Categories Stats")
    combined_subcategory_stats.to_excel(writer, sheet_name="Subcategory Breakdown")


#### ----- Coherency check with human RAs and blursday codebot
from statsmodels.stats.inter_rater import fleiss_kappa
from sklearn.metrics import cohen_kappa_score


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
        lambda x: 1 if not x or (isinstance(x, str) and x.strip() == '') or (isinstance(x, str) and x.strip() == '[]') or (isinstance(x, list) and len(x) == 0) else 0
    )
    df['ranking_missing'] = df['ranking'].apply(
        lambda x: 1 if not x or (isinstance(x, str) and x.strip() == '') or (isinstance(x, str) and x.strip() == '[]') or (isinstance(x, list) and len(x) == 0) else 0
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


def load_and_merge_datasets(file_paths, columns_of_interest, merge_keys, subset_indices=None):
    """
    Load multiple Excel files and merge them on specified keys.
    Args:
        file_paths (list): Paths to the Excel files.
        columns_of_interest (list): Columns to load from each file.
        merge_keys (list): Keys to merge datasets on.
        subset_indices (list): Indices of files to include in the subset merge (default: None).
    Returns:
        pd.DataFrame: Merged data for all files.
        pd.DataFrame: Merged data for the specified subset.
    """
    datasets = [add_category_and_ranking_columns(pd.read_excel(file, usecols=columns_of_interest, dtype=str)) for file in file_paths]
    datasets = [dataset.dropna(subset=['categories_selected']) for dataset in datasets]
    merged_all = datasets[0]
    for i, dataset in enumerate(datasets[1:], start=1):
        suffix = f"_{chr(119 + i)}"  # Generate suffixes like _x, _y, _z dynamically
        merged_all = pd.merge(merged_all, dataset, on=merge_keys, suffixes=('', suffix), how='outer')
    # Merge subset of datasets based on subset_indices
    merged_subset = pd.DataFrame() # Create an empty DataFrame
    # Merge subset of datasets based on subset_indices
    if subset_indices is not None:
        # Ensure the subset_indices is valid
        if len(subset_indices) < 2:
            raise ValueError("subset_indices must contain at least two indices.")
        if max(subset_indices) >= len(datasets) or min(subset_indices) < 0:
            raise ValueError("subset_indices contains invalid indices.")
        # Initialize merged_subset with the first dataset in the subset
        merged_subset = datasets[subset_indices[0]]
        for idx in range(0,len(subset_indices)-1):
            suffix1 = f""  # Suffix based on the index
            suffix2 = f"_{idx}"  # Suffix based on the index
            #merged_subset = pd.merge(merged_subset, datasets[idx+1], on=merge_keys, suffixes=('', suffix), how='outer')
            merged_subset = pd.merge(merged_subset, datasets[idx+1], on=merge_keys, suffixes=(suffix1, suffix2), how='outer')
    else:
        merged_subset = pd.DataFrame() # Return an empty DataFrame if no subset indices are provided
    return merged_all, merged_subset


# Reliability metrics calculation
def calculate_reliability_metrics(data, rating_columns):
    # Drop rows with missing values in the rating columns
    data = data[rating_columns].dropna()
    ratings = data[rating_columns].apply(pd.Series.value_counts, axis=1).fillna(0)
    fleiss = fleiss_kappa(ratings)
    cohens = [cohen_kappa_score(data[col1], data[col2]) for col1, col2 in zip(rating_columns[:-1], rating_columns[1:])]
    agreement = (data[rating_columns].apply(lambda x: len(set(x)) == 1, axis=1).mean()) * 100
    return fleiss, cohens, agreement


def generate_reliability_report(file_paths, columns_of_interest, merge_keys, rating_columns_all, rating_columns_subset, subset_indices=[0,2,3]): # By default, subset_indices is the humans only
    """
    Generate a detailed reliability report.
    """
    # Load and merge datasets
    merged_all, merged_subset = load_and_merge_datasets(file_paths, columns_of_interest, merge_keys, subset_indices)
    # Drop duplicates
    merged_all = merged_all.drop_duplicates(subset=['Experiment_ID', 'PID', 'UTC_Date', 'Response_translated'])  # 1,179,345 rows down to 92,849 rows
    merged_subset = merged_subset.drop_duplicates(subset=['Experiment_ID', 'PID', 'UTC_Date', 'Response_translated'])  # 1,179,345 rows down to 92,849 rows
    # Calculate reliability metrics for all raters
    fleiss_all, cohens_all, agreement_all = calculate_reliability_metrics(merged_all, rating_columns_all)
    # Calculate reliability metrics for the subset of raters
    fleiss_subset, cohens_subset, agreement_subset = calculate_reliability_metrics(merged_subset, rating_columns_subset)
    # Output results
    print(f"Ratings columns (all):{rating_columns_all}")
    print("\nAll Raters:")
    print("Fleiss' Kappa:", fleiss_all)
    print("Cohen's Kappa for each pair:", cohens_all)
    print("Percentage Agreement:", agreement_all)
    print(f"\nRatings columns (subset):{rating_columns_subset}")
    print("\nSubset of Raters (1st, 3rd, 4th):")
    print("Fleiss' Kappa:", fleiss_subset)
    print("Cohen's Kappa for each pair:", cohens_subset)
    print("Percentage Agreement:", agreement_subset)
    # Generate reports
    report_all = {
        "Fleiss' Kappa": fleiss_all,
        "Cohen's Kappa (Pairwise)": cohens_all,
        "Percentage Agreement": agreement_all
    }
    report_subset = {
        "Fleiss' Kappa": fleiss_subset,
        "Cohen's Kappa (Pairwise)": cohens_subset,
        "Percentage Agreement": agreement_subset
    }
    return report_all, report_subset, merged_all, merged_subset


def generate_reliability_report_all(file_paths, columns_of_interest, merge_keys, base_categories, suffixes_all=['','_x','_y','_z'], suffixes_subset=['','_0','_1'], subset_indices=[0,2,3]): # By default, subset_indices is the humans only
    """
    Generate a detailed reliability report.
    """
    # Load and merge datasets
    merged_all, merged_subset = load_and_merge_datasets(file_paths, columns_of_interest, merge_keys, subset_indices)
    # Drop duplicates
    merged_all = merged_all.drop_duplicates(subset=['Experiment_ID', 'PID', 'UTC_Date', 'Response_translated'])  # 1,179,345 rows down to 92,849 rows
    merged_subset = merged_subset.drop_duplicates(subset=['Experiment_ID', 'PID', 'UTC_Date', 'Response_translated'])  # 1,179,345 rows down to 92,849 rows
    # Initialise the reports dataset/dictionary
    results={}
    for base_category in base_categories:
        # Construct the full list of rating columns (_all & _subset) for the category
        rating_columns_all = [f"{base_category}{suffix}" for suffix in suffixes_all]
        rating_columns_subset = [f"{base_category}{suffix}" for suffix in suffixes_subset]
        # Drop the empty rows
        merged_all.dropna(subset=[rating_columns_all])
        merged_subset.dropna(subset=[rating_columns_subset])
        # Calculate reliability metrics for all raters
        fleiss_all, cohens_all, agreement_all = calculate_reliability_metrics(merged_all, rating_columns_all)
        # Calculate reliability metrics for the subset of raters
        fleiss_subset, cohens_subset, agreement_subset = calculate_reliability_metrics(merged_subset, rating_columns_subset)
        # Output results
        print(f"Ratings columns (all):{rating_columns_all}")
        print("\nAll Raters:")
        print("Fleiss' Kappa:", fleiss_all)
        print("Cohen's Kappa for each pair:", cohens_all)
        print("Percentage Agreement:", agreement_all)
        print(f"\nRatings columns (subset):{rating_columns_subset}")
        print("\nSubset of Raters (1st, 3rd, 4th):")
        print("Fleiss' Kappa:", fleiss_subset)
        print("Cohen's Kappa for each pair:", cohens_subset)
        print("Percentage Agreement:", agreement_subset)
       # Store results
        results[base_category] = {
            "Fleiss' Kappa - All": fleiss_all,
            "Cohen's Kappa (Pairwise) - All": cohens_all,
            "Percentage Agreement - All": agreement_all,
            "Fleiss' Kappa - Subset": fleiss_subset,
            "Cohen's Kappa (Pairwise) - Subset": cohens_subset,
            "Percentage Agreement - Subset": agreement_subset
        }
    return results, merged_all, merged_subset


def generate_comparison_reliability_table(results_list, subset_labels):
    """
    Create a structured table of reliability metrics comparing multiple subsets.
    Args:
        results_list (list): List of results dictionaries containing reliability metrics for each category.
        subset_labels (list): Labels for the subsets (e.g., "Humans Only", "Lucas and AI").
    Returns:
        pd.DataFrame: A table summarizing the reliability metrics across subsets.
    """
    rows = []
    for subset_label, results in zip(subset_labels, results_list):
        for category, metrics in results.items():
            rows.append({
                "Subset": subset_label,
                "Category": category,
                "Fleiss' Kappa (All)": metrics["Fleiss' Kappa - All"],
                "Cohen's Kappa (All Pairs)": metrics["Cohen's Kappa (Pairwise) - All"],
                "% Agreement (All)": metrics["Percentage Agreement - All"],
                "Fleiss' Kappa (Subset)": metrics["Fleiss' Kappa - Subset"],
                "Cohen's Kappa (Subset Pairs)": metrics["Cohen's Kappa (Pairwise) - Subset"],
                "% Agreement (Subset)": metrics["Percentage Agreement - Subset"]
            })
    # Convert to a DataFrame
    reliability_comparison_table = pd.DataFrame(rows)
    # Format percentages for readability
    reliability_comparison_table["% Agreement (All)"] = reliability_comparison_table["% Agreement (All)"].apply(lambda x: f"{x:.2f}%")
    reliability_comparison_table["% Agreement (Subset)"] = reliability_comparison_table["% Agreement (Subset)"].apply(lambda x: f"{x:.2f}%")
    # Sort the table for better readability
    reliability_comparison_table = reliability_comparison_table.sort_values(by=["Subset", "Category"]).reset_index(drop=True)
    return reliability_comparison_table


# Load Excel files and select specific columns
file_paths = [
    '//Users/stevenbickley/stevejbickley/data_assorted/blursday_assistant/RAs_round_1/LUCAS_blursday_PastFluency-FutureFluency-TemporalLandmarks_2023-03-11_translated.xlsx',
    '/Users/stevenbickley/stevejbickley/data_assorted/blursday_assistant/RAs_round_1/parsed_responses_chunksize_10_2apr2024.xlsx',
    '/Users/stevenbickley/stevejbickley/data_assorted/blursday_assistant/RAs_round_1/SARAH_Blursday Codebook Current Version.xlsx',
    '/Users/stevenbickley/stevejbickley/data_assorted/blursday_assistant/RAs_round_1/Sean Blursday Coded Workbook.xlsx'
]

# Define base categories and suffixes
base_categories = ['category_utilitarian', 'category_discretionary', 'category_landmark', 'category_evaluation']
suffixes_all = ['', '_x', '_y', '_z']  # Raters' suffixes all
suffixes_subset = ['', '_0', '_1']  # Raters' suffixes for subset

# Define parameters
columns_of_interest = ["Experiment_ID", "PID", "UTC_Date", "Response_translated",
                       "categories_selected", "ranking", "chosen_labels"]
merge_keys = ["Experiment_ID", "PID", "Response_translated"]

# Generate report #1 - Humans RAs Only - subset_indices=[0,2,3]
reports1, merged_all, merged_subset1 = generate_reliability_report_all(file_paths, columns_of_interest, merge_keys, base_categories, suffixes_all=['','_x','_y','_z'], suffixes_subset=['','_0','_1'], subset_indices=[0,2,3]) # Humans only

# Generate report #2 - Lucas and AI - subset_indices=[0,1]
reports2, merged_all, merged_subset2 = generate_reliability_report_all(file_paths, columns_of_interest, merge_keys, base_categories, suffixes_all=['','_x','_y','_z'], suffixes_subset=['','_0'], subset_indices=[0,1]) # Lucas and AI

# Generate report #3 - # Sarah, Sean and AI - subset_indices=[1,2,3]
reports3, merged_all, merged_subset3 = generate_reliability_report_all(file_paths, columns_of_interest, merge_keys, base_categories, suffixes_all=['','_x','_y','_z'], suffixes_subset=['','_0','_1'], subset_indices=[1,2,3]) # Sarah and Sean and AI

# (Optional) Output the report
#print("Interrater Reliability Report:")
#for key, value in reports1.items():
#    print(f"{key}: {value}")

# (Optional) Save the merged data for further analysis if needed
#merged_all.to_excel("merged_coding_data.xlsx", index=False)

# Assuming `results1`, `results2`, and `results3` are dictionaries for "Humans Only", "Lucas and AI", and "Sarah and Sean and AI"
# Subset labels
subset_labels = ["Humans Only", "Lucas and AI", "Sarah and Sean and AI"]

# Generate the comparison table
reliability_comparison_table = generate_comparison_reliability_table([reports1, reports2, reports3], subset_labels)

# Print or export the table
print(reliability_comparison_table)
reliability_comparison_table.to_csv("reliability_comparison_metrics.csv", index=False)


# -- Older function to calculate summary statistics and plot histograms
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


