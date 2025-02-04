import pandas as pd

#preprocessing stance detection 
def verify_filtering(df):
    # Check for any rows that satisfy both conditions
    
    condition = (df['majority_label'] == 'No majority') | (df['If there is a link-broken option'] == 1)
    
    if df.loc[condition].empty:
        return "Verification passed: No rows with 'No majority' and link broken are present."
    else:
        return "Verification failed: There are still rows with 'No majority' and link broken."


def clean_data(df):
    df = df.loc[~((df['majority_label'] == 'No majority') | (df['If there is a link-broken option'] == 1))]
    return df
def doc_cocat(row):
    if row['gpt_summaries'].strip() != '':
        return row['gpt_summaries']
    return row['docCont']


def combine_columns(df):
    """
    Combine 'Query', 'docTitle', and 'doc' columns into a new 'Input' column.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Query', 'docTitle', and 'doc' columns.

    Returns:
    pd.DataFrame: DataFrame with a new 'Input' column.
    """
    df['Input'] = df['Query'] + ' ' + df['docTitle'] + ' ' + df['doc']
    return df


def baseline_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the DataFrame for the baseline dataset.
    
    This function combines the answers from columns 'answer1', 'answer2', and 'answer3' 
    into a list for each row, storing them in a new 'labels' column. The resulting 
    DataFrame includes only 'docID', 'Input', 'labels', and 'majority_label' columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the original data.

    Returns:
    pd.DataFrame: The processed DataFrame with the specified columns.
    """
    df['labels'] = df[['answer1', 'answer2', 'answer3']].apply(lambda x: x.dropna().tolist(), axis=1)
    df = df[['docID', 'Query','docTitle', 'doc', 'Input', 'labels', 'majority_label']]
    return df


def multip_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the DataFrame for the multiperspective dataset.
    
    This function combines the answers from columns 'answer1', 'answer2', and 'answer3' 
    into a list, then explodes the list into separate rows for each answer, effectively 
    creating multiple rows for each original input. It also removes any rows where 
    the 'label' is 'Link-broken'.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the original data.

    Returns:
    pd.DataFrame: The processed DataFrame with exploded labels.
    """
    df['label'] = df[['answer1', 'answer2', 'answer3']].apply(lambda x: x.dropna().tolist(), axis=1)
    df = df.explode('label').reset_index(drop=True)
    df = df[['docID', 'Query' ,'docTitle', 'doc', 'Input', 'label', 'majority_label']]
    return df


def save_dataset(df_train: pd.DataFrame, df_test: pd.DataFrame, df_val: pd.DataFrame, output_dir: str, choice: str) -> None:
    """
    Save the processed DataFrames for train, test, and val splits to a specified output directory 
    based on the chosen dataset type.

    This function processes each DataFrame (train, test, val) using either the `baseline_data` or 
    `multip_data` function, and then saves each processed DataFrame to the specified output directory 
    with filenames that reflect both the dataset type and the split (train, test, or val).

    Parameters:
    df_train (pd.DataFrame): The DataFrame containing the train data.
    df_test (pd.DataFrame): The DataFrame containing the test data.
    df_val (pd.DataFrame): The DataFrame containing the validation data.
    output_dir (str): The directory where the processed DataFrames will be saved.
    choice (str): The type of dataset to generate ('baseline' or 'multip').

    Returns:
    None
    """
    # Process each DataFrame based on the choice
    if choice == 'baseline':
        processed_train = baseline_data(df_train)
        processed_test = baseline_data(df_test)
        processed_val = baseline_data(df_val)
    elif choice == 'multip':
        processed_train = multip_data(df_train)
        processed_test = multip_data(df_test)
        processed_val = multip_data(df_val)
    else:
        raise ValueError("Invalid choice! Use 'baseline' or 'multip'.")

    # Print the shapes of the processed DataFrames
    print(f"Train DataFrame shape: {processed_train.shape}")
    print(f"Test DataFrame shape: {processed_test.shape}")
    print(f"Val DataFrame shape: {processed_val.shape}")

    # Construct and save each DataFrame to the appropriate file path
    processed_train.to_csv(f"{output_dir}/train_{choice}.csv", index=False)
    processed_test.to_csv(f"{output_dir}/test_{choice}.csv", index=False)
    processed_val.to_csv(f"{output_dir}/val_{choice}.csv", index=False)

    # Confirm the files have been saved
    print(f"Train, Test, and Val datasets for '{choice}' saved to {output_dir}")
