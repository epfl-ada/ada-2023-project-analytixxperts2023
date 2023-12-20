import pandas as pd
import ast

def load_movies():
    """
    Load movie metadata from the specified TSV file.
    
    Returns:
    movies_df (DataFrame): DataFrame containing movie metadata.
    """
    # Define the file path to the movie metadata
    file_path = 'MovieSummaries/movie.metadata.tsv'

    # Define column names for the movies DataFrame
    movies_column_names = ["movie_wikipedia_id", 
                           "movie_freebase_id", 
                           "movie_name", 
                           "movie_release_date", 
                           "movie_box_office_revenue", 
                           "movie_runtime", 
                           "movie_languages", 
                           "movie_countries", 
                           "movie_genres"]

    # Read the data from the TSV file into a DataFrame
    movies_df = pd.read_csv(file_path, delimiter='\t', names=movies_column_names)

    return movies_df


def load_characters():
    """
    Load character metadata from the specified TSV file.
    
    Returns:
    characters_df (DataFrame): DataFrame containing character metadata.
    """
    # Define the file path to the character metadata
    file_path = 'MovieSummaries/character.metadata.tsv'

    # Define column names for the characters DataFrame
    characters_column_names = ["movie_wikipedia_id", 
                               "movie_freebase_id", 
                               "movie_release_date", 
                               "character_name", 
                               "actor_date_of_birth", 
                               "actor_gender", 
                               "actor_height_meters", 
                               "actor_ethnicity_freebase_id", 
                               "actor_name", 
                               "actor_age_at_movie_release", 
                               "character_actor_freebase_map_id", 
                               "character_freebase_id",
                               "actor_freebase_id"]

    # Read the data from the TSV file into a DataFrame
    characters_df = pd.read_csv(file_path, delimiter='\t', names=characters_column_names)

    return characters_df

def load_plot_summaries():
    """
    Load plot summaries from the specified text file.
    
    Returns:
    plot_summaries_df (DataFrame): DataFrame containing plot summaries.
    """
    # Define the file path to the plot summaries text file
    file_path = 'MovieSummaries/plot_summaries.txt'

    # Initialize an empty list to store the data
    data_list = []

    # Read the text file line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t', maxsplit=1)
            if len(parts) == 2:
                wikipedia_id, summary = parts
                data_list.append({"movie_wikipedia_id": wikipedia_id, "movie_plot": summary})

    # Create a DataFrame from the list of dictionaries
    plot_summaries_df = pd.DataFrame(data_list)
    
    plot_summaries_df['movie_wikipedia_id'] = plot_summaries_df['movie_wikipedia_id'].astype(int)
    plot_summaries_df['movie_plot'] = plot_summaries_df['movie_plot'].astype(str)

    return plot_summaries_df

def load_tvtropes():
    """
    Load TV Tropes data from the specified text file.
    
    Returns:
    df_tvtropes_clean (DataFrame): Cleaned DataFrame containing TV Tropes data.
    """
    # Define the file path to the TV Tropes text file
    file_path = 'MovieSummaries/tvtropes.clusters.txt'
    
    # Define column names for the TV Tropes DataFrame
    column_names = ["character_type", "data"]
    
    # Read the data from the text file into a DataFrame
    tvtropes_df = pd.read_csv(file_path, sep='\t', header=None, names=column_names)
    
    def convert_to_dict(string_repr):
        try:
            # Attempt to evaluate the string representation as a dictionary
            return ast.literal_eval(string_repr)
        except (ValueError, SyntaxError):
            # Return None if the conversion is not successful
            return None
    
    # Apply the 'convert_to_dict' function to the 'match' column and create a new 'match_dict' column
    df_tvtropes = tvtropes_df.copy()
    df_tvtropes['match_dict'] = df_tvtropes['data'].apply(convert_to_dict)
    
    # Concatenate the original DataFrame with the normalized values from the 'match_dict' column
    df_tvtropes_clean = pd.concat([df_tvtropes, pd.json_normalize(df_tvtropes['match_dict'])], axis=1)
    
    # Drop the 'match' and 'match_dict' columns as they are no longer needed
    df_tvtropes_clean.drop(['data', 'match_dict'], axis=1, inplace=True)
    
    # Rename columns for clarity with the other DataFrames
    df_tvtropes_clean.rename(columns={'char': 'character_name', 'movie': 'movie_name', 'id': 'character_actor_freebase_map_id', 
                                      'actor': 'actor_name'}, inplace=True)
    
    return df_tvtropes_clean


def count_single_character_occurrence(text, character_name):
    # Initialize a count variable to keep track of occurrences
    count = 0

    # Find the first occurrence of the target in the text
    index = text.find(character_name)

    # Continue searching for the target and updating the count until not found
    while index != -1:
        count += 1

        # Move the index forward to continue searching
        index = text.find(character_name, index + len(character_name))

    return count

def compute_apparition_frequency(character_name, text):
    """
    Compute the maximum count of the character's name in the related plot summary.
    Parameters:
    - character_name: the name of the character
    - text: the plot summary related to the character
    Return:
    -The apparition frequency of the character's name in the plot summary
    """

    # List of common prepositions to exclude from counting if some are countains in the character's name
    prepo_list = ['the', 'at', 'in', 'on', 'of', 'for', 'to', 'with', 'from', 'by', 'about', 'as', 'into', 'like', 'through', 'after', 'over', 
                  'between', 'out', 'against', 'during', 'without', 'before', 'under', 'around', 'among']
    
    # Split the character name into a list of words
    splited_name = character_name.split()
    
    # Initialize the maximum count of the character's name in the text
    max_count = 0
    
    # Iterate through each part of the character's name
    for name in splited_name:
        # Check if the count of the name in the text is greater than the current maximum count and if the name is not a preposition
        if text.count(name) > max_count and name not in prepo_list:
            max_count = text.count(name)
    
    # Return the maximum count of the character's name in the text
    return max_count

def remove_outliers(df, column_name):
    """
    Removes outliers from a specified column in a pandas DataFrame.

    Args:
    df: pandas.DataFrame
        The DataFrame from which outliers will be removed.
    column_name: str
        The name of the column in the DataFrame where outliers are to be identified and removed.

    Returns:
    pandas.DataFrame
        A new DataFrame with outliers removed from the specified column. Rows where the column value 
        is either an outlier or NaN are excluded. All other columns and rows remain unaffected.
    """

    # Calculate the first and third quartiles
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    # Interquartile range
    IQR = Q3 - Q1

    # Define the bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Keeping only the rows that do not contain outliers or are NaN
    condition = ((df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)) | df[column_name].isna()
    cleaned_df = df[condition]

    return cleaned_df

def normalize_columns_zscore(df, columns_to_normalize):
    """
    Applies Z-score normalization to specified columns in a pandas DataFrame.

    Args:
    df: pandas.DataFrame
        The DataFrame containing the columns to be normalized.
    columns_to_normalize: list of str
        A list of column names in the DataFrame that are to be normalized.

    Returns:
    pandas.DataFrame
        The DataFrame with the specified columns normalized using the Z-score method.
        This method standardizes each element in a column by subtracting the mean 
        of the column and dividing by the standard deviation. NaN values in these columns 
        are left unchanged. All other columns in the DataFrame are unaffected.
    """

    for column in columns_to_normalize:
        # Compute the mean and std of the column, skipping NaN values
        mean = df[column].mean()
        std = df[column].std()

        # Apply Z-score normalization, leaving NaN values as they are
        df[column] = df[column].apply(lambda x: (x - mean) / std if not pd.isna(x) else x)

    return df
