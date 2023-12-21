import ast
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf



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



def filtering(df, performance_param):
    ''' 
    Filter the dataframe according 
    to the performance parameter

    Parameters:
    -df: A dataframe
    -performance_param: A string
    
    Return: 
    -A dataframe
    '''
    df_copy = df.copy()
    if (performance_param == 'box_office'):
        df_copy.dropna(subset=['movie_box_office_revenue'],inplace=True)
        return df_copy

    elif (performance_param == 'rating'):
        vote_threshold = 100
        df_copy.dropna(subset=['rating_average','movie_box_office_revenue'],inplace=True)
        df_bis = df_copy[df_copy['rating_count'] > vote_threshold]
        return df_bis

    else:
        print("Invalid parameters")
        
    

def year_release_split(df, number_parts):
    ''' 
    Split the dataframe in number_parts 
    according to the year of release

    Parameters:
    -df: A dataframe
    -number_parts: An integer
    
    Return: 
    -df: A dictionary of dataframes 
    -cutoff: A list of the cutoff years
    '''
    total_size = df.shape[0]
    share = int(total_size/(number_parts))
    cumulative_data = df['movie_release_year'].value_counts().sort_index().cumsum()
    cutoff = []
    for i in range(number_parts):
        relevant_data = cumulative_data[cumulative_data > (i)*share] 
        if not relevant_data.empty:
            cutoff.append(relevant_data.index[0])

    period_dataframes = {}
    for i in range(len(cutoff)):
        if i < len(cutoff)-1:
            period_df = df[
                (df['movie_release_year'] > cutoff[i]) &
                (df['movie_release_year'] <= cutoff[i+1])
            ]
        if i == len(cutoff)-1:
            period_df = df[
                (df['movie_release_year'] > cutoff[i])
            ]

        period_dataframes[f'df_period{i+1}'] = period_df
    return period_dataframes,cutoff  


## Now we define some utility functions

def common_movie_gender(str1, str2):  
    '''
    Check if two movies have at least one common genre

    Parameters:
    -str1: A string
    -str2: A string

    Return:
    -common_elements: A boolean
    '''
    # Convert string representation of lists to actual lists
    list1 = [genre.strip() for genre in str1.strip("[]").split(",")]
    list2 = [genre.strip() for genre in str2.strip("[]").split(",")]

    set1 = set(list1)
    set2 = set(list2)
    common_elements = set1.intersection(set2)
    return common_elements

    
def min_max_scaling(df, column_name):
    '''
    Scale the values of a column between 0 and 1

    Parameters:
    -df: A dataframe
    -column_name: A list of string

    Return:
    -df: A dataframe 
    '''
    for col in column_name:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

def binarize_diversity(df, column_name):
    '''
    Binarize the diversity column using the median value

    Parameters:
    -df: A dataframe
    -column_name: A string

    Return: 
    -df: A dataframe 
    '''
    median_value = df[column_name].median()
    df['treat'] = np.where(df[column_name] > median_value, 1, 0)
    return df

def add_propensity_score(df,formula_propensity_score,diversity_name,column_name):
    '''
    Add a column with the propensity score to the dataframe

    Parameters:
    -df: A dataframe
    -formula_propensity_score: A string
    -diversity_name: A string
    -column_name: A string
    
    Return: 
    -df: A dataframe 
    '''
    df = binarize_diversity(df,diversity_name)
    df = min_max_scaling(df,column_name)
    mod = smf.logit(formula= formula_propensity_score, data=df)
    res = mod.fit(disp=False)
    df['Propensity_score'] = res.predict()
    return df

def get_similarity(propensity_score1, propensity_score2):
    '''
    Calculate similarity for instances with given propensity scores

    Parameters:
    -propensity_score1: A float
    -propensity_score2: A float
    
    Return: A float
    '''
    return 1-np.abs(propensity_score1-propensity_score2)

def compute_balance_df(df): 
    '''
    Compute a balanced dataframe using the propensity score

    Parameters:
    -df: A dataframe

    Return: 
    -df: A dataframe
    '''
    # Separate the treatment and control groups
    treatment_df = df[df['treat'] == 1]
    control_df = df[df['treat'] == 0]

    # Create an empty graph
    G = nx.Graph()

    # Loop through all the pairs of instances
    for control_id, control_row in control_df.iterrows():
        for treatment_id, treatment_row in treatment_df.iterrows():
            # Check if the two instances have the same number of languages and at least one common genre
            if (control_row['movie_languages_count'] == treatment_row['movie_languages_count']) \
                and common_movie_gender(control_row['movie_genres'], treatment_row['movie_genres']):
                # Calculate the similarity 
                similarity = get_similarity(control_row['Propensity_score'],
                                            treatment_row['Propensity_score'])

                # Add an edge between the two instances weighted by the similarity between them
                G.add_weighted_edges_from([(control_id, treatment_id, similarity)])

    # Generate and return the maximum weight matching on the generated graph
    matching = nx.max_weight_matching(G)

    matched = [i[0] for i in list(matching)] + [i[1] for i in list(matching)]
    balanced_df = df.loc[matched]
    return balanced_df

def regression(df,formula,formula_propensity_score,diversity_name,column_name):
    '''
    Perform an ols regression on a dataframe according to the formula

    Parameters:
    -df: A dataframe
    -formula: A string
    -formula_propensity_score: A string
    -diversity_name: A string
    -column_name: A lsit of string

    Return: 
    -balanced_df: A dataframe 
    -mod: A model
    '''
    df = add_propensity_score(df, formula_propensity_score, diversity_name, column_name)
    balanced_df = compute_balance_df(df)
    mod = smf.ols(formula=formula, data=balanced_df)
    return balanced_df,mod


def compute_all_regressions(period_df,formula,formula_propensity_score,diversity_name,columns):
    '''
    Perform the regression for each period and store the results in a list

    Parameters:
    -period_df: A dataframe
    -formula: A string
    -formula_propensity_score: A string
    -diversity_name: A string
    -columns: A lsit of string


    Return: 
    -balanced_dfs: A list of tuples containing the balanced dataframe
    -mods: A list of tuples containing the model
    '''
    # Create an empty list to store the results
    results = []

    # Iterate over periods
    for period_num in range(1, 11):
        period_key = f'df_period{period_num}'
        
        # Perform regression for each period
        balanced_df, mod = regression(
            period_df[period_key],
            formula=formula,
            formula_propensity_score=formula_propensity_score,
            diversity_name=diversity_name,
            column_name=columns
        )
        
        # Store the results in a tuple and append to the list
        result = (balanced_df, mod)
        results.append(result)

    # Extract the results for each period
    balanced_dfs, mods = zip(*results)
    print("done")
    return balanced_dfs,mods

def common_movie_genre(str1, str2, similarity_rate = 1.):  #check if two movies have at least one common genre
    """Parse and compare two sets of movie genres and returns true if, given a certain threshold, movie genres are
       considered to match and return False otherwise.

    Args:
        str1 (str): First movie genre string. 
        str2 (str): Second movie genre string.
        similarity_rate (float, optional): proportion of the genre set with the highest number of elements that are
                                           considered as a match.

    Returns:
        bool: True if enough genres are matching, otherwise false
    """    
    list1 = [genre.strip() for genre in str1.strip("[]").split(",")]
    list2 = [genre.strip() for genre in str2.strip("[]").split(",")]

    set1 = set(list1)
    set2 = set(list2)
    intersection_set = len(set1.intersection(set2))

    if intersection_set/max([len(set1),len(set2)]) >= similarity_rate:
        return True
    else:
        return False
    

def match_on_attributes(sample1, sample2):
    """Determines exact matching between two samples of the movie dataset based on genre.

    Args:
        sample1 (pd.Series): First sample to match
        sample2 (pd.Series): Second sample to match

    Returns:
        bool: Returns true if the samples can be matched, false otherwise
    """    
    cond = True
    cond = cond & common_movie_genre(sample1['movie_genres'],sample2['movie_genres'], similarity_rate=1/2)
    return cond

    

def add_propensity_score(df, formula):
    """Fits a linear regressor to a dataframe given a formula to compute propensity scores.

    Args:
        df (pd.Dataframe): Data on which the regression is made.
        formula (string): logical formula used for the linear regressor

    Returns:
        pd.Dataframe: _description_
    """
    #df = standardize_continuous_features(df)
    mod = smf.ols(formula= formula, data=df)
    res = mod.fit()
    df['Propensity_score'] = res.predict()
    return df

def get_similarity(propensity_score1, propensity_score2):
    '''Computes a similarity metric based on propensity scores of two samples.
       Similar elements will have a value close to 1 and different ones close to 0'''
    return 1-np.abs(propensity_score1-propensity_score2)


def filtering(df, performance_param):
    """Filters NaN values in dataframe for performance variables.

    Args:
        df (pd.Dataframe): dataframe with the values to filter
        performance_param (string): name of the columns/variable to filter

    Returns:
        pd.Dataframe: Dataframe filtered based on the performance variables 
    """    
    df_copy = df.copy()
    if (performance_param == 'movie_box_office_revenue'):
        df_copy.dropna(subset=['movie_box_office_revenue'],inplace=True)
        return df_copy

    elif (performance_param == 'rating_average'):
        df_copy.dropna(subset=['rating_average'],inplace=True)
        df_bis = df_copy[df_copy['rating_count'] > 100]
        return df_bis

    else:
        print("Invalid parameters")


def define_treat_control(df,match_on):
    """Creates treatment and control groups depending on the chosen diversity feature. 
       The split is based on the median value of the distribution.

    Args:
        df (pd.Dataframe): Dataframe containing the specified diversity feature
        match_on (str): Feature on which the treatment control split is made
    """    
    if match_on == 'ethnicity_diversity':
        threshold = df[match_on].median()
    elif match_on == 'gender_diversity':
        threshold = df[match_on].median()
    else:
        threshold=None
        print('Error in the feature that needs to be matched')

    # Assign control or treatment status depending on threshold
    df['treatment'] = np.where(df[match_on] > threshold, 1, 0)

    return df




def balanced_dataset(data, match_on, perf_var, out_df_name = None, plot_distrib=False):
    """Performs the pipeline to create control and treatment groups with matching of the confounders.
       Based on the steps of Exercise 05 - Causal analysis of observational data.
       Credits to: [Tiziano Piccardi](https://piccardi.me/) and [Kristina Gligoric](https://kristinagligoric.github.io/)

    Args:
        data (pd.Dataframe): Dataframe on which the balancing needs to be performed.
        match_on (str): Diversity variable that will be used to create control and treatment variables.
        perf_var (str): Performance variable that will be used to filter the dataframe before processing.
        out_df_name (str, optional): Will save the dataset to csv if any name is provided . Defaults to None.
        plot_distrib (bool, optional): Plots the distribution of matching features for treatment and control before matching

    Returns:
        pd.Dataframe: Dataframe of matches made based on propensity scores.
    """    
    print('filtering df')
    df = filtering(data,perf_var)
    df = define_treat_control(df=df, match_on=match_on)
    if plot_distrib:
        plot_feature_distrib(df=df)
    
    # Make the prediction of propensity scores using a linear regressor
    mod = smf.logit(formula= 'treatment ~ movie_release_year + movie_languages_count', data=df)
    res = mod.fit()
    df['Propensity_score'] = res.predict()
    
    print('Creating graph')
    treatment = df[df['treatment'] == 1]
    control = df[df['treatment'] == 0]

    G = nx.Graph()
    # Loop through all the pairs of instances
    for control_id, control_row in control.iterrows():
        for treatment_id, treatment_row in treatment.iterrows():
            if (match_on_attributes(control_row,treatment_row)):
                # Compute the similarity 
                similarity = get_similarity(control_row['Propensity_score'],
                                            treatment_row['Propensity_score'])

                # Add an edge between the two instances weighted by the similarity between them
                G.add_weighted_edges_from([(control_id, treatment_id, similarity)])

    # Generate and return the maximum weight matching on the generated graph
    print('performing matching')
    matching = nx.max_weight_matching(G)

    matched = [i[0] for i in list(matching)] + [i[1] for i in list(matching)]
    balanced_df = df.loc[matched]
    
    # Save the balanced dataframe if a name was specified
    if out_df_name:
        print('saving file')
        file = './generated/' + out_df_name
    balanced_df.to_csv(file)
    return balanced_df

def compute_results(balanced_df, perf_var, diversity_var):
    """Plots control and treatment distributions and performs a statistical analysis using linear regression.

    Args:
        balanced_df (pd.Dataframe): Dataframe with a control and treatment group(in a 'treatment' column) and matched samples
        perf_var (str): Name of the performance variable to plot and compare with linear regression.


    Returns:
        TODO: regressive model trained on the balanced dataframe  
    """    
    # Compute regressive line parameters
    treatment_balanced = balanced_df[balanced_df['treatment'] == 1]
    control_balanced = balanced_df[balanced_df['treatment'] == 0]
    mod = smf.ols(formula= '{} ~ C(treatment)'.format(perf_var), data=balanced_df)
    res = mod.fit()

    # Plot density distributions for treatment and control groups
    plt.figure()
    ax = sns.histplot(treatment_balanced[perf_var], kde=True, stat='density', color='blue', label='High diversity', log_scale=True)
    ax = sns.histplot(control_balanced[perf_var], kde=True, stat='density', color='orange', label='Low diversity',log_scale=True)
    ax.set(title='{} density distribution after matching'.format(perf_var),xlabel='z-scored {}'.format(perf_var), ylabel='Density')
    plt.legend()

    # Scatter plot with a regression line
    intercept = mod.fit().params['Intercept']
    ethnicity_coef = mod.fit().params['C(treatment)[T.1]']
    x_values = np.linspace(min(balanced_df[diversity_var]), max(balanced_df[diversity_var]), 100)
    y_values = intercept + ethnicity_coef * x_values
    plt.figure()
    plt.title('Relationship between {} and {}'.format(diversity_var,perf_var))
    plt.ylabel('z-scored {}'.format(perf_var))
    sns.scatterplot(data=balanced_df,x=diversity_var,y=perf_var)
    sns.lineplot(x=x_values,y=y_values,color='red')
    plt.tight_layout()
    plt.show()
    print(res.summary())

    return mod

def plot_feature_distrib(df):
    """Plots the distribution of features to match for treatment and control groups

    Args:
        df (pd.Dataframe): movie dataframe with identified treatment and control groups 
    """   
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(6,8), sharex=True)
    sns.violinplot(data=df,x='treatment',y='movie_release_year',ax=ax1)
    ax1.set_title('movies\' release year distributions')
    sns.violinplot(data=df,x='treatment',y='movie_languages_count',ax=ax2)
    ax2.set_title('movies\'s language count distributions')
    plt.tight_layout()
    plt.show()




def plot_regression_line(balanced_dfs,mods,
                         cutoff,param_div,param_perf,
                         xlabel='Ethnicity Diversity',ylabel='Box Office Revenue'):
    '''
    Plot the regression line and the datapoints for each period
    '''
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(25, 8))
    axes = axes.flatten()
    for i in range(10):
        df = balanced_dfs[i]
        mod = mods[i]
        #Extract the intercept and the slope
        intercept_coef = mod.fit().params['Intercept']
        ethnicity_coef = mod.fit().params[param_div]
        #Plot the regression line
        x_values = np.linspace(min(df[param_div]), max(df[param_div]), 100)
        y_values = intercept_coef + ethnicity_coef * x_values
        axes[i].plot(x_values, y_values, color='red', label='Regression Line')
        #Plot the datapoints
        axes[i].scatter(df[param_div], df[param_perf], s=5, label='Data Points')
        axes[i].set_xlabel(f'{xlabel}: pval= {mod.fit().pvalues[param_div].round(3)}, intercept pval= {mod.fit().pvalues["Intercept"].round(3)}, \n intercept={mod.fit().params["Intercept"].round(3)} ,slope= {mod.fit().params[param_div].round(3)}')
        axes[i].set_ylabel(ylabel)
        axes[i].legend()
        if i < len(cutoff)-1:
            axes[i].set_title(f'{int(cutoff[i])} - {int(cutoff[i+1])}')
        if i == len(cutoff)-1:
            axes[i].set_title(f'{int(cutoff[i])} - {int(df["movie_release_year"].max())}')
    plt.tight_layout()
    plt.show()


def plot_comparison(balanced_dfs,cutoff,param_perf,movie_charac):
    '''
    Plot the distribution of the performance parameter for each group for each period
    '''
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(25, 8))
    axes = axes.flatten()
    for i in range(10):
        df = balanced_dfs[i]
        #Recollect the movie box office revenue for each movie
        movie_charac['movie_name'].drop_duplicates(keep='first',inplace=True)
        movie_charac_reduced = movie_charac[['movie_freebase_id','movie_box_office_revenue']]
        movie_charac_reduced.rename(columns={'movie_box_office_revenue':'Box_office'},inplace=True)
        merge = pd.merge(balanced_dfs[i],movie_charac_reduced,how='left',on='movie_freebase_id')
        #Split the dataframe according to the group
        treatment_df = merge[merge['treat'] == 1]
        control_df = merge[merge['treat'] == 0]
        ax = axes[i]
        #Plot the distribution of the performance parameter for each group
        if param_perf == 'Box_office':
            sns.histplot(treatment_df[param_perf], label='Treat', kde=True, stat='density', color='blue', log_scale=True, ax=ax)
            sns.histplot(control_df[param_perf], label='Control', kde=True, stat='density', color='orange', log_scale=True, ax=ax)
        elif param_perf == 'rating_average':
            sns.histplot(treatment_df[param_perf],label='Treat', kde=True, stat='density', color='blue', ax=ax)
            sns.histplot(control_df[param_perf],label='Control', kde=True, stat='density', color='orange', ax=ax)
        if param_perf == 'Box_office':
            ax.set_xlabel('Movie Box Office Revenue')
        elif param_perf == 'rating_average':
            ax.set_xlabel('Movie Rating Average')
        ax.set_ylabel('Density')
        ax.legend()
        if i < len(cutoff)-1:
            ax.set_title(f'{int(cutoff[i])} - {int(cutoff[i+1])}')
        if i == len(cutoff)-1:
            ax.set_title(f'{int(cutoff[i])} - {int(df["movie_release_year"].max())}')
    plt.tight_layout()
    plt.show()

