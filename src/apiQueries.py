import pandas as pd
import requests
import time
from SPARQLWrapper import SPARQLWrapper, JSON
import tmdbsimple as tmdb
import multiprocessing

wiki_Qrl = 'https://query.wikidata.org/sparql'

def compare_titles(title1,title_list):
    '''
    Checks if there is a match between a str and a list of str objects.
    The comparison is performed in lower cases
    '''
    list_lower = [x.lower() for x in title_list]
    if title1.lower() in list_lower:
        return True
    else:
        return False
    
def match_movies(tmdb_response, querried_movie):
    '''
    Checks more precisely in a tmdb request if the requested movie has an 'exact' match
    A match occurs when titles (in lowercases) and runtime (+- 3min) are similar 

    '''
    for movies in tmdb_response['results']:
        q_title, q_runtime = querried_movie
        movie_details = tmdb.Movies(movies['id']).info()
        movie_title = movie_details['title'] 
        movie_original_title = movie_details['original_title']
        movie_runtime = movie_details['runtime']
        if ((compare_titles(q_title,[movie_title,movie_original_title])) 
            and (abs(q_runtime - movie_runtime)<=3)):
            return [q_title,
                    movie_details['vote_average'],
                    movie_details['vote_count'],
                    movie_details['budget']]
    return None

def multiprocess_query(movie_list, api_key, nb_workers = 5, task = None):
    '''
    Multitasks the query to the TMDB database.
    '''
    # Split dataset in chunk of equal size for workers
    chunk_size = len(movie_list) // nb_workers 
    chunks = [movie_list[i:i+chunk_size] for i in range(0, len(movie_list), chunk_size)] 
    # Create a pool with the specified number of workers and the task to accomplish
    pool = multiprocessing.Pool(nb_workers) 
    results = pool.map(task,(chunks,api_key))
    # Wait for all workers to end and concatenate their results
    pool.join() 
    merged_df = pd.concat(results) 
    return merged_df

def getInfoFromTMDB(movie_list,key = None,verbose = False):
    '''
    Takes a dataframe of movie titles along with durations and returns the 
    vote_counts, vote_averages and budget all the matches found in the 
    TMDB database
    '''
    #Define the verbose function to print informations during the process
    if verbose:
        vprint = _verboseprint
    else:
        vprint = lambda *a: None
    vprint('Starting to fetch ...')
    tmdb.API_KEY = key
    matched_results = []
    for idx, query in movie_list.iterrows():
        # Search for the list of movies containing the queried name
        search = tmdb.Search()
        response = search.movie(query=query[0])
        match = match_movies(response,query)
        if match:
            matched_results.append(match)
            vprint('Found a match for movie number {}\n{}'.format(idx,match))
    info_df = pd.DataFrame(data = matched_results,columns=['title','vote_average',
                                                           'vote_counts','budget'])
    return info_df

def getLabelFromFBID(id_list, verbose = False):
    '''Takes a list of Freebase IDs as input and returns
    a Pandas dataframe with the according freebase IDs labels
    '''
    # Allows to print items only if verbose is set to True
    # Does nothing in case of false 
    if verbose:
        vprint = _verboseprint
    else:
        vprint = lambda *a: None
    vprint('Querying the id_list to api...')
    # Transform the list of FB ids into a string formatted
    # For an SQL VALUE query, we get all labels with a single query
    freebase_ids = "\"" + "\" \"".join(id_list) + "\""
    query = '''
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wikibase: <http://wikiba.se/ontology#>

    SELECT ?sLabel ?freebaseID WHERE {
        VALUES ?freebaseID { %s }.
        ?s wdt:P646 ?freebaseID.
        ?s rdfs:label ?sLabel.
        FILTER (lang(?sLabel) = "en").
    }
    '''%(freebase_ids)
    # Define some parameters for the query wrapper
    sparql = SPARQLWrapper(wiki_Qrl)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.method = 'POST' # Post request allows more returned bytes
    results = sparql.query().convert()
    #Transform the response payload into a Pandas dataframe
    results_df = pd.json_normalize(results['results']['bindings'])
    return results_df[["freebaseID.value","sLabel.value"]]

# Ref : https://stackoverflow.com/questions/5980042/how-to-implement-the-verbose-or-v-option-into-a-script
def _verboseprint(*args):
    '''Prints all elements given as argument.'''
    for arg in args:
        print(arg)

if __name__ == '__main__':
    pass