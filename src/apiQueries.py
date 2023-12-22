import pandas as pd
import requests
import time
from SPARQLWrapper import SPARQLWrapper, JSON
import tmdbsimple as tmdb
import multiprocessing
import sys

wiki_Qrl = 'https://query.wikidata.org/sparql'

# Ref : https://stackoverflow.com/questions/5980042/how-to-implement-the-verbose-or-v-option-into-a-script
def _verboseprint(*args):
    # Print each argument separately so caller doesn't need to
    # stuff everything to be printed into a single string
    for arg in args:
        print(arg)

def compare_titles(title1,title_list):
    '''
    Checks if there is a match between a str and a list of str objects.
    The process is performed in lower cases
    '''
    list_lower = [x.lower() for x in title_list]
    if title1.lower() in list_lower:
        return True
    else:
        return False
    
def set_TMDB_key(key):
    tmdb.API_KEY = key

def multiprocess_query(movie_list, task, nb_workers = 5):
    """Allows to run the queries to TMDB in parallel

    Args:
        movie_list (pd.Dataframe): Dataframe with the required info to run the queries.
        task (fun): fun that will be applied to each dataframe element.
        nb_workers (int, optional): Number of subproccesses running. Defaults to 5.

    Returns:
        pd.Dataframe: Dataframe containing the TMDB info of every querried movie.
    """    
    num_workers = nb_workers # number of workers you want to use
    chunk_size = len(movie_list) // num_workers # size of each chunk
    chunks = [movie_list[i:i+chunk_size] for i in range(0, len(movie_list), chunk_size)] # split the dataframe into chunks
    pool = multiprocessing.Pool(num_workers) # create a pool of workers
    results = pool.map(task,chunks) # apply the worker function to each chunk
    pool.close() # close the pool
    pool.join() # wait for all the workers to finish
    merged_df = pd.concat(results) # concatenate the dataframes returned by each worker
    return merged_df

def match_movies(tmdb_response, querried_movie):
    """Checks in a tmdb request if one of the movies is an 'exact' match
    A match occurs when titles (in lowercase letters) and runtime (+- 3min) 
    are similar 

    Args:
        tmdb_response (dict): response from the TMDB server
        querried_movie (_type_): Information about the movie that needs to be matched to a tmdb entry

    Returns:
        array: information extracted from tmdb
    """
    for movies in tmdb_response['results']:
        q_title, q_runtime, q_id = querried_movie
        try:
            time.sleep(0.01)
            movie_details = tmdb.Movies(movies['id']).info()
            movie_title = movie_details['title'] 
            movie_original_title = movie_details['original_title']
            movie_runtime = movie_details['runtime']
            if ((compare_titles(q_title,[movie_title,movie_original_title])) 
                and (abs(q_runtime - movie_runtime)<=3)):
                return [q_title,
                        q_id,
                        movie_details['vote_average'],
                        movie_details['vote_count'],
                        movie_details['budget']]
        except requests.exceptions.HTTPError as e:
            print(e, file=sys.stdout)
        except Exception:
            print('General exception caught')
            print(e, file=sys.stdout)
    return None


def getInfoFromTMDB(movie_list,verbose = False):
    """Fetches a movie list from tmdb and return useful info

    Args:
        movie_list (pd.Dataframe): Movie dataframe with the info needed to perform matchings
        verbose (bool, optional):  prints additional information. Defaults to False.

    Returns:
        pd.Dataframe: dataframe with the info from tmdb for all querried movies
    """    
    if verbose:
        vprint = _verboseprint
    else:
        vprint = lambda *a: None
    vprint('Starting to fetch ...')
    matched_results = []
    for idx, query in movie_list.iterrows():
        if idx%1000 == 0:
            print('Fetching movie number {} out of {}'.format(idx,len(movie_list)))
        time.sleep(0.01)
        try:
            search = tmdb.Search()
            response = search.movie(query=query[0])
            match = match_movies(response,query)
            if match:
                matched_results.append(match)
                vprint('Found a match for movie number {}\n{}'.format(idx,match))
        except Exception:
            print('Caught an exception on search for the movie:\n{}'.format(query))
    info_df = pd.DataFrame(data = matched_results,columns=['title','movie_freebase_id','rating_average',
                                                           'rating_count','budget'])
    return info_df
                


def getLabelFromFBID(id_list, verbose = False):
    """Takes a list of Freebase IDs as input and returns
    a Pandas dataframe with the according freebase IDs

    Args:
        id_list (list): list containing freebase ids
        verbose (bool, optional): prints additional information. Defaults to False.

    Returns:
        pd.Dataframe: dataframe containing freebase ids with their corresponding labels.
    """    
    # Allows to print items only if verbose is set to True
    # Does nothing in case of false 
    if verbose:
        vprint = _verboseprint
    else:
        vprint = lambda *a: None
    vprint('Querying the id_list to api...')
    # Transform the list of FB ids into a string formatted
    # For an SQL VALUE query
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

def task(chunk):
    return getInfoFromTMDB(movie_list = chunk, verbose=False)

if __name__ == '__main__':
    pass