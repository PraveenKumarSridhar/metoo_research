from argparse import ArgumentParser
import logging
import pandas as pd
from gensim.utils import simple_preprocess
import tweepy, os, json, gzip
from pathlib import Path

logger = logging.getLogger()

auth = os.environ['TWITTER_API_KEY']
client = tweepy.Client(auth, wait_on_rate_limit=True)

def write_gz(file_pth, file_name, data):
    # write user tweets in .json.gz format in output_folder
    out_pth = os.path.join(file_pth, file_name)
    with gzip.open(out_pth, 'w') as fout:
        fout.write(json.dumps(data).encode('utf-8'))

def hit_users_api(user_names_list):
    out = {}
    results = client.get_users(usernames=user_names_list)
    for result in results.data:
        out[result['username']] = result['id']
    for error in results.errors:
        out[error['value']] = 'NOT FOUND'
    return out

def read_clean_data(input_file):
    data = pd.read_csv(input_file, sep = "\t", index_col = 0)[['Author', 'Full Text']]
    data = data.rename(columns = {
            "Author": "user_name", 
            "Full Text": "text"
            }
        )
    return data

def get_user_ids(author_data):
    # user names list can be a max of 100
    user_names_list = author_data['user_name'].to_list()
    n = 100
    batched_user_names = [user_names_list[i * n:(i + 1) * n] for i in range((len(user_names_list) + n - 1) // n )]
    user_id_map = {}
    for batch in batched_user_names:
        batch_user_id_map = hit_users_api(batch)
        user_id_map = {**user_id_map, **batch_user_id_map}
    author_data['user_id'] = author_data['user_name'].map(user_id_map)
    return author_data

def identify_less_twt_users(input_file, output_folder, output_file):
    out_pth = os.path.join(output_folder, output_file)
    if os.path.isfile(out_pth):
        author_data = pd.read_csv(out_pth)
        data = read_clean_data(input_file)
        return author_data, data

    logger.info("Reading data from input file...")
    data = read_clean_data(input_file)
    
    logger.info("Applying transformations on the input file...")
    author_data = data['user_name'].value_counts().rename_axis('user_name').reset_index(name='tweet_count')

    logger.info("Saving tweets for users with more than 200 tweets in dataset")
    save_more_twt_users(data, author_data, output_folder)

    author_data = author_data[author_data['tweet_count'] < 200]
    author_data['needed_tweets'] = 200 - author_data['tweet_count']
    
    logger.info("Hitting the get_user_id API in batches...")
    author_data = get_user_ids(author_data)

    logger.info("Saving output data to file...")
    author_data.to_csv(out_pth, index = False)
    return author_data, data

def save_more_twt_users(data, author_data, output_folder):
    ###  identify authors with more than 200 tweets and save them in {user_id}_statuses.json.gz format
    # create output folder if it does not exist create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        author_list = author_data[author_data['tweet_count'] >= 200]['user_name'].to_list()
        for author in author_list:
            sample_tweets = data[data['user_name'] == author].head(200).to_dict('records')
            write_gz(output_folder,'{}_statuses.json.gz'.format(author), sample_tweets)
    
def get_to_hit_users(output_folder, user_data):
    saved_files = [fn for fn in os.listdir(output_folder) if fn.endswith('.json.gz')]
    user_names = [fn.split('_')[0] for fn in saved_files]
    to_hit_users_data = user_data[~user_data['user_name'].isin(user_names)]
    return to_hit_users_data 

def clean_additional_tweets(recent_tweets):
    out = [dict(tweet) for tweet in recent_tweets]
    return out

def get_save_more_tweets(output_folder, data ,user_name, user_id, contained_tweets, needed_tweets):
    sample_tweets = data[data['user_name'] == user_name].head(contained_tweets).to_dict('records')
    if user_id != 'NOT FOUND':
        additional_tweets = client.get_users_tweets(id = user_id, max_results = str(needed_tweets)).data
        additional_tweets = clean_additional_tweets(additional_tweets)
        sample_tweets = sample_tweets + additional_tweets
    write_gz(output_folder, '{}_statuses.json.gz'.format(user_name), sample_tweets)   


def go(input):
    artifact_path = Path('components/artifacts/')
    # input has input_filename, output_folder, user_filename 

    # find users to get more tweets for and find users with 200 or more tweets save them in user_tweets folder 
    user_data, data = identify_less_twt_users(input['input_path'], input['user_data_folder'], input['less_twt_users_file'])
    
    logger.info("Getting to hit user data...")
    # find users to get more users for after subtracting users for which this process is completed
    to_hit_users = get_to_hit_users(input['user_data_folder'], user_data)
    logger.info(f"Need to get info for {to_hit_users.shape[0]}")
    
    logger.info("Hitting the get_tweets API for each user...")
    # hit & save twitter API to get more tweets for each user
    [get_save_more_tweets(input['user_data_folder'], data, user_name, id, twt_count, needed_twt) \
        for user_name, id, twt_count, needed_twt in \
        zip(to_hit_users['user_name'], to_hit_users['id'], to_hit_users['tweet_count'], to_hit_users['needed_tweets'])]
    
    # Check if we fetched all users data 
    to_hit_users = get_to_hit_users(input['user_data_folder'], user_data)

    logger.info(f"Still need to get data for {len(to_hit_users['user_name'])} users")
    logger.info(f"Still need to get data for {to_hit_users['user_name']}")
    

    
