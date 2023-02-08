import logging
import pandas as pd
import tweepy, os, json, gzip, re
from pathlib import Path

logger = logging.getLogger()

def write_gz(file_pth, file_name, data):
    # write user tweets in .json.gz format in output_folder
    out_pth = os.path.join(file_pth, file_name)
    with gzip.open(out_pth, 'w') as fout:
        fout.write(json.dumps(data).encode('utf-8'))

def write_csv(file_pth, data):
    if not os.path.exists(file_pth):
        data.to_csv(file_pth, index=False)
    else:
        data.to_csv(file_pth, mode='a', index=False, header=False)

def hit_users_api(user_names_list, client):
    try:
        out = {}
        results = client.get_users(usernames=user_names_list)
        for result in results.data:
            out[result['username'].lower()] = result['id']
        for error in results.errors:
            out[error['value'].lower()] = 'NOT FOUND'
        return out
    except Exception as e:
        logger.error(e)
        return {}

def read_clean_data(input_file):
    data = pd.read_csv(input_file, sep = "\t", index_col = 0)[['Author', 'Full Text']]
    data = data.rename(columns = {
            "Author": "user_name", 
            "Full Text": "text"
            }
        )
    return data

def valid_username(username):
    try:
        pattern = re.compile('^[A-Za-z0-9_]{0,14}$') 
        return True if pattern.match(username) else False
    except re.error:
        return False

def get_user_ids(author_data, users_wt_ids, out_pth, client):
    try:
        # user names list can be a max of 100
        logger.info(f'In get_users_id, prev processed users count {len(users_wt_ids)}')
        user_names_list = list(set(author_data['user_name'].to_list()) - set(users_wt_ids))
        valid_user_names_list = [user_name for user_name in user_names_list if valid_username(user_name)]
        logger.info(f'Invalid user names list: {len(set(user_names_list) - set(valid_user_names_list))}')

        logger.info(f'In get_users_id, need to process {len(valid_user_names_list)}')
        n = 100
        batched_user_names = [valid_user_names_list[i * n:(i + 1) * n] for i in range((len(valid_user_names_list) + n - 1) // n )]
        user_id_map = {}
        for index, batch in enumerate(batched_user_names):
            if index % 10 == 0:
                logger.info(f'In get_users_id, processed {index} of {len(batched_user_names)} users')
            batch_user_id_map = hit_users_api(batch, client)
            # logger.info(f'batch_user_id_map {batch_user_id_map}')
            tmp_df = author_data[author_data['user_name'].isin(batch)]
            # logger.info(f'tmp_df {tmp_df}')
            tmp_df['user_id'] = tmp_df['user_name'].apply(lambda x: batch_user_id_map[x.lower()])
            write_csv(out_pth, tmp_df)
            user_id_map = {**user_id_map, **batch_user_id_map}
        author_data['user_id'] = author_data['user_name'].map(user_id_map)
        return author_data
    except Exception as e:
        logger.error('Error in get_users_id: ' + str(e))
        raise e

def identify_less_twt_users(input_file, output_folder, output_file, clients):
    out_pth = os.path.join(output_folder, output_file)

    logger.info("Reading data from input file...")
    data = read_clean_data(input_file)

    logger.info("Applying transformations on the input file...")
    author_data = data['user_name'].value_counts().rename_axis('user_name').reset_index(name='tweet_count')

    users_wt_ids = []
    if os.path.isfile(out_pth):
        written_author_data = pd.read_csv(out_pth)
        if written_author_data.shape[0] == author_data[author_data['tweet_count'] < 200].shape[0]:
            # save_more_twt_users(data, author_data, output_folder)
            return written_author_data, data
        else:
            users_wt_ids = written_author_data[written_author_data['user_id'].notna()]['user_name'].to_list()

    logger.info("Saving tweets for users with more than 200 tweets in dataset")
    save_more_twt_users(data, author_data, output_folder)

    author_data = author_data[author_data['tweet_count'] < 200]
    author_data['needed_tweets'] = 200 - author_data['tweet_count']
    
    logger.info("Hitting the get_user_id API in batches...")
    author_data = get_user_ids(author_data, users_wt_ids, out_pth, clients)
    logger.info(f'User data without ids: {author_data[author_data["user_id"] != "NOT FOUND"].shape[0]} out of total {author_data.shape[0]}')

    logger.info("Saving output data to file...")
    full_out_pth = os.path.join(output_folder, output_file.replace('.csv','_full.csv'))
    written_author_data = pd.read_csv(out_pth)
    # fix this to read and then send
    return written_author_data, data

def save_more_twt_users(data, author_data, output_folder, tweets_to_save = 50):
    ###  identify authors with more than 200 tweets and save them in {user_id}_statuses.json.gz format
    # create output folder if it does not exist create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        author_list = author_data[author_data['tweet_count'] >= tweets_to_save]['user_name'].to_list()
        for author in author_list:
            sample_tweets = data[data['user_name'] == author].head(tweets_to_save).to_dict('records')
            write_gz(output_folder,'{}_statuses.json.gz'.format(author), sample_tweets)
    
def get_to_hit_users(output_folder, user_data):
    saved_files = [fn for fn in os.listdir(output_folder) if fn.endswith('.json.gz')]
    user_names = [fn.split('_')[0] for fn in saved_files]
    to_hit_users_data = user_data[~user_data['user_name'].isin(user_names)]
    to_hit_users_data['needed_tweets'] = to_hit_users_data['needed_tweets']\
        .apply(lambda x: [100,x-100] if x > 100 else [x])
    to_hit_users_data = to_hit_users_data.explode('needed_tweets')
    # need to fetch a min of 5 tweets per user
    to_hit_users_data['needed_tweets'] = to_hit_users_data['needed_tweets'].apply(lambda x: max(x,5))
    # mod to pull only 50
    to_hit_users_data['needed_tweets'] = to_hit_users_data['needed_tweets'].apply(lambda x: min(x,50))
    return to_hit_users_data 

def clean_additional_tweets(recent_tweets):
    out = [dict(tweet) for tweet in recent_tweets]
    return out

def get_save_more_tweets(output_folder, data ,user_name, user_id, contained_tweets, needed_tweets, client):
    sample_tweets = data[data['user_name'] == user_name].head(contained_tweets).to_dict('records')
    # logger.info(f'user id {user_id}, for user {user_name}')
    if user_id != 'NOT FOUND':
        try:
            additional_tweets = client.get_users_tweets(id = user_id, max_results = str(needed_tweets)).data
        except:
            logger.error(f'invalid user id {user_id}, for user {user_name}')
        if additional_tweets:
            additional_tweets = clean_additional_tweets(additional_tweets)
        else:
            logger.info(f'No additional tweets found for user {user_name}')
            additional_tweets = []
        sample_tweets = sample_tweets + additional_tweets
    write_gz(output_folder, '{}_statuses.json.gz'.format(user_name), sample_tweets)   


def go(input):
    artifact_path = Path('components/artifacts/')
    auth = os.environ[input['TWITTER_API_KEY_NAME']]
    client = tweepy.Client(auth, wait_on_rate_limit=True)
    # input has input_filename, output_folder, user_filename 

    # find users to get more tweets for and find users with 200 or more tweets save them in user_tweets folder 
    user_data, data = identify_less_twt_users(input['input_path'], input['user_data_folder'], input['less_twt_users_file'], client)
    
    logger.info("Getting to hit user data...")
    # find users to get more users for after subtracting users for which this process is completed
    to_hit_users = get_to_hit_users(input['user_data_folder'], user_data)
    logger.info(f"Need to get info for {to_hit_users.shape[0]}")
    
    logger.info("Hitting the get_tweets API for each user...")
    # hit & save twitter API to get more tweets for each user
    [get_save_more_tweets(input['user_data_folder'], data, user_name, user_id, twt_count, needed_twt, client) \
        for user_name, user_id, twt_count, needed_twt in \
        zip(to_hit_users['user_name'], to_hit_users['user_id'], to_hit_users['tweet_count'], to_hit_users['needed_tweets'])]
    
    # Check if we fetched all users data 
    to_hit_users = get_to_hit_users(input['user_data_folder'], user_data)

    logger.info(f"Still need to get data for {len(to_hit_users['user_name'])} users")
    logger.info(f"Still need to get data for {to_hit_users['user_name']}")
    

    
