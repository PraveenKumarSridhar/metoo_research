from argparse import ArgumentParser
import logging
import pandas as pd
from pathlib import Path


logger = logging.getLogger()

def news_iden_imp_heu(author, acc_type, news_id_list):
    auth_heuristics = 'news' in str(author).lower() or author in news_id_list
    return 'news' if auth_heuristics else acc_type

def go(input):
    artifact_path = Path('components/artifacts/')

    raw_data = pd.read_csv(input['input_path'], sep = "\t", index_col = 0)
    raw_data.drop(columns = ["Gender", "Account Type"], inplace = True)
    raw_data.rename(columns = {"Thread Entry Type": "Tweet Type"}, inplace = True)

    demo = pd.read_csv(input['demo_path'])

    # news = pd.read_html("https://memeburn.com/2010/09/the-100-most-influential-news-media-twitter-accounts/", header = 0)[0]
    news = pd.read_csv(r'../data/news_outlets-accounts.csv')
    news_id_list = [news_.lower() for news_ in news['Token'].tolist()]
    celeb_data = pd.read_csv(r'../data/celebrity.csv')
    celeb_map = {u.lower(): "celebrity" for u in celeb_data["twitter"]}
    # news["@name"] = news["@name"].str.replace("@", "").str.lower()
    # news_users = {u.lower(): "news" for u in news["@name"]}

    brands_df = pd.read_excel(input['brand_path'], sheet_name = "All 1558", usecols = ["Twitter Handle"])
    brands = (
        brands_df[brands_df["Twitter Handle"] != "NOT AVAILABLE"]
        ["Twitter Handle"]
        .str.replace("@", "")
        .str.lower()
    )
    brands_map = {k: "business" for k in brands.values}

    companies_df = pd.read_excel(input['comp_path'], usecols = ["TwitterHandle", "TwitterHandle2"])
    companies = (
        pd.concat([companies_df["TwitterHandle"], companies_df["TwitterHandle2"]], axis = 0)
        .dropna()
        .str.replace("@", "")
        .str.lower()
    )
    companies_map = {k: "business" for k in companies.values}

    demo["Account Type"] = demo["screen"].str.lower().map(celeb_map).combine_first(demo["Account Type"])
    demo["Account Type"] = (
        demo["followers_count"].apply(lambda x: "influencer" if x > input['influencer_thresh'] else "core")
        .mask(~demo["Account Type"].isin(["individual"])) # if induvidual do this else keep the already acc type
        .combine_first(demo["Account Type"])
    )
    demo["Account Type"] = demo["screen"].str.lower().map(brands_map).combine_first(demo["Account Type"])
    demo["Account Type"] = demo["screen"].str.lower().map(companies_map).combine_first(demo["Account Type"])
    demo["Account Type"] = demo[["screen","Account Type"]].apply(lambda x: news_iden_imp_heu(x[0], x[1], news_id_list))
    demo["Gender"] = demo["Gender"].mask(~demo["Account Type"].isin(["core", "influencer"]))
    demo["Ethnicity"] = demo["Ethnicity"].mask(~demo["Account Type"].isin(["core", "influencer"]))


    data = raw_data.merge(right = demo, how = "left", left_on = "Author", right_on = "screen")
    data = data.merge(right = demo, how = "left", left_on = "Thread Author", right_on = "screen", suffixes = ("", "_originator"))
    
    data.to_csv(artifact_path / input['output'], sep = "\t")
