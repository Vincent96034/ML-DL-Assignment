"""common utility functions to share across notebooks"""
import os
import ast
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix


def eval_recommendations(rec_df):
    rec_df["is_in"] = rec_df.apply(lambda x: x["ContentItemId"] in x.values, axis=1)
    rec_df["position"] = rec_df.apply(lambda row: list(row).index(
        row["ContentItemId"]) if row["is_in"] else -1, axis=1)
    rec_df["position"] = rec_df["position"].replace(400, np.nan)

    pos_counts = rec_df["position"].value_counts()

    top_10_mass = pos_counts.head(10).sum()
    top_10_pos = np.mean(list(pos_counts.head(10).index))
    top_10_mass_pos_ration = top_10_mass / top_10_pos

    top_20_mass = pos_counts.head(20).sum()
    top_20_pos = np.mean(list(pos_counts.head(20).index))
    top_20_mass_pos_ration = top_20_mass / top_20_pos

    top_50_mass = pos_counts.head(50).sum()
    top_50_pos = np.mean(list(pos_counts.head(50).index))
    top_50_mass_pos_ration = top_50_mass / top_50_pos

    return {
        "top10_mass": top_10_mass,
        "top10_pos": top_10_pos,
        "top10_ratio": top_10_mass_pos_ration,
        "top20_mass": top_20_mass,
        "top20_pos": top_20_pos,
        "top20_ratio": top_20_mass_pos_ration,
        "top50_mass": top_50_mass,
        "top50_pos": top_50_pos,
        "top50_ratio": top_50_mass_pos_ration,
    }


def load_items_data() -> pd.DataFrame:
    # ! moved to data_loaders.py
    """Load item data"""
    print("Loading items data ...")
    path = 'data/raw/content_data.csv'
    df_items = pd.read_csv(path)
    df_items["Created"] = df_items['Created'].astype('datetime64[ns]')
    df_items["Expires"] = df_items['Expires'].astype('datetime64[ns]')
    df_items["Exported"] = df_items['Exported'].astype('datetime64[ns]')
    print(f"Loaded items: {df_items.shape}")
    return df_items


def load_events_data(version: str | None = "v2") -> pd.DataFrame:
    """Load event data"""
    print("Loading events data ...")
    path = 'data_/raw/events_hashed_raw.csv'
    if version == "v2":
        path = 'data_/raw/events_hashed_clean_v2.csv'
    df_events = pd.read_csv(path)
    try:
        df_events['EventTimestamp'] = pd.to_datetime(
            df_events['EventTimestamp'], format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        df_events['EventTimestamp'] = pd.to_datetime(
            df_events['EventTimestamp'], format='mixed')
    print(f"Loaded events: {df_events.shape}")
    return df_events


def build_previous_responses(group):
    previous_responses = []
    result = []
    for item in group['ContentItemId']:
        result.append(previous_responses.copy())
        previous_responses.append(item)
    return result


def load_prev_responses_user_item_matrix(nrows: int = 100_000):
    if nrows <= 50_000:
        dataset = "small"
    elif nrows <= 100_000:
        dataset = "medium"
    elif nrows <= 800_000:
        dataset = "large"
    else:
        dataset = "full"
    item_resp_path = f"../data/prev_resp/rec_prev_items_{dataset}.csv"
    if os.path.exists(item_resp_path):
        df = pd.read_csv(item_resp_path)
        df = df.head(nrows)
        df = df.drop(columns=["Unnamed: 0"], errors='ignore')
        df["items"] = df["items"].apply(ast.literal_eval)

        items = load_items_data()
        mlb = MultiLabelBinarizer(classes=items["ItemId"])
        item_matrix = mlb.fit_transform(df['items'])
        sparse_matrix = csr_matrix(item_matrix)
        rec_item_matrix = pd.DataFrame.sparse.from_spmatrix(
            sparse_matrix, index=df['RecId'], columns=mlb.classes_)
        return rec_item_matrix
    else:
        raise FileNotFoundError("File not found.")


def load_rec_id_mapping():
    file_path = "../data/prev_resp/rec_id_mapping.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = df.drop(columns=["Unnamed: 0"], errors='ignore')
        return df
    else:
        raise FileNotFoundError("File not found.")


def create_prev_responses_test_set(events):
    # only used to create the test set
    print("WARNING: this function is deprecated. use "
          "`load_prev_responses_user_item_matrix` instead.")

    if os.path.exists("../data/prev_resp/test_set_prev_responses.csv"):
        print("df found in cache ... loading ...")
        test_set = pd.read_csv("../data/prev_resp/test_set_prev_responses.csv")
        try:
            test_set['EventTimestamp'] = pd.to_datetime(
                test_set['EventTimestamp'], format='%Y-%m-%d %H:%M:%S')
        except ValueError:
            test_set['EventTimestamp'] = pd.to_datetime(
                test_set['EventTimestamp'], format='mixed')
        test_set = test_set.drop(columns=["Unnamed: 0"], errors='ignore')
        test_set["previousResponses"] = test_set["previousResponses"].apply(
            ast.literal_eval)
    else:
        print("df not found in cache ... creating ...")
        test_set = events.copy()[["UserId", "ContentItemId", "EventTimestamp"]]
        # print(test_set.shape)
        test_set.sort_values(by=['EventTimestamp'], inplace=True)
        # drop userid - contentitemid duplicates
        test_set = test_set.drop_duplicates(
            subset=['UserId', 'ContentItemId', 'EventTimestamp'])
        # print(test_set.shape)

        # Apply the function to each group
        test_set.sort_values(by=['UserId', 'EventTimestamp'], inplace=True)
        test_set.reset_index(drop=True, inplace=True)
        print("Applying ...")
        test_set['previousResponses'] = test_set.groupby('UserId').apply(
            lambda x: build_previous_responses(x)).explode().reset_index(drop=True)
        # to csv
        test_set.to_csv("../data/test_set_prev_responses.csv")
    return test_set


def create_negative_samples(eid_lookup, item_embeddings, ratio=1):
    """Create negative samples for training data. Used to creat training data for DNN 
    model.

    Args:
        eid_lookup (pd.DataFrame): DataFrame containing the mapping between users and
            items - embedding the interactions. This DataFrame should contain the columns
            'eid', 'uid' and 'cid'.
        item_embeddings (pd.DataFrame): DataFrame containing the embeddings for each item.
            This DataFrame should contain the columns 'cid'.
        ratio (float): Ratio of negative samples to positive samples. Default is 1,
            meaning that the number of negative samples is equal to the number of positive
            samples.
    """
    users = eid_lookup['uid'].unique()
    content_items = item_embeddings['cid'].unique()
    print("Generating all user-item pairs ...")
    all_pairs = pd.MultiIndex.from_product([users, content_items], names=[
                                           'uid', 'cid']).to_frame(index=False)

    # Step 2: Identify positive samples
    event_df_positive = eid_lookup[['eid', 'uid', 'cid']].copy()
    event_df_positive['label'] = 1

    # Merge all pairs with positive samples to find negative samples
    print("Merging positive samples ...")
    merged = pd.merge(all_pairs, event_df_positive, on=[
                      'uid', 'cid'], how='left', indicator=True)

    # Step 3: Sample negative samples
    negative_samples = merged[merged['_merge'] == 'left_only'].copy()
    negative_samples = negative_samples.drop(columns=['_merge'])
    negative_samples['label'] = 0

    # Sample a subset of negative samples (e.g., equal to the number of positive samples)
    print('Sampling negative samples ...')
    num_positive_samples = int(len(event_df_positive) * ratio)
    negative_samples = negative_samples.sample(n=num_positive_samples, random_state=42)
    negative_samples['eid'] = negative_samples.uid.astype(
        str) + '_' + negative_samples.cid.astype(str) + '_nad'

    # Step 4: Combine positive and negative samples
    training_df = pd.concat([event_df_positive, negative_samples], ignore_index=True)
    return training_df


def create_negative_samples_v2(eid_lookup, item_embeddings, ratio=1):
    """Create negative samples for training data. Used to create training data for DNN 
    model.

    Args:
        eid_lookup (pd.DataFrame): DataFrame containing the mapping between users and
            items - embedding the interactions. This DataFrame should contain the columns
            'eid', 'uid' and 'cid'.
        item_embeddings (pd.DataFrame): DataFrame containing the embeddings for each item.
            This DataFrame should contain the columns 'cid'.
        ratio (float): Ratio of negative samples to positive samples. Default is 1,
            meaning that the number of negative samples is equal to the number of positive
            samples.
    """
    users = eid_lookup['uid'].unique()
    content_items = item_embeddings['cid'].unique()

    # Step 2: Identify positive samples
    event_df_positive = eid_lookup[['eid', 'uid', 'cid']].copy()
    event_df_positive['label'] = 1

    # Create a set of positive (uid, cid) pairs for fast lookup
    positive_pairs = set(zip(event_df_positive['uid'], event_df_positive['cid']))

    # Step 3: Sample negative samples
    num_positive_samples = len(event_df_positive)
    num_negative_samples = int(num_positive_samples * ratio)

    print('Sampling negative samples ...')
    np.random.seed(42)
    negative_samples = []
    while len(negative_samples) < num_negative_samples:
        sampled_uids = np.random.choice(users, num_negative_samples, replace=True)
        sampled_cids = np.random.choice(content_items, num_negative_samples, replace=True)
        sampled_pairs = zip(sampled_uids, sampled_cids)
        negative_samples.extend(
            [(uid, cid) for uid, cid in sampled_pairs if (uid, cid) not in positive_pairs])
        # Trim to the required number of negatives
        negative_samples = negative_samples[:num_negative_samples]

    negative_samples_df = pd.DataFrame(negative_samples, columns=['uid', 'cid'])
    negative_samples_df['label'] = 0
    negative_samples_df['eid'] = negative_samples_df['uid'].astype(
        str) + '_' + negative_samples_df['cid'].astype(str) + '_nad'

    # Step 4: Combine positive and negative samples
    training_df = pd.concat([event_df_positive, negative_samples_df], ignore_index=True)
    return training_df
