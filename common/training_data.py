import pandas as pd
import numpy as np

def create_negative_samples(eid_lookup, item_embeddings, ratio=1):
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
        negative_samples.extend([(uid, cid) for uid, cid in sampled_pairs if (uid, cid) not in positive_pairs])
        negative_samples = negative_samples[:num_negative_samples]  # Trim to the required number of negatives
    
    negative_samples_df = pd.DataFrame(negative_samples, columns=['uid', 'cid'])
    negative_samples_df['label'] = 0
    negative_samples_df['eid'] = negative_samples_df['uid'].astype(str) + '_' + negative_samples_df['cid'].astype(str) + '_nad'

    # Step 4: Combine positive and negative samples
    training_df = pd.concat([event_df_positive, negative_samples_df], ignore_index=True)
    return training_df


def create_user_df(df_users, eid_scope):
    # merge with eid_lookup_train on cid, to get same size as df_users
    df_users = pd.merge(eid_scope, df_users, on='uid', how='left')
    df_users.drop(columns=["uid", "cid", "label"], inplace=True)
    df_users.set_index("eid", inplace=True, drop=True)
    print("df_users shape: ", df_users.shape)
    return df_users


def create_item_df(df_items, eid_scope):
    # merge with eid_lookup_train on cid, to get same size as df_users
    df_items = pd.merge(eid_scope, df_items, on='cid', how='left')
    df_items.drop(columns=["uid", "cid", "label"], inplace=True)
    df_items.set_index("eid", inplace=True, drop=True)
    print("df_items shape: ", df_items.shape)
    return df_items


def create_label_df(df_labels, eid_scope, target_col="click"):
    TARGET_COL = target_col
    target_cols = ("engagement_time", "click")
    non_target_col = [col for col in target_cols if col not in TARGET_COL][0]
    # merge with eid_lookup_train on cid, to get same size as df_users
    df_labels = pd.merge(eid_scope, df_labels, on='eid', how='left')

    df_labels.drop(columns=["uid_x", "cid_x", "uid_y", "cid_y",
                   "Date", non_target_col], inplace=True, errors="ignore")
    df_labels.set_index("eid", inplace=True, drop=True)
    df_labels["label"] = df_labels[TARGET_COL].fillna(0)
    df_labels.drop(columns=[TARGET_COL], inplace=True)
    # if target col is click, make label int
    df_labels["label"] = df_labels["label"].astype(int)
    print("df_labels shape: ", df_labels.shape)
    return df_labels


# def create_user_df(df_users, eid_scope):
#     # aggregate user data per uid
#     df_users = df_users.groupby('uid').agg({
#         'Region': 'first',
#         'Country': 'first',
#         'DeviceLang': 'first',
#         'DeviceOS': 'first',
#         'PageReferrerDomain': 'first', }).reset_index()
#     df_users = pd.merge(eid_scope, df_users, on=['uid'], how='left')
#     df_users.set_index("eid", inplace=True, drop=True)

#     # one hot encoding categorical columns
#     df_users = pd.get_dummies(
#         df_users, columns=["Region", "Country", "DeviceLang", "DeviceOS", "PageReferrerDomain"])
#     df_users.drop(columns=['uid', 'cid', 'label'], inplace=True)
#     # true to 1 and false to 0
#     df_users = df_users * 1
#     print("df_users shape: ", df_users.shape)
#     return df_users