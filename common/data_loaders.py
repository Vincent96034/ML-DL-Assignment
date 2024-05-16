import ast
import pickle

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix


def load_items_data() -> pd.DataFrame:
    print("loading items data ...")
    path = 'data/processed/items.csv'
    df_items = pd.read_csv(path)
    df_items["Created"] = df_items['Created'].astype('datetime64[ns]')
    df_items["Expires"] = df_items['Expires'].astype('datetime64[ns]')
    df_items["Exported"] = df_items['Exported'].astype('datetime64[ns]')
    print(f"Loaded items: {df_items.shape}")
    return df_items


def load_events_data() -> pd.DataFrame:
    print("loading events data ...")
    path = 'data/processed/events.csv'
    df_events = pd.read_csv(path)
    df_events["EventTimestamp"] = df_events['EventTimestamp'].astype('datetime64[ns]')
    return df_events


def load_test_resp_eval_frame() -> pd.DataFrame:
    print("loading responses data ...")
    path = 'data/processed/responses_evaluation.pkl'
    with open(path, 'rb') as f:
        df_responses = pickle.load(f)
    return df_responses


def load_test_resp_eval_matrix():
    df = load_test_resp_eval_frame()
    items = load_items_data()
    mlb = MultiLabelBinarizer(classes=items["cid"])
    item_matrix = mlb.fit_transform(df['prev_resp'])
    sparse_matrix = csr_matrix(item_matrix)
    rec_item_matrix = pd.DataFrame.sparse.from_spmatrix(
        sparse_matrix, index=df['eid'], columns=mlb.classes_)
    return rec_item_matrix
