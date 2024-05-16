import os
import time
from typing import Optional
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import scipy

from openai import OpenAI
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor

from .base_recommender import Recommender
from .data_loaders import load_items_data

__all__ = ["OHRecommender", "TfidfRecommender", "OAIRecommender"]


class ContentBasedRecommender(Recommender):
    """Base class for content-based recommenders."""
    model_name: str = "ContentBasedRecommender"

    def __init__(self, normalize: str | None = None, use_cache: bool = False):
        self.use_cache = use_cache
        self.data = None
        self.normalize = self._get_normalize_func(normalize)
        print(f"Initialized `{self.model_name}`")

    def create_user_embedding(self, items: pd.DataFrame, method: str = "mean"):
        if method == "mean":
            user_embedding = self._mean_user_embedding_optimized(items)
        else:
            raise ValueError(f"Unknown method: {method}")
        return user_embedding

    def _mean_user_embedding_optimized(self, rec_items_df: pd.DataFrame) -> pd.DataFrame:
        """Optimized version of mean_user_embedding for DataFrame input."""
        item_embeddings = self.data.values
        rec_items_matrix = rec_items_df.to_numpy()
        # Matrix multiplication to get weighted sum of embeddings
        user_embeddings_sum = np.dot(rec_items_matrix, item_embeddings)
        # Count of seen items for each RecId; avoid division by zero by setting 0 to 1
        seen_items_count = rec_items_matrix.sum(axis=1).reshape(-1, 1)
        seen_items_count[seen_items_count == 0] = 1
        # Calculate average embeddings
        user_embeddings_avg = user_embeddings_sum / seen_items_count
        user_embeddings_df = pd.DataFrame(
            user_embeddings_avg, index=rec_items_df.index, columns=self.data.columns)
        return user_embeddings_df

    @staticmethod
    def _get_normalize_func(normalize: str | None):
        if normalize == "l2":
            def l2_normalize(df):
                return df.div(df.sum(axis=1), axis=0)
            return l2_normalize
        elif normalize is None:
            return None
        else:
            raise ValueError(f"Unknown normalization method: {normalize}")

    @staticmethod
    def _clean_html(text):
        return BeautifulSoup(str(text), 'html.parser').get_text()

    @staticmethod
    def _create_prompt(row):
        def clean_html(text):
            return BeautifulSoup(str(text), 'html.parser').get_text()

        cleaned_text = clean_html(row['Text'])
        return (
            f"ItemName: {row['ItemName']}\n"
            f"TemplateName: {row['TemplateName']}\n"
            f"Subject: {row['Subject']}\n"
            f"Heading: {row['Heading']}\n"
            f"Teaser: {row['Teaser']}\n"
            f"Text: {cleaned_text}"
        )


class OHRecommender(ContentBasedRecommender):
    model_name: str = "One-Hot-Embeddings Recommender"

    def __init__(self, use_cache: bool = True):
        super().__init__(use_cache=use_cache)

    def fit(self, *args, **kwargs):
        df_items = load_items_data()

        cols = ["cid", "TemplateName", "Category", "Subject"]
        df_items_oh = df_items[cols]
        df_items_oh = pd.get_dummies(
            df_items_oh, columns=["TemplateName", "Category", "Subject"], drop_first=True)
        df_items_oh = df_items_oh * 1  # convert to binary

        # set ItemId as index
        df_items_oh = df_items_oh.set_index("cid")
        self.data = df_items_oh

    def naive_recommend(self, items: list, n: int = 10, **kwargs):
        """Naive recommender that does not create a user-embeddings, but recommends the
        top n items from the top n items per seen item."""
        if self.data is None:
            raise ValueError("Model not fitted. Run `fit()` first.")
        # calculate cosine similarity between items
        cos_sim = cosine_similarity(self.data)
        df_sim_oh = pd.DataFrame(cos_sim, index=self.data.index,
                                 columns=self.data.index)
        self.similarity = df_sim_oh
        if len(items) == 0:
            return {}
        df_rec = pd.DataFrame()
        for item in items:
            if item not in self.data.index:
                raise ValueError(f"Item {item} not in data.")
            # get similarity scores for item and sort
            sim_scores = self._recommend_items(item, top_n=n)
            sim_scores = pd.DataFrame(sim_scores)
            sim_scores["src"] = item
            sim_scores.columns = ["sim", "src"]
            df_rec = pd.concat([df_rec, sim_scores])
        df_rec["rank"] = range(1, len(df_rec) + 1)
        df_rec.sort_values("sim", ascending=False, inplace=True)
        # remove items already in input
        df_rec = df_rec[~df_rec.index.isin(items)]
        return df_rec.head(n).T.to_dict()

    def _recommend_items(self, item_id, top_n=10):
        item_idx = self.similarity.index.get_loc(item_id)
        sim_scores = self.similarity.iloc[item_idx]
        return sim_scores.sort_values(ascending=False).head(top_n)


class TfidfRecommender(ContentBasedRecommender):
    model_name: str = "Tf-idf Recommender"

    def __init__(
        self,
        normalize: str | None = None,
        max_features: int = 1000,
        use_cache: bool = True
    ) -> None:
        super().__init__(use_cache=use_cache, normalize=normalize)
        self.max_features = max_features

    def fit(self, *args, **kwargs):
        if os.path.exists("data/embeddings/tfidf_embeddings_df.csv") and self.use_cache:
            df_items_tfidf_embed = pd.read_csv(
                "data/embeddings/tfidf_embeddings_df.csv")
        else:
            print("WARNING: embeddings not in cache, creating now ...")
            df_items_tfidf_embed = self._build_tfidf_embeddings()

        # df_items_tfidf_embed.set_index("cid", inplace=True)
        df_items_tfidf_embed.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

        if self.normalize:
            df_items_tfidf_embed = self.normalize(df_items_tfidf_embed)
        self.data = df_items_tfidf_embed

    def _build_tfidf_embeddings(self):
        df_items = load_items_data()
        # create prompt
        df_items['Prompt'] = df_items.apply(self._create_prompt, axis=1)
        df_items = df_items[["cid", "Prompt"]]

        tfidf = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=self._load_stopwords()
        )
        tfidf_matrix = tfidf.fit_transform(df_items['Prompt'])
        df_items_tfidf_embed = pd.DataFrame(
            tfidf_matrix.toarray(), index=df_items['cid'])
        df_items_tfidf_embed.to_csv("data/embeddings/tfidf_embeddings_df.csv")

        return df_items_tfidf_embed

    def _load_stopwords(self) -> list:
        try:
            with open("data/danish_stopwords.txt", "r") as f:
                stopwords = f.read().splitlines()
        except FileNotFoundError:
            print("WARNING: stopwords not found, using None.")
            return None
        return stopwords


class OAIRecommender(ContentBasedRecommender):
    model_name: str = "OpenAI Embeddings Recommender"

    def __init__(
        self,
        normalize: str | None = None,
        dotenv_path: str = None,
        embedding_model: str = "text-embedding-ada-002",
        num_workers: int = 6,
        use_cache: bool = True
    ) -> None:
        super().__init__(use_cache=use_cache, normalize=normalize)
        self.embedding_model = embedding_model
        self.num_workers = num_workers
        load_dotenv(dotenv_path)
        try:
            self.oai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        except KeyError:
            self.oai_client = None

    def init_oai_client(self, api_key: Optional[str] = None):
        self.oai_client = OpenAI(api_key=api_key or os.environ['OPENAI_API_KEY'])

    def fit(self, *args, **kwargs):
        """Creates item embeddings. Looks if embeddings are cached."""
        if os.path.exists("data/embeddings/oai_embeddings_df.csv") and self.use_cache:
            df_items_oai_embed = pd.read_csv("data/embeddings/oai_embeddings_df.csv")
        else:
            print("WARNING: embeddings not in cache, creating now ...")
            df_items_oai_embed = self._build_oai_embeddings()

        df_items_oai_embed.set_index("cid", inplace=True)
        df_items_oai_embed.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
        df_items_oai_embed = df_items_oai_embed.iloc[:, 1:]

        if self.normalize:
            df_items_oai_embed = self.normalize(df_items_oai_embed)
        self.data = df_items_oai_embed

    def _build_oai_embeddings(
            self,
            df_items: pd.DataFrame | None = None,
            output_path: str | None = None
    ) -> pd.DataFrame:
        if self.oai_client is None:
            raise ValueError(
                "OpenAI API key not set. Re-initialize with `init_oai_client()` first.")
        if df_items is None:
            df_items = load_items_data()
        # create prompt
        df_items['Prompt'] = df_items.apply(self._create_prompt, axis=1)
        df_items = df_items[["cid", "Prompt"]]
        df_items_oai_embed = self._parallelize_embeddings(
            df_items, num_workers=self.num_workers)
        # save to csv
        if output_path is None:
            output_path = "data/embeddings/oai_embeddings_df.csv"
        df_items_oai_embed.to_csv(output_path)
        return df_items_oai_embed

    def _get_embedding(self, text):
        response = self.oai_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding

    def _process_row(self, row):
        embedding = self._get_embedding(row['Prompt'])
        return row['cid'], embedding

    def _parallelize_embeddings(self, df, num_workers):
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._process_row, row): row
                       for _, row in df.iterrows()}
            for future in futures:
                try:
                    itemid, embedding = future.result()
                    results.append((itemid, embedding))
                except Exception as e:
                    print(f"Error processing row: {e}")
        results_df = pd.DataFrame(results, columns=['cid', 'embedding'])
        embeddings_df = pd.DataFrame(
            results_df['embedding'].to_list(), index=results_df['cid'])
        merged_df = df.merge(embeddings_df, left_on='cid', right_index=True)
        return merged_df
