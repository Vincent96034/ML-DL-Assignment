from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# similarity measures
COSINE = "cosine"
DOT = "dot"


class Recommender(ABC):
    """Base class for recommenders. All recommenders should inherit from this class and
    implement the abstract methods. Note that the `data` attribute should be set after
    fitting the model. This attribute should contain the item embeddings.
    """
    model_name: str = "BaseRecommender"

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Must be implemented by subclass.

        Fit or setup the model. Ideally, load model from disk if available. This method
        must set the `data` attribute to the item embeddings.
        """
        self.data = None

    @abstractmethod
    def create_user_embedding(user_items_df, *args, **kwargs):
        """Must be implemented by subclass.

        Create user embedding from user-item matrix. The output should be a 2D array with
        shape (n_recommendations, n_features). The number of features should be the same
        as the item embeddings.
        """
        pass

    def recommend(self, items: pd.DataFrame, n: int = 10, method: str = "cosine", **kwargs):
        """Create recommendations.

        Args:
            user_items_df (pd.DataFrame): RecId-item matrix. Per row, a recommendation should be
                made. The columns represent seen items at that point in time. Example:

                       | item1  item2  item3                
                RecId  |----------------------
                rec1   |   0      0      1
                ...

        Returns:
            pd.DataFrame with recommendations for each RecId as well as respective scores.
        """
        return self.batch_recommend(items, n, method, **kwargs)

    def batch_recommend(self, user_items_df: pd.DataFrame, n: int = 10, method="cosine", **kwargs):
        """Create multiple recommendations.

        Args:
            user_items_df (pd.DataFrame): RecId-item matrix. Per row, a recommendation should be
                made. The columns represent seen items at that point in time. Example:

                       | item1  item2  item3                
                RecId  |----------------------
                rec1   |   0      0      1
                rec2   |   1      1      0
                rec3   |   0      0      0
                ...

        Returns:
            pd.DataFrame with recommendations for each RecId as well as respective scores.
        """
        if self.data is None:
            raise ValueError("Model not fitted. Run `fit()` first.")
        user_embeddings = self.create_user_embedding(user_items_df)
        print(f"Created user embeddings: {user_embeddings.shape}")

        if method == COSINE:
            similarity = cosine_similarity(user_embeddings, self.data)
        elif method == DOT:
            similarity = np.dot(user_embeddings, self.data.T)
        else:
            raise ValueError(f"Method {method} not supported.")

        # set similarity of seen items to -1 to not recommend them again
        similarity[user_items_df.to_numpy() == 1] = -10
        print(f"Calculated cosine similarity: {similarity.shape}; now sorting ...")

        # get top n indices and scores; similarity is sorted in desc. order (cos & dot)
        sorted_indices = np.argsort(-similarity, axis=1)
        top_n_indices = sorted_indices[:, :n]
        top_n_scores = np.take_along_axis(similarity, top_n_indices, axis=1)

        # map indices to actual item ids
        index_values = self.data.index.values
        index_length = len(index_values)
        adjusted_array = top_n_indices % index_length
        mapped_array = np.vectorize(lambda x: index_values[x])(adjusted_array)

        # return dict with item ids as keys and dict of top n items and scores
        rec_dict = {}
        for i, rec_id in enumerate(user_items_df.index):
            rec_dict[rec_id] = dict(zip(mapped_array[i], top_n_scores[i]))
        return rec_dict

    def __repr__(self):
        return f"<{self.model_name}>"
