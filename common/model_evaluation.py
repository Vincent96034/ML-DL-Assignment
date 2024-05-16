import pandas as pd
import numpy as np


def calculate_metrics(row, ks=[10, 20, 50, 100, 200]):
    future_resp = row['future_resp']
    recommended_ids = row['recommended_ids']

    metrics = {}

    for k in ks:
        top_k_recs = recommended_ids[:k]
        correct_recommendations = [item for item in future_resp if item in top_k_recs]
        num_correct = len(correct_recommendations)

        if num_correct > 0:
            positions = [top_k_recs.index(item) + 1 for item in correct_recommendations]
            avg_position = sum(positions) / num_correct
            reciprocal_ranks = [1 / pos for pos in positions]
        else:
            avg_position = np.nan
            reciprocal_ranks = []

        metrics[f'num_correct_{k}'] = num_correct
        metrics[f'avg_position_{k}'] = avg_position
        metrics[f'reciprocal_ranks_{k}'] = reciprocal_ranks

    return pd.Series(metrics)


def calculate_global_metrics(df, ks=[10, 20, 50, 100, 200]):
    global_metrics = {}

    for k in ks:
        num_correct_col = df[f'num_correct_{k}']
        reciprocal_ranks_col = df[f'reciprocal_ranks_{k}']
        future_resp_col = df['future_resp']

        # Precision@k
        precision_at_k = num_correct_col.sum() / (len(df) * k)

        # Recall@k
        recall_at_k = num_correct_col.sum() / future_resp_col.apply(len).sum()

        # Mean Reciprocal Rank (MRR)
        mrr = reciprocal_ranks_col.apply(lambda x: np.mean(x) if x else 0).mean()

        # Mean Average Precision (MAP)
        map_score = (reciprocal_ranks_col.apply(lambda x: np.mean(
            x) if x else 0) / future_resp_col.apply(len)).mean()

        # Hit Rate
        hit_rate = (num_correct_col > 0).mean()

        # Coverage
        unique_recommended_items = df['recommended_ids'].apply(
            lambda x: set(x[:k])).explode().nunique()
        total_unique_items = df['recommended_ids'].explode().nunique()
        coverage = unique_recommended_items / total_unique_items

        global_metrics[k] = {
            'Precision@k': precision_at_k,
            'Recall@k': recall_at_k,
            'MRR': mrr,
            'MAP': map_score,
            'Hit Rate': hit_rate,
            'Coverage': coverage
        }

    return pd.DataFrame(global_metrics).transpose()


def evaluate_recommendation(recs, test_frame, ks=[10, 20, 50, 100, 200]):
    # Merge the dataframes on 'eid'
    merged_df = pd.merge(test_frame, recs, on='eid')
    # metrics_df = merged_df.apply(calculate_metrics, ks=ks, axis=1)

    metrics_df = merged_df.apply(calculate_metrics, ks=ks, axis=1)

    # Combine metrics_df with the merged_df to keep all original data
    merged_metrics_df = pd.concat([merged_df, metrics_df], axis=1)

    # Apply the function to the merged dataframe
    # merged_df[['num_correct', 'avg_position']] = merged_df.apply(calculate_metrics, axis=1)
    return merged_metrics_df
