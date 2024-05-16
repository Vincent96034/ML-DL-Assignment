import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def create_user_item_matrix(data, users, items):
    """Create a user-item matrix for given data."""
    matrix = pd.pivot_table(data, index='uid', columns='cid',
                            values='engagement_time', fill_value=0)
    # Ensure all users and items are represented
    matrix = matrix.reindex(index=users, columns=items, fill_value=0)
    return matrix.values


# adapted the code from here: http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/#source-code

def matrix_factorization(R_train, U, V, d, steps=5000, alpha=0.0002, beta=0.02, log_steps=100):
    V = V.T
    train_losses = []
    for step in range(steps):
        for i in range(len(R_train)):
            for j in range(len(R_train[i])):
                if R_train[i][j] > 0:
                    eij = R_train[i][j] - np.dot(U[i, :], V[:, j])
                    for k in range(d):
                        U[i][k] += alpha * (2 * eij * V[k][j] - beta * U[i][k])
                        V[k][j] += alpha * (2 * eij * U[i][k] - beta * V[k][j])

        if step % log_steps == 0 or step == steps - 1:
            train_loss = compute_loss(R_train, U, V)
            train_losses.append(train_loss)
            print(f"Step {step}: Training loss = {train_loss}")

        if train_loss < 0.001:
            break

    return U, V.T, train_losses


def compute_loss(R, U, V):
    """Calculate the total mean squared error on the dataset."""
    error = 0
    count = 0
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j] > 0:
                error += pow(R[i][j] - np.dot(U[i, :], V[:, j]), 2)
                count += 1
    return error / count if count != 0 else float('inf')


def compute_mse(R, nR):
    """ Compute Mean Squared Error """
    mse = 0
    count = 0
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i][j] > 0:  # Only consider non-zero entries
                prediction = nR[i][j]
                mse += (R[i][j] - prediction) ** 2
                count += 1
    return mse / count if count != 0 else 0


def recommend_items(user_id, user_ids, user_matrix, item_matrix, metric='cosine', top_n=10):
    """
    Generate item recommendations for a given user based on a similarity metric.

    Args:
    - user_id (int/str): Actual user ID as per your dataset
    - user_ids (np.array): Array of user IDs corresponding to the indices in the user_matrix
    - user_matrix (np.array): Matrix of user feature vectors (nP from matrix factorization)
    - item_matrix (np.array): Matrix of item feature vectors (nQ from matrix factorization)
    - metric (str): Similarity metric ('cosine' or 'dot'). Default is 'cosine'.

    Returns:
    - recommendations (np.array): Sorted array of item indices based on similarity scores
    """
    # Find the matrix index for the given user_id
    try:
        user_index = np.where(user_ids == user_id)[0][0]
    except IndexError:
        raise ValueError("User ID not found in the user IDs array.")

    # Fetch the user feature vector
    user_vector = user_matrix[user_index, :]

    # Compute similarity scores
    if metric == 'cosine':
        scores = cosine_similarity(user_vector.reshape(1, -1), item_matrix)[0]
    elif metric == 'dot':
        scores = np.dot(user_vector, item_matrix.T)
    else:
        raise ValueError("Unsupported similarity metric. Choose 'cosine' or 'dot'.")

    # Sort item indices based on scores in descending order
    recommended_item_indices = np.argsort(-scores)[:top_n]

    return recommended_item_indices, scores[recommended_item_indices]
