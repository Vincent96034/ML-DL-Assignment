from abc import ABC, abstractmethod

import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


class BaseTTRecommender(ABC):
    def __init__(self, **kwargs):
        self.model = self.build_model(**kwargs)

    @abstractmethod
    def build_model(self):
        pass

    def fit(
            self,
            features_train,
            labels_train,
            epochs=8,
            batch_size=32,
            validation_split=0.2,
            callbacks=None
    ) -> dict:
        history = self.model.fit(
            features_train,
            labels_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks)
        self.history = history
        return history

    def recommend_batch(self, user_embeddings, item_embeddings, use_ids=False, k=200, batch_size=128, n_jobs=-1):
        item_features = item_embeddings.iloc[:, 1:].values.astype(
            'float32')  # Extract item features
        item_ids = item_embeddings.iloc[:, 0].values  # Extract item ids
        print(f"#users {len(user_embeddings)}, #items {len(item_embeddings)}")

        def process_user_batch(user_batch):

            user_features = user_batch.iloc[:, 2:].values.astype(
                'float32')  # Extract user features
            user_features_repeated = np.repeat(user_features, len(item_features), axis=0)
            item_features_repeated = np.tile(item_features, (len(user_batch), 1))

            # Predict scores for all items for the batch of users
            input_ = [user_features_repeated, item_features_repeated]
            if use_ids:
                user_ids_repeated = np.repeat(user_batch.uid.values, len(item_features))
                item_ids_repeated = np.tile(item_ids, len(user_batch))
                input_ = [user_ids_repeated, item_ids_repeated,
                          user_features_repeated, item_features_repeated]
            predictions = self.model.predict(input_, batch_size=batch_size)

            results = []
            for i in range(len(user_batch)):
                start = i * len(item_features)
                end = (i + 1) * len(item_features)
                user_predictions = predictions[start:end].flatten()

                # Get top k item ids and scores
                top_k_indices = np.argsort(user_predictions)[-k:][::-1]
                top_k_ids = item_ids[top_k_indices]
                top_k_scores = user_predictions[top_k_indices]

                results.append({
                    'eid': user_batch.iloc[i]['eid'],
                    'recommended_ids': top_k_ids.tolist(),
                    'scores': top_k_scores.tolist()
                })

            return results

        # Split user_embeddings into batches
        user_batches = np.array_split(
            user_embeddings, np.ceil(len(user_embeddings) / batch_size))

        # Use parallel processing to handle batches
        results = Parallel(n_jobs=n_jobs)(delayed(process_user_batch)(batch)
                                          for batch in user_batches)

        # Flatten the list of results
        recommendations = {}
        for result in results:
            for res in result:
                recommendations[res['eid']] = {
                    'recommended_ids': res['recommended_ids'],
                    'scores': res['scores']
                }
        return recommendations

    def evaluate(self, input_features, labels):
        return self.model.evaluate(input_features, labels)

    def predict(self, input_features):
        return self.model.predict(input_features)

    def plot_model_history(self):
        if not hasattr(self, 'history'):
            raise ValueError(
                'Model has not been trained yet. Please call the `train` method first.')
        _, ax = plt.subplots(1, 2, figsize=(10, 4))
        # Plot the loss on the first subplot
        ax[0].plot(self.history.history['loss'], label='train')
        ax[0].plot(self.history.history['val_loss'], label='validation')
        ax[0].set_title('Model Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        # Plot the accuracy on the second subplot
        ax[1].plot(self.history.history['accuracy'], label='train')
        ax[1].plot(self.history.history['val_accuracy'], label='validation')
        ax[1].set_title('Model Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        plt.tight_layout()
        plt.show()

    def recommend(self, user_embeddings, item_embeddings, k=200):

        item_features = item_embeddings.iloc[:, 1:].values  # Extract item features
        item_ids = item_embeddings.iloc[:, 0].values  # Extract item ids
        print(f"#users {len(user_embeddings)}, #items {len(item_embeddings)}")

        recommendations = {}

        for _, user_row in user_embeddings.iterrows():
            eid = user_row['eid']
            user_feature = user_row.iloc[1:].values.reshape(
                1, -1)  # Extract user features
            user_feature = user_feature.astype('float32')

            # Predict scores for all items for the current user
            user_features_eval_repeated = np.repeat(
                user_feature, item_features.shape[0], axis=0)
            predictions = self.model.predict([user_features_eval_repeated, item_features])

            # Get top k item ids and scores
            top_k_indices = np.argsort(predictions.flatten())[-k:][::-1]
            top_k_ids = item_ids[top_k_indices]
            top_k_scores = predictions.flatten()[top_k_indices]

            recommendations[eid] = {
                'recommended_ids': top_k_ids.tolist(),
                'scores': top_k_scores.tolist()
            }

        return recommendations
