import numpy as np
from lightfm import LightFM
from lightfm.datasets import fetch_movielens


def sample_recommendation(model, data, user_ids):
    # number of users and movies in training data
    n_users, n_items = data['train'].shape

    # generate recommendations for each user we input
    for user_id in user_ids:

        # movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:10]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:10]:
            print("        %s" % x)


def init_movie_lens():
    data = fetch_movielens(min_rating=4.0)

    # print(repr(data['test']))
    # print(repr(data['train']))

    model = LightFM(loss='warp')  # weighted approx rank pairwise
    model.fit(data['train'], epochs=30, num_threads=2)
    sample_recommendation(model, data, [3, 25, 450])


if __name__ == '__main__':
    init_movie_lens()
