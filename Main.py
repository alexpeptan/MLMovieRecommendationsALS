import implicit
import pandas as pd
import heapq
from scipy.sparse import coo_matrix

datafile = 'resources/u.data'
data = pd.read_csv(datafile, sep="\t", header=None, usecols=[0, 1, 2], names=['userId', 'itemId', 'rating'])

# Construct Users vs Items matrix
data['userId'] = data['userId'].astype("category")
data['itemId'] = data['itemId'].astype("category")
rating_matrix = coo_matrix((data['rating'].astype(float),
                            (data['itemId'].cat.codes.copy(),
                             data['userId'].cat.codes.copy())))

# Training Phase

# Deprecated
# user_factors, item_factors = implicit.alternating_least_squares(rating_matrix, factors=10, regularization=0.01)
# WARNING:implicit:This method is deprecated. Please use the AlternatingLeastSquares class instead

als = implicit.als.AlternatingLeastSquares(factors=30, regularization=1e-9)
als.fit(rating_matrix)
user_factors = als.user_factors
item_factors = als.item_factors


# Testing Phase

# Array of ratings for all the movies for user 196
# user196=item_factors.dot(user_factors[196])
# print(user196)

def topk_for_user(topk, user_id):
    user = item_factors.dot(user_factors[user_id])

    return heapq.nlargest(topk, range(len(user)), user.take)


# Print top 5 recommended movies for user with id 196
print(topk_for_user(5, 196))
# Print top 3 recommended movies for user with id 007
print(topk_for_user(3, 7))

# TODO: See how I can make such that the recommendations will remain the same on rerunning the fitting algorithm.
# It makes all the sense in the world for a recommendation algorithm to produce reproducible results
# Or is the current dataset  "too relaxed" (factors=10, regularization=0.01)?
# It allows for too much of a variety of solutions(too few factors?)

# This configuration(factors=30, regularization=1e-9) appears to provide more stable/reproducible results.
# Still far from perfect.
# More analysis needed to better calibrate the ALS algorithm
