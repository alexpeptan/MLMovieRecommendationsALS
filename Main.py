import implicit
import pandas as pd
import heapq
from scipy.sparse import coo_matrix

datafile = 'resources/u.data'
data = pd.read_csv(datafile, sep="\t", header=None, usecols=[0, 1, 2], names=['userId', 'itemId', 'rating'])

# Construct Users vs Movies rating matrix
data['userId'] = data['userId'].astype("category")
data['itemId'] = data['itemId'].astype("category")
rating_matrix = coo_matrix((data['rating'].astype(float),
                            (data['itemId'].cat.codes.copy(),
                             data['userId'].cat.codes.copy())))


# Training Phase

# user_factors, item_factors = implicit.alternating_least_squares(rating_matrix, factors=10, regularization=0.01)
# WARNING:implicit:This method is deprecated. Please use the AlternatingLeastSquares class instead

als = implicit.als.AlternatingLeastSquares(factors=30, regularization=1e-9)
als.fit(rating_matrix)
user_factors = als.user_factors
item_factors = als.item_factors


# Testing Phase

# Returns top k most rated movies for user_id
def topk_for_user(topk, user_id):
    user = item_factors.dot(user_factors[user_id])

    return heapq.nlargest(topk, range(len(user)), user.take)


print(topk_for_user(5, 196))
print(topk_for_user(3, 7))

# TODO: How can I stabilize the received recommendations upon repeated reruns of the fitting algorithm?
# We should have reproducible results
# Or does a model configured with (factors=10, regularization=0.01) not capture that complexity?
# Does it allow for too many of solutions to exist? (too few factors/too small of a regularization constant?)

# This configuration(factors=30, regularization=1e-9) appears to provide more stable results.
# Still far from perfect.
# More analysis needed to better calibrate the ALS algorithm
