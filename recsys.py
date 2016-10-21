#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 16:20:58 2016

@author: matt-666
"""

from lightfm.datasets import movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import time
from scipy.sparse import coo_matrix, csr_matrix, eye, diags, csc_matrix
from scipy.sparse.linalg import spsolve
import pandas as pd
import boto3
import json

def bm25_weight(data, K1=100, B=0.8):
    """ Weighs each row of the matrix data by BM25 weighting """
    # calculate idf per term (user)
    N = float(data.shape[0])
    idf = np.log(N / (1 + np.bincount(data.col)))

    # calculate length_norm per document (artist)
    row_sums = np.squeeze(np.asarray(data.sum(1)))
    average_length = row_sums.sum() / N
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    ret = coo_matrix(data)
    ret.data = ret.data * (K1 + 1.0) / (K1 * length_norm[ret.row] + ret.data) * idf[ret.col]
    return ret
# Alternating Least squares
def alternating_least_squares(Cui, factors, regularization=0.01,
                              iterations=15, use_native=True, num_threads=0,
                              dtype=np.float64):
    """ factorizes the matrix Cui using an implicit alternating least squares
    algorithm
    Args:
        Cui (csr_matrix): Confidence Matrix
        factors (int): Number of factors to extract
        regularization (double): Regularization parameter to use
        iterations (int): Number of alternating least squares iterations to
        run
        num_threads (int): Number of threads to run least squares iterations.
        0 means to use all CPU cores.
    Returns:
        tuple: A tuple of (row, col) factors
    """
    #_check_open_blas()

    users, items = Cui.shape

    X = np.random.rand(users, factors).astype(dtype) * 0.01
    Y = np.random.rand(items, factors).astype(dtype) * 0.01

    Cui, Ciu = Cui.tocsr(), Cui.T.tocsr()

    solver = least_squares

    for iteration in range(iterations):
        s = time.time()
        solver(Cui, X, Y, regularization, num_threads)
        solver(Ciu, Y, X, regularization, num_threads)
        print "finished iteration %i in %s" % (iteration, time.time() - s)

    return X, Y


def least_squares(Cui, X, Y, regularization, num_threads):
    """ For each user in Cui, calculate factors Xu for them
    using least squares on Y.
    Note: this is at least 10 times slower than the cython version included
    here.
    """
    users, factors = X.shape
    YtY = Y.T.dot(Y)

    for u in range(users):
        # accumulate YtCuY + regularization*I in A
        A = YtY + regularization * np.eye(factors)

        # accumulate YtCuPu in b
        b = np.zeros(factors)

        for i, confidence in nonzeros(Cui, u):
            factor = Y[i]
            A += (confidence - 1) * np.outer(factor, factor)
            b += confidence * factor

        # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
        X[u] = np.linalg.solve(A, b)


def nonzeros(m, row):
    """ returns the non zeroes of a row in csr_matrix """
    for index in range(m.indptr[row], m.indptr[row+1]):
        yield m.indices[index], m.data[index]

class TopRelated_useruser(object):
    def __init__(self, user_factors):
        # fully normalize artist_factors, so can compare with only the dot product
        norms = np.linalg.norm(user_factors, axis=-1)
        self.factors = user_factors / norms[:, np.newaxis]

    def get_related(self, movieid, N=10):
        scores = self.factors.dot(self.factors[movieid])
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best]), key=lambda x: -x[1])

class TopRelated_itemitem(object):
    def __init__(self, movie_factors):
        # fully normalize artist_factors, so can compare with only the dot product
        norms = np.linalg.norm(movie_factors, axis=-1)
        self.factors = movie_factors / norms[:, np.newaxis]

    def get_related(self, movieid, N=10):
        scores = self.factors.T.dot(self.factors.T[movieid])
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best]), key=lambda x: -x[1])

class ImplicitMF():

    def __init__(self, counts, num_factors=40, num_iterations=30,
                 reg_param=0.8):
        self.counts = counts
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param

    def train_model(self):
        self.user_vectors = np.random.normal(size=(self.num_users,
                                                   self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items,
                                                   self.num_factors))

        for i in xrange(self.num_iterations):
            t0 = time.time()
            print 'Solving for user vectors...'
            self.user_vectors = self.iteration(True, csr_matrix(self.item_vectors))
            print 'Solving for item vectors...'
            self.item_vectors = self.iteration(False, csr_matrix(self.user_vectors))
            t1 = time.time()
            print 'iteration %i finished in %f seconds' % (i + 1, t1 - t0)

    def iteration(self, user, fixed_vecs):
        num_solve = self.num_users if user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        YTY = fixed_vecs.T.dot(fixed_vecs)
        eye1 = eye(num_fixed)
        lambda_eye = self.reg_param * eye(self.num_factors)
        solve_vecs = np.zeros((num_solve, self.num_factors))

        t = time.time()
        for i in xrange(num_solve):
            if user:
                counts_i = self.counts[i].toarray()
            else:
                counts_i = self.counts[:, i].T.toarray()
            CuI = diags(counts_i, [0])
            pu = counts_i.copy()
            pu[np.where(pu != 0)] = 1.0
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
            YTCupu = fixed_vecs.T.dot(CuI + eye1).dot(csr_matrix(pu).T)
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
            solve_vecs[i] = xu
            if i % 1000 == 0:
                print 'Solved %i vecs in %d seconds' % (i, time.time() - t)
                t = time.time()

        return solve_vecs
        
# ==============================================================================




# on beer data =================================================================
beer_data = pd.read_csv('beer_reviews/beer_reviews.csv')

test_data = beer_data.groupby('review_profilename', as_index=False).apply(lambda x: x.loc[np.random.choice(x.index, 1, replace=False),:])
l1 = [x[1] for x in test_data.index.tolist()]

train_data = beer_data.drop(beer_data.index[l1]).dropna()

train_data['review_profilename'] = train_data['review_profilename'].astype("category")
train_data['beer_name'] = train_data['beer_name'].astype("category")

print "Unique users: %s" % (len(train_data['review_profilename'].unique()))
print "Unique beers: %s" % (len(train_data['beer_name'].unique()))

# create a sparse matrix of all the artist/user/play triples
reviews = csc_matrix((train_data['review_overall'].astype(float), 
                   (train_data['beer_name'].cat.codes, 
                    train_data['review_profilename'].cat.codes)))

beerid2beername = dict(enumerate(train_data['beer_name'].cat.categories))
beername2beerid = {v: k for k, v in beerid2beername.items()}

userid2username = dict(enumerate(train_data['review_profilename'].cat.categories))
username2userid  = {v: k for k, v in userid2username.items()}

#SVD ============
denseVecSize = 25
beer_factors, s, userbeer_factors = svds(bm25_weight(reviews.tocoo()), denseVecSize)
Related_beers_ii_svd25 = TopRelated_itemitem(beer_factors.T)

denseVecSize = 50
beer_factors_50, s, userbeer_factors_50 = svds(bm25_weight(reviews.tocoo()), denseVecSize)
Related_beers_ii_svd50 = TopRelated_itemitem(beer_factors_50.T)

denseVecSize = 100
beer_factors_100, s, userbeer_factors_100 = svds(bm25_weight(reviews.tocoo()), denseVecSize)
Related_beers_ii_svd100 = TopRelated_itemitem(beer_factors_100.T)

# Implicit =========================
impl = ImplicitMF(reviews.tocsr())
impl.train_model()
# user vectors is beers
impl_ii = TopRelated_itemitem(impl.user_vectors.T)

# ALS =================
als_userbeer_factors, als_beer_factors = alternating_least_squares(bm25_weight(reviews.tocoo()), 50)
als_ii = TopRelated_itemitem(als_userbeer_factors.T)


# Push to s3 - This won't work unless you set up amazon CLI on your computer
s3 = boto3.resource('s3')
for i in range(beer_factors.shape[0]):
    beer_recs_ii_svd25 = [{"value":beerid2beername[rec[0]].replace('"',''), "users":rec[1]} 
                           for rec in Related_beers_ii_svd25.get_related(i)]
    beer_recs_ii_svd50 = [{"value":beerid2beername[rec[0]].replace('"',''), "users":rec[1]} 
                           for rec in Related_beers_ii_svd50.get_related(i)]
    beer_recs_ii_svd100 = [{"value":beerid2beername[rec[0]].replace('"',''), "users":rec[1]} 
                            for rec in Related_beers_ii_svd100.get_related(i)]
    beer_recs_impl_ii = [{"value":beerid2beername[rec[0]].replace('"',''), "users":rec[1]} 
                          for rec in impl_ii.get_related(i)]
    beer_recs_als_ii = [{"value":beerid2beername[rec[0]].replace('"',''), "users":rec[1]} 
                         for rec in als_ii.get_related(i)]
    
    # remove spaces
    beername = beerid2beername[i].replace(' ', '_').replace('"','')
    beer_recs = {"svd25-item":beer_recs_ii_svd25, "implicit":beer_recs_impl_ii, 
                 "svd50-item":beer_recs_ii_svd50, "als":beer_recs_als_ii, 
                 "svd100-item":beer_recs_ii_svd100}
    # jsonify - just works better - always get double quotes etc
    with open('temp.json', 'wb') as fp:
        json.dump(beer_recs, fp)  
    try:
        s3.Object('beer-reco', beername+'.json').put(Body=open('temp.json', 'rb'), ACL='public-read')
    except UnicodeDecodeError:
        print ("can't assign: %s" % (beername))


# Get top ~10k 
count_data = beer_data['beer_name'].value_counts()    
count_data_top10k = count_data[count_data>15]    
beer_recs_all = []
for index, value in count_data_top10k.iteritems():
    beer_recs_all.append({"value":index, "users":value})
s3.Object('beer-reco', 'top10k').put(Body = str(beer_recs_all), 
          ACL='public-read', ContentType='string')



# if movies are your thing ==================================================== 
movie_data = movielens.fetch_movielens()
n_users, n_items = movie_data['train'].shape

model = LightFM(loss='warp')
model.fit(movie_data['train'], epochs=30, num_threads=2, user_features=None, 
          item_features=None)

print("Train precision: %.2f" % precision_at_k(model, movie_data['train'], k=5).mean())
print("Test precision: %.2f" % precision_at_k(model, movie_data['test'], k=5).mean())

train_prec, test_prec = [], []
epochs = np.linspace(1, 30, 30)

for epoch in epochs:
    model.fit(movie_data['train'], epochs = int(epoch), num_threads = 2)
    train_prec.append(precision_at_k(model, movie_data['train'], k=5).mean())
    test_prec.append(precision_at_k(model, movie_data['test'], k=5).mean())
    print(epoch)
    
plt.figure()
plt.plot(epochs, train_prec)
plt.plot(epochs, test_prec, 'g')

# Find most popular
md = movie_data['train'].toarray()
md[np.isnan(md)] = 0
nonzero_ratings = np.ma.masked_array(md, md==0).mean(axis=0)
top_movies = np.asarray(np.argsort(nonzero_ratings))[-22:-2]

print 'Top Movies: %s' % movie_data['item_labels'][top_movies]


# svd ======================================================================
denseVecSize = 25
user_factors, s, movie_factors = svds(bm25_weight(movie_data['train']), denseVecSize)


Related_uu = TopRelated_useruser(user_factors)
mov = 1
print 'Movie pick: %s' % movie_data['item_labels'][mov]
recs = [rec[0] for rec in Related_uu.get_related(mov)]
print 'Recomendations: %s' % movie_data['item_labels'][recs[1:]]


Related_ii = TopRelated_itemitem(movie_factors)
print 'Movie pick: %s' % movie_data['item_labels'][mov]
recs = [rec[0] for rec in Related_ii.get_related(mov)]
print 'Recomendations: %s' % movie_data['item_labels'][recs[1:]]

# Implicit MF ============================================================================
gg= ImplicitMF(movie_data['train'].tocsr())
gg.train_model()
gg_ii = TopRelated_itemitem(gg.item_vectors.T)

print 'Movie pick: %s' % movie_data['item_labels'][mov]
recs_gg = [rec[0] for rec in gg_ii.get_related(mov)]
print 'Recomendations: %s' % movie_data['item_labels'][recs_gg[1:]]

# Alernating Least squares ==============================================================
als_user_factors, als_movie_factors = alternating_least_squares(bm25_weight(movie_data['train']), 50)
als_ii = TopRelated_itemitem(als_movie_factors.T)

print 'Movie pick: %s' % movie_data['item_labels'][mov]
als_ii = [rec[0] for rec in als_ii.get_related(mov)]
print 'Recomendations: %s' % movie_data['item_labels'][als_ii[1:]]