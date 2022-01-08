# -*- coding: utf-8 -*-
# @Author: xueqiao
# @Date:   2021-08-29 15:36:32
# @Last Modified by:   xueqiao
# @Last Modified time: 2022-01-07 01:52:14
import numpy as np
np.seterr(divide='ignore', invalid='ignore')



def get_Euclidean_Similarity(interaction_matrix):
    X=np.mat(interaction_matrix)
    row_matrix=np.power(interaction_matrix,2).sum(axis=1)
    distance_matrix=row_matrix+row_matrix.T-2*np.dot(X,X.T)
    distance_matrix=np.sqrt(distance_matrix)
    ones_matrix = np.ones(distance_matrix.shape)
    similarity_matrix=np.divide(np.mat(ones_matrix),(distance_matrix+ones_matrix))

    return matrix_normalize(similarity_matrix)
    

def get_Jaccard_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    E = np.ones_like(X.T)
    denominator=X * E + E.T * X.T - X * X.T
    denominator_zero_index=np.where(denominator==0)
    denominator[denominator_zero_index]=1
    result = X * X.T / denominator
    result[denominator_zero_index]=0
    result = result - np.diag(np.diag(result))
    # return matrix_normalize(result)
    return result


def get_Cosin_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    alpha = np.multiply(X, X).sum(axis=1)
    similarity_matrix = X * X.T / (np.sqrt(alpha * alpha.T))
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    for i in range(similarity_matrix.shape[0]):
        similarity_matrix[i, i] = 0

    return matrix_normalize(similarity_matrix)



def get_Pearson_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    X = X - (np.divide(X.sum(axis=1),X.shape[1]))
    similarity_matrix = get_Cosin_Similarity(X)
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    for i in range(similarity_matrix.shape[0]):
        similarity_matrix[i, i] = 0
    return similarity_matrix


def get_Gauss_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    delta = 1 / np.mean(np.power(X,2), 0).sum()
    alpha = np.power(X, 2).sum(axis=1)
    result = np.exp(np.multiply(-delta, alpha + alpha.T - 2 * X * X.T))
    result[np.isnan(result)] = 0
    result = result - np.diag(np.diag(result))

    return matrix_normalize(result)


def matrix_normalize(sim, Es = 1e-6):
    g0 = sim
    n = sim.shape[0]
    diff = 1
    while diff >= Es:
        for i in (0,200):
            temp1 = (np.ones((n,1)) * np.ones((1,n)) * g0) / n
            temp2 = np.eye(n) - g0
            g0[g0 < 0] = 0
            g1 = g0 + ((temp1 + temp2) * np.ones((n,1)) * np.ones((n,1))) / n - temp1
            
        J1 = np.trace(g1.T * g1 - 2 * sim.T * g1)
        J0 = np.trace(g0.T * g0 - 2 * sim.T * g0)
    
        diff = np.absolute(J1 - J0)
        g0 = g1
    
    similarity_matrix = g0
    return similarity_matrix
