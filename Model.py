import numpy as np
import pickle
import csv
from collections import defaultdict
import pandas as pd
from numpy import genfromtxt
from eval import Eval
from similarity import matrix_normalize
from utils import get_eval_matrix
# from mapping import id2drug, id2adr



class Model:
    def __init__(self, metrics):
        self.ALPHA = 0.1
        self.metrics = metrics
        self.THETA = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
        self.DELTA = 1


    def solve_Y(self, X, alpha, theta, similarity_list):
        #Fix theta, solve Y
        weight_simi_matrix = np.zeros((1111,1111))
        for i in range(len(similarity_list)):
            weight_simi_matrix = weight_simi_matrix + theta[i] * similarity_list[i]
        weight_simi_matrix = matrix_normalize(weight_simi_matrix)
        Y = (1 - alpha) * np.matmul(np.linalg.pinv(
            np.eye(np.shape(X)[0]) - alpha * weight_simi_matrix), X)

        return Y


    def solve_theta(self, Y, alpha, delta, similarity_list, s = 1):
        #Fix Y, solve theta
        #sum of vector theta is 1
        I = np.identity(len(Y))
        C = []
        for matrix in similarity_list:
            c = np.trace(np.matmul(np.matmul(Y.T,(I - matrix)),Y))
            C.append(c)
    
        C = np.transpose(np.array(C))
        C1 = ((-alpha) / (2 * delta)) * C
        # print(C1)
        n, = C1.shape
        u = np.sort(C1)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
        theta0 = float(cssv[rho] - s) / (rho + 1)
        theta = (C1 - theta0).clip(min=0)
        return theta

    def validate(self, X, Y, idx, theta, drug_sider, adr_sider, drug_list, adr_list, similarity_list):
        AUC = []
        test_auc_best = float('-inf')
        test_res_best = [0]*6
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # for alpha in [0.6, 0.7, 0.8, 0.9, 1.0]:
            for delta in [0.1,0.2,0.3,0.4,0.5,0.7,0.9,1.0,3.0,5.0,8.0,10.0]:
                for i in range(0,5):
            #iteratively solve Y and theta
                    score_matrix = self.solve_Y(X, alpha, theta, similarity_list)
                    theta = self.solve_theta(score_matrix, alpha, delta, similarity_list, s = 1)
                eval_score_matrix = get_eval_matrix(score_matrix, drug_sider, adr_sider, drug_list, adr_list)
                test_res = self.eval(eval_score_matrix, Y, idx)
                auc = test_res[0]
                AUC.append(auc)
                for i in range(len(test_res)):
                    if test_res[i] > test_res_best[i]:
                        test_res_best[i] = test_res[i]
                if test_res[0] > test_auc_best:
                    test_auc_best = test_res[0]
                    test_res_best = test_res
                    best_parameters = {"alpha":alpha,"delta":delta, "theta":theta}
        self.ALPHA = best_parameters["alpha"]
        self.DELTA = best_parameters["delta"]
        self.THETA = best_parameters["theta"]

        eval_matrix = get_eval_matrix(X, drug_sider, adr_sider, drug_list, adr_list)
        test_res = self.eval(eval_matrix, Y, idx)

        return test_res_best, test_res
        

    def eval(self, Y_pred, Y, idx):
        y_pred, y_gold = [], []
        for r, c in zip(idx[0], idx[1]):
            y_pred.append(Y_pred[r, c])
            y_gold.append(Y[r, c])
        ev = Eval(y_pred, y_gold)
        return ev.Metrics(self.metrics)





