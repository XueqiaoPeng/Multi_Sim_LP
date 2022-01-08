import os, glob
import numpy as np
import pandas as pd
import random
from numpy import genfromtxt
from similarity import matrix_normalize

def load_data(file, drug_adr_pair):
    drug_list = list(set(drug for (drug, adr) in drug_adr_pair))
    adr_list = list(set(adr for (drug, adr) in drug_adr_pair))

    id2drug = {i: drug for i, drug in enumerate(drug_list)}
    drug2id = {drug: i for i, drug in enumerate(drug_list)}

    id2adr = {i: adr for i, adr in enumerate(adr_list)}
    adr2id = {adr: i for i, adr in enumerate(adr_list)}

    x = np.zeros(shape=(len(drug_list), len(adr_list)))
    drug_adr_pair = set()

    with open(file, 'r') as f:
        next(f)
        for row in f:
            row = row.strip().split(',')
            drug, adr, score = row[0], row[1], row[2]
            if score == 'NA' or score=='Inf':
                continue

            drug_id, adr_id = drug2id.get(drug), adr2id.get(adr)
            if drug in drug_list and adr in adr_list:
                x[drug_id, adr_id] = score
                drug_adr_pair.add((drug, adr))

    return x, drug_list, adr_list


def load_data_sider(sider_eval_pairs):
    drug_sider = list(set(drug for (drug, adr) in sider_eval_pairs))
    adr_sider = list(set(adr for (drug, adr) in sider_eval_pairs))

    id2drug = {i: drug for i, drug in enumerate(drug_sider)}
    drug2id = {drug: i for i, drug in enumerate(drug_sider)}

    id2adr = {i: adr for i, adr in enumerate(adr_sider)}
    adr2id = {adr: i for i, adr in enumerate(adr_sider)}

    y = np.zeros(shape=(len(drug_sider), len(adr_sider)))
    for drug, adr in sider_eval_pairs:
        drug_id, adr_id = drug2id.get(drug), adr2id.get(adr)
        y[drug_id, adr_id] = 1
    y = np.asarray(y)
    return y, drug_sider, adr_sider


def get_eval_matrix(Y, drug_sider, adr_sider, drug_list, adr_list):
    # get the evaluation matrix
    sider_df = pd.DataFrame(drug_sider)
    sider_df.columns = ['drugid']
    y_df = pd.DataFrame(Y, index = drug_list, columns = adr_list)
    merged_df = sider_df.merge(y_df, how = "inner" , left_on = 'drugid', right_index= True)
    df_res = pd.DataFrame()
    for adr in adr_sider:
        df0 = merged_df.loc[:,[adr]]
        df_res = pd.concat([df_res,df0],axis=1)
    
    eval_matrix = df_res.to_numpy()

    return eval_matrix


def sample_zeros(y):
    zeros_r_idx, zeros_c_idx = np.where(y==0)
    ones_r_idx, ones_c_idx = np.where(y==1)


    sample_zeros_pos = random.sample(range(len(zeros_r_idx)), len(ones_r_idx)*2)
    sample_zeros_r_idx, sample_zeros_c_idx = zeros_r_idx[sample_zeros_pos], zeros_c_idx[sample_zeros_pos]

    sample_r_idx = np.append(ones_r_idx, sample_zeros_r_idx)
    sample_c_idx = np.append(ones_c_idx, sample_zeros_c_idx)

    return (sample_r_idx, sample_c_idx)

def load_similarity():
    path = r'./data_all/'
    file = glob.glob(os.path.join(path, "*_similarity.csv"))
    similarity_list = []
    for f in file:
        similarity_matrix = genfromtxt(f, delimiter=',')
        # similarity_matrix = matrix_normalize(similarity_matrix)
        similarity_list.append(similarity_matrix)

    return similarity_list














