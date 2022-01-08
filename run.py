from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import numpy as np

import pickle
from Model import Model
from utils import load_data,load_data_sider, sample_zeros, load_similarity


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--eval_metrics', required=True, choices=['all', 'specificity-sensitivity'],
                        help='Evaluation metrics')
    parser.add_argument('--split', type=bool, default=False)
    parser.add_argument('--output')

    args = parser.parse_args()
    return args


def pretty_print_eval(res, metrics):
    if metrics == 'all':
        print('All metrics: ' + ','.join(np.round(res,3).astype(str)))
    else:
        print('fixed_sensitivity: ' + ','.join(np.round(res[1],3).astype(str)))
        print('fixed_specificity: ' + ','.join(np.round(res[2],3).astype(str)))


def main(args):

    drug_adr_pairs = pickle.load(open('./data_all/drug_adr_go.pickle','rb'))
    sider_eval_pairs = pickle.load(open('/users/PCON0023/xueqiao/drugsimilarity/Multi_sim_LP/data_all/sider_go.pickle', 'rb'))
    method = 'gps_04'
    file = '/users/PCON0023/xueqiao/drugsimilarity/LP-SDA/OriginalSignal/GPS/{}.csv'.format(method)
    print('#' * 50)
    print('Signal Detection Algorithm: {}'.format(method))
    print('#' * 50)

    similarity_list = load_similarity()
    Y, drug_sider, adr_sider = load_data_sider(sider_eval_pairs)
    all_idx = sample_zeros(Y)
    X, drug_list, adr_list = load_data(file, drug_adr_pairs)

    model = Model(args.eval_metrics)
    lp_res, base_res = model.validate(X, Y, all_idx, model.THETA, drug_sider, adr_sider, drug_list, adr_list, similarity_list)

    print('LP-{}:'.format(method))
    pretty_print_eval(lp_res, args.eval_metrics)

    print('baseline-{}:'.format(method))
    pretty_print_eval(base_res, args.eval_metrics)

def more_main():
    args = parse_args()
    main(args)

if __name__ == '__main__':
    more_main()
