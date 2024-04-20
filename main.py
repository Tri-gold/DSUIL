import time
import argparse
import pickle
from model import *
from utils import *


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='diginetica/Nowplaying/Tmall')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=2)
parser.add_argument('--is_time', type=int, default=0)
parser.add_argument('--gamma', type=float, default=0.2)
parser.add_argument('--is_add_two_edge', type=int, default=1)
parser.add_argument('--is_meaning', type=int, default=0)
opt = parser.parse_args()


def main():
    init_seed(2020)

    if opt.dataset == 'diginetica':
        num_node = 43098
        opt.n_iter = 2



    elif opt.dataset == 'Tmall':
        num_node = 16162
        pad_beh = 3

 
    elif opt.dataset == 'UB':
        pad_beh = 4
        num_node = 32838
  
    elif opt.dataset == 'RC15':
        pad_beh =2
        num_node = 9620
    elif opt.dataset == 'ML1M':
        pad_beh =2
        num_node = 2357
  
    else:
        num_node = 310

    train_data = pickle.load(open(opt.dataset + '/train.txt', 'rb'))

    test_data = pickle.load(open(opt.dataset + '/test.txt', 'rb'))

    train_data = Data(train_data, pad_item=num_node, pad_beh=pad_beh, dataset = opt.dataset)
    test_data = Data(test_data, pad_item=num_node, pad_beh=pad_beh, dataset = opt.dataset)
    model = trans_to_cuda(CombineGraph(opt, num_node))

    print(opt)
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        result = train_test(model, train_data, test_data)
        hit10, ndcg10, hit20, ndcg20 = result
        flag = 0
        if hit10 >= best_result[0]:
            best_result[0] = hit10
            best_epoch[0] = epoch
            flag = 1
        if ndcg10 >= best_result[1]:
            best_result[1] = ndcg10
            best_epoch[1] = epoch
            flag = 1
        print('Current Result:')
        print('\tHit@10:\t%.4f\tNDCG@10:\t%.4f\n \tHit@20:\t%.4f\tNGCD@20:\t%.4f' % (hit10, ndcg10, hit20, ndcg20))
        print('Best Result:')
        print('\tHit@10:\t%.4f\tNDCG@10:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
