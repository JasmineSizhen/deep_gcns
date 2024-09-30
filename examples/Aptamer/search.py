from bayes_opt import BayesianOptimization
from main import main
from args import ArgsInit
import random
import logging

def func(num_layers, mlp_layers, hidden_channels, gcn_aggr, batch_size):
    """Function with unknown internals we wish to maximize.
    """
    args = ArgsInit().save_exp()
    args.num_layers = int(num_layers)
    args.mlp_layers = int(mlp_layers)
    args.hidden_channels = 2 ** int(hidden_channels)
    args.gcn_aggr = ['mean', 'max', 'add', 'softmax', 'softmax_sg', 'power'][int(gcn_aggr)]
    args.batch_size = 2 ** int(hidden_channels)
    
    args.epochs = 100
    args.use_gpu = True
    args.add_virtual_node= False
    
    best_pear = 0.
    valid_spear = 0.
    test_pear = 0.
    test_spear = 0.0
    i = 0
    while i < 10:
        args.val_set = random.randint(1, 10)
        args.test_set = random.randint(1, 10)
        if args.val_set == args.test_set:
            continue

        results = main(args)
        best_pear += results["highest_valid_pear"]
        valid_spear += results["valid_spear"]
        test_pear += results["test_pear"]
        test_spear += results["test_spear"]
        i += 1
        
    logging.info('%s' % args)
    print(best_pear/10, valid_spear/10, test_pear/10, test_spear/10)

    return best_pear / 10


# Bounded region of parameter space
pbounds = {'num_layers': (3, 15), "mlp_layers": (1, 4), "hidden_channels": (6, 9), "gcn_aggr": (0, 6), "batch_size": (5, 8)}

optimizer = BayesianOptimization(
    f=func,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=10,
    n_iter=50,
)