import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from loader_simple import Data, DataGenerator
from util import print_level, gen_header
from training_functions import make_model, train_network
from scipy.stats import mode


def diff(x, y):
    return np.abs((y - x) / x)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        config_path = 'config.yaml'
        print_level(f'path to config file not specified, defaulting to {config_path}',
        1,
        True)  # we don't know d.verbose yet
    else:
        config_path = sys.argv[1]

    d = Data(config_path)
    
    print_level('making model',
                1,
                d.verbose)

    model = make_model(final_layer=d.n_params)

    if d.train:
        print_level('training',
                    1,
                    d.verbose)

        shuffle = True
        if d.generator:
            x_train = DataGenerator(D=d, ds='train', shuffle=shuffle)
            x_valid = DataGenerator(D=d, ds='valid', shuffle=shuffle)
            y_train = None
            y_valid = None

        else:
            x_train, y_train = d.load_data('train', shuffle=shuffle)
            x_valid, y_valid = d.load_data('valid', shuffle=shuffle)

            # Quick data check
            for i in [0, -1]:
                print(f'<xHI>[{i:d}]:', y_train[i])
                print('min(dTb)', x_train[i, :, :, 0].min())
                print('max(dTb)', x_train[i, :, :, 0].max())
                # plt.imshow(x_train[i, :, :, 0])
                # plt.show()
        
        train_network(model,
                      x_train=x_train,
                      y_train=y_train,
                      x_val=x_valid,
                      y_val=y_valid,
                      batch_size=d.batch_size,
                      epochs=d.epochs,
                      save_weights_only=True,
                      dirname=d.weight_dir)

    if d.test:
        print_level('testing',
                    1,
                    d.verbose)
        model.load_weights(os.path.join(d.weight_dir, 'cp.ckpt'))
        x_test, y_test = d.load_data('test')
        y_pred = model.predict(x_test)

        # Check the bad outliers
        fail, _ = mode(y_pred)
        fail = float(fail)
        print(fail)
        idxs = np.argwhere(np.abs(y_pred - fail) < 1e-6)[:, 0]
        print(idxs)
        for i, idx in enumerate(idxs):
            if (i < 100): #(idx % 1000 == 0):
                print(idx)
                plt.imsave(os.path.join(d.out_dir, f'{idx:05d}.png'),
                           x_test[idx, :, :, 0])
        
        np.savetxt(os.path.join(d.out_dir, 'inp_pred.txt'),
                   y_test)
        np.savetxt(os.path.join(d.out_dir, 'out_pred.txt'),
                   y_pred)
