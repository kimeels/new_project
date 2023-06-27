import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from loader import Data, DataGenerator
from util import print_level, gen_header
from training_functions import make_model, train_network, make_model_fine_tune

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

    # print_level('LC TESTING FINE TUNING', 1, d.verbose)
    # fine_tune = True
    # learning_rate = 1e-2

    
    # model = make_model_fine_tune(final_layer=d.n_params,
    #                              learning_rate=learning_rate,
    #                              fine_tune=fine_tune,
    #                              weights_path=os.path.join(d.out_dir, 'cp.ckpt'
    #                                                         ))
    model = make_model(final_layer=d.n_params)

    if d.train:
        print_level('training',
                    1,
                    d.verbose)

        if d.generator:
            print_level('using a data generator',
                        2,
                        d.verbose)
 
            x_train = DataGenerator(d, 'train')
            y_train = None
            x_valid = DataGenerator(d, 'valid')
            y_valid = None

        else:
            print_level('loading all the data',
                        2,
                        d.verbose)

            x_train, y_train = d.load_data('train', shuffle=True)
            x_valid, y_valid = d.load_data('valid', shuffle=True)

            if d.max_n_slice > 0:
                s = x_train.shape[0]
                # Calculate ratio of sizes to determine how many
                # slices to take to keep ratieos -- same for test
                i1_train = d.max_n_slice
                i1_valid = int(d.max_n_slice * x_valid.shape[0] / s)
                x_train = x_train[0:i1_train]
                y_train = y_train[0:i1_train]
                x_valid = x_valid[0:i1_valid]
                y_valid = y_valid[0:i1_valid]
                
        train_network(model,
                      x_train=x_train,
                      y_train=y_train,
                      x_val=x_valid,
                      y_val=y_valid,
                      batch_size=d.batch_size,
                      epochs=d.epochs,
                      save_weights_only=True,
                      dirname=d.out_dir)

    if d.test:
        print_level('testing',
                    1,
                    d.verbose)
        model.load_weights(os.path.join(d.out_dir, 'cp.ckpt'))
        
        # ret_z = True
        # x_test, y_test, z_test = d.load_data('test', ret_z=ret_z)
        ret_z = False
        x_test, y_test = d.load_data('test', ret_z=ret_z)

        # if d.max_n_slice > 0:
        #     i1_test = int(x_test.shape[0] // s) * d.max_n_slice
        #     x_test = x_test[0:i1_test]
        #     y_test = y_test[0:i1_test]
            
        # Make a network prediction
        y_pred = model.predict(x_test)
    
        # Rescale to original range
        for i in range(d.n_params):
            y_pred[:, i] = (y_pred[:, i] *
                            d.norms['sig'][0, i]) + d.norms['mu'][0, i]
            y_test[:, i] = (y_test[:, i] *
                            d.norms['sig'][0, i]) + d.norms['mu'][0, i]

        # fig, ax = plt.subplots(figsize=(4, 4))
        # ax.hist(diff(y_pred[:, 0], y_test[:, 0]), bins=300)
        # ax.set_yscale('log')
        # ax.set_xlabel('diff')
        # fig.savefig('diff.pdf')

        # bad_pred = 0.008693039417266846
        # bad_idxs = np.abs(y_pred[:, 0] - bad_pred) < 1e-4
        # bad_slcs = x_test[bad_idxs]
        # good_slcs = x_test[~bad_idxs]
        # print('mean/median/min/max bad_slcs', np.mean(bad_slcs), np.median(bad_slcs), bad_slcs.min(), bad_slcs.max())
        # print('mean/median/min/max good_slcs', np.mean(good_slcs), np.median(good_slcs), good_slcs.min(), good_slcs.max())
        # for i in range(128):
        #     plt.imsave(f'/tmp/{i:03d}.png', bad_slcs[i, :, :, 0])
        #     plt.imsave(f'/tmp/{i:03d}_good.png', good_slcs[i, :, :, 0])
        
        header = gen_header(d)
        np.savetxt(os.path.join(d.out_dir, 'inp_pred.txt'),
                   y_test,
                   header=header)
        np.savetxt(os.path.join(d.out_dir, 'out_pred.txt'),
                   y_pred,
                   header=header)
        if ret_z:
            np.savetxt(os.path.join(d.out_dir, 'inp_zs.txt'),
                       z_test,
                       header=header)
            
