import os
import sys
import numpy as np
from loader import Data, DataGenerator
from util import print_level, gen_header
from training_functions import make_model, train_network

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
            
        x_test, y_test = d.load_data('test')

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

        header = gen_header(d)
        np.savetxt(os.path.join(d.out_dir, 'inp_pred.txt'),
                   y_test,
                   header=header)
        np.savetxt(os.path.join(d.out_dir, 'out_pred.txt'),
                   y_pred,
                   header=header)
