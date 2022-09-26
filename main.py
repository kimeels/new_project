import os
import sys
import numpy as np
from loader import Data
from util import print_level
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
        
        x_train, y_train = d.load_data('train', shuffle=True)
        x_valid, y_valid = d.load_data('valid', shuffle=True)
        
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
        
        # Make a network prediction
        y_pred = model.predict(x_test)
    
        # Rescale to original range
        for i in range(d.n_params):
            y_pred[:, i] = (y_pred[:, i] *
                            d.norms['sig'][i]) + d.norms['mu'][i]
            y_test[:, i] = (y_test[:, i] *
                            d.norms['sig'][i]) + d.norms['mu'][i]

        header = d.param_keys[0]
        for i in range(1, d.n_params):
            header += ' ' + d.param_keys[i]
        np.savetxt(os.path.join(d.out_dir, 'inp_pred.txt'), y_test, header=header)
        np.savetxt(os.path.join(d.out_dir, 'out_pred.txt'), y_pred, header=header)
