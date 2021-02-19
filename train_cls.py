from __future__ import print_function
from collections import OrderedDict
import os
import neural_cls
from neural_cls.util import Trainer, Loader
from neural_cls.models import BiLSTM
from neural_cls.models import CNN
from neural_cls.models import CNN_BB
import matplotlib.pyplot as plt
import torch

import hydra
import numpy as np
import logging
import warnings

warnings.filterwarnings('ignore')
log = logging.getLogger(__name__)

@hydra.main(config_path=os.environ['HYDRA_CONFIG_PATH'])
def main(config):
    os.chdir(hydra.utils.get_original_cwd())

    usemodel = config.parameters['model']
    use_dataset = config.parameters.dataset
    if usemodel == 'BiLSTM' and use_dataset == 'trec':
        config.parameters['dpout'] = 0.5
        config.parameters['wldim'] = 200
        config.parameters['nepch'] = 10
        config.parameters['lrate'] = 0.001
        config.parameters['batch_size'] = 50
        config.parameters['opsiz'] = 7

    elif usemodel == 'BiLSTM' and use_dataset == 'mareview':
        config.parameters['dpout'] = 0.5
        config.parameters['wldim'] = 200
        config.parameters['nepch'] = 5
        config.parameters['lrate'] = 0.001
        config.parameters['batch_size'] = 50
        config.parameters['opsiz'] = 2

    elif usemodel == 'CNN' and use_dataset == 'trec':
        config.parameters['dpout'] = 0.5
        config.parameters['wlchl'] = 100
        config.parameters['nepch'] = 15
        config.parameters['lrate'] = 0.001
        config.parameters['batch_size'] = 50
        config.parameters['opsiz'] = 7

    elif usemodel == 'CNN' and use_dataset == 'mareview':
        config.parameters['dpout'] = 0.5
        config.parameters['wlchl'] = 100
        config.parameters['nepch'] = 5
        config.parameters['lrate'] = 0.001
        config.parameters['batch_size'] = 50
        config.parameters['opsiz'] = 2

    elif usemodel == 'CNN_BB' and use_dataset == 'trec':
        config.parameters['wlchl'] = 100
        config.parameters['nepch'] = 10
        config.parameters['lrate'] = 0.001
        config.parameters['batch_size'] = 50
        config.parameters['opsiz'] = 7
        config.parameters['sigmp'] = float(np.exp(-3))

    elif usemodel == 'CNN_BB' and use_dataset == 'mareview':
        config.parameters['wlchl'] = 100
        config.parameters['nepch'] = 5
        config.parameters['lrate'] = 0.001
        config.parameters['batch_size'] = 50
        config.parameters['opsiz'] = 2
        config.parameters['sigmp'] = float(np.exp(-3))

    else:
        raise NotImplementedError()
    dataset_path = os.path.join('datasets', use_dataset)
    result_path = os.path.join(config.parameters.result_path, use_dataset)
    model_load = config.opt.reload


    loader = Loader()
    print('Model:', usemodel)
    print('Dataset:', use_dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(os.path.join(result_path, usemodel)):
        os.makedirs(os.path.join(result_path, usemodel))
    if use_dataset == 'trec':
        train_data, test_data, mappings = loader.load_trec(dataset_path, config.parameters['ptrnd'],
                                                           config.parameters['wrdim'])
    elif use_dataset == 'mareview':
        train_data, test_data, mappings = loader.load_mareview(dataset_path, config.parameters['ptrnd'],
                                                               config.parameters['wrdim'])
    else:
        raise NotImplementedError()
    word_to_id = mappings['word_to_id']
    tag_to_id = mappings['tag_to_id']
    word_embeds = mappings['word_embeds']
    print('Load Complete')
    if model_load:
        print('Loading Saved Weights....................................................................')
        model_path = os.path.join(result_path, usemodel, opt.checkpoint, 'modelweights')
        model = torch.load(model_path)
    else:
        print('Building Model............................................................................')
        if (usemodel == 'BiLSTM'):
            print('BiLSTM')
            word_vocab_size = len(word_to_id)
            word_embedding_dim = config.parameters['wrdim']
            word_hidden_dim = config.parameters['wldim']
            output_size = config.parameters['opsiz']

            model = BiLSTM(word_vocab_size, word_embedding_dim, word_hidden_dim,
                           output_size, pretrained=word_embeds)

        elif (usemodel == 'CNN'):
            print('CNN')
            word_vocab_size = len(word_to_id)
            word_embedding_dim = config.parameters['wrdim']
            word_out_channels = config.parameters['wlchl']
            output_size = config.parameters['opsiz']

            model = CNN(word_vocab_size, word_embedding_dim, word_out_channels,
                        output_size, pretrained=word_embeds)

        elif (usemodel == 'CNN_BB'):
            print('CNN_BB')
            word_vocab_size = len(word_to_id)
            word_embedding_dim = config.parameters['wrdim']
            word_out_channels = config.parameters['wlchl']
            output_size = config.parameters['opsiz']
            sigma_prior = config.parameters['sigmp']

            model = CNN_BB(word_vocab_size, word_embedding_dim, word_out_channels,
                           output_size, sigma_prior=sigma_prior, pretrained=word_embeds)
    model.cuda()
    learning_rate = config.parameters['lrate']
    num_epochs = config.parameters['nepch']
    print('Initial learning rate is: %s' % (learning_rate))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, optimizer, result_path, usemodel, tag_to_id, usedataset=use_dataset)
    losses, all_F = trainer.train_model(num_epochs, train_data, test_data, learning_rate,
                                        batch_size=config.parameters['batch_size'])
    plt.plot(losses)
    plt.savefig(os.path.join(result_path, usemodel, 'lossplot.png'))


if __name__ == '__main__':
    main()
