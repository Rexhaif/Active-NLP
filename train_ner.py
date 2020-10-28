from __future__ import print_function
from collections import OrderedDict
import os
import datetime
import neural_ner
from neural_ner.util import Trainer, Loader
from neural_ner.models import CNN_BiLSTM_CRF
from neural_ner.models import CNN_BiLSTM_CRF_MC
from neural_ner.models import CNN_BiLSTM_CRF_BB
from neural_ner.models import CNN_CNN_LSTM
from neural_ner.models import CNN_CNN_LSTM_MC
from neural_ner.models import CNN_CNN_LSTM_BB
import matplotlib.pyplot as plt
import torch
import random
import hydra

import numpy as np


import warnings
warnings.filterwarnings('ignore')



@hydra.main(config_path=os.environ['HYDRA_CONFIG_PATH'])  # os.environ['HYDRA_CONFIG_PATH']
def main(config):
    os.chdir(hydra.utils.get_original_cwd())
    config.parameters.model = config.opt.usemodel
    config.parameters.wrdim = config.opt.worddim
    config.parameters.ptrnd = config.opt.pretrnd

    use_dataset = config.opt.dataset
    dataset_path = os.path.join('datasets', use_dataset)

    tz = datetime.timezone(datetime.timedelta(hours=3))
    dt = datetime.datetime.now(tz=tz)
    date_path = f'{dt.date()}'
    time_path = f'{dt.time()}'.replace(':', '-').split('.')[0]

    result_path = os.path.join(config.opt.result_path, use_dataset, date_path, time_path)
    model_name = config.opt.usemodel
    model_load = config.opt.reload
    loader = Loader()

    print('Model:', model_name)
    print('Dataset:', use_dataset)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if not os.path.exists(os.path.join(result_path, model_name)):
        os.makedirs(os.path.join(result_path, model_name))

    with open(os.path.join(result_path, model_name, 'params'), 'wt') as fobj:
        for param in config.parameters:
            fobj.write(f'{param}:\t{config.parameters[param]}\n')

    if config.opt.dataset == 'conll':
        train_data, dev_data, test_data, test_train_data, mappings = loader.load_conll(dataset_path, config.parameters)
    elif config.opt.dataset == 'ontonotes':
        train_data, dev_data, test_data, mappings = loader.load_ontonotes(dataset_path, config.parameters)
        test_train_data = train_data[-10000:]
    else:
        raise NotImplementedError()

    word_to_id = mappings['word_to_id']
    tag_to_id = mappings['tag_to_id']
    char_to_id = mappings['char_to_id']
    word_embeds = mappings['word_embeds']

    print('Load Complete')

    if model_load:
        print('Loading Saved Weights....................................................................')
        model_path = os.path.join(result_path, model_name, config.opt.checkpoint, 'modelweights')
        model = torch.load(model_path)
    else:
        print('Building Model............................................................................')
        if (model_name == 'CNN_BiLSTM_CRF'):
            print('CNN_BiLSTM_CRF')
            word_vocab_size = len(word_to_id)
            word_embedding_dim = config.parameters.wrdim
            word_hidden_dim = config.parameters.wldim
            char_vocab_size = len(char_to_id)
            char_embedding_dim = config.parameters.chdim
            char_out_channels = config.parameters['cnchl']

            model = CNN_BiLSTM_CRF(word_vocab_size, word_embedding_dim, word_hidden_dim, char_vocab_size,
                                   char_embedding_dim, char_out_channels, tag_to_id, pretrained=word_embeds)

        elif (model_name == 'CNN_BiLSTM_CRF_MC'):
            print('CNN_BiLSTM_CRF_MC')
            word_vocab_size = len(word_to_id)
            word_embedding_dim = config.parameters['wrdim']
            word_hidden_dim = config.parameters['wldim']
            char_vocab_size = len(char_to_id)
            char_embedding_dim = config.parameters['chdim']
            char_out_channels = config.parameters['cnchl']

            model = CNN_BiLSTM_CRF_MC(word_vocab_size, word_embedding_dim, word_hidden_dim, char_vocab_size,
                                      char_embedding_dim, char_out_channels, tag_to_id, pretrained=word_embeds)

        elif (model_name == 'CNN_BiLSTM_CRF_BB'):
            print('CNN_BiLSTM_CRF_BB')
            word_vocab_size = len(word_to_id)
            word_embedding_dim = config.parameters['wrdim']
            word_hidden_dim = config.parameters['wldim']
            char_vocab_size = len(char_to_id)
            char_embedding_dim = config.parameters['chdim']
            char_out_channels = config.parameters['cnchl']
            sigma_prior = config.parameters['sigmp']

            model = CNN_BiLSTM_CRF_BB(word_vocab_size, word_embedding_dim, word_hidden_dim, char_vocab_size,
                                      char_embedding_dim, char_out_channels, tag_to_id, sigma_prior=sigma_prior,
                                      pretrained=word_embeds)

        elif (model_name == 'CNN_CNN_LSTM'):
            print('CNN_CNN_LSTM')
            word_vocab_size = len(word_to_id)
            word_embedding_dim = config.parameters['wrdim']
            word_out_channels = config.parameters['wdchl']
            char_vocab_size = len(char_to_id)
            char_embedding_dim = config.parameters['chdim']
            char_out_channels = config.parameters['cnchl']
            decoder_hidden_units = config.parameters['dchid']

            model = CNN_CNN_LSTM(word_vocab_size, word_embedding_dim, word_out_channels, char_vocab_size,
                                 char_embedding_dim, char_out_channels, decoder_hidden_units,
                                 tag_to_id, pretrained=word_embeds)

        elif (model_name == 'CNN_CNN_LSTM_MC'):
            print('CNN_CNN_LSTM_MC')
            word_vocab_size = len(word_to_id)
            word_embedding_dim = config.parameters['wrdim']
            word_out_channels = config.parameters['wdchl']
            char_vocab_size = len(char_to_id)
            char_embedding_dim = config.parameters['chdim']
            char_out_channels = config.parameters['cnchl']
            decoder_hidden_units = config.parameters['dchid']

            model = CNN_CNN_LSTM_MC(word_vocab_size, word_embedding_dim, word_out_channels, char_vocab_size,
                                    char_embedding_dim, char_out_channels, decoder_hidden_units,
                                    tag_to_id, pretrained=word_embeds)

        elif (model_name == 'CNN_CNN_LSTM_BB'):
            print('CNN_CNN_LSTM_BB')
            word_vocab_size = len(word_to_id)
            word_embedding_dim = config.parameters['wrdim']
            word_out_channels = config.parameters['wdchl']
            char_vocab_size = len(char_to_id)
            char_embedding_dim = config.parameters['chdim']
            char_out_channels = config.parameters['cnchl']
            decoder_hidden_units = config.parameters['dchid']
            sigma_prior = config.parameters['sigmp']

            model = CNN_CNN_LSTM_BB(word_vocab_size, word_embedding_dim, word_out_channels, char_vocab_size,
                                    char_embedding_dim, char_out_channels, decoder_hidden_units,
                                    tag_to_id, sigma_prior=sigma_prior, pretrained=word_embeds)

    model.cuda()
    learning_rate = config.parameters['lrate']
    print('Initial learning rate is: %s' % (learning_rate))
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    trainer = Trainer(model, optimizer, result_path, model_name, usedataset=config.opt.dataset, mappings=mappings)
    losses, all_F = trainer.train_model(config.opt.num_epochs, train_data, dev_data, test_train_data, test_data,
                                        learning_rate=learning_rate, batch_size=config.parameters['batch_size'],
                                        lr_decay=0.05)

    plt.plot(losses)
    plt.savefig(os.path.join(result_path, model_name, 'lossplot.png'))

if __name__ == "__main__":

    main()