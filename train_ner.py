from __future__ import print_function
import os
import datetime
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
import logging
import warnings

warnings.filterwarnings('ignore')
log = logging.getLogger(__name__)

def make_model(config, mappings, result_path):
    word_to_id = mappings['word_to_id']
    tag_to_id = mappings['tag_to_id']
    char_to_id = mappings['char_to_id']
    word_embeds = mappings['word_embeds']
    if config.opt.reload:
        log.info('Loading Saved Weights....................................................................')
        model_path = os.path.join(result_path, config.opt.usemodel, config.opt.checkpoint, 'modelweights')
        model = torch.load(model_path)
    else:
        log.info('Building Model............................................................................')
        log.info(config.opt.usemodel)
        word_vocab_size = len(word_to_id)
        char_vocab_size = len(char_to_id)
        word_embedding_dim = config.parameters.wrdim
        char_out_channels = config.parameters.cnchl
        char_embedding_dim = config.parameters.chdim

        if (config.opt.usemodel == 'CNN_BiLSTM_CRF'):
            word_hidden_dim = config.parameters.wldim
            model = CNN_BiLSTM_CRF(word_vocab_size, word_embedding_dim, word_hidden_dim, char_vocab_size,
                                   char_embedding_dim, char_out_channels, tag_to_id, pretrained=word_embeds)

        elif (config.opt.usemodel == 'CNN_BiLSTM_CRF_MC'):
            word_hidden_dim = config.parameters['wldim']
            model = CNN_BiLSTM_CRF_MC(word_vocab_size, word_embedding_dim, word_hidden_dim, char_vocab_size,
                                      char_embedding_dim, char_out_channels, tag_to_id, pretrained=word_embeds)

        elif (config.opt.usemodel == 'CNN_BiLSTM_CRF_BB'):
            word_hidden_dim = config.parameters['wldim']
            sigma_prior = config.parameters['sigmp']
            model = CNN_BiLSTM_CRF_BB(word_vocab_size, word_embedding_dim, word_hidden_dim, char_vocab_size,
                                      char_embedding_dim, char_out_channels, tag_to_id, sigma_prior=sigma_prior,
                                      pretrained=word_embeds)

        elif (config.opt.usemodel == 'CNN_CNN_LSTM'):
            word_out_channels = config.parameters['wdchl']
            decoder_hidden_units = config.parameters['dchid']
            model = CNN_CNN_LSTM(word_vocab_size, word_embedding_dim, word_out_channels, char_vocab_size,
                                 char_embedding_dim, char_out_channels, decoder_hidden_units,
                                 tag_to_id, pretrained=word_embeds)

        elif (config.opt.usemodel == 'CNN_CNN_LSTM_MC'):
            word_out_channels = config.parameters['wdchl']
            decoder_hidden_units = config.parameters['dchid']
            model = CNN_CNN_LSTM_MC(word_vocab_size, word_embedding_dim, word_out_channels, char_vocab_size,
                                    char_embedding_dim, char_out_channels, decoder_hidden_units,
                                    tag_to_id, pretrained=word_embeds)

        elif (config.opt.usemodel == 'CNN_CNN_LSTM_BB'):
            word_out_channels = config.parameters['wdchl']
            decoder_hidden_units = config.parameters['dchid']
            sigma_prior = config.parameters['sigmp']
            model = CNN_CNN_LSTM_BB(word_vocab_size, word_embedding_dim, word_out_channels, char_vocab_size,
                                    char_embedding_dim, char_out_channels, decoder_hidden_units,
                                    tag_to_id, sigma_prior=sigma_prior, pretrained=word_embeds)
        else:
            raise KeyError
    return model

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
    # model_name = config.opt.usemodel
    # model_load = config.opt.reload
    loader = Loader()

    log.info(f'Model: {config.opt.usemodel}')
    log.info(f'Dataset: {use_dataset}')

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if not os.path.exists(os.path.join(result_path, config.opt.usemodel)):
        os.makedirs(os.path.join(result_path, config.opt.usemodel))

    with open(os.path.join(result_path, config.opt.usemodel, 'params'), 'wt') as fobj:
        for param in config.parameters:
            fobj.write(f'{param}:\t{config.parameters[param]}\n')

    if use_dataset == 'conll':
        train_data, dev_data, test_data, test_train_data, mappings = loader.load_conll(dataset_path, config.parameters)
    elif use_dataset == 'ontonotes':
        train_data, dev_data, test_data, mappings = loader.load_ontonotes(dataset_path, config.parameters)
        test_train_data = train_data[-10000:]
    else:
        raise KeyError('unknown dataset name')

    log.info('Load Complete')

    model = make_model(config, mappings, result_path)

    model.cuda()
    learning_rate = config.parameters['lrate']
    log.info(f'Initial learning rate is: {learning_rate:.6f}')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    trainer = Trainer(model, optimizer, result_path, config.opt.usemodel, usedataset=config.opt.dataset, mappings=mappings)
    losses, all_F = trainer.train_model(config.opt.num_epochs, train_data, dev_data, test_train_data, test_data,
                                        learning_rate=learning_rate, batch_size=config.parameters['batch_size'],
                                        lr_decay=config.opt.lr_decay)

    plt.plot(losses)
    plt.savefig(os.path.join(result_path, config.opt.usemodel, 'lossplot.png'))

if __name__ == "__main__":
    main()