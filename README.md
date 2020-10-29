# Active-NLP

Implementation of different acquisition functions for NER, classification and SRL task. (Machine Translation in progress). I ll soon add a proper readme file (this weekend). For now, You can peek into train_ner.py/active_ner.py and figure out arguments according to help in argparse.

For example to run a CNN_BiLSTM_CRF model on Conll dataset on full dataset, you can run

        $ HYDRA_CONFIG_PATH=./configs/cnn_bilstm_crf.yaml python train_ner.py

or in docker:

        $ nvidia-docker run --rm -ti  -e CUDA_VISIBLE_DEVICES=4  -e HYDRA_CONFIG_PATH=./configs/cnn_bilstm_crf.yaml -v `pwd`:/workspace/biomed_ie/src schokoro/al4ner python train_ner.py

and to run active learning for CNN_BiLSTM_CRF model on  Conll dataset with "MNLP" acquisition function, you can run

     $ python active_ner.py --usemodel CNN_BiLSTM_CRF --dataset conll --acquiremethod mnlp
