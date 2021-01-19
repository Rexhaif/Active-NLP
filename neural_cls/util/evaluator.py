import os
import codecs
import torch
from .utils import *
import torch
from torch.autograd import Variable
from pdb import set_trace
from sklearn.metrics import f1_score

class Evaluator(object):
    def __init__(self, result_path, model_name, usecuda=True):
        self.result_path = result_path
        self.model_name = model_name
        self.usecuda = usecuda

    def evaluate(self, model, dataset, best_F, checkpoint_folder='.', batch_size = 32):
        predicted_ids = []
        ground_truth_ids = []
        
        save = False
        new_F = 0.0
        
        data_batches = create_batches(dataset, batch_size = batch_size)

        for data in data_batches:

            words = data['words']

            if self.usecuda:
                words = Variable(torch.LongTensor(words)).cuda()
            else:
                words = Variable(torch.LongTensor(words))

            wordslen = data['wordslen']
            
            _, out = model.predict(words, wordslen, usecuda = self.usecuda)         
            
            ground_truth_ids.extend(data['tags'])
            predicted_ids.extend(out)
#         set_trace()

        new_F = np.mean(np.array(ground_truth_ids) == np.array(predicted_ids))
        f1_weighted = f1_score(np.array(ground_truth_ids), np.array(predicted_ids),average = 'weighted' )
        f1_macro = f1_score(np.array(ground_truth_ids), np.array(predicted_ids),average = 'macro' )
        
        if new_F > best_F:
            best_F = new_F
            save = True
        
        print('*'*100)
        print(f'Accuracy: {new_F:.8f}, Best Accuracy: {best_F:.8f}, F1 weighted: {f1_weighted:.8f}, F1 macro: {f1_macro:.8f}')
        
        print('*'*100)
            
        return best_F, new_F, save
