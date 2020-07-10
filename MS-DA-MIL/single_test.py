# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import csv
import utils
import os



def test(model, device, test_data, output_file):
    model.eval() # test mode
    for data in test_data:
        # loading data
        input_tensor, class_label, instance_list = utils.test_data_load(data)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            class_prob, class_hat, A = model(input_tensor, 'test', 0)

        class_softmax = F.softmax(class_prob, dim=1).squeeze(0)
        class_softmax = class_softmax.tolist() # convert to list

        # write prediction results for bag and attention weights for patches
        bag_id = data[0][0].split('/')[8]
        f = open(output_file, 'a')
        f_writer = csv.writer(f, lineterminator='\n')
        slideid_tlabel_plabel = [bag_id, int(class_label), class_hat] + class_softmax # [Bag name, ground truth, predicted label] + [y_prob[1], y_prob[2]]
        f_writer.writerow(slideid_tlabel_plabel)
        f_writer.writerow(instance_list) # write instance
        attention_weights = A.squeeze(0) # reduce 1st dim[1,100] -> [100]
        attention_weights_list = attention_weights.tolist()
        f_writer.writerow(attention_weights_list) # write attention weights for each instance
        f.close()

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == "__main__":
    import sys
    argvs = sys.argv
    mag = argvs[1]
    train_slide = argvs[2]
    test_slide = argvs[3]
    DArate = float(argvs[4])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_params = f'model_params/{mag}_train-{train_slide}_DArate-{DArate}.pth'

    import dataset as ds
    train_DLBCL, train_nonDLBCL, test_DLBCL, test_nonDLBCL = ds.slide_split(train_slide, test_slide)
    train_domain = train_DLBCL + train_nonDLBCL
    domain_num = len(train_domain)

    # load model
    from model import feature_extractor, class_predictor, domain_predictor, DAMIL
    # declare each block
    feature_extractor = feature_extractor()
    class_predictor = class_predictor()
    domain_predictor = domain_predictor(domain_num)
    # DAMIL
    model = DAMIL(feature_extractor, class_predictor, domain_predictor)
    # load model parameters
    model.load_state_dict(torch.load(model_params))
    model = model.to(device)

    # output files
    makedir('test_result')
    test_result = f'test_result/{mag}_train-{train_slide}_test-{test_slide}_DArate-{DArate}.csv'
    # validation
    f = open(test_result, 'w')
    f.close()
    test_data = utils.make_testdata(test_DLBCL, 1, mag) + utils.make_testdata(test_nonDLBCL, 0, mag)
    test(model, device, test_data, test_result)
