# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import csv
import os

# confirmation of correct or incorrect (correct: ans=1, incorrect: ans=0)
def eval_ans(y_hat, label):
    true_label = int(label)
    if(y_hat == true_label):
        ans = 1
    if(y_hat != true_label):
        ans = 0
    return ans

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

import random
import utils
def train(model, device, loss_fn, optimizer, train_data, DArate):
    model.train() # train mode
    train_class_loss = 0.0
    train_domain_loss = 0.0
    correct_num = 0
    # shuffle bag
    random.shuffle(train_data)
    for data in train_data:
        # loading data
        input_tensor, class_label, domain_label = utils.data_load(data)
        # to device
        input_tensor = input_tensor.to(device)
        class_label = class_label.to(device)
        domain_label = domain_label.to(device)

        optimizer.zero_grad() # initialize gradient
        class_prob, domain_prob, class_hat = model(input_tensor, 'train', DArate)
        # calculate loss
        class_loss = loss_fn(class_prob, class_label)
        domain_loss = loss_fn(domain_prob, domain_label)
        total_loss = class_loss + domain_loss

        train_class_loss += class_loss.item()
        train_domain_loss += domain_loss.item()

        total_loss.backward() # backpropagation
        optimizer.step() # renew parameters
        correct_num += eval_ans(class_hat, class_label)

    train_class_loss = train_class_loss / len(train_data)
    train_domain_loss = train_domain_loss / len(train_data)
    train_acc = correct_num / len(train_data)
    return train_class_loss, train_domain_loss, train_acc

def test(model, device, test_data, output_file):
    model.eval() # test mode
    for data in test_data:
        # load data
        input_tensor, class_label, instance_list = utils.test_data_load(data)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            class_prob, class_hat, A = model(input_tensor, 'test', 0)

        class_softmax = F.softmax(class_prob, dim=1).squeeze(0)
        class_softmax = class_softmax.tolist() # convert to list
        # write predicton results for bag and attention weights for each patches
        bag_id = data[0][0].split('/')[8]
        f = open(output_file, 'a')
        f_writer = csv.writer(f, lineterminator='\n')
        slideid_tlabel_plabel = [bag_id, int(class_label), class_hat] + class_softmax # [bag name, ground truth, predicted label] + [y_prob[1], y_prob[2]]
        f_writer.writerow(slideid_tlabel_plabel)
        f_writer.writerow(instance_list) # write instance
        attention_weights = A.squeeze(0) # reduce 1st dim [1,100] -> [100]
        attention_weights_list = attention_weights.tolist()
        f_writer.writerow(attention_weights_list) # write attention weights for each instance
        f.close()



if __name__ == "__main__":
    ##################experimental setup#######################################
    train_slide = '123'
    valid_slide = '4'
    mag = 'x10' # ('x10' or 'x20')
    EPOCHS = 5
    DArate = 0.001
    device = 'cuda:0'
    ################################################################
    # split slides for training and validation
    import dataset as ds
    train_DLBCL, train_nonDLBCL, valid_DLBCL, valid_nonDLBCL = ds.slide_split(train_slide, valid_slide)
    train_domain = train_DLBCL + train_nonDLBCL
    domain_num = len(train_domain)
    # provide class and domain labels for training data
    train_dataset = []
    for slideID in train_DLBCL:
        domain_idx = train_domain.index(slideID)
        train_dataset.append([slideID, 1, domain_idx])
    for slideID in train_nonDLBCL:
        domain_idx = train_domain.index(slideID)
        train_dataset.append([slideID, 0, domain_idx])

    # output files
    makedir('train_log')
    log = f'train_log/log_{mag}_train-{train_slide}_DArate-{DArate}.csv'
    makedir('valid_result')
    valid_result = f'valid_result/{mag}_train-{train_slide}_valid-{valid_slide}_DArate-{DArate}.csv'
    makedir('model_params')
    model_params = f'model_params/{mag}_train-{train_slide}_DArate-{DArate}.pth'
    f = open(log, 'w')
    f_writer = csv.writer(f, lineterminator='\n')
    csv_header = ["epoch", "class_loss", "domain_loss", "train_acc"]
    f_writer.writerow(csv_header)
    f.close()

    torch.backends.cudnn.benchmark=True #cudnn benchmark mode

    # load model
    from model import feature_extractor, class_predictor, domain_predictor, DAMIL
    # declare each block
    feature_extractor = feature_extractor()
    class_predictor = class_predictor()
    domain_predictor = domain_predictor(domain_num)
    # DAMIL
    model = DAMIL(feature_extractor, class_predictor, domain_predictor)
    model = model.to(device)

    # use cross entropy loss function
    loss_fn = nn.CrossEntropyLoss()
    # use SGDmomentum for optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0)


    # start training
    for epoch in range(EPOCHS):
        # generate bags
        train_data = []
        for data in train_dataset:
            slideID = data[0]
            class_label = data[1]
            domain_label = data[2]
            bag_list = utils.build_bag_single(slideID, class_label, domain_label, mag, 100, 50)
            train_data = train_data + bag_list
        # calculate domain regularization parameter
        p = ((epoch+1) / (EPOCHS)) * DArate
        lamda = (2 / (1 + np.exp(-10*p))) - 1
        # training
        class_loss, domain_loss, acc = train(model, device, loss_fn, optimizer, train_data, lamda)
        # write log
        f = open(log, 'a')
        f_writer = csv.writer(f, lineterminator='\n')
        f_writer.writerow([epoch, class_loss, domain_loss, acc])
        f.close()
        # save parameters
        torch.save(model.state_dict(), model_params)

    # validation
    f = open(valid_result, 'w')
    f.close()
    valid_data = utils.make_testdata(valid_DLBCL, 1, mag) + utils.make_testdata(valid_nonDLBCL, 0, mag)
    test(model, device, valid_data, valid_result)
