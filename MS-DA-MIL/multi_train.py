# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
import torch.nn.functional as F
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
def train(model, device, loss_fn, optimizer, train_data):
    model.train() # train mode
    train_class_loss = 0.0
    correct_num = 0
    # shuffle bag
    random.shuffle(train_data)
    for data in train_data:
        # loading data
        input_tensor_x10, input_tensor_x20, class_label = utils.data_load_multi(data)
        # to device
        input_tensor_x10 = input_tensor_x10.to(device)
        input_tensor_x20 = input_tensor_x20.to(device)
        class_label = class_label.to(device)

        optimizer.zero_grad() # initialize gradient
        class_prob, class_hat, A = model(input_tensor_x10, input_tensor_x20)
        # culculate loss function
        class_loss = loss_fn(class_prob, class_label)
        train_class_loss += class_loss.item()

        class_loss.backward() # backpropagation
        optimizer.step() # renew parameters
        correct_num += eval_ans(class_hat, class_label)

    train_class_loss = train_class_loss / len(train_data)
    train_acc = correct_num / len(train_data)
    return train_class_loss, train_acc

def test(model, device, test_data, output_file):
    model.eval() # test mode
    for data in test_data:
        # loading data
        input_x10, input_x20, inst_name_x10, inst_name_x20, class_label = utils.test_data_load_multi(data)
        input_x10 = input_x10.to(device)
        input_x20 = input_x20.to(device)

        with torch.no_grad():
            class_prob, class_hat, A = model(input_x10, input_x20)

        instance_list = inst_name_x10 + inst_name_x20

        class_softmax = F.softmax(class_prob, dim=1).squeeze(0)
        class_softmax = class_softmax.tolist() # convert to list

        # output prediction results for bag and attention weights for each patch
        bag_id = data[0][0].split('/')[8]
        f = open(output_file, 'a')
        f_writer = csv.writer(f, lineterminator='\n')
        slideid_tlabel_plabel = [bag_id, int(class_label), class_hat] + class_softmax # [bag ID, ground truch, predicted label] + [y_prob[1], y_prob[2]]
        f_writer.writerow(slideid_tlabel_plabel)
        f_writer.writerow(instance_list) # write instance
        attention_weights = A.squeeze(0) # reduce 1st dim [1,100] -> [100]
        attention_weights_list = attention_weights.tolist()
        f_writer.writerow(attention_weights_list) # write attention weight for each instance
        f.close()



if __name__ == "__main__":
    ##################experimental setup#######################################
    train_slide = '123'
    test_slide = '5'
    EPOCHS = 5
    device = 'cuda:0'
    # model parameters for each scale
    x10_params = 'model_params/x10_train-123_DArate-0.0001.pth'
    x20_params = 'model_params/x10_train-123_DArate-0.0001.pth'
    ################################################################
    # split slides for training and validation
    import dataset as ds
    train_DLBCL, train_nonDLBCL, valid_DLBCL, valid_nonDLBCL = ds.slide_split(train_slide, test_slide)
    domain_num = len(train_DLBCL+train_nonDLBCL)
    # provide class and domain labels for training slides
    train_dataset = []
    for slideID in train_DLBCL:
        train_dataset.append([slideID, 1])
    for slideID in train_nonDLBCL:
        train_dataset.append([slideID, 0])

    # output files
    makedir('multi_train_log')
    log = f'multi_train_log/log_train-{train_slide}.csv'
    makedir('multi_test_result')
    test_result = f'multi_test_result/train-{train_slide}_test-{test_slide}.csv'
    makedir('multi_model_params')
    model_params = f'multi_model_params/train-{train_slide}.pth'
    f = open(log, 'w')
    f_writer = csv.writer(f, lineterminator='\n')
    csv_header = ["epoch", "class_loss", "train_acc"]
    f_writer.writerow(csv_header)
    f.close()

    torch.backends.cudnn.benchmark=True # cudnn benchmark mode

    # loading model
    from model import feature_extractor, class_predictor, domain_predictor, DAMIL, MSDAMIL
    # declare each block
    feature_extractor = feature_extractor()
    domain_predictor = domain_predictor(domain_num)
    class_predictor = class_predictor()
    # DAMIL
    DAMIL = DAMIL(feature_extractor, class_predictor, domain_predictor)
    # loading x10 parameters
    DAMIL.load_state_dict(torch.load(x10_params))
    # use only feature extractor
    feature_extractor_x10 = DAMIL.feature_extractor
    # loading x20 parameters
    DAMIL.load_state_dict(torch.load(x20_params))
    # use only feature extractor
    feature_extractor_x20 = DAMIL.feature_extractor
    # MSDAMIL
    model = MSDAMIL(feature_extractor_x10, feature_extractor_x20, class_predictor)
    model = model.to(device)

    # use cross entropy loss function
    loss_fn = nn.CrossEntropyLoss()
    # use SGDmomentum
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0)


    # start training
    for epoch in range(EPOCHS):
        # generate bag
        train_data = []
        for data in train_dataset:
            slideID = data[0]
            class_label = data[1]
            bag_list = utils.build_bag_multi(slideID, class_label, 100, 50)
            train_data = train_data + bag_list
        # training
        class_loss, acc = train(model, device, loss_fn, optimizer, train_data)
        # write log
        f = open(log, 'a')
        f_writer = csv.writer(f, lineterminator='\n')
        f_writer.writerow([epoch, class_loss, acc])
        f.close()
        # save model parameters for each epoch
        torch.save(model.state_dict(), model_params)

    # validation
    f = open(test_result, 'w')
    f.close()
    valid_data = utils.make_testdata_mult(valid_DLBCL, 1) + utils.make_testdata_mult(valid_nonDLBCL, 0)
    test(model, device, valid_data, test_result)
