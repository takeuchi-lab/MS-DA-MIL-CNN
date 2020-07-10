# -*- coding: utf-8 -*-

import csv
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sys

# return a list of probability and label for each slide
def get_slide_prob_label(csv_file):
    pred_corpus = {}
    label_corpus = {}
    slide_id_list = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if(len(row)==5):
                slide_id = row[0].split('_')[0]
                prob_list = [float(row[3]), float(row[4])] # [probability of non-DLBCL, probability of DLBCL]
                if(slide_id not in pred_corpus):
                    pred_corpus[slide_id] = []
                    label_corpus[slide_id] = int(row[1])

                pred_corpus[slide_id].append(prob_list)
                if(slide_id not in slide_id_list):
                    slide_id_list.append(slide_id)
    # calculation of posterior probability for each slide
    slide_prob = []
    true_label_list = []
    pred_label_list = []

    for slide_id in slide_id_list:
        prob_list= pred_corpus[slide_id]
        bag_num = len(prob_list) # number of bag

        add_DLBCL = 0.0
        add_non_DLBCL = 0.0
        for prob in prob_list:
            add_non_DLBCL = add_non_DLBCL + np.log(float(prob[0]))
            add_DLBCL = add_DLBCL + np.log(float(prob[1]))
        DLBCL_prob = np.exp(add_DLBCL / bag_num)
        non_DLBCL_prob = np.exp(add_non_DLBCL / bag_num)
        slide_prob.append([non_DLBCL_prob, DLBCL_prob])
        true_label_list.append(label_corpus[slide_id])

        # decision of predicted label
        if(DLBCL_prob > non_DLBCL_prob):
            pred_label_list.append(1)
        else:
            pred_label_list.append(0)
    return slide_id_list, slide_prob, true_label_list, pred_label_list

def cal_recall(true_label_list, pred_label_list):
    slide_num = len(true_label_list)
    correct = 0
    DLBCL_num = 0
    n_correct = 0
    n_DLBCL_num = 0
    for i in range(slide_num):
        if(true_label_list[i]==1):
            DLBCL_num += 1
            if(pred_label_list[i]==1):
                correct += 1
        if(true_label_list[i]==0):
            n_DLBCL_num += 1
            if(pred_label_list[i]==0):
                n_correct += 1
    return n_correct/n_DLBCL_num, correct/DLBCL_num

def cal_precision(true_label_list, pred_label_list):
    slide_num = len(true_label_list)
    correct = 0
    DLBCL_num = 0
    n_correct = 0
    n_DLBCL_num = 0
    for i in range(slide_num):
        if(pred_label_list[i]==1):
            DLBCL_num += 1
            if(true_label_list[i]==1):
                correct += 1
        if(pred_label_list[i]==0):
            n_DLBCL_num += 1
            if(true_label_list[i]==0):
                n_correct += 1
    return n_correct/n_DLBCL_num, correct/DLBCL_num

def cal_acc(true_label_list, pred_label_list):
    cor_num = 0
    slide_num = len(true_label_list)
    for i in range(slide_num):
        if(true_label_list[i]==pred_label_list[i]):
            cor_num += 1
    return cor_num / slide_num


def eval(csv_file):
    slide_id_list, slide_prob, true_label_list, pred_label_list = get_slide_prob_label(csv_file)
    #print(slide_id_list)
    print(pred_label_list)
    print('recall', cal_recall(true_label_list, pred_label_list))
    print('precision', cal_precision(true_label_list, pred_label_list))
    print('acc', cal_acc(true_label_list, pred_label_list))


if __name__ == "__main__":
    argvs = sys.argv
    result_file = argvs[1]

    eval(result_file)
