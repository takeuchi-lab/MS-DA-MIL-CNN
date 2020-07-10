import numpy as np
from PIL import Image, ImageStat, ImageDraw
import argparse
import os, re, shutil, sys, time
import numpy as np
import csv

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def str_to_float(l):
    return float(l)

'''
Visualization of attention weights normalized for each bag
result_file: output file of single_train
attention_dir: directory for saving normalized attention
'''
def norm_attention(result_file, attention_dir):
    makedir(attention_dir)
    slide_list = []
    with open(result_file, 'r') as f:
        reader = csv.reader(f)
        #header = next(reader)  # when header should be skipped
        for row in reader:
            if(len(row)==5):
                slideID = row[0].split('_')[0]
                summary_file = f'{attention_dir}/{slideID}_norm.csv'
                if not os.path.exists(summary_file):
                    f = open(summary_file, 'w')
                    f.close()
                    slide_list.append(slideID)
            elif('_' in row[0]):
                instance_name_list = row
            elif('_' not in row[0]):
                attention_list = list(map(str_to_float, row))
                max_value = max(attention_list)
                # normalize attention weights between 0 and 1
                norm_attention_list = []
                for attention in attention_list:
                    norm_attention_list.append(attention/max_value)
                # write normalized attention weights
                f = open(summary_file, 'a')
                f_writer = csv.writer(f, lineterminator='\n')
                for i in range(100):
                    f_writer.writerow([instance_name_list[i], norm_attention_list[i]])
                f.close()
    return slide_list

def red(x):
    if(x<0.352):
        return 0
    if(x>=0.352 and x<0.662):
        return 822.58*x - 289.55
    if(x>=0.662 and x<0.89):
        return 255
    if(x>=0.89):
        return -1159*x + 1286.5

def green(x):
    if(x<0.137):
        return 0
    if(x>=0.137 and x<0.376):
        return 1066.9*x - 146.16
    if(x>=0.376 and x<0.639):
        return 255
    if(x>=0.639 and x<0.908):
        return -944.4*x + 858.45
    if(x>=0.908):
        return 0

def blue(x):
    if(x<0.117):
        return 1089.7*x + 127.5
    if(x>=0.117 and x<0.341):
        return 255
    if(x>=0.341 and x<0.65):
        return -825.2*x + 536.39
    if(x>=0.65):
        return 0

def attention_to_rgb(attention):
    r = red(attention)
    g = green(attention)
    b = blue(attention)
    return (int(r), int(g), int(b))

def draw_heatmap(slideID, attention_dir, save_dir):
    b_size = 224 # size of image patch
    t_size = 16 # size of a block in thumbnail of WSI
    thumb = Image.open(f'../Lymphoma/thumb/{slideID}_thumb.tif')
    w, h = thumb.size
    w_num = w // t_size
    h_num = h // t_size
    draw = ImageDraw.Draw(thumb)
    makedir(save_dir)

    attention_file = f'{attention_dir}/{slideID}_norm.csv'
    with open(attention_file, 'r') as f:
        reader = csv.reader(f)
        #header = next(reader)  # when header should be skipped
        for row in reader:
            block = row[0].split('_')[1].split('.')[0] # patch ID
            if('r' in block):
                continue # remove rotated image patches
            w_pos = int(block) % w_num
            h_pos = int(block) //w_num
            draw = ImageDraw.Draw(thumb)
            color = attention_to_rgb(float(row[1]))
            for i in range(2):
                draw.rectangle((w_pos*t_size+i, h_pos*t_size+i, (w_pos+1)*t_size-i, (h_pos+1)*t_size-i), fill=color, outline=color)
    thumb.save(f'{save_dir}/{slideID}_map.tif')


if __name__ == "__main__":
    argvs = sys.argv
    result_file = argvs[1]
    attention_dir = argvs[2]
    save_dir = argvs[3]

    slide_list = norm_attention(result_file, attention_dir)

    for slide in slide_list:
        draw_heatmap(slide, attention_dir, save_dir)
