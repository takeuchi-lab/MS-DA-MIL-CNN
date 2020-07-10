import cv2
import torch
import os
import numpy as np
import random


# make lists for patch ID of all slides
data_dir = '../Lymphoma/patches/x10' # or /x20
patch_map = {} # key:slideID
for slideID in os.listdir(data_dir): # make patch ID lists for all slides
    patch_map[slideID] = []
    for patch in os.listdir(f'{data_dir}/{slideID}'):
        patch_i = patch.split('_')[1].split('.')[0] # get patch ID from image filename
        patch_map[slideID].append(patch_i)

'''
bag-generating function for single-scale
class_label: {0, 1}
mag: magnification ('x10' or 'x20')
inst_num: The number of patches for a bag (basically 100)
max_bag: The maximum number of bags for a slide (basically 50)
'''
def build_bag_single(slideID, class_label, domain_label, mag, inst_num, max_bag):
    patch_dir = f'../Lymphoma/patches/{mag}' # directory for a selected magnification
    patch_i_list = patch_map[slideID] # list of patch ID for a slide
    patch_num = len(patch_i_list) # the number of patches for a slide
    bag_num = int(patch_num/inst_num) # the possible number of bags
    if bag_num < max_bag:
        max_bag = bag_num
    random.shuffle(patch_i_list) # shuffle patch ID
    bag_list = []
    for i in range(max_bag):
        bag = []
        start = i*inst_num
        end = start + inst_num
        for j in range(start, end):
            patch_i = patch_i_list[j]
            patch_path = f'{patch_dir}/{slideID}/{slideID}_{patch_i}.tif'
            bag.append(patch_path)
        bag_list.append([bag, class_label, domain_label])
    return bag_list

'''
data loading function for training bag
data = [patch_list, class_label, domain_label]
'''
def data_load(data):
    # read image -> add to list as tensor
    tensor_list = [torch.Tensor(cv2.imread(patch_path).transpose((2, 1, 0))) for patch_path in data[0]]
    # generate bag tensor by stacking all tensors in a bag (size = [# instance,3,224,224])
    input_tensor = torch.stack(tensor_list, dim=0)
    # generate class label
    class_label = torch.tensor([data[1]], dtype=torch.int64)
    # generate domain label
    inst_num = len(data[0])
    domain_label = np.full(inst_num, data[2])
    domain_label = torch.from_numpy(domain_label)
    return input_tensor, class_label, domain_label

'''
bag-generating function for testing (vaidating) set
class_label: {0, 1}
mag: magnification ('x5' or 'x10' or 'x20')
'''
def make_testdata(slideID_list, class_label, mag):
    test_data_dir = f'../Lymphoma/patches/test_bags/{mag}'
    bag_list = []
    for slideID in slideID_list:
        for bag in os.listdir(f'{test_data_dir}/{slideID}'):
            patch_list = []
            for patch in os.listdir(f'{test_data_dir}/{slideID}/{bag}'):
                patch_path = f'{test_data_dir}/{slideID}/{bag}/{patch}'
                patch_list.append(patch_path)
            bag_list.append([patch_list, class_label])
    return bag_list

'''
data loading fuction for testing bag
return tensor, class_label, and instance ID for testing bag
data = [patch_list, label]
'''
def test_data_load(data):
    tensor_list = []
    instance_list = [] # list for Instance ID in a bag
    for image_path in data[0]:
        patch_name = image_path.split('/')[9]
        instance_list.append(patch_name)
        # loading image
        img = cv2.imread(image_path).transpose((2, 1, 0)) #  shape=(224,224,3)
        img_tensor = torch.Tensor(img)
        tensor_list.append(img_tensor)
    # generate tensor by stacking (size = [# instance,3,224,224])
    input_tensor = torch.stack(tensor_list, dim=0)
    class_label = torch.tensor([data[1]], dtype=torch.int64)
    return input_tensor, class_label, instance_list

# bag-generating function for multi_scale
def build_bag_multi(slideID, class_label, inst_num, max_bag):
    patch_dir_x10 = f'../Lymphoma/patches/x10' # directories for available scales
    patch_dir_x20 = f'../Lymphoma/patches/x20'
    patch_i_list = patch_map[slideID] # list of patch ID for a slide
    patch_num = len(patch_i_list) # the number of patches for a bag
    bag_num = int(patch_num/inst_num) # the possible number of bag
    if bag_num < max_bag:
        max_bag = bag_num
    random.shuffle(patch_i_list) # shuffle patch ID
    bag_list = []
    for i in range(max_bag):
        bag_x10 = []
        bag_x20 = []
        start = i*inst_num
        end = start + inst_num
        for j in range(start, end):
            patch_i = patch_i_list[j]
            bag_x10.append(f'{patch_dir_x10}/{slideID}/{slideID}_{patch_i}.tif')
            bag_x20.append(f'{patch_dir_x20}/{slideID}/{slideID}_{patch_i}.tif')
        bag_list.append([bag_x10, bag_x20, class_label])
    return bag_list

'''
data-loading function for multi-scale training
data = [bag_x10, bag_x20, class_label]
'''
def data_load_multi(data):
    tensor_list_x10 = [torch.Tensor(cv2.imread(img_path).transpose((2, 1, 0))) for img_path in data[0]]
    tensor_list_x20 = [torch.Tensor(cv2.imread(img_path).transpose((2, 1, 0))) for img_path in data[1]]
    # generate bags for each magnification [# instance,3,224,224]
    input_tensor_x10 = torch.stack(tensor_list_x10, dim=0)
    input_tensor_x20 = torch.stack(tensor_list_x20, dim=0)
    class_label = torch.tensor([data[2]], dtype=torch.int64)
    return input_tensor_x10, input_tensor_x20, class_label

# bag-generating function for testing (validation)
def make_testdata_mult(slideID_list, label):
    test_dir = f'../Lymphoma/patches/test_bags'
    bag_list = []
    for slideID in slideID_list:
        for bag in os.listdir(f'{test_dir}/x5/{slideID}'):
            bag_x10 = []
            bag_x20 = []
            for patch in os.listdir(f'{test_dir}/x10/{slideID}/{bag}'):
                bag_x10.append(f'{test_dir}/x10/{slideID}/{bag}/{patch}')
            for patch in os.listdir(f'{test_dir}/x20/{slideID}/{bag}'):
                bag_x20.append(f'{test_dir}/x20/{slideID}/{bag}/{patch}')
            bag_list.append([bag_x10, bag_x20, label])
    return bag_list


# data loading function for multi-scale testing
# data = [bag_x10, bag_x20, class_label]
def test_data_load_multi(data):
    # tensors for each scale
    tensor_list_x10 = []
    tensor_list_x20 = []
    # patch ID in a bag
    inst_name_list_x10 = []
    inst_name_list_x20 = []
    # generate bag for x10 (if you need, add other scales)
    for image_path in data[0]:
        patch_name = image_path.split('/')[9] # get image file name
        inst_name_list_x10.append(f'x10_{patch_name}')
        img = cv2.imread(image_path).transpose((2, 1, 0)) # read image shape=(224,224,3)
        img_tensor = torch.Tensor(img)
        tensor_list_x10.append(img_tensor)
    # generate bag for x20
    for image_path in data[1]:
        patch_name = image_path.split('/')[9] # get image file name
        inst_name_list_x20.append(f'x20_{patch_name}')
        img = cv2.imread(image_path).transpose((2, 1, 0)) # read image shape=(224,224,3)
        img_tensor = torch.Tensor(img)
        tensor_list_x20.append(img_tensor)
    input_tensor_x10 = torch.stack(tensor_list_x10, dim=0) # [# instance,3,224,224]
    input_tensor_x20 = torch.stack(tensor_list_x20, dim=0) # [# instance,3,224,224]
    class_label = torch.tensor([data[2]], dtype=torch.int64)
    return input_tensor_x10, input_tensor_x20, inst_name_list_x10, inst_name_list_x20, class_label
