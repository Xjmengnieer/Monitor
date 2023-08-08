import cv2
import os
import numpy as np

from PIL import Image

'''
function one

First calculate the mean, then calculate the standard deviation

eg: mean = mean_std()
    std = mean_std(mean, m_or_s ='std')
    
root_path is the Parent directory of the Parent directory of picture
    eg: /home/xjm/xjm/xjm_pycharm/explain/custom_data/train/
    
'''

root_path = '/home/data/monitor/init/train/'

R_mean = 0
G_mean = 0
B_mean = 0

def mean_std( mean = [0,0,0], m_or_s = 'mean', root = None):

    '''


    :param mean: the mean of dataset which has been computed
    :param m_or_s: compute mean or std
    :return: mean or std
    '''

    R_img_mean = []
    G_img_mean = []
    B_img_mean = []

    R_mean = mean[0]
    G_mean = mean[1]
    B_mean = mean[2]

    R_img_std = []
    G_img_std = []
    B_img_std = []

    '''
    dir is class directory
    '''
    for dir in os.listdir(root_path):
        if dir == 'cat':
            print(1)
        sub_dir = os.path.join(root_path,dir)
        for img_name in os.listdir(sub_dir):
            img_path = os.path.join(sub_dir, img_name)

            if img_name != 'Thumbs.db':
                # img = cv2.imread(img_path)
                img = Image.open(img_path).convert('RGB')
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.asarray(img)
                img = img / 255
            else:
                continue

            if m_or_s == 'mean':
                R_img_mean.append(np.mean(img[:, :, 0]))
                G_img_mean.append(np.mean(img[:, :, 1]))
                B_img_mean.append(np.mean(img[:, :, 2]))
            else:
                assert m_or_s == 'std',  'only for mean or std'
                R_img_std.append(np.mean((img - R_mean) ** 2))
                G_img_std.append(np.mean((img - G_mean) ** 2))
                B_img_std.append(np.mean((img - B_mean) ** 2))
    if m_or_s == 'mean':
        R_mean = np.mean(R_img_mean)
        G_mean = np.mean(G_img_mean)
        B_mean = np.mean(B_img_mean)

        print('mean :',[R_mean, G_mean, B_mean])
        return [R_mean, G_mean, B_mean]
    elif  m_or_s == 'std':
        R_std = np.mean(R_img_std)
        G_std = np.mean(G_img_std)
        B_std = np.mean(B_img_std)

        print('std: ', [R_std, G_std, B_std])
        return [R_std, G_std, B_std]


'''
use the function
'''
mean = mean_std(root = root_path)
std = mean_std(mean, m_or_s = 'std', root = root_path)



'''
function two:only for mean
    
    The standard deviation cannot be calculated in this way because it is not an average
    
'''
def only_for_mean(root_path):
    mean = []
    # std = []
    R_mean_list = []
    G_mean_list = []
    B_mean_list = []

    # R_std_list = []
    # G_std_list = []
    # B_std_list = []
    '''
    dir is class directory
    '''
    for dir in os.listdir(root_path):
        sub_dir = os.path.join(root_path, dir)
        for img_name in os.listdir(sub_dir):
            img_path = os.path.join(sub_dir, img_name)

            if img_name != 'Thumbs.db':
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255
            else:
                continue

            (R_img_mean, R_img_std) = cv2.meanStdDev(img[:, :, 0])
            (G_img_mean, G_img_std) = cv2.meanStdDev(img[:, :, 1])
            (B_img_mean, B_img_std) = cv2.meanStdDev(img[:, :, 2])

            R_mean_list.append(R_img_mean)
            G_mean_list.append(G_img_mean)
            B_mean_list.append(B_img_mean)

            # R_std_list.append(R_img_std)
            # G_std_list.append(G_img_std)
            # B_std_list.append(B_img_std)
    mean.append(np.mean(R_mean_list))
    mean.append(np.mean(G_mean_list))
    mean.append(np.mean(B_mean_list))
    '''
    The standard deviation cannot be calculated in this way because it is not an average
    '''
    # std.append(np.mean(R_std_list))
    # std.append(np.mean(G_std_list))
    # std.append(np.mean(B_std_list))

    print('mean:', mean)
    # print('std: ', std)
'''
use of only_for_mean
'''
# only_for_mean(root_path)