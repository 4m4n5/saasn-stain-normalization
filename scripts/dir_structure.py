import os, random
import os
from random import randint
import shutil
import random

# +
# Change this to folder which contains train and valid
src_path = '/project/DSone/as3ek/data/patches/500/unnorm/'
target_path = '/project/DSone/as3ek/data/ganstain/500_one_one/'
dir_names = ['trainA', 'trainB', 'testA', 'testB']
source_target_map = {'EE': 'A', 'Normal': 'B', 'train':'train', 'valid':'test'}
two_domains_in_ee = False

os.mkdir(target_path)


# -

def ee_check(file, src, folder, two_domains_in_ee):
    if two_domains_in_ee:
        return True
    if src != 'EE':
        return True
    else:
        if len(file.split('___')[0]) != 2:
            return False
        else:
            return True


for folder in os.listdir(src_path):
    for src in os.listdir(src_path+folder):
        if src not in source_target_map.keys():
            continue
        if not os.path.exists(target_path + source_target_map[folder] + source_target_map[src]):
            os.mkdir(target_path + source_target_map[folder] + source_target_map[src])
        for file in os.listdir(src_path+folder + '/' + src):
            if ee_check(file, src, folder, two_domains_in_ee):
                shutil.copy(src_path + folder + '/' + src + '/' + file, 
                            target_path + source_target_map[folder] + source_target_map[src] + '/' + file)

len('34___2500_5750'.split('___')[0])

