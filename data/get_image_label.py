import os
import sys
import os.path as osp

root = '/mnt/lustre/share_data'
lst_path = '/mnt/lustre/share_data/OCR/data/EN/train/train1.txt'

img_out_path = open('images.txt', 'w')
lb_out_path = open('labels.txt', 'w')

for line in open(lst_path).readlines():
    strs = line.strip().split()
    img_path = osp.join(root, strs[0])
    label = strs[1]
    img_out_path.write(img_path + '\n')
    lb_out_path.write(label + '\n')
