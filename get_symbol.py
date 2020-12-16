import os
import shutil

names=[]
labels=[]
with open('img_label_symbol.txt','r') as file:
    lines=file.readlines()
    for line in lines:
        words=line.split('\t')
        names.append(words[0])
        labels.append(words[-1].rstrip('\n'))

save_dir = './mnist/processed/symbol'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
for name in names:
    old_path = './data_png_/'+name
    new_path = save_dir+'/'+name
    shutil.copy(old_path,new_path)