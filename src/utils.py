import os
import numpy as np
import librosa
import librosa.display
import json
import random
import matplotlib.pyplot as plt

def list2txt(output_file,out_list):
    with open(output_file,'w') as o:
        for label in out_list:
            o.write(label+'\n')

def mkdir(directory):
    if not os.path.exists(directory):
        print('making dir:{0}'.format(directory))
        os.makedirs(directory)
    else:
        print('already exist: {0}'.format(directory))

def draw(img_name, spec_list, config):
    song =len(spec_list)
    instru = len(spec_list[0])
    print(song, instru)
    #hei, wid = spec_list[0].shape
    #plt.figure(1, figsize=(7*(wid/302), 7/302*256)) # the resolution of y-axis goes up with hei
    plt.figure(1,figsize=(7*4,4*4))
    plt.clf()
    for i in range(song):
        for j in range(instru):
            # plot spectrogram
            plt.subplot(song, instru, (i*8)+j+1)
            #librosa.display.specshow(spec_list[i][j], y_axis='mel', x_axis='time', hop_length=config['hop_length'])
            librosa.display.specshow(spec_list[i][j], hop_length=config['hop_length'])
            plt.title('instrument '+ str(j))
    #plt.colorbar()
        
    plt.savefig(img_name, dpi='figure', bbox_inches='tight')
    plt.clf()

def load_npy(data_path):
    data=np.load(data_path)
    return data

def read_data_minmax(data_path):
    data=load_npy('../dataset/npy/train/'+data_path+'/data.npy')
    print(max(np.ravel(data)))
    print(min(np.ravel(data)))

def compute_mean_std(data_path):  ####train dataset
    data=load_npy('../dataset/npy/train/'+data_path+'/data.npy')
    pixels=np.ravel(data)
    print('Data Mean:{}, Variance:{}'.format(np.mean(pixels),np.std(pixels)))
    return np.mean(pixels), np.std(pixels)

## random sample 1000 data from npy.py/label.py
def ran_sample_data2cluster(data_path):
    number=2000
    out_data_list=[]
    out_label_list=[]
    index_list=[]
    data=load_npy('../dataset/npy'+data_path+'/data.npy')
    label=load_npy('../dataset/npy'+data_path+'/label.npy')
    #print(label)
    #print(len(label))
    for i in range(number):
        r=random.randint(0,len(label)-1)
        while index_list.count(r)!=0:
            r=random.randint(0,len(label)-1)
        index_list.append(r)
        out_label_list.append(label[r])
        out_data_list.append(data[r])
    #print(len(out_data_list))
    mkdir('../dataset/cluster_npy'+data_path)
    print('create cluster data.....')
    np.save('../dataset/cluster_npy'+data_path+'/data.npy',out_data_list)
    np.save('../dataset/cluster_npy'+data_path+'/label.npy',out_label_list)


