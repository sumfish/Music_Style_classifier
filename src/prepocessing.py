import numpy as np
import os 
import glob
import librosa
from utils import *

### settings
config = {
    # basic parameters
    'sr': 22050,
    'n_fft': 1024, # window size     #remy2048,512 #suli 2048,256
    'hop_length': 512,

    # mels
    'n_mels': 128,

    # for slicing and overlapping (now is no overlapping)
    'audio_samples_frame_size': 44100, # sec * sr
    'audio_samples_hop_length': 44100,

    # file/path
    # change path for test/training data
    ##### dataset
    #'dataset_path' : '../dataset/test',
    ##### experiment
    'dataset_path' : '../dataset/raw_song',
    'npy_path' : '../dataset/npy/raw',
    #'npy_path' : '../dataset/npy/test/data_1s_ver2',  ####different preprocessing
    'label_txt' : 'label.txt',
    'data_npy' : 'data.npy',
    'label_npy' : 'label.npy',
    'visualize path':'/img'
}
print(config)

def get_melspec(y, config):
    S = librosa.feature.melspectrogram(y, sr=config['sr'], n_fft=config['n_fft'], 
                                    hop_length=config['hop_length'], n_mels=config['n_mels'])
    S = np.log(1+10000*S) #filter
    #print(S.shape) #(128,T)
    return S

def audio2feature():
    data_list = []
    label_list = []
    draw_list = []
    # read file
    #files = glob.glob(config['dataset_path']+'/*/*.mp3')
    paths = glob.glob(config['dataset_path']+'/*')
    for p in paths:
        print(p)
        #input()
        '''
        #draw
        if(p.find('train\\04'))!=-1:
                break
        temp=[]
        '''
        files = glob.glob(p+'/*.mp3')
        for filename in files:
            print(filename)
            '''
            if(filename.find('04.mp3'))!=-1:
                break
            '''
            #mkdir(filename[:-4]) # dir for instrument of the song
            instru_class = filename[-6:-4] #class label
            y, sr = librosa.core.load(filename, offset=1.0, sr=config['sr']) # read after 1 seconds #mono=True(convert signal to mono)
            
            Len = y.shape[0]
            count = 0
            st_idx = 0 
            end_idx = st_idx + config['audio_samples_frame_size']
            next_idx = st_idx + config['audio_samples_hop_length']
            
            while  st_idx < Len:
                if end_idx > Len:
                    end_idx = Len ####last is too short?
                audio = np.zeros(config['audio_samples_frame_size'], dtype='float32')
                audio[:end_idx-st_idx] = y[st_idx:end_idx]

                ''' test silence clip'''

                feature = get_melspec(audio,config)
                data_list.append(feature)
                label_list.append(instru_class)

                '''
                # output audio
                outputname=os.path.join(filename[:-4],'cut{:03d}.wav'.format(count))
                print(outputname)
                librosa.output.write_wav(outputname,audio,config['sr'])
                '''
                '''
                #draw
                if(count==5):
                    temp.append(feature)
                '''
                count +=1 
                st_idx = next_idx
                end_idx = st_idx + config['audio_samples_frame_size']
                next_idx = st_idx + config['audio_samples_hop_length']
        '''
        #draw
        draw_list.append(temp) 
        '''
    # save
    data_name=os.path.join(config['npy_path'],config['data_npy'])
    label_name=os.path.join(config['npy_path'],config['label_npy'])
    np.save(data_name,data_list)
    np.save(label_name,label_list)
    
    np.save('draw.npy',draw_list)
    #list2txt(config['label_txt'],label)
    #print(print(data[52]))
    #print(len(label))
    return draw_list

### raw data for label no.8
def for_raw_test():
    data_list = []
    label_list = []
    # read file
    
    paths = glob.glob(config['dataset_path']+'/*')
    for instrument in paths:
        print(instrument[-2:])
        instru_class = instrument[-2:] #class label
        
        files = glob.glob(instrument+'/*.mp3')
        for file in files:
            print(file)
            #input()
            y, _ = librosa.core.load(file, offset=1.0, sr=config['sr']) # read after 1 seconds #mono=True(convert signal to mono)
          
            Len = y.shape[0]
            count = 0
            st_idx = 0 
            end_idx = st_idx + config['audio_samples_frame_size']
            next_idx = st_idx + config['audio_samples_hop_length']

            while  st_idx < Len:
                if end_idx > Len:
                    end_idx = Len ####last is too short?
                audio = np.zeros(config['audio_samples_frame_size'], dtype='float32')
                audio[:end_idx-st_idx] = y[st_idx:end_idx]

                feature = get_melspec(audio,config)
                data_list.append(feature)
                label_list.append(instru_class)

                '''
                # output audio
                outputname=os.path.join(filename[:-4],'cut{:03d}.wav'.format(count))
                print(outputname)
                librosa.output.write_wav(outputname,audio,config['sr'])
                '''
                
                count +=1 
                st_idx = next_idx
                end_idx = st_idx + config['audio_samples_frame_size']
                next_idx = st_idx + config['audio_samples_hop_length']

        # save
        #data_name=os.path.join(config['npy_path'],str(instrument[-2:]),config['data_npy'])
        #label_name=os.path.join(config['npy_path'],str(instrument[-2:]),config['label_npy'])
        data_path=os.path.join(config['npy_path'],str(instrument[-2:]))   
        mkdir(data_path)   
        data_name=os.path.join(data_path,config['data_npy'])
        label_name=os.path.join(data_path,config['label_npy'])
        print('producing... : {}\n'.format(data_name))
        np.save(data_name,data_list)
        np.save(label_name,label_list)

def main():
    
    mkdir(config['npy_path'])
    #audio2feature()
    for_raw_test()
    
    #### generate random cluster data
    #ran_sample_data2cluster('/train/data_1s')
    
    
    '''
    #draw
    compare=np.load('draw.npy')
    draw('compare.png',compare ,config)
    '''

if __name__ == '__main__':
    main()

