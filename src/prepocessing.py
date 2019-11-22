import numpy as np
import os 
import glob
import librosa
from utils import mkdir, list2txt, draw

### settings
config = {
    # basic parameters
    'sr': 22050,
    'n_fft': 1024, # window size
    'hop_length': 512,
    
    # mels
    'n_mels': 128,

    # for slicing and overlapping (now is no overlapping)
    'audio_samples_frame_size': 66150, # 3sec * sr
    'audio_samples_hop_length': 66150,

    # file/path
    'dataset_path' : '../dataset/train',
    'label_txt' : 'label.txt',
    'data_npy' : 'data.npy',
    'label_npy' : 'label.npy',
    'avg_npy' : 'avg.npy',
    'std_npy' : 'std.npy'
}
print(config)

def get_melspec(y, config):
    S = librosa.feature.melspectrogram(y, sr=config['sr'], n_fft=config['n_fft'], 
                                    hop_length=config['hop_length'], n_mels=config['n_mels'])
    return S

def gen_normalize_file(audio_list):
    avg = np.mean(np.array([np.mean(x, axis=1) for x in audio_list]), axis=0)
    std = np.mean(np.array([np.std(x, axis=1) for x in audio_list]), axis=0)
    avg_file=os.path.join(config['dataset_path'],config['avg_npy'])
    std_file=os.path.join(config['dataset_path'],config['std_npy'])
    np.save(avg_file, avg)
    np.save(std_file, std)

def audio2feature():
    data_list = []
    label_list = []
    draw_list = []
    # read file
    #files = glob.glob(config['dataset_path']+'/*/*.mp3')
    paths = glob.glob(config['dataset_path']+'/*')
    for p in paths:
        print(p)
        ''' 
        #draw
        if(p.find('train\\04'))!=-1:
                break
        temp=[]
        '''
        files = glob.glob(p+'/*.mp3')
        for filename in files:
            print(filename)
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
        # draw_list.append(temp) #draw

    # save
    data_name=os.path.join(config['dataset_path'],config['data_npy'])
    label_name=os.path.join(config['dataset_path'],config['label_npy'])
    np.save(data_name,data_list)
    np.save(label_name,label_list)

    # std & avg data
    print('Generate std/avg data....')
    gen_normalize_file(data_list)
    
    np.save('draw.npy',draw_list)
    #list2txt(config['label_txt'],label)
    #print(print(data[52]))
    #print(len(label))
    return draw_list

def main():
    audio2feature()
    '''
    #draw
    compare=np.load('draw.npy')
    draw('compare.png',compare ,config)
    '''

if __name__ == '__main__':
    main()