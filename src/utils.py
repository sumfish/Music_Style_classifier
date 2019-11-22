import os
import librosa
import librosa.display
import json
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

def get_spectrogram(data, config, win_len=None):
    ### get spectrogram according to the configuration and window_length
    ### we first calculate the power2-spectrum,
    ### and then get the Mel-spectrogram via the Mel-Filter banks
    stft_matrix = librosa.stft(data, n_fft=config['n_fft'], hop_length=config['hop_length'], win_length=win_len)
    mag_D = np.abs(stft_matrix)
    pwr = mag_D**2
    
    mel_basis = librosa.filters.mel(sr=config['sr'], n_fft=config['n_fft'], n_mels=config['n_mels'])
    mel_pwr = np.dot(mel_basis, pwr)
    chk_NaN(mel_pwr)
    # last, apply the gamma-power to approxiate Steven power law.
    return power_to_outputType(mel_pwr, config)