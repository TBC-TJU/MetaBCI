from scipy.fftpack import fft
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import mne 

class FrequencyAnalysis():
    def __init__(self,data,meta,event,srate,latency=0,channel = 'all'):
        '''
       -author: Zhou hongzhan
       -Create on:2022-8-9
       -update log:
        2022-8-11 by Zhou hongzhan
        
        Args:
            1.data:EEG data (nTrials, nChannels, nTimes)
                A matrix fullfilled with timepoint voltage
            2.meta:DataFrame
                Concrete message of data,including subject ID,the events correspond to  specific trials,etc.
            3.event:String
                Events needed to be extracted
            4.srate:Int
                The sample rate of data
            5.latency:Float
                The start timepoint of experiment (latency=0 indicate that the data recording begin with the stimuli performs),default value=0
            6.channel:String
                The wanted channel.if 'all',all channel will be extracted.default value = 'all'

        '''


        sub_meta = meta[meta['event']==event]
        event_id = sub_meta.index.to_numpy()
        self.data_length=np.round(data.shape[2]/srate)
        if channel=='all':
            self.data=data[event_id,:,:]
        else:
            self.data=data[event_id,channel,:]
        self.latency=latency
        self.fs=srate
    
    def stacking_average(self, data = [], _axis = 0):
        '''
        -author: Zhou hongzhan
        -Create on:2022-8-9
        -update log:
            2022-8-11 by Zhou hongzhan

        Args:
            data : np.array (nTrials, nChannels, nTimes)
                 EEG origin data. The default is [].
            _axis : int
                 The dimension need to be stacked. The default is 0.

        Returns:
            data_mean : np.array
                 The data after stacked.

        '''
        if data ==[]:
            data = self.data
        data_mean = data
        data_mean = np.mean(data, axis=_axis)
        return data_mean

    def power_spectrum_periodogram(self,x):
        '''
        -author: Zhou hongzhan & He Jiatong
        -Create on:2022-8-9
        -update log:
            2022-8-31 by Zhou hongzhan
         
        Args:
            x : np.array
               1D data.

        Returns:
            f : np.array
               An array of frequencies
            Pxx_den : np.array
               The amplitude array respectively correspond to frequency array

        '''
        f, Pxx_den = signal.periodogram(x, self.fs,window='boxcar',scaling='spectrum')
        plt.plot(f, Pxx_den)
        plt.title('Power Spectral Density')
        plt.xlim([0, 60])
        plt.ylim([0, 0.5])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2]')
        plt.show()
        return f,Pxx_den

    def sum_y(self,x,y,x_inf,x_sup):
        '''
        -author: Zhou hongzhan
        -Create on:2022-8-9
        -update log:
            2022-8-11 by Zhou hongzhan

        Args:
            x : np.array(1D)
               An array of frequencies
            y : np.array(1D,SAME TYPE WITH X)
               The amplitude array respectively correspond to frequency array
            x_inf : int
               Infimum of freq.
            x_sup : int
               Supremum of freq.

        Returns:
           np.mean(sum_A): int
               freq parameter,topomap procedure needed
        '''
        sum_A=[]
        for i,freq in enumerate(x):
            if freq<=x_sup and freq>=x_inf:
                sum_A.append(y[i])     
        return np.mean(sum_A)

    def plot_topomap(self,data,ch_names,srate=-1,ch_types = 'eeg'):
        '''
        -author: Zhou hongzhan & He Jiatong
        -Create on:2022-8-9
        -update log:
            2022-8-31 by Zhou hongzhan

        Args:
           data : np.array, 1D array
               eeg data. The default is [].
           ch_names : list
               interested channels 
           srate : int
               sample rate. The default is -1.if set as default ,the initial sample ratio will be applied
           ch_types : string
               Type of channels,default value='eeg'
        '''

        if srate == -1:
            srate = self.fs
        info  = mne.create_info(ch_names=ch_names, sfreq = srate,ch_types = ch_types)
        evoked = mne.EvokedArray(data, info)
        evoked.set_montage('standard_1005')
        mne.viz.plot_topomap(evoked.data[:, 0], evoked.info,show=True)
    
    def signal_noise_ratio(self,data=[],srate=-1,T=[],channel=[]): 
        '''
        -author: Zhou hongzhan & He Jiatong
        -Create on:2022-8-9
        -update log:
            2022-8-31 by Zhou hongzhan

        Args:
            data : np.array, 1D array
               eeg data. The default is [].
            srate : int
               sample rate. The default is -1.if set as default ,the initial sample ratio will be applied
            T : int, ms
               the during time of data. The default is [].
            channel : string
               interested channels 

        Returns:
            X1 : np.array
               frequency sequece.
            snr : np.array
               SNR sequence

        '''
        if srate == -1:
            srate = self.fs
        num_fft = srate*T
        df = srate/num_fft
        n = np.arange(0, num_fft-1, 1)
        fx = n*df
        fx = fx[None,:]
        Y = fft(data[channel,:],num_fft)
        Y = np.abs(Y)*2/num_fft
        Y = Y[None,:]
        X1 = fx[0,0:num_fft//2]
        Y1 = Y[0,0:num_fft//2]
        plt.plot(X1, Y1)
        plt.title('fft transform')
        plt.xlim(0,60)
        plt.ylim(0,2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (μV)')
        plt.show()

        nn1 = np.linspace(start=5, stop=round(60/df),  num = round(60/df), endpoint=False).astype(int)
        snr=[]
        for center_freq in nn1:
            if center_freq<round(60/df):
                sum_denominator=[]
                snr_nominator=[]
                sum_denominator=np.mean(Y1[center_freq-5:center_freq-1])+np.mean(Y1[center_freq+1:center_freq+5])
                snr_nominator=Y1[center_freq]
                SNR=20*np.log10(snr_nominator/sum_denominator)
                snr.append(SNR)
        plt.plot(X1[0:round(60/df)], snr)
        plt.title('Signal Noise Ratio')
        plt.xlim(5,60)
        plt.ylim(-35,10)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Signal Noise Ratio (dB)')
        plt.show()
        return X1,snr
