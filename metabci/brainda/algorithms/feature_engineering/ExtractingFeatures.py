# emg features
import numpy as np
# import pywt
import operator
from nitime import algorithms as alg
from scipy import stats
from numpy import linalg as LA
# from pyhht import pyhht
from scipy import signal as sig
from scipy import linalg
from pyts.image import RecurrencePlot
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
# import operator
from math import pi
from sampen import sampen2
# emg features
import numpy as np
# import pywt
import operator
from nitime import algorithms as alg
from scipy import stats
from numpy import linalg as LA
# from pyhht import pyhht
from scipy import signal as sig
from scipy import linalg
from pyts.image import RecurrencePlot
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
# import operator
from math import pi
from sampen import sampen2
from sklearn.base import BaseEstimator

Feature_list = ['ssc','wl','mean','rms','log','mnf_MEDIAN_POWER']

class Extracting_Feature(BaseEstimator):
    def __init__(self, threshold: float = 1e-5, EPS: float = 1e-5, fs=100, type='MEDIAN_POWER'):
        self.threshold = threshold
        self.type = type
        self.fs = fs
    def emg_ssc(self,signal):
        signal = np.array(signal)
        temp = [(signal[i] - signal[i - 1]) * (signal[i] - signal[i + 1]) for i in range(1, signal.shape[0] - 1, 1)]
        temp = np.array(temp)

        temp = temp[temp >= self.threshold].shape[0]
        return temp

    def emg_wl(self,signal):
        signal = np.array(signal)
        length = signal.shape[0]
        wl = [abs(signal[i + 1] - signal[i]) for i in range(length - 1)]
        return np.sum(wl)

    def emg_mean(self,signal):
        signal = np.array(signal)
        signal_mean = np.mean(signal)
        length = signal.shape[0]
        if length <= 1:
            return 0
        return float(signal_mean)

 
    def emg_mav(self,signal):
        signal_abs = [abs(s) for s in signal]
        signal_abs = np.array(signal_abs)
        if len(signal_abs) == 0:
            return 0
        else:
            return np.mean(signal_abs)

    def emg_rms(self,signal):
        signal = np.array(signal)
        ssi = self.emg_ssi(signal)
        length = signal.shape[0]
        if length <= 0:
            return 0
        return np.sqrt(float(ssi) / length)


    def emg_log(self,signal):
        signal = np.array(signal)
        signal_log = []
        for s in signal:
            if abs(s) == 0:
                signal_log.append(1e-6)
            else:
                signal_log.append(np.log(abs(s)))
        value = np.mean(signal_log)
        return np.exp(value)

    def emg_mnf_MEDIAN_POWER(self,signal):
        if self.type == 'MEDIAN_POWER':
            cc, freq = self.emg_fft_power(signal)
        else:
            cc, freq = self.emg_fft(signal)

        ccsum = np.sum(cc)
        fp = cc * freq
        if np.sum(cc) == 0:
            return 0

        return np.sum(fp) / ccsum



    def emg_mav1(self,signal):
        signal_abs = [abs(s) for s in signal]
        signal_abs = np.array(signal_abs)
        N = len(signal)
        w = []
        for i in range(1,N+1,1):
            if i >= 0.25*N and i <= 0.75*N:
                w.append(1)
            else:
                w.append(0.5)
        w = np.array(w)
        return np.mean(signal_abs*w)

    def emg_mav2(self,signal):
        signal_abs = [abs(s) for s in signal]
        signal_abs = np.array(signal_abs)
        N = len(signal)
        w = []
        for i in range(1,N+1,1):
            if i >= 0.25*N and i <= 0.75*N:
                w.append(1)
            elif i < 0.25*N:
                w.append(4.0*i/N)
            else:
                w.append(4.0*(i-N)/N)
        w = np.array(w)
        return np.mean(signal_abs*w)

  
    def emg_mavslp(self,signal, sub_window_length=100, sub_window_overlap=50):
        frame_length = len(signal)
        if sub_window_length > frame_length:
            sub_window_length = frame_length

        window_step = int(sub_window_length*float(1-sub_window_overlap))

        mavs = []
        start = 0
        flag = 0
        for i in range(0, frame_length, sub_window_length):
            if (start+sub_window_length) >= frame_length:
                end = frame_length
                flag = 1
            else:
                end = start + sub_window_length

            each_frame = signal[start:end]

            start = start + window_step

            each_mav = self.emg_mav(each_frame)
            mavs.append(each_mav)
            if flag == 1:
                break

        if len(mavs) == 1:
            return mavs
        else:
            newmavs = []
            for i in range(1, len(mavs), 1):
                newmavs.append(mavs[i] - mavs[i - 1])
        return newmavs


    def emg_ssi(self,signal):
        signal_squ = [s * s for s in signal]
        signal_squ = np.array(signal_squ)
        return np.sum(signal_squ)


    def emg_var(self,signal):
        signal = np.array(signal)
        ssi = self.emg_ssi(signal)
        length = signal.shape[0]
        if length <= 1:
            return 0
        return float(ssi) / (length - 1)




    def emg_zc(self,signal, zc_threshold=0):
        sign = [[signal[i] * signal[i - 1], abs(signal[i] - signal[i - 1])] for i in range(1, len(signal), 1)]

        sign = np.array(sign)
        sign = sign[sign[:, 0] < 0]
        if sign.shape[0] == 0:
            return 0
        sign = sign[sign[:, 1] >= zc_threshold]
        return sign.shape[0]

 

    def emg_wamp(self,signal, threshold=5e-3):
        signal = np.array(signal)
        temp = [abs(signal[i] - signal[i - 1]) for i in range(1, signal.shape[0], 1)]
        temp = np.array(temp)

        temp = temp[temp >= threshold]
        return temp.shape[0]

 

    def emg_arc(self,signal, order=4):
        if order >= len(signal):
            rd = len(signal)-1
        else:
            rd = order
        arc, ars = alg.AR_est_YW(signal, rd)
        arc = np.array(arc)
        return arc


    def emg_cc(self,signal, order=4):
        arc = self.emg_arc(signal, order)
        cc = []
        cc.append(-arc[0])
        cc = np.array(cc)
        for i in range(1, arc.shape[0], 1):
            cp = cc[0:i]
            cp = cp[::-1]
            num = range(1, i + 1, 1)
            num = np.array(num)
            num = -num / float(i + 1) + 1
            cp = cp * num
            cp = np.sum(cp)
            cc = np.append(cc, -arc[i] * (1 + cp))
        return cc
    

    def emg_fft(self,signal):
        fft_size = signal.shape[0]

        freqs = np.linspace(0, self.fs//2, fft_size//2+1)

        xf = np.fft.rfft(signal)/fft_size
        cc = np.clip(np.abs(xf), 1e-20, 1e100)
        # pl.scatter(freqs, cc)
        # pl.show()
        return cc, freqs

    def emg_rfft2d(self,signal):

        fft_size = signal.shape[0]*signal.shape[1]
        xf = np.fft.rfft2(signal)/fft_size
        cc = np.clip(np.abs(xf), 1e-20, 1e100)
    #     fshift = np.fft.fftshift(cc)
    #     cc = 20.0*np.log(np.abs(fshift))
    #     cc = np.abs(fshift)

        freqs_1 = np.linspace(0, self.fs//2, signal.shape[1]//2+1)

        freqs = [np.linspace(ff, self.fs, signal.shape[0]) for ff in freqs_1]

        freqs = np.array(freqs)

        freqs = np.transpose(freqs)

        return cc , freqs

    def emg_fft_power(self,signal):

        fft_size = signal.shape[0]
        cc, freq = self.emg_fft(signal)
        cc = cc * cc
        cc = cc / float(fft_size)

        cc = np.array(cc)
        # if cc.all() == 0:
        #     cc[cc == 0] = 0
        #     cc[cc != 0] = 10 * np.log10(cc[cc != 0])
        #     cc = 0
        # else:
        #     cc = 10 * np.log10(cc)
        return cc, freq

    def emg_rfft_power2d(self,signal, fs=100):
        fft_size = signal.shape[0]*signal.shape[1]
        cc, freq = self.emg_rfft2d(signal, fs)
        cc = cc * cc
        cc = cc / float(fft_size)

        cc = np.array(cc)
        # if cc.all() == 0:
        #     cc[cc == 0] = 0
        #     cc[cc != 0] = 10 * np.log10(cc[cc != 0])
        #     cc = 0
        # else:
        #     cc = 10 * np.log10(cc)
        return cc, freq

    def emg_fft_power_db1(self,signal):
        [cc, freqs] =  self.emg_fft_power(signal)
        return cc

    def emg_rfft2d_power_db1(self,signal):
        [cc, freqs] =  self.emg_rfft_power2d(signal,100)
        return cc


    def emg_mdf_MEDIAN_POWER(self,signal, fs=100, mtype='MEDIAN_POWER'):
        if mtype == 'MEDIAN_POWER':
            cc, freq = self.emg_fft_power(signal)
        else:
            cc, freq = self.emg_fft(signal)
        csum = 0
        pre_csum = 0
        index = 0
        ccsum = np.sum(cc)

        for i in range(cc.shape[0]):
            pre_csum = csum
            csum = csum + cc[i]
            if csum >= ccsum / 2:
                if (ccsum / 2 - pre_csum) < (csum - ccsum / 2):
                    index = i - 1
                else:
                    index = i
                break
        return freq[index]

    
    def emg_pkf(self,signal, fs=100):
        cc, freq = self.emg_fft_power(signal)

        max_index, max_power = max(enumerate(cc), key=operator.itemgetter(1))
        return freq[max_index]

    def emg_smn1(self,signal):
        return self.emg_smn(signal, order=1)

    def emg_smn2(self,signal):
        return self.emg_smn(signal, order=2)

    def emg_smn3(self,signal):
        return self.emg_smn(signal, order=3)


    def emg_psr(self,signal, fs=100, prange=5):
        cc, freq = self.emg_fft_power(signal)
        max_index, max_power = max(enumerate(cc), key=operator.itemgetter(1))
        if max_index-prange < 0:
            start = 0
        else:
            start = max_index - prange
        if max_index+prange >len(signal):
            end = len(signal)
        else:
            end = max_index + prange
        range_value = cc[start:end]
        range_value = np.sum(range_value)
        sum_value = np.sum(cc)
        if sum_value == 0:
            return 0
        return range_value / sum_value

    
    def ar_generator(self,coefs, prior, N):
        ar_order = len(coefs)
        signal = []
        signal.extend(prior)
        for i in range(ar_order, N, 1):
            temp = 0
            for j in range(ar_order):
                temp = temp + coefs[j] * signal[i - j - 1]
            signal.append(temp)
        return signal


    def emg_arr(self,signal, ar_order=4):
        arc = self.emg_arc(signal, ar_order)
        signal_inverse = self.ar_generator(arc, signal[0:ar_order], len(signal))
        ar_residue = signal - signal_inverse
        return ar_residue

    def emg_arr_mean(self,signal, ar_order=4):
        arc = self.emg_arc(signal, ar_order)
        signal_inverse = self.self.ar_generator(arc, signal[0:ar_order], len(signal))
        ar_residue = signal - signal_inverse
        return  np.mean(ar_residue)

    def emg_arr_std(self,signal, ar_order=4):
        arc = self.emg_arc(signal, ar_order)
        signal_inverse = self.ar_generator(arc, signal[0:ar_order], len(signal))
        ar_residue = signal - signal_inverse
        return  np.std(ar_residue)

    def emg_arr_Range(self,signal, ar_order=4):
        arc = self.emg_arc(signal, ar_order)
        signal_inverse = self.ar_generator(arc, signal[0:ar_order], len(signal))
        ar_residue = signal - signal_inverse
        RH, H = max(enumerate(ar_residue), key=operator.itemgetter(1))
        RL, L = min(enumerate(ar_residue), key=operator.itemgetter(1))
        Range = H - L
        return Range

    def emg_arr_Rangen(self,signal, ar_order=4):
        arc = self.emg_arc(signal, ar_order)
        signal_inverse = self.ar_generator(arc, signal[0:ar_order], len(signal))
        ar_residue = signal - signal_inverse
        RH, H = max(enumerate(ar_residue), key=operator.itemgetter(1))
        RL, L = min(enumerate(ar_residue), key=operator.itemgetter(1))
        RD = abs(RH - RL)
        Rangen = (H - L) / RD
        return Rangen

    def emg_arr_H(self,signal, ar_order=4):
        arc = self.emg_arc(signal, ar_order)
        signal_inverse = self.ar_generator(arc, signal[0:ar_order], len(signal))
        ar_residue = signal - signal_inverse
        RH, H = max(enumerate(ar_residue), key=operator.itemgetter(1))
        RL, L = min(enumerate(ar_residue), key=operator.itemgetter(1))
        return H

    def emg_arr_meanraise(self,signal, ar_order=4):
        arc = self.emg_arc(signal, ar_order)
        signal_inverse = self.ar_generator(arc, signal[0:ar_order], len(signal))
        ar_residue = signal - signal_inverse
        raising = []
        raising.append(ar_residue[0])
        N=len(ar_residue)
        for i in range(1, N, 1):
            if ar_residue[i] >= ar_residue[i - 1]:
                raising.append(ar_residue[i])

        mean_raise = np.mean(raising)
        return mean_raise

    def emg_arr_stdraise(self,signal, ar_order=4):
        arc = self.emg_arc(signal, ar_order)
        signal_inverse = self.ar_generator(arc, signal[0:ar_order], len(signal))
        ar_residue = signal - signal_inverse
        raising = []
        raising.append(ar_residue[0])
        N=len(ar_residue)
        for i in range(1, N, 1):
            if ar_residue[i] >= ar_residue[i - 1]:
                raising.append(ar_residue[i])

        std_raise = np.std(raising)
        return std_raise

    def emg_arr_offon(self,signal, ar_order=4):
        arc = self.emg_arc(signal, ar_order)
        signal_inverse = self.ar_generator(arc, signal[0:ar_order], len(signal))
        ar_residue = signal - signal_inverse

        onset = ar_residue[0]
        offset = ar_residue[-1]
        off_on = abs(offset - onset)

        return off_on

    def emg_arr_offset(self,signal, ar_order=4):
        arc = self.emg_arc(signal, ar_order)
        signal_inverse = self.ar_generator(arc, signal[0:ar_order], len(signal))
        ar_residue = signal - signal_inverse

        return ar_residue[-1]

 
    def emg_mnf(self,signal):
        a, f = self.emg_hht(signal)

        mif = []

        a2 = a * a
        for i in range(f.shape[0]):
            a2f = a2[i, :] * f[i, :]
            if sum(a2[i, :]) == 0:
                mif.append(0)
            else:
                mif.append(sum(a2f) / sum(a2[i, :]))
        mif = np.array(mif)
        # print 'mif:',mif

        anorm = []
        for i in range(a.shape[0]):
            anorm.append(linalg.norm(a[i, :]))
        anorm = np.array(anorm)
        # print 'anorm:',anorm

        if sum(anorm) == 0:
            mnfhht = 0
        else:
            mnfhht = sum(mif * anorm) / sum(anorm)

        return mnfhht

    def emg_hht(self,signal):
        imfs = self.pyhht.emd(signal, 10)
        a = self.pyhht.getinstamp(imfs)

        omega = self.pyhht.getinstfreq(imfs)

        f = omega / 2 * pi
        return a, f


 
    def predict(self,signal,feature_name):
        func = 'self.emg_'+feature_name
        result = eval(str(func))(signal)
        # print("result:",result)
        return result
