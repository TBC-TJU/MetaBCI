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

#print [x for x in dir(sig)]

EPS=np.finfo(float).eps

def emg_sampen(signal):
    sampen = sampen2(signal)
    sampen = [sampen[0][1],sampen[1][1],sampen[2][1]]
    return np.array(sampen)

def emg_iemg(signal):
    signal_abs = [abs(s) for s in signal]
    signal_abs = np.array(signal_abs)
    return np.sum(signal_abs)


def emg_mav(signal):
    signal_abs = [abs(s) for s in signal]
    signal_abs = np.array(signal_abs)
    if len(signal_abs) == 0:
        return 0
    else:
        return np.mean(signal_abs)

def emg_mav1(signal):
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

def emg_mav2(signal):
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


def emg_mavslpphinyomark(signal):
    frame_length = len(signal)
    sub_window_length =  int(np.ceil(frame_length / 3.0))
    sub_window_overlap = 0
    return emg_mavslp(signal, sub_window_length, sub_window_overlap)

def emg_mavslp10(signal):
    frame_length = len(signal)
    sub_window_length = int(np.ceil(frame_length / 10.0))
    sub_window_overlap = 0
    return emg_mavslp(signal, sub_window_length, sub_window_overlap)

def emg_mavslpframewise(signal):
    sub_window_length = 1
    sub_window_overlap = 0
    return emg_mavslp(signal, sub_window_length, sub_window_overlap)

def emg_mavslpsegmentwise2(signal):
    sub_window_length = int(np.ceil(len(signal)/3))
    sub_window_overlap = 0
    return emg_mavslp(signal, sub_window_length, sub_window_overlap)

def emg_mavslpsegmentwise(signal):
    sub_window_length = int(np.ceil(len(signal)/2))
    sub_window_overlap = 0
    return emg_mavslp(signal, sub_window_length, sub_window_overlap)

def emg_mavslp(signal, sub_window_length, sub_window_overlap):
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

         each_mav = emg_mav(each_frame)
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


def emg_ssi(signal):
     signal_squ = [s * s for s in signal]
     signal_squ = np.array(signal_squ)
     return np.sum(signal_squ)


def emg_var(signal):
     signal = np.array(signal)
     ssi = emg_ssi(signal)
     length = signal.shape[0]
     if length <= 1:
         return 0
     return float(ssi) / (length - 1)

def emg_mean(signal):
     signal = np.array(signal)
     signal_mean = np.mean(signal)
     length = signal.shape[0]
     if length <= 1:
         return 0
     return float(signal_mean)

def emg_rms(signal):
     signal = np.array(signal)
     ssi = emg_ssi(signal)
     length = signal.shape[0]
     if length <= 0:
         return 0
     return np.sqrt(float(ssi) / length)


def emg_mavtm(signal, order):
     signal = np.array(signal)
     signal_order = [s ** order for s in signal]
     return abs(np.mean(signal_order))

def emg_mavtm3(signal):
    return emg_mavtm(signal, order=3)

def emg_mavtm4(signal):
    return emg_mavtm(signal, order=4)

def emg_mavtm5(signal):
    return emg_mavtm(signal, order=5)


def emg_vorder(signal):
    signal = np.array(signal)
    signal_order = [s ** 2 for s in signal]
    value = np.mean(signal_order)
    return value ** (float(1) / 2)


def emg_log(signal):
    signal = np.array(signal)
    signal_log = []
    for s in signal:
        if abs(s) == 0:
            signal_log.append(1e-6)
        else:
            signal_log.append(np.log(abs(s)))
    value = np.mean(signal_log)
    return np.exp(value)


def emg_wl(signal):
     signal = np.array(signal)
     length = signal.shape[0]
     wl = [abs(signal[i + 1] - signal[i]) for i in range(length - 1)]
     return np.sum(wl)


def emg_aac(signal):
     signal = np.array(signal)
     length = signal.shape[0]
     wl = [abs(signal[i + 1] - signal[i]) for i in range(length - 1)]
     return np.mean(wl)


def emg_zc(signal, zc_threshold=0):
    sign = [[signal[i] * signal[i - 1], abs(signal[i] - signal[i - 1])] for i in range(1, len(signal), 1)]

    sign = np.array(sign)
    sign = sign[sign[:, 0] < 0]
    if sign.shape[0] == 0:
        return 0
    sign = sign[sign[:, 1] >= zc_threshold]
    return sign.shape[0]


def emg_wl_dasdv(signal):
     signal = np.array(signal)
     length = signal.shape[0]
     wl = [(signal[i + 1] - signal[i]) ** 2 for i in range(length - 1)]
     sum_squ = np.sum(wl)
     if length <= 1:
         return 0
     return np.sqrt(sum_squ / (length - 1))


def emg_afb9(signal):
    a_value = emg_afb(signal, hamming_window_length=9)
    return a_value


def emg_afb(signal, hamming_window_length):
     hamming_window = np.hamming(hamming_window_length)
     signal_length = len(signal)
     signal_after_filter = []
     end_flag = 0

     for i in range(signal_length):
         start = i
         end = i + hamming_window_length
         if end >= signal_length:
             end = signal_length
             end_flag = 1

         signal_seg = signal[start:end]
         signal_seg = np.array(signal_seg)
         signal_after_filter.append(np.sum(signal_seg * signal_seg * hamming_window) / np.sum(hamming_window))

         if end_flag == 1:
             end_flag = 0
             break
     signal_after_filter = np.array(signal_after_filter)

     a_value = signal_after_filter[0]

     for i in range(1, len(signal_after_filter) - 1, 1):
         if signal_after_filter[i] > signal_after_filter[i - 1] and signal_after_filter[i] > signal_after_filter[i + 1]:
             a_value = signal_after_filter[i]
             break
     return a_value


def emg_myop(signal, threshold=0.5):
     signal = np.array(signal)
     length = signal.shape[0]

     signal = signal[signal >= threshold]
     count = signal.shape[0]
     if length <= 0:
         return 0
     return float(count) / length

#def emg_sscbestninapro1(signal):
#    return emg_ssc(signal, 1e-5)

def emg_ssc(signal, threshold=1e-5):
     signal = np.array(signal)
     temp = [(signal[i] - signal[i - 1]) * (signal[i] - signal[i + 1]) for i in range(1, signal.shape[0] - 1, 1)]
     temp = np.array(temp)

     temp = temp[temp >= threshold]
     return temp.shape[0]


def emg_wamp(signal, threshold=5e-3):
     signal = np.array(signal)
     temp = [abs(signal[i] - signal[i - 1]) for i in range(1, signal.shape[0], 1)]
     temp = np.array(temp)

     temp = temp[temp >= threshold]
     return temp.shape[0]


def emg_hemg(signal, bins):
     signal = np.array(signal)
     hist, bin_edge = np.histogram(signal, bins)
     return hist

def emg_hemg15(signal):
     return emg_hemg(signal, bins=15)

def emg_hemg20(signal):
     return emg_hemg(signal, bins=20)

#def emg_hemg20(signal):
#     return emg_hemg(signal, bins=20)


def emg_mhw_energy(signal):
     signal = np.array(signal)
     window_length = signal.shape[0]
     sub_window_len = int(window_length/2.4)
     mhwe = []
     start = 0
     for i in range(3):
         end = min(start+sub_window_len, window_length)

         sub_signal = signal[start:end]
         start = int(start + sub_window_len*0.7)
         hamming_window = np.hamming(len(sub_signal))
         sub_signal = sub_signal * hamming_window
         sub_signal = sub_signal **2
         mhwe.append(np.sum(sub_signal))

     mhwe = np.array(mhwe)
     return mhwe


def emg_mtw_energy(signal, first_percent=0.2, second_percent=0.7):
     signal = np.array(signal)
     window_length = signal.shape[0]
     sub_window_len = int(window_length/2.4)
     mtwe = []
     start = 0
     for i in range(3):
         end = min(start+sub_window_len, window_length)
         sub_signal = signal[start:end]
         start = int(start + sub_window_len*0.7)

         t_window = []
         k1 = 1 / float(len(sub_signal) * first_percent)
         k2 = 1 / float(len(sub_signal) * (second_percent - 1))
         b2 = 1 / float(1 - second_percent)
         first_point = int(len(sub_signal) * first_percent)
         second_point = int(len(sub_signal) * second_percent)
         for i in range(len(sub_signal)):
             if i >= 0 and i < first_point:
                 y = k1 * i
             elif i >= second_point and i <= len(sub_signal):
                 y = k2 * i + b2
             else:
                 y = 1
             t_window.append(y)
         t_window = np.array(t_window)
         sub_signal = sub_signal * t_window
         sub_signal = sub_signal ** 2
         mtwe.append(np.sum(sub_signal))
     mtwe = np.array(mtwe)

     return mtwe


def emg_arc(signal, order=4):
     if order >= len(signal):
         rd = len(signal)-1
     else:
         rd = order
     arc, ars = alg.AR_est_YW(signal, rd)
     arc = np.array(arc)
     return arc


def emg_cc(signal, order=4):
     arc = emg_arc(signal, order)
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

def emg_spectrogram_stft(signal, fs, nperseg, noverlap):
#    from scipy import signal
    f, t, Sxx = sig.stft(x=signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

#    print t.shape
#    print Sxx.shape
    return f,t,Sxx

def emg_spectrogram_stft_db1_seg10overlap9(signal):
    [f,t,Sxx] =  emg_spectrogram(signal,100,10,9)
#    print Sxx.shape
    return Sxx.flatten()

def emg_spectrogram(signal, fs, nperseg, noverlap):
#    from scipy import signal
    f, t, Sxx = sig.spectrogram(x=signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

#    print t.shape
#    print Sxx.shape
    return f,t,Sxx

def emg_spectrogram_db1_seg10overlap9(signal):
    [f,t,Sxx] =  emg_spectrogram(signal,100,10,9)
#    print Sxx.shape
    return Sxx.flatten()

def emg_spectrogram_db1_seg10overlap9_nolowpass(signal):
    [f,t,Sxx] =  emg_spectrogram(signal,100,10,9)
#    print Sxx.shape
    return Sxx.flatten()

def emg_fft(signal, fs):
    fft_size = signal.shape[0]

    freqs = np.linspace(0, fs//2, fft_size//2+1)

    xf = np.fft.rfft(signal)/fft_size
    cc = np.clip(np.abs(xf), 1e-20, 1e100)
    # pl.scatter(freqs, cc)
    # pl.show()
    return cc, freqs

def emg_rfft2d(signal, fs):

     fft_size = signal.shape[0]*signal.shape[1]
     xf = np.fft.rfft2(signal)/fft_size
     cc = np.clip(np.abs(xf), 1e-20, 1e100)
#     fshift = np.fft.fftshift(cc)
#     cc = 20.0*np.log(np.abs(fshift))
#     cc = np.abs(fshift)

     freqs_1 = np.linspace(0, fs//2, signal.shape[1]//2+1)

     freqs = [np.linspace(ff, fs, signal.shape[0]) for ff in freqs_1]

     freqs = np.array(freqs)

     freqs = np.transpose(freqs)

     return cc , freqs

def emg_fft_power(signal, fs=1000):
    fft_size = signal.shape[0]
    cc, freq = emg_fft(signal, fs)
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

def emg_rfft_power2d(signal, fs=100):
    fft_size = signal.shape[0]*signal.shape[1]
    cc, freq = emg_rfft2d(signal, fs)
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

def emg_fft_power_db1(signal):
    [cc, freqs] =  emg_fft_power(signal,100)
    return cc

def emg_rfft2d_power_db1(signal):
    [cc, freqs] =  emg_rfft_power2d(signal,100)
    return cc


def emg_mdf_MEDIAN_POWER(signal, fs=100, mtype='MEDIAN_POWER'):
     if mtype == 'MEDIAN_POWER':
         cc, freq = emg_fft_power(signal, fs)
     else:
         cc, freq = emg_fft(signal, fs)
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


#def emg_mdf_MEDIAN_POWER_2d(signal, fs=100, mtype='MEDIAN_POWER'):
#     if mtype == 'MEDIAN_POWER':
#         cc, freq = emg_fft_power2d(signal, fs)
#     else:
#         cc, freq = emg_fft2d(signal, fs)
#     csum = 0
#     pre_csum = 0
#     index = 0
#     ccsum = np.sum(cc)
#
#
#     for kk in range(cc.shape[0]*cc.shape[1]):
#             pre_csum = csum
#             csum = csum + cc[kk // cc.shape[1]][kk % cc.shape[1]]
#             if csum >= ccsum / 2:
#                 if (ccsum / 2 - pre_csum) < (csum - ccsum / 2):
#                     index = kk - 1
#                 else:
#                     index = kk
#                 break
#
#     return freq[index // cc.shape[1]][index % cc.shape[1]]

def emg_mnf_MEDIAN_POWER(signal, fs=100, type='MEDIAN_POWER'):
     if type == 'MEDIAN_POWER':
         cc, freq = emg_fft_power(signal, fs)
     else:
         cc, freq = emg_fft(signal, fs)

     ccsum = np.sum(cc)
     fp = cc * freq
     if np.sum(cc) == 0:
         return 0

     return np.sum(fp) / ccsum


def emg_pkf(signal, fs=100):
     cc, freq = emg_fft_power(signal, fs)

     max_index, max_power = max(enumerate(cc), key=operator.itemgetter(1))
     return freq[max_index]


def emg_mnp(signal, fs=100):
    cc, freq = emg_fft_power(signal, fs)
    return np.mean(cc)


def emg_ttp(signal, fs=100):
     cc, freq = emg_fft_power(signal, fs)
     return np.sum(cc)

def emg_smn1(signal):
    return emg_smn(signal, order=1)

def emg_smn2(signal):
    return emg_smn(signal, order=2)

def emg_smn3(signal):
    return emg_smn(signal, order=3)

def emg_smn(signal, order):
     cc, freq = emg_fft_power(signal, fs=100)
     freq = freq ** order
     cc = cc * freq
     return np.sum(cc)


def emg_fr(signal, fs=100, low_down=0, low_up=20, high_down=60, high_up=90):
     cc, freq = emg_fft_power(signal, fs)

     maxfre = np.max(freq)
     minfre = np.min(freq)

     ld = minfre + (maxfre - minfre) * low_down / 100
     lu = minfre + (maxfre - minfre) * low_up / 100
     hd = minfre + (maxfre - minfre) * high_down / 100
     hu = minfre + (maxfre - minfre) * high_up / 100

     low = cc[(freq >= ld) & (freq <= lu)]

     high = cc[(freq >= hd) & (freq <= hu)]

     if len(high) == 0 | len(low) == 0:
         return 0

     if np.sum(high) == 0:
         return 0

     return np.sum(low) / np.sum(high)


def emg_psr(signal, fs=100, prange=5):
     cc, freq = emg_fft_power(signal, fs)
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


def emg_vcf(signal, fs=100):
     sm2 = emg_smn(signal, 2)
     sm1 = emg_smn(signal, 1)
     sm0 = emg_smn(signal, 0)
     if sm0 == 0:
         return 0

     return sm2 / sm0 - (sm1 / sm0) ** 2


def emg_hos2(signal, fs, t1):
     cc, freq = emg_fft_power(signal, fs)
     cc = np.array(cc)

     signalt = np.zeros(cc.shape[0])
     length = cc.shape[0]
     if t1 >= 0:
         signalt[0:(length - t1)] = cc[t1:]
     else:
         signalt[-t1:] = cc[0:(length + t1)]

     signalt = cc * signalt
     return np.mean(signalt)


def emg_hos3(signal, fs, t1, t2):
     cc, freq = emg_fft_power(signal, fs)

     cc = np.array(cc)
     length = cc.shape[0]
     signalt1 = np.zeros(length)
     signalt2 = np.zeros(length)
     signalt1[0:(length - t1)] = cc[t1:]
     signalt2[0:(length - t2)] = cc[t2:]
     signalt = cc * signalt1 * signalt2
     return np.mean(signalt)


def emg_hos4(signal, fs, t1, t2, t3):
     cc, freq = emg_fft_power(signal, fs)

     cc = np.array(cc)
     length = cc.shape[0]
     signalt1 = np.zeros(length)
     signalt2 = np.zeros(length)
     signalt3 = np.zeros(length)
     signalt1[0:(length - t1)] = cc[t1:]
     signalt2[0:(length - t2)] = cc[t2:]
     signalt3[0:(length - t3)] = cc[t3:]

     signalt = cc * signalt1 * signalt2 * signalt3
     mean4 = np.mean(signalt)
     result = mean4 - emg_hos2(signal, fs, t1) * emg_hos2(signal, fs, t2 - t3) - emg_hos2(signal, fs, t2) * emg_hos2(
         signal, fs, t3 - t1) - emg_hos2(signal, fs, t3) * emg_hos2(signal, fs, t1 - t2)
     return result


def emg_hos(signal, fs=100):
     hos = []
     hos.append(emg_hos2(signal, fs, 0))
     hos.append(emg_hos2(signal, fs, 1))
     hos.append(emg_hos2(signal, fs, 2))
     for i in [0, 1, 2]:
         for j in range(i, 3):
             hos.append(emg_hos3(signal, fs, i, j))
     for i in [0, 1, 2]:
         for j in range(i, 3):
             for k in range(j, 3):
                 hos.append(emg_hos4(signal, fs, i, j, k))
     return hos


# def emg_dwt4(signal):
# #    print signal.shape
#     wavelet_level = 4 #int(np.log2(len(signal)))
#     coeffs = pywt.wavedec(signal, 'db1', level=wavelet_level)
#     return np.hstack(coeffs)

# def emg_dwt(signal):
# #    print signal.shape
#     wavelet_level = int(np.log2(len(signal)))
#     coeffs = pywt.wavedec(signal, 'db1', level=wavelet_level)
#     return np.hstack(coeffs)

# def emg_dwt2d(signal):
#     coeffs = pywt.wavedec2(signal, 'db1')
#     feature = []
#     feature.append(coeffs[0].flatten())
#     for i in range(1,len(coeffs)):
#        for j in range(len(coeffs[i])):
#            feature.append(coeffs[i][j].flatten())
#     feature = np.hstack(feature)
#     return feature


# # # return all the energy of dwt coeffs
# def emg_dwt_energy(signal):
#      coeffs = emg_dwt(signal)
# #     print coeffs[0].shape
#      energys = []
#      for c in coeffs:
#          c_squ = [cc ** 2 for cc in c]
#          c_squ = np.array(c_squ)
#          energys.append(np.sum(c_squ))
#      energys = np.array(energys)
#      return energys

# def emg_dwt2d_energy(signal):
#      coeffs = pywt.wavedec2(signal, 'db1')
# #     print coeffs[0].shape
#      energys = []
#      for c in coeffs:
#          c_squ = [cc ** 2 for cc in c]
#          c_squ = np.array(c_squ)
#          energys.append(np.sum(c_squ))
#      energys = np.array(energys)
#      return energys

# def emg_cwt(signal, width=8):
#      wavelet = sig.ricker
#      widths = np.arange(1, width + 1)
#      cwt = sig.cwt(signal, wavelet, widths)

     return cwt.flatten()

# def emg_cwt6(signal, width=6):
#      wavelet = sig.ricker
#      widths = np.arange(1, width + 1)
#      cwt = sig.cwt(signal, wavelet, widths)

#      return cwt.flatten()

# def emg_cwt4(signal, width=4):
#      wavelet = sig.ricker
#      widths = np.arange(1, width + 1)
#      cwt = sig.cwt(signal, wavelet, widths)

#      return cwt.flatten()
# def emg_dwpt4(signal, wavelet_name='db1'):
#     wavelet_level = 4# int(np.log2(len(signal)))
#     wp = pywt.WaveletPacket(signal, wavelet_name, mode='sym')
#     coeffs = []
#     level_coeff = wp.get_level(wavelet_level)
#     for i in range(len(level_coeff)):
#         coeffs.append(level_coeff[i].data)
#     coeffs = np.array(coeffs)
#     coeffs = coeffs.flatten()
#     return coeffs


# def emg_dwpt(signal, wavelet_name='db1'):
#     wavelet_level =  int(np.log2(len(signal)))
#     wp = pywt.WaveletPacket(signal, wavelet_name, mode='sym')
#     coeffs = []
#     level_coeff = wp.get_level(wavelet_level)
#     for i in range(len(level_coeff)):
#         coeffs.append(level_coeff[i].data)
#     coeffs = np.array(coeffs)
#     coeffs = coeffs.flatten()
#     return coeffs

# def emg_dwpt2d(signal, wavelet_name='db1'):
#     wavelet_level = 3
#     wp = pywt.WaveletPacket2D(signal, wavelet_name, mode='sym')
#     coeffs = []
#     level_coeff = wp.get_level(wavelet_level)
#     for i in range(len(level_coeff)):
#         coeffs.append(level_coeff[i].data.flatten())
#     coeffs = np.hstack(coeffs)
# #    coeffs = coeffs.flatten()
#     return coeffs

#def emg_dwpt2d(signal, wavelet_name='db1'):
#    wavelet_level = int(np.log2(len(signal)))
#    wp = pywt.WaveletPacket(signal, wavelet_name, mode='sym')
#    coeffs = []
#    level_coeff = wp.get_level(wavelet_level)
#    for i in range(len(level_coeff)):
#        coeffs.append(level_coeff[i].data)
#    coeffs = np.array(coeffs)
#    coeffs = coeffs.flatten()
#    return coeffs

def emg_dwpt_mean(signal):
    coeffs = emg_dwpt(signal)
    return np.mean(coeffs)


def emg_dwpt_sd(signal):
    coeffs = emg_dwpt(signal)
    return np.std(coeffs)

def emg_dwpt_energy(signal):
     coeffs = emg_dwpt(signal)
     coeffs = coeffs ** 2
     return np.sum(coeffs)


def emg_dwpt_skewness(signal):
     coeffs = emg_dwpt(signal)
     coeffs = np.array(coeffs)
     skew = stats.skew(coeffs)
     return skew


def emg_dwpt_kurtosis(signal):
     coeffs = emg_dwpt(signal)
     coeffs = np.array(coeffs)
     kurtosis = stats.kurtosis(coeffs)
     return kurtosis

def emg_dwpt_m2(signal):
    return emg_dwpt_m(signal, order=2)

def emg_dwpt_m3(signal):
    return emg_dwpt_m(signal, order=3)

def emg_dwpt_m4(signal):
    return emg_dwpt_m(signal, order=4)

def emg_dwpt_m(signal, order):
     coeffs = emg_dwpt(signal)
     coeffs = np.array(coeffs)
     length = coeffs.shape[0]
     a = range(1, length + 1, 1)
     a = np.array(a)
     a = (a / float(length)) ** order
     coeffs = coeffs * a
     return np.sum(coeffs)

def emg_dwpt2d_mean(signal):
    coeffs = emg_dwpt2d(signal)
    return np.mean(coeffs)


def emg_dwpt2d_sd(signal):
    coeffs = emg_dwpt2d(signal)
    return np.std(coeffs)

def emg_dwpt2d_energy(signal):
     coeffs = emg_dwpt2d(signal)
     coeffs = coeffs ** 2
     return np.sum(coeffs)

def emg_dwpt2d_skewness(signal):
     coeffs = emg_dwpt2d(signal)
     coeffs = np.array(coeffs)
     skew = stats.skew(coeffs)
     return skew

def emg_dwpt2d_kurtosis(signal):
     coeffs = emg_dwpt2d(signal)
     coeffs = np.array(coeffs)
     kurtosis = stats.kurtosis(coeffs)
     return kurtosis

def emg_dwpt2d_m2(signal):
    return emg_dwpt2d_m(signal, order=2)

def emg_dwpt2d_m3(signal):
    return emg_dwpt2d_m(signal, order=3)

def emg_dwpt2d_m4(signal):
    return emg_dwpt2d_m(signal, order=4)

def emg_dwpt2d_m(signal, order):
     coeffs = emg_dwpt2d(signal)
     coeffs = np.array(coeffs)
     length = coeffs.shape[0]
     a = range(1, length + 1, 1)
     a = np.array(a)
     a = (a / float(length)) ** order
     coeffs = coeffs * a
     return np.sum(coeffs)

def emg_mdwt(signal, wavelet_name='db1'):
     coeffs = pywt.wavedec(signal, wavelet_name)
     mdwt = []
     for detail_coeff in range(1, len(coeffs)):
         coeff_abs = [abs(c) for c in coeffs[detail_coeff]]
         coeff_abs = np.array(coeff_abs)
         mdwt.append(np.sum(coeff_abs))
     mdwt = np.array(mdwt)
     return mdwt

def emg_mdwtdb7ninapro(signal):
     coeffs = pywt.wavedec(signal, wavelet='db7', mode='symmetric', level=3)
     mdwt = []
     for detail_coeff in range(1, len(coeffs)):
         coeff_abs = [abs(c) for c in coeffs[detail_coeff]]
         coeff_abs = np.array(coeff_abs)
         mdwt.append(np.sum(coeff_abs))
     mdwt = np.array(mdwt)
     return mdwt

def emg_mdwtdb1ninapro(signal):
     coeffs = pywt.wavedec(signal, wavelet='db1', mode='symmetric', level=3)
     mdwt = []
     for detail_coeff in range(1, len(coeffs)):
         coeff_abs = [abs(c) for c in coeffs[detail_coeff]]
         coeff_abs = np.array(coeff_abs)
         mdwt.append(np.sum(coeff_abs))
     mdwt = np.array(mdwt)
     return mdwt

def emg_mdwtdb1ninapro_2d(signal):
     coeffs = pywt.wavedec2(signal, wavelet='db1', mode='symmetric', level=3)
     mdwt = []
     for detail_coeff in range(1, len(coeffs)):
         coeff_abs = [abs(c) for c in coeffs[detail_coeff]]
         coeff_abs = np.array(coeff_abs)
         mdwt.append(np.sum(coeff_abs))
     mdwt = np.array(mdwt)
     return mdwt

def emg_mdwtdb7ninapro_2d(signal):
     coeffs = pywt.wavedec2(signal, wavelet='db7', mode='symmetric', level=3)
     mdwt = []
     for detail_coeff in range(1, len(coeffs)):
         coeff_abs = [abs(c) for c in coeffs[detail_coeff]]
         coeff_abs = np.array(coeff_abs)
         mdwt.append(np.sum(coeff_abs))
     mdwt = np.array(mdwt)
     return mdwt

def emg_mrwa(signal, wavelet_name='db1'):
     coeffs = pywt.wavedec(signal, wavelet_name)
     coeffs = np.array(coeffs)
     mrwa = []
     mrwa.append(LA.norm(coeffs[0]))
     for i in range(1, coeffs.shape[0], 1):
         detail = coeffs[i]
         detail_squ = [d * d for d in detail]
         detail_squ = np.array(detail_squ)
         mrwa.append(np.sum(detail_squ) / detail_squ.shape[0])
     mrwa = np.array(mrwa)
     return mrwa

def emg_mrwa2d(signal):
#     coeffs = emg_dwt2d(signal)
     coeffs = pywt.wavedec2(signal, wavelet='db1')
#     coeffs = np.array(coeffs)
     mrwa = []
     mrwa.append(LA.norm(coeffs[0]))
     for i in range(1, len(coeffs), 1):
         detail = coeffs[i]
         detail_squ = [d * d for d in detail]
         detail_squ = np.array(detail_squ)
         detail_squ = detail_squ.flatten()
         mrwa.append(np.sum(detail_squ) / len(detail_squ))
     mrwa = np.array(mrwa)
     return mrwa


def emg_apen(signal, sub_length=10, threshold=0.002):
     return fai(signal, sub_length, threshold) - fai(signal, sub_length + 1, threshold)


def fai(signal, sub_length, threshold):
     dist = []
     signal = np.array(signal)
     N = signal.shape[0]

     if (N - sub_length + 1) == 0:
         return 0

     for i in range(0, N - sub_length + 1, 1):
         sub1 = signal[i:(i + sub_length)]
         row_dist = []
         for j in range(0, N - sub_length + 1, 1):
             sub2 = signal[j:(j + sub_length)]
             dist_value = abs(sub1 - sub2)
             dist_value = np.max(dist_value)
             row_dist.append(dist_value)
         row_dist = np.array(row_dist)
         dist.append(row_dist)
     dist = np.array(dist)
     cmr = [d[d <= threshold].shape[0] for d in dist]
     cmr = np.array(cmr)
     cmr = cmr / float(N - sub_length + 1)
     cmr = np.log(cmr)
     cmr = np.sum(cmr)

     cmr = cmr / float(N - sub_length + 1)
     return cmr


def emg_wte(signal, width=8):
     wavelet = sig.ricker
     widths = np.arange(1, width + 1)
     cwt = sig.cwt(signal, wavelet, widths)
     wte = []
     for i in range(cwt.shape[1]):
         col = cwt[:, i]
         col = col * col
         col_energy = np.sum(col)
         col = col / float(col_energy)
         col = -(col * np.log(col))
         wte.append(np.sum(col))
     wte = np.array(wte)
     return wte


def emg_wfe(signal, width=8):
     wavelet = sig.ricker
     widths = np.arange(1, width + 1)
     cwt = sig.cwt(signal, wavelet, widths)
     wfe = []
     for i in range(cwt.shape[0]):
         row = cwt[i, :]
         row = row * row
         row_energy = np.sum(row)
         row = row / float(row_energy)
         row = -(row * np.log(row))
         wfe.append(np.sum(row))
     wfe = np.array(wfe)
     return wfe




def ar_generator(coefs, prior, N):
    ar_order = len(coefs)
    signal = []
    signal.extend(prior)
    for i in range(ar_order, N, 1):
        temp = 0
        for j in range(ar_order):
            temp = temp + coefs[j] * signal[i - j - 1]
        signal.append(temp)
    return signal


def emg_arr(signal, ar_order=4):
    arc = emg_arc(signal, ar_order)
    signal_inverse = ar_generator(arc, signal[0:ar_order], len(signal))
    ar_residue = signal - signal_inverse
    return ar_residue

def emg_arr_mean(signal, ar_order=4):
    arc = emg_arc(signal, ar_order)
    signal_inverse = ar_generator(arc, signal[0:ar_order], len(signal))
    ar_residue = signal - signal_inverse
    return  np.mean(ar_residue)

def emg_arr_std(signal, ar_order=4):
    arc = emg_arc(signal, ar_order)
    signal_inverse = ar_generator(arc, signal[0:ar_order], len(signal))
    ar_residue = signal - signal_inverse
    return  np.std(ar_residue)

def emg_arr_Range(signal, ar_order=4):
    arc = emg_arc(signal, ar_order)
    signal_inverse = ar_generator(arc, signal[0:ar_order], len(signal))
    ar_residue = signal - signal_inverse
    RH, H = max(enumerate(ar_residue), key=operator.itemgetter(1))
    RL, L = min(enumerate(ar_residue), key=operator.itemgetter(1))
    Range = H - L
    return Range

def emg_arr_Rangen(signal, ar_order=4):
    arc = emg_arc(signal, ar_order)
    signal_inverse = ar_generator(arc, signal[0:ar_order], len(signal))
    ar_residue = signal - signal_inverse
    RH, H = max(enumerate(ar_residue), key=operator.itemgetter(1))
    RL, L = min(enumerate(ar_residue), key=operator.itemgetter(1))
    RD = abs(RH - RL)
    Rangen = (H - L) / RD
    return Rangen

def emg_arr_H(signal, ar_order=4):
    arc = emg_arc(signal, ar_order)
    signal_inverse = ar_generator(arc, signal[0:ar_order], len(signal))
    ar_residue = signal - signal_inverse
    RH, H = max(enumerate(ar_residue), key=operator.itemgetter(1))
    RL, L = min(enumerate(ar_residue), key=operator.itemgetter(1))
    return H

def emg_arr_meanraise(signal, ar_order=4):
    arc = emg_arc(signal, ar_order)
    signal_inverse = ar_generator(arc, signal[0:ar_order], len(signal))
    ar_residue = signal - signal_inverse
    raising = []
    raising.append(ar_residue[0])
    N=len(ar_residue)
    for i in range(1, N, 1):
        if ar_residue[i] >= ar_residue[i - 1]:
            raising.append(ar_residue[i])

    mean_raise = np.mean(raising)
    return mean_raise

def emg_arr_stdraise(signal, ar_order=4):
    arc = emg_arc(signal, ar_order)
    signal_inverse = ar_generator(arc, signal[0:ar_order], len(signal))
    ar_residue = signal - signal_inverse
    raising = []
    raising.append(ar_residue[0])
    N=len(ar_residue)
    for i in range(1, N, 1):
        if ar_residue[i] >= ar_residue[i - 1]:
            raising.append(ar_residue[i])

    std_raise = np.std(raising)
    return std_raise

def emg_arr_offon(signal, ar_order=4):
    arc = emg_arc(signal, ar_order)
    signal_inverse = ar_generator(arc, signal[0:ar_order], len(signal))
    ar_residue = signal - signal_inverse

    onset = ar_residue[0]
    offset = ar_residue[-1]
    off_on = abs(offset - onset)

    return off_on

def emg_arr_offset(signal, ar_order=4):
    arc = emg_arc(signal, ar_order)
    signal_inverse = ar_generator(arc, signal[0:ar_order], len(signal))
    ar_residue = signal - signal_inverse

    return ar_residue[-1]



def emg_arr29(signal, ar_order=4):
    arc = emg_arc(signal, ar_order)
    signal_inverse = ar_generator(arc, signal[0:ar_order], len(signal))
    ar_residue = signal - signal_inverse
    # print ar_residue

    arr_features = emg_stat29(ar_residue)

    return arr_features


def emg_hht58(signal):
    a, f = emg_hht(signal)
    sum_a = np.sum(a, axis=0)
    sum_f = np.sum(f, axis=0)
    # print sum_a
    # print sum_f
    sum_a29 = emg_stat29(sum_a)
    sum_f29 = emg_stat29(sum_f)
    features = np.hstack((sum_a29, sum_f29))
    # print np.shape(features)
    return features

def emg_mnf(signal):
    a, f = emg_hht(signal)

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

def emg_hht(signal):
    imfs = pyhht.emd(signal, 10)
    a = pyhht.getinstamp(imfs)

    omega = pyhht.getinstfreq(imfs)

    f = omega / 2 * pi
    return a, f




def emg_stat29(signal):
    RH, H = max(enumerate(signal), key=operator.itemgetter(1))
    RL, L = min(enumerate(signal), key=operator.itemgetter(1))
    RD = abs(RH - RL)
    Range = H - L

    if RD == 0:
       Rangen = 0
    else:
       Rangen = (H - L) / RD

    mean = np.mean(signal)
    std = np.std(signal)
    skewness = stats.skew(signal)
    kurtosis = stats.kurtosis(signal)

    ordered_signal = np.sort(signal)
    N = len(signal)
    q1 = ordered_signal[N // 4]
    q2 = ordered_signal[N // 2]
    q3 = ordered_signal[3 * N // 4]
    IQR = q3 - q1
    IQR_STD = abs(IQR - std)

    onset = signal[0]
    target = signal[N // 2]
    offset = signal[-1]
    tar_on = abs(target - onset)
    off_on = abs(offset - onset)
    off_tar = abs(offset - target)

    raising = []
    raising.append(signal[0])
    falling = []
    for i in range(1, N, 1):
        if signal[i] >= signal[i - 1]:
            raising.append(signal[i])
        else:
            falling.append(signal[i])

    mean_raise = np.mean(raising)
    std_raise = np.std(raising)
    mean_fall = np.mean(falling)
    std_fall = np.std(falling)

    e = np.array(range(N), float) / 100
    cc = np.polyfit(e, signal, 1)
    reg_slop = cc[0]

    features = [H, RH, L, RL, RD, Range, Rangen, mean, std, skewness, kurtosis, \
                q1, q2, q3, IQR, IQR_STD, reg_slop, onset, target, offset, tar_on, \
                off_on, off_tar, mean_raise, std_raise, mean_fall, \
                std_fall]

    features.extend(raising)
    features.extend(falling)
    features = np.array(features)

    where_are_nan = np.isnan(features)
    where_are_inf = np.isinf(features)
    features[where_are_nan] = 0
    features[where_are_inf] = 0

    return features


def emg_d1(signal):
    tmp = signal[1:]-signal[0:-1]
    return np.concatenate(([0],tmp))

def emg_d2(signal):
    return emg_d1(emg_d1(signal))

def emg_power_trans(signal, flag=False):
    if flag:
        N = len(signal)
        return np.sqrt(np.sum(np.abs(signal)**2)/N)
    else:
        return np.sqrt(np.sum(np.abs(signal)**2))

def emg_teo(signal):
    tmp = np.concatenate(([signal[0]], signal, [signal[-1]]))
    res = []
    for i in range(1,len(tmp)-1,1):
        res.append(tmp[i]**2-tmp[i-1]*tmp[i+1])
    return np.sum(res)

def emg_tdd(signal):
    d1 = emg_d1(signal)
    d2 = emg_d2(signal)
    m0 = emg_power_trans(signal,False)
    m2 = emg_power_trans(d1, True)
    m4 = emg_power_trans(d2, True)
    std = np.std(signal)
    mean = np.mean(signal)
    if mean == 0:
        cov = 0
    else:
        cov = np.abs(std/(mean))
    teo = emg_teo(signal)
    if np.sqrt(m0-m2)*np.sqrt(m0-m4) == 0:
        S = 0
    else:
        S = m0/(np.sqrt(m0-m2)*np.sqrt(m0-m4))
    if m0*m4 == 0:
        IF = 0
    else:
        IF = np.sqrt(m2*m2/(m0*m4))

    f1 = np.log(np.abs(m0)+EPS)
    f2 = np.log(np.abs(m0-m2)+EPS)
    f3 = np.log(np.abs(m0-m4)+EPS)
    f4 = np.log(np.abs(S)+EPS)
    f5 = np.log(np.abs(IF)+EPS)
    f6 = np.log(np.abs(cov)+EPS)
    f7 = np.log(np.abs(teo)+EPS)
    # print [f1,f2,f3,f4,f5,f6,f7]
    # print cov
    return np.array([f1,f2,f3,f4,f5,f6,f7])

def emg_tdd_cor(signal):
    a = emg_tdd(signal)
    b = emg_tdd(np.log(signal**2+EPS))
    res = (a*b)/(np.sqrt(np.sum(a**2))*np.sqrt(np.sum(b**2)))
    return res

def emg_tsd_v1(signal):
    # channel * frame
    res = []
    for i in range(signal.shape[0]):
        res.append(emg_tdd_cor(signal[i,:]))
    for i in range(signal.shape[0]):
        for j in range(i+1,signal.shape[0]):
            res.append(emg_tdd_cor(signal[i,:]-signal[j,:]))
    return np.vstack(res)

def emg_mtf(signal):
    mtf = MarkovTransitionField(image_size=400)
    print(signal.shape)
    # print(signal)
    # print(np.array(signal))
    # print(signal,signal.shape[1])
    # feature = [mtf.fit_transform([signal[:,i]]) for i in range(signal.shape[2])]
    feature = mtf.fit_transform([signal])
    print(feature.shape)
    return feature

def emg_rp(signal):
    rp = RecurrencePlot(threshold='point', percentage=20)
    feature = [rp.fit_transform([signal[:,i]]) for i in range(signal.shape[1])]
    print(np.array(feature).shape)
    return feature

def emg_gadf(signal):
    gadf = GramianAngularField(image_size=400, method='difference')
    feature = [gadf.fit_transform([signal[:,i]]) for i in range(signal.shape[1])]
    print(np.array(feature).shape)
    return feature

def emg_gasf(signal):
    gasf = GramianAngularField(image_size=400, method='summation')
    feature = [gasf.fit_transform([signal[:,i]]) for i in range(signal.shape[1])]
    print(np.array(feature).shape)
    return feature

