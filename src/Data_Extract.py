import numpy as np
import matplotlib.pyplot as plt

class data_extraction:
    def __init__(self, data_set, **kwargs):
        self.data_set=data_set
        self.sampling_rate=kwargs.get('sampling_rate', 100)
        self.amp_limit=kwargs.get('amp_limit', 0.1)
        self.low_frq_limit=kwargs.get('low_frq_limit', 10)
        self.valid_freq=[]
        self.valid_amp=[]
    
    def extract_core_feature(self):
        self.fourier_trans(self.data_set[:,3])

        max_freq=None
        max_index=1
        for i in range (1,len(self.valid_amp)):
            if(self.valid_amp[max_index]<self.valid_amp[i]):
                max_index=i
        max_freq = self.valid_freq[max_index]
        Tmid=int(1/max_freq * 50)

        mid=len(self.data_set)//2
        self.data_set=self.data_set[mid-Tmid:mid+Tmid]

        return self.extract_feature()


    def extract_feature(self):
        traditional_feature=self.stat_dt()
        fft_feature=[]

        for i in range(4):
            raw_freq, raw_amp = self.fourier_trans(self.data_set[:, i])
            fft_feature+=(self.stat_fft_amp(self.filter_amp(raw_amp)))
        return np.concatenate((fft_feature,traditional_feature))
    
    def stat_dt(self): #raw한 데이터를 전통적인 방법으로 특징 추출
        d_mean=[]
        d_std=[]
        d_max=[]
        d_min=[]
        d_range=[]
    
        for i in range(4):
            d_mean.append(np.mean(self.data_set[:, i]))
            d_std.append(np.std(self.data_set[:, i]))
            d_max.append(np.max(self.data_set[:, i]))
            d_min.append(np.min(self.data_set[:, i]))
            d_range.append(d_max[i] - d_min[i])
    
        return np.concatenate((d_mean, d_std, d_max, d_min, d_range))
    
    def fourier_trans(self, data_signal): #fft를 통해 한 축의 가속도 그래프에서 freq와 amp를 반환
        amp=np.fft.fft(data_signal)
        freq=np.fft.fftfreq(len(data_signal), d=1/self.sampling_rate)
        
    
        v_freq = (freq >= 0) & (freq <= self.low_frq_limit)
        a_amp = np.abs(amp)
        valid_amp = a_amp[v_freq]
        valid_freq = freq[v_freq]
        
        self.valid_amp=valid_amp
        self.valid_freq=valid_freq
        return valid_freq, valid_amp
    
    def filter_ampNfreq(self, freq, amp): #amp랑 freq둘다 가져갈때
    
        f_amp=(amp>=self.amp_limit)
        fil_amp=amp[f_amp]
        fil_freq=freq[f_amp]
    
        return fil_freq, fil_amp
    
    def filter_amp(self, amp): #amp만 가져갈때 아마 먼저 이것을 사용할듯
    
        f_amp=(amp>=self.amp_limit)
        fil_amp=amp[f_amp]
    
        return fil_amp
    
    def stat_fft_amp(self, filtered_amp): #filtered amp를 feature추출
        return [np.mean(filtered_amp), np.std(filtered_amp), np.max(filtered_amp), np.min(filtered_amp), np.max(filtered_amp)-np.min(filtered_amp)]
    
    def show(self):
        plt.figure() 
        plt.plot(self.valid_freq, self.valid_amp)
        plt.show()

