from __future__ import division


import numpy as np
import pandas as pd
import os
from datetime import datetime
from scipy.linalg import norm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt, resample
from scipy.signal import hilbert
from numpy.fft import rfft
from numpy import argmax, log
from scipy.signal import blackmanharris
from scipy.signal import find_peaks
from scipy.signal import welch
from sklearn.metrics import mean_squared_error
import math
from collections import defaultdict
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from numpy import array, sign, zeros
from scipy.interpolate import interp1d
from pyts.metrics import sakoe_chiba_band
from scipy.fftpack import fft
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

from tslearn.barycenters import dtw_barycenter_averaging

class dynamicEnsemble_BoostedDTW:
    def __init__(self, template_set, save_path, max_templates=1):
        self.template_set = template_set
        self.save_path = save_path
        self.max_templates = max_templates
        self.template_ID = 0
        self.align_count = 0

    def performDBA(self,series, n_iterations=10):
        n_series = len(series)
        max_length = reduce(max, map(len, series))

        cost_mat = np.zeros((max_length, max_length))
        delta_mat = np.zeros((max_length, max_length))
        path_mat = np.zeros((max_length, max_length), dtype=np.int8)

        medoid_ind = approximate_medoid_index(series, cost_mat, delta_mat)
        center = series[medoid_ind]

        for i in range(0, n_iterations):
            center = DBA_update(center, series, cost_mat, path_mat, delta_mat)

        return center

    def approximate_medoid_index(self,series, cost_mat, delta_mat):
        if len(series) <= 50:
            indices = range(0, len(series))
        else:
            indices = np.random.choice(range(0, len(series)), 50, replace=False)

        medoid_ind = -1
        best_ss = 1e20
        for index_candidate in indices:
            candidate = series[index_candidate]
            ss = sum_of_squares(candidate, series, cost_mat, delta_mat)
            if (medoid_ind == -1 or ss < best_ss):
                best_ss = ss
                medoid_ind = index_candidate
        return medoid_ind

    def sum_of_squares(self,s, series, cost_mat, delta_mat):
        return sum(map(lambda t: squared_DTW(s, t, cost_mat, delta_mat), series))

    def DTW(self,s, t, cost_mat, delta_mat):
        return np.sqrt(squared_DTW(s, t, cost_mat, delta_mat))

    def squared_DTW(self,s, t, cost_mat, delta_mat):
        s_len = len(s)
        t_len = len(t)
        length = len(s)
        fill_delta_mat_dtw(s, t, delta_mat)
        cost_mat[0, 0] = delta_mat[0, 0]
        for i in range(1, s_len):
            cost_mat[i, 0] = cost_mat[i - 1, 0] + delta_mat[i, 0]

        for j in range(1, t_len):
            cost_mat[0, j] = cost_mat[0, j - 1] + delta_mat[0, j]

        for i in range(1, s_len):
            for j in range(1, t_len):
                diag, left, top = cost_mat[i - 1, j - 1], cost_mat[i, j - 1], cost_mat[i - 1, j]
                if (diag <= left):
                    if (diag <= top):
                        res = diag
                    else:
                        res = top
                else:
                    if (left <= top):
                        res = left
                    else:
                        res = top
                cost_mat[i, j] = res + delta_mat[i, j]
        return cost_mat[s_len - 1, t_len - 1]

    def fill_delta_mat_dtw(self,center, s, delta_mat):
        slim = delta_mat[:len(center), :len(s)]
        np.subtract.outer(center, s, out=slim)
        np.square(slim, out=slim)

    def DBA_update(self,center, series, cost_mat, path_mat, delta_mat):
        options_argmin = [(-1, -1), (0, -1), (-1, 0)]
        updated_center = np.zeros(center.shape)
        n_elements = np.array(np.zeros(center.shape), dtype=int)
        center_length = len(center)
        for s in series:
            s_len = len(s)
            fill_delta_mat_dtw(center, s, delta_mat)
            cost_mat[0, 0] = delta_mat[0, 0]
            path_mat[0, 0] = -1

            for i in range(1, center_length):
                cost_mat[i, 0] = cost_mat[i - 1, 0] + delta_mat[i, 0]
                path_mat[i, 0] = 2

            for j in range(1, s_len):
                cost_mat[0, j] = cost_mat[0, j - 1] + delta_mat[0, j]
                path_mat[0, j] = 1

            for i in range(1, center_length):
                for j in range(1, s_len):
                    diag, left, top = cost_mat[i - 1, j - 1], cost_mat[i, j - 1], cost_mat[i - 1, j]
                    if (diag <= left):
                        if (diag <= top):
                            res = diag
                            path_mat[i, j] = 0
                        else:
                            res = top
                            path_mat[i, j] = 2
                    else:
                        if (left <= top):
                            res = left
                            path_mat[i, j] = 1
                        else:
                            res = top
                            path_mat[i, j] = 2

                    cost_mat[i, j] = res + delta_mat[i, j]

            i = center_length - 1
            j = s_len - 1

            while (path_mat[i, j] != -1):
                updated_center[i] += s[j]
                n_elements[i] += 1
                move = options_argmin[path_mat[i, j]]
                i += move[0]
                j += move[1]
            assert (i == 0 and j == 0)
            updated_center[i] += s[j]
            n_elements[i] += 1

        return np.divide(updated_center, n_elements)


    def parabolic(self, f, x):
        """

        This is a helper function that performs parabolic interpolation to
        help achieve high accuracy dominant frequency classification.

        Parameters
        ----------
        f : Logarithmic absolute value of output of Fourier Transform.
        x : Candidate peak.

        Returns
        -------
        xv : X-axis coordinate of predicted peak.
        yv : Y-axis coordinate of predicted peak.

        """

        xv = 1 / 2. * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
        yv = f[x] - 1 / 4. * (f[x - 1] - f[x + 1]) * (xv - x)

        return (xv, yv)

    def fft_robust(self, x, t, debug=0):
        """
        Purpose: Calculates and visualizes FFT
        Input: x as signal array, t as time of the signal array, debug =1 for plotting and 0 for pass
        Output: PSD, frequency
        Note: this function is not used, it is for self reference and can be used in future
        """

        # subtract t_ini to start from 0 - not necessary
        t_first = t[0]
        t = t - t_first
        L = len(t)
        # length correction if necessary
        if not L % 2:
            x = np.delete(x, -1)
            t = np.delete(t, -1)
            L = len(t)
        T = t[-1] / L
        Fs = 1 / T
        # define a frequency space starting from 0
        f = Fs * np.linspace(1, L // 2, L // 2) / L
        # define a frequency space starting from -Fmax
        f_true = Fs * np.linspace(-L // 2, L // 2, L) / L
        y = fft(x)
        P2 = abs(y / L)
        P1 = P2[0:L // 2]

        # calculate db
        logP1 = 20 * np.array([math.log10(x) for x in P1])

        peaks, _ = find_peaks(logP1)
        peak_amps = logP1[peaks]
        peak_freqs = f[peaks]

        top_two_peaks = peak_amps.argsort()[-2:][::-1]

        # print(peak_freqs[np.amax(top_two_peaks)]/peak_freqs[np.amin(top_two_peaks)])
        # print(peak_freqs[np.amax(top_two_peaks)])
        # print(peak_freqs[np.amin(top_two_peaks)])

        '''
        if 1.8 < (peak_freqs[np.amax(top_two_peaks)]/peak_freqs[np.amin(top_two_peaks)]) < 2.35:
            dominant_freq = peak_freqs[np.amin(top_two_peaks)]
        else:
            dominant_freq = peak_freqs[top_two_peaks[0]]
        '''

        dominant_freq = peak_freqs[top_two_peaks[0]]

        # print(dominant_freq)

        if debug:  # #plotting - NOT USED
            # pass
            plt.figure()
            plt.plot(f, logP1)
            plt.scatter(f[peaks], logP1[peaks], color='red')
            plt.title('FFT')
            plt.ylabel('Amplitude [db]')
            plt.xlabel('Frequency [Hz]')
            plt.axis('tight')
            # mng = plt.get_current_fig_manager()
            # mng.frame.Maximize(True)
            plt.show()
            plt.savefig("mygraph.png")
        return dominant_freq

    def get_minPeakDist(self, bvp_part):
        """

        This function determines the minimum cardiac cycle constraint based
        on average heart rate for the segment.

        Parameters
        ----------
        bvp_seg : Clean BVP segment.

        Returns
        -------
        minPeakDist : Minimum cardiac cycle constraint.

        """

        # Estimate average heart rate based on dominant frequency
        # Estimate average cycle length
        # Minimum cycle length constraint is taken as 0.7 of the average

        time_dummy = np.linspace(1 // 125, len(bvp_part[:, 0]) // 125, len(bvp_part[:, 0]))

        dominant_freq = self.fft_robust(bvp_part[:, 1], time_dummy)

        avg_len_cycle = 125 / dominant_freq

        return avg_len_cycle

    def bisect_right(self, a, x, lo=0, hi=None):
        """Return the index where to insert item x in list a, assuming a is sorted.
        The return value i is such that all e in a[:i] have e <= x, and all e in
        a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
        insert just after the rightmost x already there.
        Optional args lo (default 0) and hi (default len(a)) bound the
        slice of a to be searched.
        """

        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if x < a[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def findCandidateEndpoints(self, bvp_seg):
        zero_crossings_1 = np.where(np.diff(np.sign(bvp_seg[:, 2])) == 2)[0]
        zero_crossings_2 = np.where(np.diff(np.sign(bvp_seg[:, 3])))[0]
        '''
        fig,axs = plt.subplots(2, dpi=350, sharex=True)
        #plt.figure(dpi=350)
        axs[0].plot(bvp_seg[:,2])
        #xmin, xmax, ymin, ymax = plt.axis()
        for i in zero_crossings_1:
            axs[0].axvline(i)
        axs[0].axhline(0)
        axs[1].plot(bvp_seg[:,3])
        for j in zero_crossings_2:
            axs[1].axvline(j)
        axs[1].axhline(0)
        plt.show()
        exit()
        '''
        seg_mean = np.mean(bvp_seg[:, 1])
        scaler = MinMaxScaler()
        grad_scores = scaler.fit_transform(np.reshape(bvp_seg[:, 2], (len(bvp_seg), 1)))
        candidate_endpoints = np.zeros((len(bvp_seg[:, 1]), 1))
        candidate_grads_idx, candidate_grads = list(), list()
        for z in zero_crossings_1:
            if bvp_seg[z, 1] < seg_mean:
                idx_gradient = self.bisect_right(zero_crossings_2, z)
                if idx_gradient >= len(zero_crossings_2):
                    candidate_grads_idx.append(z)
                    candidate_grads.append(grad_scores[-1])
                else:
                    candidate_grads_idx.append(z)
                    candidate_grads.append(grad_scores[zero_crossings_2[idx_gradient]])
        candidate_endpoints[candidate_grads_idx] = candidate_grads
        '''
        fig,axs = plt.subplots(4, dpi=350, sharex=True)
        axs[0].plot(bvp_seg[:,1])
        axs[1].plot(candidate_endpoints)
        axs[2].plot(bvp_seg[:,2])
        axs[3].plot(bvp_seg[:,3])
        plt.show()

        print('check check boi boi')
        exit()
        '''

        return candidate_endpoints

    def update_DTW(self, step, d_mat, init_flag, len_template):
        """
        Update the DTW distance and starting point matrices sequentially.

        Parameters
        ----------
        step : Input step of the stream (matrix column) to update.

        Returns
        -------
        None. DTW matrices are global class variables.

        """

        if init_flag == True:
            for t in range(0, len_template):
                if t == 0:
                    continue
                else:
                    d_mat[t, step] += d_mat[t - 1, step]
            init_flag = False
        else:
            for t in range(0, len_template):
                if t == 0:
                    continue
                else:
                    d_best = [d_mat[t - 1, step], d_mat[t, step - 1], d_mat[t - 1, step - 1]]
                    d_best_idx = d_best.index(min(d_best))
                    d_mat[t, step] += min(d_best)

        return d_mat, init_flag

    def map_pts(self, bvp_part, d_mat, template_sys, template_dicr, template_dia, template,debug=True):
        window_size = 0.35
        #window_size = 0.99
        region = sakoe_chiba_band(d_mat.shape[1], d_mat.shape[0], window_size)
        mask = np.full((d_mat.shape[0], d_mat.shape[1]), np.inf)
        # print(mask)
        # exit()
        # exit()
        for i, (j, k) in enumerate(region.T):
            mask[j:k, i] = 1.
        # plt.matshow(mask)
        # exit()
        new_d_mat = np.multiply(d_mat, mask)
        # print(new_d_mat)
        # plt.matshow(new_d_mat)
        # exit()
        i = new_d_mat.shape[0] - 1
        j = new_d_mat.shape[1] - 1
        alignment_dict = {}
        sys_list, dicr_list, dia_list = list(), list(), list()
        while ((i != 0) and (j != 0)):
            d_best = [new_d_mat[i - 1, j], new_d_mat[i, j - 1], new_d_mat[i - 1, j - 1]]
            if min(d_best) == new_d_mat[i - 1, j]:
                i = i - 1
                j = j

                if i == template_sys:
                    sys_list.append(j)
                if i == template_dicr:
                    dicr_list.append(j)
                if i == template_dia:
                    dia_list.append(j)

            elif min(d_best) == new_d_mat[i, j - 1]:
                i = i
                j = j - 1

                if i == template_sys:
                    sys_list.append(j)
                if i == template_dicr:
                    dicr_list.append(j)
                if i == template_dia:
                    dia_list.append(j)

            elif min(d_best) == new_d_mat[i - 1, j - 1]:
                i = i - 1
                j = j - 1

                if i == template_sys:
                    sys_list.append(j)
                if i == template_dicr:
                    dicr_list.append(j)
                if i == template_dia:
                    dia_list.append(j)

            alignment_dict.update({(i, j): min(d_best)})

        sys = None
        dicr = None
        dia = None
        sys_time = None
        dicr_time = None
        dia_time = None
        if len(sys_list) != 0:
            #sys = int(sum(sys_list) / len(sys_list))
            sub_bvp = bvp_part[sys_list, 1]
            #sub_bvp = np.absolute(sub_bvp)
            sys = sys_list[np.argmax(sub_bvp)]
            sys_time = bvp_part[int(sys), 0]

        if len(dicr_list) != 0:
            sub_bvp = bvp_part[dicr_list,2]
            dicr = dicr_list[np.argmax(sub_bvp)]
            #dicr = int(sum(dicr_list) / len(dicr_list))
            dicr_time = bvp_part[int(dicr), 0]

        if len(dia_list) != 0:
            dia = int(sum(dia_list) / len(dia_list))
            dia_time = bvp_part[int(dia), 0]


        if debug == False:
            #if not os.path.exists(self.save_path + 'alignments/'):
            #    os.makedirs(self.save_path + 'alignments/')
            #print(alignment_dict)
            #exit()
            plt.figure()
            plt.plot(template[:,1])
            plt.show()
            
            template_show = np.add(template[:,1],0)
            #template_show = template[:,0]
            plt.figure(dpi=350)
            plt.plot(template_show, color='red', label='template',linewidth=4)
            plt.plot(bvp_part[:,2], color='blue', label='segment',linewidth=4)
            for p in alignment_dict:
                plt.plot([p[0],p[1]],[template_show[p[0]],bvp_part[p[1],2]], color='black')
            #plt.scatter([template_sys,template_dicr,template_dia],[template_show[template_sys],template_show[template_dicr],template_show[template_dia]],color='green')
            #plt.scatter([sys,dicr,dia],[bvp_part[sys,1],bvp_part[dicr,1],bvp_part[dia,1]],color='green')
            plt.legend()
            plt.show()
            exit()
            #plt.savefig(self.save_path + 'alignments/' + str(self.align_count) + '.png')
            #plt.close()


            plt.figure(dpi=350)

            warp_mat = np.full((d_mat.shape[0],d_mat.shape[1]), np.inf)

            for i, (j, k) in enumerate(region.T):
                warp_mat[j:k, i] = 0.

            for p in alignment_dict:
                warp_mat[p] = 1
            plt.matshow(warp_mat,fignum=2)
            plt.show()
            exit()



        return alignment_dict, sys, dicr, dia, sys_time, dicr_time, dia_time

    def updateEnsemble(self, bvp_seg, df_subs, partStart, avg_len_cycle):
        scaler = StandardScaler()
        new_temp_dict = defaultdict(dict)
        self.template_ID += 1
        tempTemplate = self.template_set[0]['seg']

        segs_list = []
        for idx, row in df_subs.iterrows():
            tmp_seg = bvp_seg[int(row['step_start'] - partStart):int(row['step_end'] - partStart + 1), 1:]
            tmp_seg = scaler.fit_transform(tmp_seg)
            segs_list.append(tmp_seg[:,0])

        new_temp = dtw_barycenter_averaging(segs_list, max_iter=1, barycenter_size=int(avg_len_cycle))

        new_temp = new_temp.flatten()

        b, a = butter(4, 8, btype='lowpass', fs=125)
        #b, a = butter(4, [0.5, 5], btype='band', fs=125)
        new_temp = filtfilt(b, a, new_temp)

        temp_df = pd.DataFrame(new_temp)

        temp_df.rename(columns={temp_df.columns[0]: 'new_temp'}, inplace=True)
        temp_df['new_temp_1'] = np.gradient(temp_df['new_temp'].values)
        temp_df['new_temp_2'] = np.gradient(temp_df['new_temp_1'].values)
        temp_df.dropna(inplace=True)
        smooth_temp = scaler.fit_transform(temp_df.values)


        upsamp_temp = resample(self.template_set[0]['seg'][:, 0],int(avg_len_cycle))
        b, a = butter(4, 8, btype='lowpass', fs=125)
        upsamp_temp = filtfilt(b, a, upsamp_temp)

        up_temp_df = pd.DataFrame(upsamp_temp)

        up_temp_df.rename(columns={up_temp_df.columns[0]: 'new_temp'}, inplace=True)
        up_temp_df['new_temp_1'] = np.gradient(up_temp_df['new_temp'].values)
        up_temp_df['new_temp_2'] = np.gradient(up_temp_df['new_temp_1'].values)
        up_temp_df.dropna(inplace=True)
        smooth_up_temp = scaler.fit_transform(up_temp_df.values)

        dist_new, path_new = fastdtw(self.template_set[0]['seg'][:, 1], smooth_up_temp[:, 1], radius=1, dist=euclidean)

        sys_list, dicr_list, dia_list = list(), list(), list()
        for p in path_new:
            if p[0] == self.template_set[0]['sys']:
                sys_list.append(p[1])
            elif p[0] == self.template_set[0]['dicr']:
                dicr_list.append(p[1])
            elif p[0] == self.template_set[0]['dia']:
                dia_list.append(p[1])




        sub_bvp = smooth_up_temp[sys_list, 0]
        up_temp_sys = sys_list[np.argmax(sub_bvp)]

        sub_bvp = smooth_up_temp[dicr_list,1]
        up_temp_dicr = dicr_list[np.argmax(sub_bvp)]

        up_temp_dia = int(sum(dia_list) / len(dia_list))

       

        dist_new, path_new = fastdtw(smooth_up_temp[:, 1], smooth_temp[:, 1], radius=1, dist=euclidean)



        if len(self.template_set) < self.max_templates:
            new_key = len(self.template_set)
            self.template_set[new_key]['seg'] = smooth_temp
            sys_list, dicr_list, dia_list = list(), list(), list()
            for p in path_new:
                if p[0] == up_temp_sys:
                    sys_list.append(p[1])
                elif p[0] == up_temp_dicr:
                    dicr_list.append(p[1])
                elif p[0] == up_temp_dia:
                    dia_list.append(p[1])

            sub_bvp = smooth_temp[sys_list, 0]
            #sub_bvp = np.absolute(sub_bvp)
            sys = sys_list[np.argmax(sub_bvp)]
            sub_bvp = smooth_temp[dicr_list, 1]
            dicr = dicr_list[np.argmax(sub_bvp)]
            dia = int(sum(dia_list) / len(dia_list))
            self.template_set[new_key]['sys'] = sys
            self.template_set[new_key]['dicr'] = dicr
            self.template_set[new_key]['dia'] = dia
            self.template_set[new_key]['frequency'] = 0
            self.template_set[new_key]['template_ID'] = self.template_ID

        else:
            lru_temp = 1
            for temp in range(1, len(self.template_set)):
                if self.template_set[temp]['frequency'] < self.template_set[lru_temp]['frequency']:
                    lru_temp = temp
            self.template_set[lru_temp]['seg'] = smooth_temp
            sys_list, dicr_list, dia_list = list(), list(), list()
            for p in path_new:
                if p[0] == up_temp_sys:
                    sys_list.append(p[1])
                elif p[0] == up_temp_dicr:
                    dicr_list.append(p[1])
                elif p[0] == up_temp_dia:
                    dia_list.append(p[1])

            sub_bvp = smooth_temp[sys_list, 0]
            #sub_bvp = np.absolute(sub_bvp)
            sys = sys_list[np.argmax(sub_bvp)]
            sub_bvp = smooth_temp[dicr_list, 1]
            dicr = dicr_list[np.argmax(sub_bvp)]
            dia = int(sum(dia_list) / len(dia_list))
            self.template_set[lru_temp]['sys'] = sys
            self.template_set[lru_temp]['dicr'] = dicr
            self.template_set[lru_temp]['dia'] = dia
            self.template_set[lru_temp]['frequency'] = 0
            self.template_set[lru_temp]['template_ID'] = self.template_ID


        self.align_count += 1

    def ensemble_alg(self, bvp_part, partStart, partEnd, start_trunc, end_trunc, df_subsequences,
                     first_pass=True, skip_res=False):
        # bvp_part = np.copy(bvp_part_orig)
        avg_len_cycle = self.get_minPeakDist(bvp_part)
        p_avgCycleLengths = np.repeat(avg_len_cycle, len(bvp_part))
        p_avgCycleLengths = np.reshape(p_avgCycleLengths, (len(p_avgCycleLengths), 1))

        # scaler = StandardScaler()
        # bvp_part[:,1:] = scaler.fit_transform(bvp_part[:,1:])
        p_candidateEndpoints = self.findCandidateEndpoints(bvp_part)
        init_start = np.argmax(p_candidateEndpoints > 0)

        df_results_tmps = defaultdict(dict)
        for template_count in range(0, len(self.template_set)):
            if (template_count == 0) and (first_pass == False):
                df_results_tmps[template_count]['d_mins'] = float('inf')
                continue
            self.template_set[template_count]['d_mat_tmp'] = np.copy(
                self.template_set[template_count]['d_mat_absolute'])

            df_results_tmps[template_count]['df_subs'] = pd.DataFrame(
                columns=['start_trunc', 'end_trunc', 't_start', 't_end', 'step_start', 'step_end', 'template_ID',
                         'alignment', 'sys_step', 'dicr_step', 'dia_step', 'sys_time', 'dicr_time', 'dia_time'])
            
            
            df_results_tmps[template_count]['d_min'] = float('inf')

            init_flag = True
            # d_min = float('inf')
            start_cycle = (0, 0)
            end_cycle = (0, 0)
            step_track = 0
            step = init_start
            df_results_tmps[template_count]['p_endPoints'] = np.zeros((len(bvp_part), 1))
            df_results_tmps[template_count]['p_DTWs'] = np.zeros((len(bvp_part), 1))
            df_results_tmps[template_count]['d_lasts'] = np.zeros((len(bvp_part), 1))
            df_results_tmps[template_count]['d_mins'] = list()
            while (step < (len(bvp_part) - 1)):
                self.template_set[template_count]['d_mat_tmp'], init_flag = self.update_DTW(step, self.template_set[
                    template_count]['d_mat_tmp'], init_flag, len(self.template_set[template_count]['seg']))
                if self.template_set[template_count]['d_mat_tmp'][-1, step] < df_results_tmps[template_count]['d_min']:
                    df_results_tmps[template_count]['d_min'] = self.template_set[template_count]['d_mat_tmp'][-1, step]
                p_DTW = np.exp(-(1 / 5000) * df_results_tmps[template_count]['d_min'])
                p_endPoint = p_candidateEndpoints[step] * p_DTW
                df_results_tmps[template_count]['p_endPoints'][step, 0] = p_endPoint
                step_track += 1
                df_results_tmps[template_count]['p_DTWs'][step, 0] = p_DTW
                df_results_tmps[template_count]['d_lasts'][step, 0] = self.template_set[template_count]['d_mat_tmp'][
                    -1, step]

                if (step_track < 0.7 * avg_len_cycle):
                    if (p_endPoint > start_cycle[0]):
                        start_cycle = (p_endPoint, step)
                        end_cycle = (0, step)
                        step_track = 1
                        step += 1
                    else:
                        step += 1
                elif 0.7 * avg_len_cycle < step_track < 1.3 * avg_len_cycle:
                    if p_endPoint > end_cycle[0]:
                        end_cycle = (p_endPoint, step)
                        step += 1
                    else:
                        step += 1
                else:
                    if start_cycle[1] == end_cycle[1]:
                        self.template_set[template_count]['d_mat_tmp'] = np.copy(
                            self.template_set[template_count]['d_mat_absolute'])
                        init_flag = True
                        df_results_tmps[template_count]['d_min'] = float('inf')
                        step = step
                        start_cycle = (0, step)
                        end_cycle = (0, step)
                        step_track = 0
                    else:
                        df_results_tmps[template_count]['d_mins'].append(df_results_tmps[template_count]['d_min'])
                        alignment_dict, sys_step, dicr_step, dia_step, sys_time, dicr_time, dia_time = self.map_pts(
                            bvp_part[start_cycle[1]:end_cycle[1] + 1],
                            self.template_set[template_count]['d_mat_tmp'][:, start_cycle[1]:end_cycle[1] + 1],
                            self.template_set[template_count]['sys'], self.template_set[template_count]['dicr'],
                            self.template_set[template_count]['dia'], self.template_set[template_count]['seg'])
                        df_results_tmps[template_count]['df_subs'] = df_results_tmps[template_count]['df_subs'].append(
                            {'start_trunc': start_trunc, 'end_trunc': end_trunc, 't_start': bvp_part[start_cycle[1], 0],
                             't_end': bvp_part[end_cycle[1], 0], 'step_start': partStart + start_cycle[1],
                             'step_end': partStart + end_cycle[1], 'template_ID': self.template_set[template_count]['template_ID'],
                             'alignment': alignment_dict, 'sys_step': sys_step, 'dicr_step': dicr_step,
                             'dia_step': dia_step, 'sys_time': sys_time, 'dicr_time': dicr_time, 'dia_time': dia_time},
                            ignore_index=True)
                        
                        self.template_set[template_count]['d_mat_tmp'] = np.copy(
                            self.template_set[template_count]['d_mat_absolute'])
                        init_flag = True
                        df_results_tmps[template_count]['d_min'] = float('inf')
                        step = end_cycle[1]
                        start_cycle = (0, end_cycle[1])
                        end_cycle = (0, end_cycle[1])
                        step_track = 0

            df_results_tmps[template_count]['d_mins'] = sum(df_results_tmps[template_count]['d_mins']) / len(
                df_results_tmps[template_count]['d_mins'])
        min_dmin = float('inf')
        min_temp = 0
        for template_count in range(0, len(self.template_set)):
            if df_results_tmps[template_count]['d_mins'] < min_dmin:
                min_dmin = df_results_tmps[template_count]['d_mins']
                min_temp = template_count

        if (self.max_templates > 1) and (first_pass == True):
            if (min_temp == 0) or (len(self.template_set) < self.max_templates):
                self.updateEnsemble(bvp_part, df_results_tmps[min_temp]['df_subs'], partStart, avg_len_cycle)
                self.generate_dmat(bvp_part)
                df_subsequences = self.ensemble_alg(bvp_part, partStart, partEnd, start_trunc,
                                                                      end_trunc, df_subsequences,
                                                                      first_pass=False)
                skip_res = True
        if skip_res == False:
            
            df_subsequences = df_subsequences.append(df_results_tmps[min_temp]['df_subs'], ignore_index=True)
            self.template_set[min_temp]['frequency'] += 1

        # if self.max_templates > 1:
        #    if (min_temp == 0) or (len(self.template_set) < self.max_templates):
        #        self.updateEnsemble(bvp_part, df_results_tmps[min_temp]['df_subs'], partStart)

        return df_subsequences

    def generate_dmat(self, bvp_seg):
        for template_count in range(0, len(self.template_set)):

            dif_list = list()
            # for i in range(1, self.template_set[template_count]['seg'].shape[1]):
            for i in range(1, 2):
                
                template_mat = np.reshape(self.template_set[template_count]['seg'][:, i],
                                          (len(self.template_set[template_count]['seg'][:, i]), 1)).astype(float)
                template_mat = np.repeat(template_mat, repeats=len(bvp_seg), axis=1)

                bvp_mat = np.reshape(bvp_seg[:, (i + 1)], (1, len(bvp_seg[:, (i + 1)]))).astype(float)
                bvp_mat = np.repeat(bvp_mat, repeats=len(self.template_set[template_count]['seg']), axis=0)

                dif_list.append(np.square(np.subtract(template_mat, bvp_mat)))
            d_mat = sum(dif_list)
            d_mat = np.sqrt(d_mat)
            self.template_set[template_count]['d_mat_absolute'] = d_mat

    def runRobustSpringDTW(self, bvp_seg, start_seg, end_seg, start_trunc, end_trunc, df_subsequences):
        scaler = StandardScaler()
        bvp_seg[:, 1:] = scaler.fit_transform(bvp_seg[:, 1:])

        self.generate_dmat(bvp_seg)

        df_subsequences = self.ensemble_alg(bvp_seg, start_seg, end_seg, start_trunc, end_trunc,
                                                              df_subsequences)

        return df_subsequences

    def extractFeatures(self, bvp):
        df_subsequences = pd.DataFrame(
            columns=['start_trunc', 'end_trunc', 't_start', 't_end', 'step_start', 'step_end', 'template_ID',
                     'alignment', 'sys_step', 'dicr_step', 'dia_step', 'sys_time', 'dicr_time', 'dia_time'])
        
        # Iterate through clean segments to extract features
        #print('Starting loop...')
        start_seg = 0
        len_seg = 125 * 5
        end_seg = start_seg + len_seg
        bvp.reset_index(drop=False, inplace=True)
        bvp.rename(columns={bvp.columns[0]: 'time'})
        bvp = np.copy(bvp.values)
        while end_seg <= len(bvp):
            
            
            bvp_seg = np.copy(bvp[start_seg:end_seg])

            try:
                df_subsequences = self.runRobustSpringDTW(bvp_seg, start_seg, end_seg, bvp_seg[0, 0],
                                                                        bvp_seg[0, -1], df_subsequences)
            except:
                pass 

            start_seg = end_seg
            end_seg = start_seg + len_seg
            if (end_seg - start_seg) < len_seg:
                break

        
        return df_subsequences
