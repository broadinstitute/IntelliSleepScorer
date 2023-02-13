import numpy as np
import pandas as pd
from mne.io import read_raw_edf
import shap
import sqlite3
import os
import pickle


def printW (string):
    f = open("log", "a")
    f.write(str(string) + '\n\n')
    f.close()


def cut_epochs (full_single_trace, sfreq, epoch_len = 10):
    epoch_length = (epoch_len * sfreq)
    n_epochs = len(full_single_trace) // epoch_length
    output = np.zeros((n_epochs,epoch_length))
    for index in range(n_epochs):
        data_this_epoch = np.array([
            full_single_trace[index*epoch_length: (index+1)*epoch_length]
        ])
        output[index] = data_this_epoch
    return output


def get_epoch_abs_mean (full_single_trace, sfreq, epoch_len = 10):
    epoch_length = (epoch_len * sfreq)
    n_epochs = len(full_single_trace) // epoch_length
    output = np.zeros(n_epochs)
    for index in range(n_epochs):
        data_this_epoch = np.array(
            full_single_trace[index*epoch_length: (index+1)*epoch_length]
        )
        output[index] = np.abs(data_this_epoch).mean()
    return output


def get_epoch_abs_median (full_single_trace, sfreq, epoch_len = 10):
    epoch_length = (epoch_len * sfreq)
    n_epochs = len(full_single_trace) // epoch_length
    output = np.zeros(n_epochs)
    for index in range(n_epochs):
        data_this_epoch = np.array(
            full_single_trace[index*epoch_length: (index+1)*epoch_length]
        )
        output[index] = np.median(np.abs(data_this_epoch))
    return output


def get_epoch_abs_max (full_single_trace, sfreq, epoch_len = 10):
    epoch_length = (epoch_len * sfreq)
    n_epochs = len(full_single_trace) // epoch_length
    output = np.zeros(n_epochs)
    for index in range(n_epochs):
        data_this_epoch = np.array(
            full_single_trace[index*epoch_length: (index+1)*epoch_length]
        )
        output[index] = np.abs(data_this_epoch).max()
    return output


def get_epoch_abs_std (full_single_trace, sfreq, epoch_len = 10):
    epoch_length = (epoch_len * sfreq)
    n_epochs = len(full_single_trace) // epoch_length
    output = np.zeros(n_epochs)
    for index in range(n_epochs):
        data_this_epoch = np.array(
            full_single_trace[index*epoch_length: (index+1)*epoch_length]
        )
        output[index] = np.abs(data_this_epoch).std()
    return output


def get_epoch_psd (full_single_trace, sfreq, epoch_len = 10):
    epoch_length = (epoch_len * sfreq)
    n_epochs = len(full_single_trace) // epoch_length
    output = np.zeros((6, n_epochs))
    for index in range(n_epochs):
        data_this_epoch = np.array(
            full_single_trace[index*epoch_length: (index+1)*epoch_length]
        )
        psd = np.abs(np.fft.fft(data_this_epoch))**2
        time_step = 1 / sfreq

        freqs = np.fft.fftfreq(data_this_epoch.size, time_step)
        idx = np.argsort(freqs)
        freqs = freqs[idx]
        psd = psd[idx]
        
        delta = np.sum(psd[np.logical_and(freqs >= 1, freqs <= 4)])
        theta = np.sum(psd[np.logical_and(freqs >= 4, freqs <= 8)])
        alpha = np.sum(psd[np.logical_and(freqs >= 8, freqs <= 12)])
        sigma = np.sum(psd[np.logical_and(freqs >= 12, freqs <= 15)])
        beta  = np.sum(psd[np.logical_and(freqs >= 15, freqs <= 30)])
        gamma = np.sum(psd[freqs >= 30])

        output[0,index] = delta
        output[1,index] = theta
        output[2,index] = alpha
        output[3,index] = sigma
        output[4,index] = beta
        output[5,index] = gamma
    return output


def rmsValue(arr): 
    square = 0
    mean = 0.0
    root = 0.0
    n = len(arr)
    #Calculate square 
    for i in range(0,n): 
        square += (arr[i]**2) 
      
    #Calculate Mean  
    mean = (square / (float)(n)) 
      
    #Calculate Root 
    root = mean**0.5 
      
    return root 


def get_epoch_rms (full_single_trace, sfreq, epoch_len = 10):
    epoch_length = (epoch_len * sfreq)
    n_epochs = len(full_single_trace) // epoch_length
    output = np.zeros(n_epochs)
    for index in range(n_epochs):
        data_this_epoch = np.array(
            full_single_trace[index*epoch_length: (index+1)*epoch_length]
        )
        output[index] = rmsValue(data_this_epoch)
    return output


def remove_channel_outlier(df_input, channel):
    df = df_input.copy()
    stats = df.groupby(['subject_id'])[channel].agg(['mean', 'count', 'std'])
    
    ci_hi = []
    ci_lo = []
    
    for i in stats.index:
        m, c, s = stats.loc[i]
        ci_hi.append(m + 3*s)
        ci_lo.append(m - 3*s)
    
    stats['ci_hi'] = ci_hi
    stats['ci_lo'] = ci_lo
    
    df['ci_hi'] = df['subject_id'].map(stats['ci_hi'])
    df['ci_lo'] = df['subject_id'].map(stats['ci_lo'])
    
    df.loc[df[channel] > df['ci_hi'], channel] = np.nan
    df.loc[df[channel] < df['ci_lo'], channel] = np.nan

    # print(df['ci_hi'])
    # print(df['ci_lo'])
    
    # print(df.shape)
    # print(df[channel].isna().sum())
    
    return df


def remove_outlier(df_input):
    df = remove_channel_outlier(df_input, 'eeg1_rms')
    df = remove_channel_outlier(df, 'eeg1_rms')
    
    df = remove_channel_outlier(df_input, 'eeg2_rms')
    df = remove_channel_outlier(df, 'eeg2_rms')

    df = remove_channel_outlier(df_input, 'emg_rms')
    df = remove_channel_outlier(df, 'emg_rms')
    
    df = df[~df['eeg1_rms'].isna() & ~df['eeg2_rms'].isna() & ~df['emg_rms'].isna()]
    
    return df



def save_single_edf_to_csv_2eeg(edf_filepath=None, epoch_len=10, model_name=None, test_run=False, include_score=False):
    file_firstname = edf_filepath.split("/")[-1].split('.edf')[0]
    edf_folderpath = edf_filepath.split(file_firstname)[0]
    
    if include_score:
        db3_filepath = edf_path + file_firstname + ".db3"
        connection = sqlite3.connect(db3_filepath)
        score_list = pd.read_sql_query(f"SELECT * from sleep_scores_table", connection)['score'].values
        print("score_list_shape", score_list.shape)
    else:
        score_list = np.nan

    # Save a downsampled data file
    downsampled_files = f"{edf_folderpath}{file_firstname}_{model_name}_rs_100hz.npy"
    if not os.path.exists(downsampled_files):
        raw = read_raw_edf(edf_filepath, preload=True)
        sfreq = int(raw.info["sfreq"])
        print(f"sfreq:{sfreq}")
        raw.resample(sfreq=100)
        resampled_data = raw.get_data()
        with open(downsampled_files, 'wb') as f:
            np.save(f, resampled_data)

    raw = read_raw_edf(edf_filepath, preload=True)
    raw_data_length = raw.get_data().shape[1]
    sfreq = int(raw.info["sfreq"])

    if include_score:
        print("raw_data_shape", raw_data_length)
        if raw_data_length != len(score_list) * 10 * sfreq:
            print("score list and raw data have different number of epochs")
            return
    if test_run:
        return

    raw.filter(1., 40., fir_design='firwin')
    raw_data = raw.get_data()
    
    df = pd.DataFrame()
    
    print("get basic features")
    df['eeg1_abs_mean'] = get_epoch_abs_mean(raw_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_abs_mean'] = get_epoch_abs_mean(raw_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['emg_abs_mean'] = get_epoch_abs_mean(raw_data[2], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_abs_median'] = get_epoch_abs_median(raw_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_abs_median'] = get_epoch_abs_median(raw_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['emg_abs_median'] = get_epoch_abs_median(raw_data[2], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_abs_max'] = get_epoch_abs_max(raw_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_abs_max'] = get_epoch_abs_max(raw_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['emg_abs_max'] = get_epoch_abs_max(raw_data[2], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_abs_std'] = get_epoch_abs_std(raw_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_abs_std'] = get_epoch_abs_std(raw_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['emg_abs_std'] = get_epoch_abs_std(raw_data[2], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_rms'] = get_epoch_rms(raw_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_rms'] = get_epoch_rms(raw_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['emg_rms'] = get_epoch_rms(raw_data[2], sfreq=sfreq, epoch_len=epoch_len)
    
    df['eeg1_abs_mean_n'] = df['eeg1_abs_mean']/df['eeg1_abs_mean'].median()
    df['eeg2_abs_mean_n'] = df['eeg2_abs_mean']/df['eeg2_abs_mean'].median()
    df['emg_abs_mean_n'] = df['emg_abs_mean']/df['emg_abs_mean'].median()
    df['eeg1_abs_median_n'] = df['eeg1_abs_median']/df['eeg1_abs_median'].median()
    df['eeg2_abs_median_n'] = df['eeg2_abs_median']/df['eeg2_abs_median'].median()
    df['emg_abs_median_n'] = df['emg_abs_median']/df['emg_abs_median'].median()
    df['eeg1_abs_max_n'] = df['eeg1_abs_max']/df['eeg1_abs_max'].median()
    df['eeg2_abs_max_n'] = df['eeg2_abs_max']/df['eeg2_abs_max'].median()
    df['emg_abs_max_n'] = df['emg_abs_max']/df['emg_abs_max'].median()
    df['eeg1_abs_std_n'] = df['eeg1_abs_std']/df['eeg1_abs_std'].median()
    df['eeg2_abs_std_n'] = df['eeg2_abs_std']/df['eeg2_abs_std'].median()
    df['emg_abs_std_n'] = df['emg_abs_std']/df['emg_abs_std'].median()
    df['eeg1_rms_n'] = df['eeg1_rms']/df['eeg1_rms'].median()
    df['eeg2_rms_n'] = df['eeg2_rms']/df['eeg2_rms'].median()
    df['emg_rms_n'] = df['emg_rms']/df['emg_rms'].median()
    
    df['eeg1_abs_mean_n2'] = df['eeg1_abs_mean']/df['eeg1_abs_mean'].mean()
    df['eeg2_abs_mean_n2'] = df['eeg2_abs_mean']/df['eeg2_abs_mean'].mean()
    df['emg_abs_mean_n2'] = df['emg_abs_mean']/df['emg_abs_mean'].mean()
    df['eeg1_abs_median_n2'] = df['eeg1_abs_median']/df['eeg1_abs_median'].mean()
    df['eeg2_abs_median_n2'] = df['eeg2_abs_median']/df['eeg2_abs_median'].mean()
    df['emg_abs_median_n2'] = df['emg_abs_median']/df['emg_abs_median'].mean()
    df['eeg1_abs_max_n2'] = df['eeg1_abs_max']/df['eeg1_abs_max'].mean()
    df['eeg2_abs_max_n2'] = df['eeg2_abs_max']/df['eeg2_abs_max'].mean()
    df['emg_abs_max_n2'] = df['emg_abs_max']/df['emg_abs_max'].mean()
    df['eeg1_abs_std_n2'] = df['eeg1_abs_std']/df['eeg1_abs_std'].mean()
    df['eeg2_abs_std_n2'] = df['eeg2_abs_std']/df['eeg2_abs_std'].mean()
    df['emg_abs_std_n2'] = df['emg_abs_std']/df['emg_abs_std'].mean()
    df['eeg1_rms_n2'] = df['eeg1_rms']/df['eeg1_rms'].mean()
    df['eeg2_rms_n2'] = df['eeg2_rms']/df['eeg2_rms'].mean()
    df['emg_rms_n2'] = df['emg_rms']/df['emg_rms'].mean()
    
    # PSD
    print("get psd features")
    eeg1_psd = get_epoch_psd(raw_data[0], sfreq=sfreq, epoch_len=epoch_len)
    eeg2_psd = get_epoch_psd(raw_data[1], sfreq=sfreq, epoch_len=epoch_len)
    
    df['eeg1_delta'] = eeg1_psd[0]
    df['eeg1_theta'] = eeg1_psd[1]
    df['eeg1_alpha'] = eeg1_psd[2]
    df['eeg1_sigma'] = eeg1_psd[3]
    df['eeg1_beta'] = eeg1_psd[4]
    df['eeg1_gamma'] = eeg1_psd[5]
    df['eeg2_delta'] = eeg2_psd[0]
    df['eeg2_theta'] = eeg2_psd[1]
    df['eeg2_alpha'] = eeg2_psd[2]
    df['eeg2_sigma'] = eeg2_psd[3]
    df['eeg2_beta'] = eeg2_psd[4]
    df['eeg2_gamma'] = eeg2_psd[5]
    
    df['eeg1_delta_n'] = df['eeg1_delta']/df['eeg1_delta'].median()
    df['eeg1_theta_n'] = df['eeg1_theta']/df['eeg1_theta'].median()
    df['eeg1_alpha_n'] = df['eeg1_alpha']/df['eeg1_alpha'].median()
    df['eeg1_sigma_n'] = df['eeg1_sigma']/df['eeg1_sigma'].median()
    df['eeg1_beta_n'] = df['eeg1_beta']/df['eeg1_beta'].median()
    df['eeg1_gamma_n'] = df['eeg1_gamma']/df['eeg1_gamma'].median()
    df['eeg2_delta_n'] = df['eeg2_delta']/df['eeg2_delta'].median()
    df['eeg2_theta_n'] = df['eeg2_theta']/df['eeg2_theta'].median()
    df['eeg2_alpha_n'] = df['eeg2_alpha']/df['eeg2_alpha'].median()
    df['eeg2_sigma_n'] = df['eeg2_sigma']/df['eeg2_sigma'].median()
    df['eeg2_beta_n'] = df['eeg2_beta']/df['eeg2_beta'].median()
    df['eeg2_gamma_n'] = df['eeg2_gamma']/df['eeg2_gamma'].median()
    
    df['eeg1_delta_n2'] = df['eeg1_delta']/df['eeg1_delta'].mean()
    df['eeg1_theta_n2'] = df['eeg1_theta']/df['eeg1_theta'].mean()
    df['eeg1_alpha_n2'] = df['eeg1_alpha']/df['eeg1_alpha'].mean()
    df['eeg1_sigma_n2'] = df['eeg1_sigma']/df['eeg1_sigma'].mean()
    df['eeg1_beta_n2'] = df['eeg1_beta']/df['eeg1_beta'].mean()
    df['eeg1_gamma_n2'] = df['eeg1_gamma']/df['eeg1_gamma'].mean()
    df['eeg2_delta_n2'] = df['eeg2_delta']/df['eeg2_delta'].mean()
    df['eeg2_theta_n2'] = df['eeg2_theta']/df['eeg2_theta'].mean()
    df['eeg2_alpha_n2'] = df['eeg2_alpha']/df['eeg2_alpha'].mean()
    df['eeg2_sigma_n2'] = df['eeg2_sigma']/df['eeg2_sigma'].mean()
    df['eeg2_beta_n2'] = df['eeg2_beta']/df['eeg2_beta'].mean()
    df['eeg2_gamma_n2'] = df['eeg2_gamma']/df['eeg2_gamma'].mean()
                 
    df['eeg1_theta_delta_ratio'] = df['eeg1_theta']/df['eeg1_delta']
    df['eeg2_theta_delta_ratio'] = df['eeg2_theta']/df['eeg2_delta']        
    df['eeg1_theta_delta_ratio_n'] = df['eeg1_theta_n']/df['eeg1_delta_n']
    df['eeg2_theta_delta_ratio_n'] = df['eeg2_theta_n']/df['eeg2_delta_n']  
    df['eeg1_theta_delta_ratio_n2'] = df['eeg1_theta_n2']/df['eeg1_delta_n2']
    df['eeg2_theta_delta_ratio_n2'] = df['eeg2_theta_n2']/df['eeg2_delta_n2']
    
    df['eeg1_alpha_delta_ratio'] = df['eeg1_alpha']/df['eeg1_delta']
    df['eeg2_alpha_delta_ratio'] = df['eeg2_alpha']/df['eeg2_delta']        
    df['eeg1_alpha_delta_ratio_n'] = df['eeg1_alpha_n']/df['eeg1_delta_n']
    df['eeg2_alpha_delta_ratio_n'] = df['eeg2_alpha_n']/df['eeg2_delta_n']  
    df['eeg1_alpha_delta_ratio_n2'] = df['eeg1_alpha_n2']/df['eeg1_delta_n2']
    df['eeg2_alpha_delta_ratio_n2'] = df['eeg2_alpha_n2']/df['eeg2_delta_n2']
    
    df['eeg1_sigma_delta_ratio'] = df['eeg1_sigma']/df['eeg1_delta']
    df['eeg2_sigma_delta_ratio'] = df['eeg2_sigma']/df['eeg2_delta']        
    df['eeg1_sigma_delta_ratio_n'] = df['eeg1_sigma_n']/df['eeg1_delta_n']
    df['eeg2_sigma_delta_ratio_n'] = df['eeg2_sigma_n']/df['eeg2_delta_n']  
    df['eeg1_sigma_delta_ratio_n2'] = df['eeg1_sigma_n2']/df['eeg1_delta_n2']
    df['eeg2_sigma_delta_ratio_n2'] = df['eeg2_sigma_n2']/df['eeg2_delta_n2']
    
    df['eeg1_beta_delta_ratio'] = df['eeg1_beta']/df['eeg1_delta']
    df['eeg2_beta_delta_ratio'] = df['eeg2_beta']/df['eeg2_delta']        
    df['eeg1_beta_delta_ratio_n'] = df['eeg1_beta_n']/df['eeg1_delta_n']
    df['eeg2_beta_delta_ratio_n'] = df['eeg2_beta_n']/df['eeg2_delta_n']  
    df['eeg1_beta_delta_ratio_n2'] = df['eeg1_beta_n2']/df['eeg1_delta_n2']
    df['eeg2_beta_delta_ratio_n2'] = df['eeg2_beta_n2']/df['eeg2_delta_n2']
    
    df['eeg1_gamma_delta_ratio'] = df['eeg1_gamma']/df['eeg1_delta']
    df['eeg2_gamma_delta_ratio'] = df['eeg2_gamma']/df['eeg2_delta']        
    df['eeg1_gamma_delta_ratio_n'] = df['eeg1_gamma_n']/df['eeg1_delta_n']
    df['eeg2_gamma_delta_ratio_n'] = df['eeg2_gamma_n']/df['eeg2_delta_n']  
    df['eeg1_gamma_delta_ratio_n2'] = df['eeg1_gamma_n2']/df['eeg1_delta_n2']
    df['eeg2_gamma_delta_ratio_n2'] = df['eeg2_gamma_n2']/df['eeg2_delta_n2']
    
    # firwin band pass filter
    print("get firwin band passed features")
    raw_firwin_delta = raw.copy()
    raw_firwin_delta.filter(1, 4, fir_design='firwin')
    raw_firwin_delta_data = raw_firwin_delta.get_data()
    df['eeg1_firwin_delta_abs_mean'] = get_epoch_abs_mean(raw_firwin_delta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_delta_abs_mean'] = get_epoch_abs_mean(raw_firwin_delta_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_delta_abs_median'] = get_epoch_abs_median(raw_firwin_delta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_delta_abs_median'] = get_epoch_abs_median(raw_firwin_delta_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_delta_abs_max'] = get_epoch_abs_max(raw_firwin_delta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_delta_abs_max'] = get_epoch_abs_max(raw_firwin_delta_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_delta_abs_std'] = get_epoch_abs_std(raw_firwin_delta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_delta_abs_std'] = get_epoch_abs_std(raw_firwin_delta_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_delta_rms'] = get_epoch_rms(raw_firwin_delta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_delta_rms'] = get_epoch_rms(raw_firwin_delta_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_delta_abs_mean_n'] = df['eeg1_firwin_delta_abs_mean']/df['eeg1_firwin_delta_abs_mean'].median()
    df['eeg2_firwin_delta_abs_mean_n'] = df['eeg2_firwin_delta_abs_mean']/df['eeg2_firwin_delta_abs_mean'].median()
    df['eeg1_firwin_delta_abs_median_n'] = df['eeg1_firwin_delta_abs_median']/df['eeg1_firwin_delta_abs_median'].median()
    df['eeg2_firwin_delta_abs_median_n'] = df['eeg2_firwin_delta_abs_median']/df['eeg2_firwin_delta_abs_median'].median()
    df['eeg1_firwin_delta_abs_max_n'] = df['eeg1_firwin_delta_abs_max']/df['eeg1_firwin_delta_abs_max'].median()
    df['eeg2_firwin_delta_abs_max_n'] = df['eeg2_firwin_delta_abs_max']/df['eeg2_firwin_delta_abs_max'].median()
    df['eeg1_firwin_delta_abs_std_n'] = df['eeg1_firwin_delta_abs_std']/df['eeg1_firwin_delta_abs_std'].median()
    df['eeg2_firwin_delta_abs_std_n'] = df['eeg2_firwin_delta_abs_std']/df['eeg2_firwin_delta_abs_std'].median()
    df['eeg1_firwin_delta_rms_n'] = df['eeg1_firwin_delta_rms']/df['eeg1_firwin_delta_rms'].median()
    df['eeg2_firwin_delta_rms_n'] = df['eeg2_firwin_delta_rms']/df['eeg2_firwin_delta_rms'].median()
    df['eeg1_firwin_delta_abs_mean_n2'] = df['eeg1_firwin_delta_abs_mean']/df['eeg1_firwin_delta_abs_mean'].mean()
    df['eeg2_firwin_delta_abs_mean_n2'] = df['eeg2_firwin_delta_abs_mean']/df['eeg2_firwin_delta_abs_mean'].mean()
    df['eeg1_firwin_delta_abs_median_n2'] = df['eeg1_firwin_delta_abs_median']/df['eeg1_firwin_delta_abs_median'].mean()
    df['eeg2_firwin_delta_abs_median_n2'] = df['eeg2_firwin_delta_abs_median']/df['eeg2_firwin_delta_abs_median'].mean()
    df['eeg1_firwin_delta_abs_max_n2'] = df['eeg1_firwin_delta_abs_max']/df['eeg1_firwin_delta_abs_max'].mean()
    df['eeg2_firwin_delta_abs_max_n2'] = df['eeg2_firwin_delta_abs_max']/df['eeg2_firwin_delta_abs_max'].mean()
    df['eeg1_firwin_delta_abs_std_n2'] = df['eeg1_firwin_delta_abs_std']/df['eeg1_firwin_delta_abs_std'].mean()
    df['eeg2_firwin_delta_abs_std_n2'] = df['eeg2_firwin_delta_abs_std']/df['eeg2_firwin_delta_abs_std'].mean()
    df['eeg1_firwin_delta_rms_n2'] = df['eeg1_firwin_delta_rms']/df['eeg1_firwin_delta_rms'].mean()
    df['eeg2_firwin_delta_rms_n2'] = df['eeg2_firwin_delta_rms']/df['eeg2_firwin_delta_rms'].mean()
    del raw_firwin_delta
    del raw_firwin_delta_data
    
    raw_firwin_theta = raw.copy()
    raw_firwin_theta.filter(4, 8, fir_design='firwin')
    raw_firwin_theta_data = raw_firwin_theta.get_data()
    df['eeg1_firwin_theta_abs_mean'] = get_epoch_abs_mean(raw_firwin_theta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_theta_abs_mean'] = get_epoch_abs_mean(raw_firwin_theta_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_theta_abs_median'] = get_epoch_abs_median(raw_firwin_theta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_theta_abs_median'] = get_epoch_abs_median(raw_firwin_theta_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_theta_abs_max'] = get_epoch_abs_max(raw_firwin_theta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_theta_abs_max'] = get_epoch_abs_max(raw_firwin_theta_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_theta_abs_std'] = get_epoch_abs_std(raw_firwin_theta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_theta_abs_std'] = get_epoch_abs_std(raw_firwin_theta_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_theta_rms'] = get_epoch_rms(raw_firwin_theta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_theta_rms'] = get_epoch_rms(raw_firwin_theta_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_theta_abs_mean_n'] = df['eeg1_firwin_theta_abs_mean']/df['eeg1_firwin_theta_abs_mean'].median()
    df['eeg2_firwin_theta_abs_mean_n'] = df['eeg2_firwin_theta_abs_mean']/df['eeg2_firwin_theta_abs_mean'].median()
    df['eeg1_firwin_theta_abs_median_n'] = df['eeg1_firwin_theta_abs_median']/df['eeg1_firwin_theta_abs_median'].median()
    df['eeg2_firwin_theta_abs_median_n'] = df['eeg2_firwin_theta_abs_median']/df['eeg2_firwin_theta_abs_median'].median()
    df['eeg1_firwin_theta_abs_max_n'] = df['eeg1_firwin_theta_abs_max']/df['eeg1_firwin_theta_abs_max'].median()
    df['eeg2_firwin_theta_abs_max_n'] = df['eeg2_firwin_theta_abs_max']/df['eeg2_firwin_theta_abs_max'].median()
    df['eeg1_firwin_theta_abs_std_n'] = df['eeg1_firwin_theta_abs_std']/df['eeg1_firwin_theta_abs_std'].median()
    df['eeg2_firwin_theta_abs_std_n'] = df['eeg2_firwin_theta_abs_std']/df['eeg2_firwin_theta_abs_std'].median()
    df['eeg1_firwin_theta_rms_n'] = df['eeg1_firwin_theta_rms']/df['eeg1_firwin_theta_rms'].median()
    df['eeg2_firwin_theta_rms_n'] = df['eeg2_firwin_theta_rms']/df['eeg2_firwin_theta_rms'].median()
    df['eeg1_firwin_theta_abs_mean_n2'] = df['eeg1_firwin_theta_abs_mean']/df['eeg1_firwin_theta_abs_mean'].mean()
    df['eeg2_firwin_theta_abs_mean_n2'] = df['eeg2_firwin_theta_abs_mean']/df['eeg2_firwin_theta_abs_mean'].mean()
    df['eeg1_firwin_theta_abs_median_n2'] = df['eeg1_firwin_theta_abs_median']/df['eeg1_firwin_theta_abs_median'].mean()
    df['eeg2_firwin_theta_abs_median_n2'] = df['eeg2_firwin_theta_abs_median']/df['eeg2_firwin_theta_abs_median'].mean()
    df['eeg1_firwin_theta_abs_max_n2'] = df['eeg1_firwin_theta_abs_max']/df['eeg1_firwin_theta_abs_max'].mean()
    df['eeg2_firwin_theta_abs_max_n2'] = df['eeg2_firwin_theta_abs_max']/df['eeg2_firwin_theta_abs_max'].mean()
    df['eeg1_firwin_theta_abs_std_n2'] = df['eeg1_firwin_theta_abs_std']/df['eeg1_firwin_theta_abs_std'].mean()
    df['eeg2_firwin_theta_abs_std_n2'] = df['eeg2_firwin_theta_abs_std']/df['eeg2_firwin_theta_abs_std'].mean()
    df['eeg1_firwin_theta_rms_n2'] = df['eeg1_firwin_theta_rms']/df['eeg1_firwin_theta_rms'].mean()
    df['eeg2_firwin_theta_rms_n2'] = df['eeg2_firwin_theta_rms']/df['eeg2_firwin_theta_rms'].mean()
    del raw_firwin_theta
    del raw_firwin_theta_data
    
    raw_firwin_alpha = raw.copy()
    raw_firwin_alpha.filter(8, 12, fir_design='firwin')
    raw_firwin_alpha_data = raw_firwin_alpha.get_data()
    df['eeg1_firwin_alpha_abs_mean'] = get_epoch_abs_mean(raw_firwin_alpha_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_alpha_abs_mean'] = get_epoch_abs_mean(raw_firwin_alpha_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_alpha_abs_median'] = get_epoch_abs_median(raw_firwin_alpha_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_alpha_abs_median'] = get_epoch_abs_median(raw_firwin_alpha_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_alpha_abs_max'] = get_epoch_abs_max(raw_firwin_alpha_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_alpha_abs_max'] = get_epoch_abs_max(raw_firwin_alpha_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_alpha_abs_std'] = get_epoch_abs_std(raw_firwin_alpha_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_alpha_abs_std'] = get_epoch_abs_std(raw_firwin_alpha_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_alpha_rms'] = get_epoch_rms(raw_firwin_alpha_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_alpha_rms'] = get_epoch_rms(raw_firwin_alpha_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_alpha_abs_mean_n'] = df['eeg1_firwin_alpha_abs_mean']/df['eeg1_firwin_alpha_abs_mean'].median()
    df['eeg2_firwin_alpha_abs_mean_n'] = df['eeg2_firwin_alpha_abs_mean']/df['eeg2_firwin_alpha_abs_mean'].median()
    df['eeg1_firwin_alpha_abs_median_n'] = df['eeg1_firwin_alpha_abs_median']/df['eeg1_firwin_alpha_abs_median'].median()
    df['eeg2_firwin_alpha_abs_median_n'] = df['eeg2_firwin_alpha_abs_median']/df['eeg2_firwin_alpha_abs_median'].median()
    df['eeg1_firwin_alpha_abs_max_n'] = df['eeg1_firwin_alpha_abs_max']/df['eeg1_firwin_alpha_abs_max'].median()
    df['eeg2_firwin_alpha_abs_max_n'] = df['eeg2_firwin_alpha_abs_max']/df['eeg2_firwin_alpha_abs_max'].median()
    df['eeg1_firwin_alpha_abs_std_n'] = df['eeg1_firwin_alpha_abs_std']/df['eeg1_firwin_alpha_abs_std'].median()
    df['eeg2_firwin_alpha_abs_std_n'] = df['eeg2_firwin_alpha_abs_std']/df['eeg2_firwin_alpha_abs_std'].median()
    df['eeg1_firwin_alpha_rms_n'] = df['eeg1_firwin_alpha_rms']/df['eeg1_firwin_alpha_rms'].median()
    df['eeg2_firwin_alpha_rms_n'] = df['eeg2_firwin_alpha_rms']/df['eeg2_firwin_alpha_rms'].median()
    df['eeg1_firwin_alpha_abs_mean_n2'] = df['eeg1_firwin_alpha_abs_mean']/df['eeg1_firwin_alpha_abs_mean'].mean()
    df['eeg2_firwin_alpha_abs_mean_n2'] = df['eeg2_firwin_alpha_abs_mean']/df['eeg2_firwin_alpha_abs_mean'].mean()
    df['eeg1_firwin_alpha_abs_median_n2'] = df['eeg1_firwin_alpha_abs_median']/df['eeg1_firwin_alpha_abs_median'].mean()
    df['eeg2_firwin_alpha_abs_median_n2'] = df['eeg2_firwin_alpha_abs_median']/df['eeg2_firwin_alpha_abs_median'].mean()
    df['eeg1_firwin_alpha_abs_max_n2'] = df['eeg1_firwin_alpha_abs_max']/df['eeg1_firwin_alpha_abs_max'].mean()
    df['eeg2_firwin_alpha_abs_max_n2'] = df['eeg2_firwin_alpha_abs_max']/df['eeg2_firwin_alpha_abs_max'].mean()
    df['eeg1_firwin_alpha_abs_std_n2'] = df['eeg1_firwin_alpha_abs_std']/df['eeg1_firwin_alpha_abs_std'].mean()
    df['eeg2_firwin_alpha_abs_std_n2'] = df['eeg2_firwin_alpha_abs_std']/df['eeg2_firwin_alpha_abs_std'].mean()
    df['eeg1_firwin_alpha_rms_n2'] = df['eeg1_firwin_alpha_rms']/df['eeg1_firwin_alpha_rms'].mean()
    df['eeg2_firwin_alpha_rms_n2'] = df['eeg2_firwin_alpha_rms']/df['eeg2_firwin_alpha_rms'].mean()
    del raw_firwin_alpha
    del raw_firwin_alpha_data
    
    raw_firwin_sigma = raw.copy()
    raw_firwin_sigma.filter(12, 15, fir_design='firwin')
    raw_firwin_sigma_data = raw_firwin_sigma.get_data()
    df['eeg1_firwin_sigma_abs_mean'] = get_epoch_abs_mean(raw_firwin_sigma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_sigma_abs_mean'] = get_epoch_abs_mean(raw_firwin_sigma_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_sigma_abs_median'] = get_epoch_abs_median(raw_firwin_sigma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_sigma_abs_median'] = get_epoch_abs_median(raw_firwin_sigma_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_sigma_abs_max'] = get_epoch_abs_max(raw_firwin_sigma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_sigma_abs_max'] = get_epoch_abs_max(raw_firwin_sigma_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_sigma_abs_std'] = get_epoch_abs_std(raw_firwin_sigma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_sigma_abs_std'] = get_epoch_abs_std(raw_firwin_sigma_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_sigma_rms'] = get_epoch_rms(raw_firwin_sigma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_sigma_rms'] = get_epoch_rms(raw_firwin_sigma_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_sigma_abs_mean_n'] = df['eeg1_firwin_sigma_abs_mean']/df['eeg1_firwin_sigma_abs_mean'].median()
    df['eeg2_firwin_sigma_abs_mean_n'] = df['eeg2_firwin_sigma_abs_mean']/df['eeg2_firwin_sigma_abs_mean'].median()
    df['eeg1_firwin_sigma_abs_median_n'] = df['eeg1_firwin_sigma_abs_median']/df['eeg1_firwin_sigma_abs_median'].median()
    df['eeg2_firwin_sigma_abs_median_n'] = df['eeg2_firwin_sigma_abs_median']/df['eeg2_firwin_sigma_abs_median'].median()
    df['eeg1_firwin_sigma_abs_max_n'] = df['eeg1_firwin_sigma_abs_max']/df['eeg1_firwin_sigma_abs_max'].median()
    df['eeg2_firwin_sigma_abs_max_n'] = df['eeg2_firwin_sigma_abs_max']/df['eeg2_firwin_sigma_abs_max'].median()
    df['eeg1_firwin_sigma_abs_std_n'] = df['eeg1_firwin_sigma_abs_std']/df['eeg1_firwin_sigma_abs_std'].median()
    df['eeg2_firwin_sigma_abs_std_n'] = df['eeg2_firwin_sigma_abs_std']/df['eeg2_firwin_sigma_abs_std'].median()
    df['eeg1_firwin_sigma_rms_n'] = df['eeg1_firwin_sigma_rms']/df['eeg1_firwin_sigma_rms'].median()
    df['eeg2_firwin_sigma_rms_n'] = df['eeg2_firwin_sigma_rms']/df['eeg2_firwin_sigma_rms'].median()
    df['eeg1_firwin_sigma_abs_mean_n2'] = df['eeg1_firwin_sigma_abs_mean']/df['eeg1_firwin_sigma_abs_mean'].mean()
    df['eeg2_firwin_sigma_abs_mean_n2'] = df['eeg2_firwin_sigma_abs_mean']/df['eeg2_firwin_sigma_abs_mean'].mean()
    df['eeg1_firwin_sigma_abs_median_n2'] = df['eeg1_firwin_sigma_abs_median']/df['eeg1_firwin_sigma_abs_median'].mean()
    df['eeg2_firwin_sigma_abs_median_n2'] = df['eeg2_firwin_sigma_abs_median']/df['eeg2_firwin_sigma_abs_median'].mean()
    df['eeg1_firwin_sigma_abs_max_n2'] = df['eeg1_firwin_sigma_abs_max']/df['eeg1_firwin_sigma_abs_max'].mean()
    df['eeg2_firwin_sigma_abs_max_n2'] = df['eeg2_firwin_sigma_abs_max']/df['eeg2_firwin_sigma_abs_max'].mean()
    df['eeg1_firwin_sigma_abs_std_n2'] = df['eeg1_firwin_sigma_abs_std']/df['eeg1_firwin_sigma_abs_std'].mean()
    df['eeg2_firwin_sigma_abs_std_n2'] = df['eeg2_firwin_sigma_abs_std']/df['eeg2_firwin_sigma_abs_std'].mean()
    df['eeg1_firwin_sigma_rms_n2'] = df['eeg1_firwin_sigma_rms']/df['eeg1_firwin_sigma_rms'].mean()
    df['eeg2_firwin_sigma_rms_n2'] = df['eeg2_firwin_sigma_rms']/df['eeg2_firwin_sigma_rms'].mean()
    del raw_firwin_sigma
    del raw_firwin_sigma_data
    
    raw_firwin_beta = raw.copy()
    raw_firwin_beta.filter(15, 30, fir_design='firwin')
    raw_firwin_beta_data = raw_firwin_beta.get_data()
    df['eeg1_firwin_beta_abs_mean'] = get_epoch_abs_mean(raw_firwin_beta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_beta_abs_mean'] = get_epoch_abs_mean(raw_firwin_beta_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_beta_abs_median'] = get_epoch_abs_median(raw_firwin_beta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_beta_abs_median'] = get_epoch_abs_median(raw_firwin_beta_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_beta_abs_max'] = get_epoch_abs_max(raw_firwin_beta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_beta_abs_max'] = get_epoch_abs_max(raw_firwin_beta_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_beta_abs_std'] = get_epoch_abs_std(raw_firwin_beta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_beta_abs_std'] = get_epoch_abs_std(raw_firwin_beta_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_beta_rms'] = get_epoch_rms(raw_firwin_beta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_beta_rms'] = get_epoch_rms(raw_firwin_beta_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_beta_abs_mean_n'] = df['eeg1_firwin_beta_abs_mean']/df['eeg1_firwin_beta_abs_mean'].median()
    df['eeg2_firwin_beta_abs_mean_n'] = df['eeg2_firwin_beta_abs_mean']/df['eeg2_firwin_beta_abs_mean'].median()
    df['eeg1_firwin_beta_abs_median_n'] = df['eeg1_firwin_beta_abs_median']/df['eeg1_firwin_beta_abs_median'].median()
    df['eeg2_firwin_beta_abs_median_n'] = df['eeg2_firwin_beta_abs_median']/df['eeg2_firwin_beta_abs_median'].median()
    df['eeg1_firwin_beta_abs_max_n'] = df['eeg1_firwin_beta_abs_max']/df['eeg1_firwin_beta_abs_max'].median()
    df['eeg2_firwin_beta_abs_max_n'] = df['eeg2_firwin_beta_abs_max']/df['eeg2_firwin_beta_abs_max'].median()
    df['eeg1_firwin_beta_abs_std_n'] = df['eeg1_firwin_beta_abs_std']/df['eeg1_firwin_beta_abs_std'].median()
    df['eeg2_firwin_beta_abs_std_n'] = df['eeg2_firwin_beta_abs_std']/df['eeg2_firwin_beta_abs_std'].median()
    df['eeg1_firwin_beta_rms_n'] = df['eeg1_firwin_beta_rms']/df['eeg1_firwin_beta_rms'].median()
    df['eeg2_firwin_beta_rms_n'] = df['eeg2_firwin_beta_rms']/df['eeg2_firwin_beta_rms'].median()
    df['eeg1_firwin_beta_abs_mean_n2'] = df['eeg1_firwin_beta_abs_mean']/df['eeg1_firwin_beta_abs_mean'].mean()
    df['eeg2_firwin_beta_abs_mean_n2'] = df['eeg2_firwin_beta_abs_mean']/df['eeg2_firwin_beta_abs_mean'].mean()
    df['eeg1_firwin_beta_abs_median_n2'] = df['eeg1_firwin_beta_abs_median']/df['eeg1_firwin_beta_abs_median'].mean()
    df['eeg2_firwin_beta_abs_median_n2'] = df['eeg2_firwin_beta_abs_median']/df['eeg2_firwin_beta_abs_median'].mean()
    df['eeg1_firwin_beta_abs_max_n2'] = df['eeg1_firwin_beta_abs_max']/df['eeg1_firwin_beta_abs_max'].mean()
    df['eeg2_firwin_beta_abs_max_n2'] = df['eeg2_firwin_beta_abs_max']/df['eeg2_firwin_beta_abs_max'].mean()
    df['eeg1_firwin_beta_abs_std_n2'] = df['eeg1_firwin_beta_abs_std']/df['eeg1_firwin_beta_abs_std'].mean()
    df['eeg2_firwin_beta_abs_std_n2'] = df['eeg2_firwin_beta_abs_std']/df['eeg2_firwin_beta_abs_std'].mean()
    df['eeg1_firwin_beta_rms_n2'] = df['eeg1_firwin_beta_rms']/df['eeg1_firwin_beta_rms'].mean()
    df['eeg2_firwin_beta_rms_n2'] = df['eeg2_firwin_beta_rms']/df['eeg2_firwin_beta_rms'].mean()
    del raw_firwin_beta
    del raw_firwin_beta_data

    raw_firwin_gamma = raw.copy()
    raw_firwin_gamma.filter(30, 40, fir_design='firwin')
    raw_firwin_gamma_data = raw_firwin_gamma.get_data()
    df['eeg1_firwin_gamma_abs_mean'] = get_epoch_abs_mean(raw_firwin_gamma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_gamma_abs_mean'] = get_epoch_abs_mean(raw_firwin_gamma_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_gamma_abs_median'] = get_epoch_abs_median(raw_firwin_gamma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_gamma_abs_median'] = get_epoch_abs_median(raw_firwin_gamma_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_gamma_abs_max'] = get_epoch_abs_max(raw_firwin_gamma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_gamma_abs_max'] = get_epoch_abs_max(raw_firwin_gamma_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_gamma_abs_std'] = get_epoch_abs_std(raw_firwin_gamma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_gamma_abs_std'] = get_epoch_abs_std(raw_firwin_gamma_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_gamma_rms'] = get_epoch_rms(raw_firwin_gamma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg2_firwin_gamma_rms'] = get_epoch_rms(raw_firwin_gamma_data[1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg1_firwin_gamma_abs_mean_n'] = df['eeg1_firwin_gamma_abs_mean']/df['eeg1_firwin_gamma_abs_mean'].median()
    df['eeg2_firwin_gamma_abs_mean_n'] = df['eeg2_firwin_gamma_abs_mean']/df['eeg2_firwin_gamma_abs_mean'].median()
    df['eeg1_firwin_gamma_abs_median_n'] = df['eeg1_firwin_gamma_abs_median']/df['eeg1_firwin_gamma_abs_median'].median()
    df['eeg2_firwin_gamma_abs_median_n'] = df['eeg2_firwin_gamma_abs_median']/df['eeg2_firwin_gamma_abs_median'].median()
    df['eeg1_firwin_gamma_abs_max_n'] = df['eeg1_firwin_gamma_abs_max']/df['eeg1_firwin_gamma_abs_max'].median()
    df['eeg2_firwin_gamma_abs_max_n'] = df['eeg2_firwin_gamma_abs_max']/df['eeg2_firwin_gamma_abs_max'].median()
    df['eeg1_firwin_gamma_abs_std_n'] = df['eeg1_firwin_gamma_abs_std']/df['eeg1_firwin_gamma_abs_std'].median()
    df['eeg2_firwin_gamma_abs_std_n'] = df['eeg2_firwin_gamma_abs_std']/df['eeg2_firwin_gamma_abs_std'].median()
    df['eeg1_firwin_gamma_rms_n'] = df['eeg1_firwin_gamma_rms']/df['eeg1_firwin_gamma_rms'].median()
    df['eeg2_firwin_gamma_rms_n'] = df['eeg2_firwin_gamma_rms']/df['eeg2_firwin_gamma_rms'].median()
    df['eeg1_firwin_gamma_abs_mean_n2'] = df['eeg1_firwin_gamma_abs_mean']/df['eeg1_firwin_gamma_abs_mean'].mean()
    df['eeg2_firwin_gamma_abs_mean_n2'] = df['eeg2_firwin_gamma_abs_mean']/df['eeg2_firwin_gamma_abs_mean'].mean()
    df['eeg1_firwin_gamma_abs_median_n2'] = df['eeg1_firwin_gamma_abs_median']/df['eeg1_firwin_gamma_abs_median'].mean()
    df['eeg2_firwin_gamma_abs_median_n2'] = df['eeg2_firwin_gamma_abs_median']/df['eeg2_firwin_gamma_abs_median'].mean()
    df['eeg1_firwin_gamma_abs_max_n2'] = df['eeg1_firwin_gamma_abs_max']/df['eeg1_firwin_gamma_abs_max'].mean()
    df['eeg2_firwin_gamma_abs_max_n2'] = df['eeg2_firwin_gamma_abs_max']/df['eeg2_firwin_gamma_abs_max'].mean()
    df['eeg1_firwin_gamma_abs_std_n2'] = df['eeg1_firwin_gamma_abs_std']/df['eeg1_firwin_gamma_abs_std'].mean()
    df['eeg2_firwin_gamma_abs_std_n2'] = df['eeg2_firwin_gamma_abs_std']/df['eeg2_firwin_gamma_abs_std'].mean()
    df['eeg1_firwin_gamma_rms_n2'] = df['eeg1_firwin_gamma_rms']/df['eeg1_firwin_gamma_rms'].mean()
    df['eeg2_firwin_gamma_rms_n2'] = df['eeg2_firwin_gamma_rms']/df['eeg2_firwin_gamma_rms'].mean()
    del raw_firwin_gamma
    del raw_firwin_gamma_data

    df['epoch_id'] = file_firstname + '___' + df.index.astype('str')
    df['subject_id'] = file_firstname
    df['score'] = score_list
    df.to_csv(edf_folderpath + file_firstname + "_" + model_name + '_features.csv')
    

def save_single_edf_to_csv_1eeg(edf_filepath=None, epoch_len=10, model_name=None, test_run=False, include_score=False):
    file_firstname = edf_filepath.split("/")[-1].split('.edf')[0]
    edf_folderpath = edf_filepath.split(file_firstname)[0]

    if include_score:
        db3_filepath = edf_path + file_firstname + ".db3"
        connection = sqlite3.connect(db3_filepath)
        score_list = pd.read_sql_query(f"SELECT * from sleep_scores_table", connection)['score'].values
        print("score_list_shape", score_list.shape)
    else:
        score_list = np.nan
    
    # Save a downsampled data file
    downsampled_files = f"{edf_folderpath}{file_firstname}_{model_name}_rs_100hz.npy"
    if not os.path.exists(downsampled_files):
        raw = read_raw_edf(edf_filepath, preload=True)
        sfreq = int(raw.info["sfreq"])
        print(f"sfreq:{sfreq}")
        raw.resample(sfreq=100)
        resampled_data = raw.get_data()
        with open(downsampled_files, 'wb') as f:
            np.save(f, resampled_data)

    raw = read_raw_edf(edf_filepath, preload=True)
    raw_data_length = raw.get_data().shape[1]
    sfreq = int(raw.info["sfreq"])

    if include_score:
        print("raw_data_shape", raw_data_length)
        if raw_data_length != len(score_list) * 10 * sfreq:
            print("score list and raw data have different number of epochs")
            return
    if test_run:
        return

    raw.filter(1., 40., fir_design='firwin')
    raw_data = raw.get_data()
    
    df = pd.DataFrame()
    
    print("get basic features")
    df['eeg_abs_mean'] = get_epoch_abs_mean(raw_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['emg_abs_mean'] = get_epoch_abs_mean(raw_data[-1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_abs_median'] = get_epoch_abs_median(raw_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['emg_abs_median'] = get_epoch_abs_median(raw_data[-1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_abs_max'] = get_epoch_abs_max(raw_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['emg_abs_max'] = get_epoch_abs_max(raw_data[-1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_abs_std'] = get_epoch_abs_std(raw_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['emg_abs_std'] = get_epoch_abs_std(raw_data[-1], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_rms'] = get_epoch_rms(raw_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['emg_rms'] = get_epoch_rms(raw_data[-1], sfreq=sfreq, epoch_len=epoch_len)
    
    df['eeg_abs_mean_n'] = df['eeg_abs_mean']/df['eeg_abs_mean'].median()
    df['emg_abs_mean_n'] = df['emg_abs_mean']/df['emg_abs_mean'].median()
    df['eeg_abs_median_n'] = df['eeg_abs_median']/df['eeg_abs_median'].median()
    df['emg_abs_median_n'] = df['emg_abs_median']/df['emg_abs_median'].median()
    df['eeg_abs_max_n'] = df['eeg_abs_max']/df['eeg_abs_max'].median()
    df['emg_abs_max_n'] = df['emg_abs_max']/df['emg_abs_max'].median()
    df['eeg_abs_std_n'] = df['eeg_abs_std']/df['eeg_abs_std'].median()
    df['emg_abs_std_n'] = df['emg_abs_std']/df['emg_abs_std'].median()
    df['eeg_rms_n'] = df['eeg_rms']/df['eeg_rms'].median()
    df['emg_rms_n'] = df['emg_rms']/df['emg_rms'].median()
    
    df['eeg_abs_mean_n2'] = df['eeg_abs_mean']/df['eeg_abs_mean'].mean()
    df['emg_abs_mean_n2'] = df['emg_abs_mean']/df['emg_abs_mean'].mean()
    df['eeg_abs_median_n2'] = df['eeg_abs_median']/df['eeg_abs_median'].mean()
    df['emg_abs_median_n2'] = df['emg_abs_median']/df['emg_abs_median'].mean()
    df['eeg_abs_max_n2'] = df['eeg_abs_max']/df['eeg_abs_max'].mean()
    df['emg_abs_max_n2'] = df['emg_abs_max']/df['emg_abs_max'].mean()
    df['eeg_abs_std_n2'] = df['eeg_abs_std']/df['eeg_abs_std'].mean()
    df['emg_abs_std_n2'] = df['emg_abs_std']/df['emg_abs_std'].mean()
    df['eeg_rms_n2'] = df['eeg_rms']/df['eeg_rms'].mean()
    df['emg_rms_n2'] = df['emg_rms']/df['emg_rms'].mean()
    
    # PSD
    print("get psd features")
    eeg_psd = get_epoch_psd(raw_data[0], sfreq=sfreq, epoch_len=epoch_len)
    
    df['eeg_delta'] = eeg_psd[0]
    df['eeg_theta'] = eeg_psd[1]
    df['eeg_alpha'] = eeg_psd[2]
    df['eeg_sigma'] = eeg_psd[3]
    df['eeg_beta'] = eeg_psd[4]
    df['eeg_gamma'] = eeg_psd[5]
    
    df['eeg_delta_n'] = df['eeg_delta']/df['eeg_delta'].median()
    df['eeg_theta_n'] = df['eeg_theta']/df['eeg_theta'].median()
    df['eeg_alpha_n'] = df['eeg_alpha']/df['eeg_alpha'].median()
    df['eeg_sigma_n'] = df['eeg_sigma']/df['eeg_sigma'].median()
    df['eeg_beta_n'] = df['eeg_beta']/df['eeg_beta'].median()
    df['eeg_gamma_n'] = df['eeg_gamma']/df['eeg_gamma'].median()
    
    df['eeg_delta_n2'] = df['eeg_delta']/df['eeg_delta'].mean()
    df['eeg_theta_n2'] = df['eeg_theta']/df['eeg_theta'].mean()
    df['eeg_alpha_n2'] = df['eeg_alpha']/df['eeg_alpha'].mean()
    df['eeg_sigma_n2'] = df['eeg_sigma']/df['eeg_sigma'].mean()
    df['eeg_beta_n2'] = df['eeg_beta']/df['eeg_beta'].mean()
    df['eeg_gamma_n2'] = df['eeg_gamma']/df['eeg_gamma'].mean()
                 
    df['eeg_theta_delta_ratio'] = df['eeg_theta']/df['eeg_delta']
    df['eeg_theta_delta_ratio_n'] = df['eeg_theta_n']/df['eeg_delta_n']
    df['eeg_theta_delta_ratio_n2'] = df['eeg_theta_n2']/df['eeg_delta_n2']
    
    df['eeg_alpha_delta_ratio'] = df['eeg_alpha']/df['eeg_delta']
    df['eeg_alpha_delta_ratio_n'] = df['eeg_alpha_n']/df['eeg_delta_n']
    df['eeg_alpha_delta_ratio_n2'] = df['eeg_alpha_n2']/df['eeg_delta_n2']
    
    df['eeg_sigma_delta_ratio'] = df['eeg_sigma']/df['eeg_delta']
    df['eeg_sigma_delta_ratio_n'] = df['eeg_sigma_n']/df['eeg_delta_n']
    df['eeg_sigma_delta_ratio_n2'] = df['eeg_sigma_n2']/df['eeg_delta_n2']
    
    df['eeg_beta_delta_ratio'] = df['eeg_beta']/df['eeg_delta']
    df['eeg_beta_delta_ratio_n'] = df['eeg_beta_n']/df['eeg_delta_n']
    df['eeg_beta_delta_ratio_n2'] = df['eeg_beta_n2']/df['eeg_delta_n2']
    
    df['eeg_gamma_delta_ratio'] = df['eeg_gamma']/df['eeg_delta']
    df['eeg_gamma_delta_ratio_n'] = df['eeg_gamma_n']/df['eeg_delta_n']
    df['eeg_gamma_delta_ratio_n2'] = df['eeg_gamma_n2']/df['eeg_delta_n2']
    
    # firwin band pass filter
    print("get firwin band passed features")
    raw_firwin_delta = raw.copy()
    raw_firwin_delta.filter(1, 4, fir_design='firwin')
    raw_firwin_delta_data = raw_firwin_delta.get_data()
    df['eeg_firwin_delta_abs_mean'] = get_epoch_abs_mean(raw_firwin_delta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_delta_abs_median'] = get_epoch_abs_median(raw_firwin_delta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_delta_abs_max'] = get_epoch_abs_max(raw_firwin_delta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_delta_abs_std'] = get_epoch_abs_std(raw_firwin_delta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_delta_rms'] = get_epoch_rms(raw_firwin_delta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_delta_abs_mean_n'] = df['eeg_firwin_delta_abs_mean']/df['eeg_firwin_delta_abs_mean'].median()
    df['eeg_firwin_delta_abs_median_n'] = df['eeg_firwin_delta_abs_median']/df['eeg_firwin_delta_abs_median'].median()
    df['eeg_firwin_delta_abs_max_n'] = df['eeg_firwin_delta_abs_max']/df['eeg_firwin_delta_abs_max'].median()
    df['eeg_firwin_delta_abs_std_n'] = df['eeg_firwin_delta_abs_std']/df['eeg_firwin_delta_abs_std'].median()
    df['eeg_firwin_delta_rms_n'] = df['eeg_firwin_delta_rms']/df['eeg_firwin_delta_rms'].median()
    df['eeg_firwin_delta_abs_mean_n2'] = df['eeg_firwin_delta_abs_mean']/df['eeg_firwin_delta_abs_mean'].mean()
    df['eeg_firwin_delta_abs_median_n2'] = df['eeg_firwin_delta_abs_median']/df['eeg_firwin_delta_abs_median'].mean()
    df['eeg_firwin_delta_abs_max_n2'] = df['eeg_firwin_delta_abs_max']/df['eeg_firwin_delta_abs_max'].mean()
    df['eeg_firwin_delta_abs_std_n2'] = df['eeg_firwin_delta_abs_std']/df['eeg_firwin_delta_abs_std'].mean()
    df['eeg_firwin_delta_rms_n2'] = df['eeg_firwin_delta_rms']/df['eeg_firwin_delta_rms'].mean()
    del raw_firwin_delta
    del raw_firwin_delta_data
    
    raw_firwin_theta = raw.copy()
    raw_firwin_theta.filter(4, 8, fir_design='firwin')
    raw_firwin_theta_data = raw_firwin_theta.get_data()
    df['eeg_firwin_theta_abs_mean'] = get_epoch_abs_mean(raw_firwin_theta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_theta_abs_median'] = get_epoch_abs_median(raw_firwin_theta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_theta_abs_max'] = get_epoch_abs_max(raw_firwin_theta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_theta_abs_std'] = get_epoch_abs_std(raw_firwin_theta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_theta_rms'] = get_epoch_rms(raw_firwin_theta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_theta_abs_mean_n'] = df['eeg_firwin_theta_abs_mean']/df['eeg_firwin_theta_abs_mean'].median()
    df['eeg_firwin_theta_abs_median_n'] = df['eeg_firwin_theta_abs_median']/df['eeg_firwin_theta_abs_median'].median()
    df['eeg_firwin_theta_abs_max_n'] = df['eeg_firwin_theta_abs_max']/df['eeg_firwin_theta_abs_max'].median()
    df['eeg_firwin_theta_abs_std_n'] = df['eeg_firwin_theta_abs_std']/df['eeg_firwin_theta_abs_std'].median()
    df['eeg_firwin_theta_rms_n'] = df['eeg_firwin_theta_rms']/df['eeg_firwin_theta_rms'].median()
    df['eeg_firwin_theta_abs_mean_n2'] = df['eeg_firwin_theta_abs_mean']/df['eeg_firwin_theta_abs_mean'].mean()
    df['eeg_firwin_theta_abs_median_n2'] = df['eeg_firwin_theta_abs_median']/df['eeg_firwin_theta_abs_median'].mean()
    df['eeg_firwin_theta_abs_max_n2'] = df['eeg_firwin_theta_abs_max']/df['eeg_firwin_theta_abs_max'].mean()
    df['eeg_firwin_theta_abs_std_n2'] = df['eeg_firwin_theta_abs_std']/df['eeg_firwin_theta_abs_std'].mean()
    df['eeg_firwin_theta_rms_n2'] = df['eeg_firwin_theta_rms']/df['eeg_firwin_theta_rms'].mean()
    del raw_firwin_theta
    del raw_firwin_theta_data
    
    raw_firwin_alpha = raw.copy()
    raw_firwin_alpha.filter(8, 12, fir_design='firwin')
    raw_firwin_alpha_data = raw_firwin_alpha.get_data()
    df['eeg_firwin_alpha_abs_mean'] = get_epoch_abs_mean(raw_firwin_alpha_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_alpha_abs_median'] = get_epoch_abs_median(raw_firwin_alpha_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_alpha_abs_max'] = get_epoch_abs_max(raw_firwin_alpha_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_alpha_abs_std'] = get_epoch_abs_std(raw_firwin_alpha_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_alpha_rms'] = get_epoch_rms(raw_firwin_alpha_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_alpha_abs_mean_n'] = df['eeg_firwin_alpha_abs_mean']/df['eeg_firwin_alpha_abs_mean'].median()
    df['eeg_firwin_alpha_abs_median_n'] = df['eeg_firwin_alpha_abs_median']/df['eeg_firwin_alpha_abs_median'].median()
    df['eeg_firwin_alpha_abs_max_n'] = df['eeg_firwin_alpha_abs_max']/df['eeg_firwin_alpha_abs_max'].median()
    df['eeg_firwin_alpha_abs_std_n'] = df['eeg_firwin_alpha_abs_std']/df['eeg_firwin_alpha_abs_std'].median()
    df['eeg_firwin_alpha_rms_n'] = df['eeg_firwin_alpha_rms']/df['eeg_firwin_alpha_rms'].median()
    df['eeg_firwin_alpha_abs_mean_n2'] = df['eeg_firwin_alpha_abs_mean']/df['eeg_firwin_alpha_abs_mean'].mean()
    df['eeg_firwin_alpha_abs_median_n2'] = df['eeg_firwin_alpha_abs_median']/df['eeg_firwin_alpha_abs_median'].mean()
    df['eeg_firwin_alpha_abs_max_n2'] = df['eeg_firwin_alpha_abs_max']/df['eeg_firwin_alpha_abs_max'].mean()
    df['eeg_firwin_alpha_abs_std_n2'] = df['eeg_firwin_alpha_abs_std']/df['eeg_firwin_alpha_abs_std'].mean()
    df['eeg_firwin_alpha_rms_n2'] = df['eeg_firwin_alpha_rms']/df['eeg_firwin_alpha_rms'].mean()
    del raw_firwin_alpha
    del raw_firwin_alpha_data
    
    raw_firwin_sigma = raw.copy()
    raw_firwin_sigma.filter(12, 15, fir_design='firwin')
    raw_firwin_sigma_data = raw_firwin_sigma.get_data()
    df['eeg_firwin_sigma_abs_mean'] = get_epoch_abs_mean(raw_firwin_sigma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_sigma_abs_median'] = get_epoch_abs_median(raw_firwin_sigma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_sigma_abs_max'] = get_epoch_abs_max(raw_firwin_sigma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_sigma_abs_std'] = get_epoch_abs_std(raw_firwin_sigma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_sigma_rms'] = get_epoch_rms(raw_firwin_sigma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_sigma_abs_mean_n'] = df['eeg_firwin_sigma_abs_mean']/df['eeg_firwin_sigma_abs_mean'].median()
    df['eeg_firwin_sigma_abs_median_n'] = df['eeg_firwin_sigma_abs_median']/df['eeg_firwin_sigma_abs_median'].median()
    df['eeg_firwin_sigma_abs_max_n'] = df['eeg_firwin_sigma_abs_max']/df['eeg_firwin_sigma_abs_max'].median()
    df['eeg_firwin_sigma_abs_std_n'] = df['eeg_firwin_sigma_abs_std']/df['eeg_firwin_sigma_abs_std'].median()
    df['eeg_firwin_sigma_rms_n'] = df['eeg_firwin_sigma_rms']/df['eeg_firwin_sigma_rms'].median()
    df['eeg_firwin_sigma_abs_mean_n2'] = df['eeg_firwin_sigma_abs_mean']/df['eeg_firwin_sigma_abs_mean'].mean()
    df['eeg_firwin_sigma_abs_median_n2'] = df['eeg_firwin_sigma_abs_median']/df['eeg_firwin_sigma_abs_median'].mean()
    df['eeg_firwin_sigma_abs_max_n2'] = df['eeg_firwin_sigma_abs_max']/df['eeg_firwin_sigma_abs_max'].mean()
    df['eeg_firwin_sigma_abs_std_n2'] = df['eeg_firwin_sigma_abs_std']/df['eeg_firwin_sigma_abs_std'].mean()
    df['eeg_firwin_sigma_rms_n2'] = df['eeg_firwin_sigma_rms']/df['eeg_firwin_sigma_rms'].mean()
    del raw_firwin_sigma
    del raw_firwin_sigma_data
    
    raw_firwin_beta = raw.copy()
    raw_firwin_beta.filter(15, 30, fir_design='firwin')
    raw_firwin_beta_data = raw_firwin_beta.get_data()
    df['eeg_firwin_beta_abs_mean'] = get_epoch_abs_mean(raw_firwin_beta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_beta_abs_median'] = get_epoch_abs_median(raw_firwin_beta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_beta_abs_max'] = get_epoch_abs_max(raw_firwin_beta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_beta_abs_std'] = get_epoch_abs_std(raw_firwin_beta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_beta_rms'] = get_epoch_rms(raw_firwin_beta_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_beta_abs_mean_n'] = df['eeg_firwin_beta_abs_mean']/df['eeg_firwin_beta_abs_mean'].median()
    df['eeg_firwin_beta_abs_median_n'] = df['eeg_firwin_beta_abs_median']/df['eeg_firwin_beta_abs_median'].median()
    df['eeg_firwin_beta_abs_max_n'] = df['eeg_firwin_beta_abs_max']/df['eeg_firwin_beta_abs_max'].median()
    df['eeg_firwin_beta_abs_std_n'] = df['eeg_firwin_beta_abs_std']/df['eeg_firwin_beta_abs_std'].median()
    df['eeg_firwin_beta_rms_n'] = df['eeg_firwin_beta_rms']/df['eeg_firwin_beta_rms'].median()
    df['eeg_firwin_beta_abs_mean_n2'] = df['eeg_firwin_beta_abs_mean']/df['eeg_firwin_beta_abs_mean'].mean()
    df['eeg_firwin_beta_abs_median_n2'] = df['eeg_firwin_beta_abs_median']/df['eeg_firwin_beta_abs_median'].mean()
    df['eeg_firwin_beta_abs_max_n2'] = df['eeg_firwin_beta_abs_max']/df['eeg_firwin_beta_abs_max'].mean()
    df['eeg_firwin_beta_abs_std_n2'] = df['eeg_firwin_beta_abs_std']/df['eeg_firwin_beta_abs_std'].mean()
    df['eeg_firwin_beta_rms_n2'] = df['eeg_firwin_beta_rms']/df['eeg_firwin_beta_rms'].mean()
    del raw_firwin_beta
    del raw_firwin_beta_data

    raw_firwin_gamma = raw.copy()
    raw_firwin_gamma.filter(30, 40, fir_design='firwin')
    raw_firwin_gamma_data = raw_firwin_gamma.get_data()
    df['eeg_firwin_gamma_abs_mean'] = get_epoch_abs_mean(raw_firwin_gamma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_gamma_abs_median'] = get_epoch_abs_median(raw_firwin_gamma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_gamma_abs_max'] = get_epoch_abs_max(raw_firwin_gamma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_gamma_abs_std'] = get_epoch_abs_std(raw_firwin_gamma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_gamma_rms'] = get_epoch_rms(raw_firwin_gamma_data[0], sfreq=sfreq, epoch_len=epoch_len)
    df['eeg_firwin_gamma_abs_mean_n'] = df['eeg_firwin_gamma_abs_mean']/df['eeg_firwin_gamma_abs_mean'].median()
    df['eeg_firwin_gamma_abs_median_n'] = df['eeg_firwin_gamma_abs_median']/df['eeg_firwin_gamma_abs_median'].median()
    df['eeg_firwin_gamma_abs_max_n'] = df['eeg_firwin_gamma_abs_max']/df['eeg_firwin_gamma_abs_max'].median()
    df['eeg_firwin_gamma_abs_std_n'] = df['eeg_firwin_gamma_abs_std']/df['eeg_firwin_gamma_abs_std'].median()
    df['eeg_firwin_gamma_rms_n'] = df['eeg_firwin_gamma_rms']/df['eeg_firwin_gamma_rms'].median()
    df['eeg_firwin_gamma_abs_mean_n2'] = df['eeg_firwin_gamma_abs_mean']/df['eeg_firwin_gamma_abs_mean'].mean()
    df['eeg_firwin_gamma_abs_median_n2'] = df['eeg_firwin_gamma_abs_median']/df['eeg_firwin_gamma_abs_median'].mean()
    df['eeg_firwin_gamma_abs_max_n2'] = df['eeg_firwin_gamma_abs_max']/df['eeg_firwin_gamma_abs_max'].mean()
    df['eeg_firwin_gamma_abs_std_n2'] = df['eeg_firwin_gamma_abs_std']/df['eeg_firwin_gamma_abs_std'].mean()
    df['eeg_firwin_gamma_rms_n2'] = df['eeg_firwin_gamma_rms']/df['eeg_firwin_gamma_rms'].mean()
    del raw_firwin_gamma
    del raw_firwin_gamma_data

    df['epoch_id'] = file_firstname + '___' + df.index.astype('str')
    df['subject_id'] = file_firstname
    df['score'] = score_list
    df.to_csv(edf_folderpath + file_firstname + "_" + model_name + '_features.csv')
    

def make_prediction (X, y_truth, model):
    # y_prediction_prob = model.predict_proba(X)
    # y_prediction = np.zeros(y_prediction_prob[:,0].shape)
    # y_prediction = y_prediction.astype('str')
    # y_prediction[y_prediction_prob[:,0] > 1/3] = 'nrem'
    # y_prediction[y_prediction_prob[:,2] > 1/3] = 'wake'
    # y_prediction[y_prediction_prob[:,1] > param['rem_threshold']] = 'rem'
    y_prediction = model.predict(X)
    report = classification_report(y_truth, y_prediction)
    printW(report)
    return y_prediction

def get_shap (df, features, model):
    explainer = shap.TreeExplainer(model)
    df_500samples = df.sample(500)
    indices_500samples = df_500samples.index.values
    shap_values_500samples = explainer.shap_values(df_500samples[features])
    return explainer, shap_values_500samples, indices_500samples

