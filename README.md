# IntelliSleepScorer
<img src="https://sites.broadinstitute.org/files/styles/original/public/pan-lab/files/logo.webp?itok=yV81aERP" alt="logo" width="300">

# Download the Windows Executable

We have generated a Windows Executable for IntelliSleepScorer using PyInstaller. Below are the download links.

<a href="https://storage.googleapis.com/grins-public/LeiWang-20230208/IntelliSleepScorer%20v1.1.1.zip">IntelliSleepScorer v1.1.1 for Windows</a>

<a href="https://storage.googleapis.com/grins-public/LeiWang-20230208/IntelliSleepScorer.zip">IntelliSleepScorer v1.1 for Windows</a>

Users of MacOS or Linux, please use the source code to launch IntelliSleepScorer.

You can use the link below to download two example EDF files.

<a href="https://storage.googleapis.com/grins-public/LeiWang-20230208/edf_examples.zip">Example EDF files</a>


# Run IntelliSleepScorer using the source code

If you are using Macbook or Linux machines, you can launch IntelliSleepScorer using the source code. Note that this repository does not include the "models" folder (due to size limit). You need to download the , unzip it, and copy the "models" folder inside the repositary. Otherwise, the software will crash due to missing model files.

# EDF/EDF+ format requirement

IntelliSleepScorer uses <a href="https://mne.tools/stable/index.html">MNE-Python</a> package to read EDF/EDF+ files. Please follow the <a href="https://www.edfplus.info/specs/edf.html">stardard EDF/EDF+ specification </a> when generating the EDF/EDF+ files. In addition to the standard specification, IntelliSleepScorer has a few specific requirements:
1. The EDF/EDF+ annotations must be encoded in UTF-8. Otherwise, the software will crash.
2. If you use the LightGBM-2EEG model for scoring, Your EDF/EDF+ should have three channels organized in the following order: 1) an EEG channel recorded in the parietal area; 2) an EEG channel recorded in the frontal area; 3) an EMG channel. If you use LightGBM-1EEG model, Your EDF/EDF+ should have two channels organized in the following order: 1) an EEG channel; 2) an EMG channel. If your EDF/EDF+ files have more than two channels when LightGBM-1EEG model, IntelliSleepScorer will use the first channel as the EEG channel and the last channel as the EMG channel to extract features and make predictions.
3. We have tested EDF/EDF+ files sampled 100-1000Hz with lengths up to 24 hours. We did not set any limits on the length of data stored in each channel or the total size of EDF/EDF+ that can be run. Depending on the specs of your PC,files larger than what we have tested may cause issues.

# Workflow

1. Launch IntelliSleepScorer using the Windows executable (IntelliSleepScorer.ext located in the root folder) or the source code.
2. Click "Select EDF/EDF+ File(s)" to select the files you want to score. If you selected any files by mistake, you can click the "Clear" button to clear the selected file list.
3. By default, the sleep stages are encoded as Wake: 1, NREM: 2; REM: 3 in the output score files. The default epoch length is set at 10 sec. The current version (v1.1) of IntelliSleepScorer does not allow changing stage encodings or epoch length. These functionalities will be added in future releases.
4. Select the model you want to use for sleep scoring.
5. Click "Score All Files". IntelliSleepScorer will automatically score all the EDF/EDF+ files and calculate the global SHAP valuess (for interpreting the scoring decisions) in the list. Please refer to <a href="https://github.com/slundberg/shap#citations">this link</a> for details of SHAP values. During the scoring process, the following files will be generated and saved to the same folder where your EDF/EDF+ files are located.
   1. "{EDF/EDF+ file name}_{model_name}_features.csv"; this file stores all the extracted feature values.
   2. "{EDF/EDF+ file name}_{model_name}_scores.csv"; this file stores the predicted sleep stages.
   3. "{EDF/EDF+ file name}_{model_name}_rs_100hz.npy"; this file stores a copy of the resampled/downsampled signals (100hz). To improve the speed of visualization, IntelliSleepScorer uses the downsampled signal instead of the original signal when plotting the signal.
   4. "{EDF/EDF+ file name}_{model_name}_explainer.pickle"; "{EDF/EDF+ file name}_{model_name}_shap_500samples.pickle"; "{EDF/EDF+ file name}_{model_name}_indicies_500samples.npy"; these files will be used when plotting the global SHAP values and epoch-level SHAP values.
6. After finishing the scoring process, you can click on "Visualize the Selected File" to visualize the EEG/EMG signals and a hypnogram time-aligned with the signals. Use the provided navigation buttons to move forward or backward. If LightGBM-2EEG was used for scoring, you can also view the global and epoch-level SHAP values. Right-click on an epoch to plot the epoch-level SHAP values. Currently, SHAP values have not been implemented for LightGBM-1EEG model.

# Contributors

Lei A. Wang, Ryan Kern, Eunah Yu, Soonwook Choi, Jen Q Pan

# License

IntelliSleepScorer software was released under the Creative Commons Attribution-NonCommercial-ShareAlike (CC-BY-NC-SA) license. It is free to academic users. For commercial use, please contact the authors for licenses.

# Contact

jpan@broadinstitute.org