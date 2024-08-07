from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import variable
import numpy as np
import pandas as pd
import sys
import os
import time
import mne
import joblib
import seaborn as sns
from PyQt5.QtWidgets import QInputDialog, QMainWindow, QApplication, QFileDialog

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from ui import Ui_MainWindow
from my_functions import *
import io
from contextlib import redirect_stdout

buf = io.StringIO()
redirect_stdout(buf)

PARAMETERS = pd.read_csv("./parameters.csv")
print(PARAMETERS)
P_STAGE_CODE = PARAMETERS[ PARAMETERS["parameter"].isin(["Wake","NREM","REM"]) ]
print(P_STAGE_CODE)
STAGE_CODE = P_STAGE_CODE.set_index("value").to_dict()["parameter"]
STAGE_CODE_REVERSE = P_STAGE_CODE.set_index("parameter").to_dict()["value"]
PARAMETERS = PARAMETERS.set_index("parameter").to_dict()["value"]
PARAMETERS['epoch_length'] = int(PARAMETERS['epoch_length'])


class Thread_run_all_files(QThread):
    signal = pyqtSignal('PyQt_PyObject')

    def __init__(self):
        QThread.__init__(self)
        self.filepath_list = []
        self.num_files = 0
        self.model_name = None
        

    def run(self):

        progress = 0
        self.signal.emit([progress,f"Selected Model: {self.model_name}.pkl"])

        self.num_files = len(self.filepath_list)
        self.model = joblib.load(f"./models/{self.model_name}.pkl")
        df = pd.DataFrame()
        for index, filepath in enumerate(self.filepath_list):

#            try:
            progress = (index+0.1)/self.num_files * 100
            starttime = time.time()
            self.signal.emit([progress,f"########\nStarted processing {filepath}"])
            self.signal.emit([progress,f"--{self.model_name}"])
            
            edfname = filepath.split("/")[-1]
            firstname = edfname.split('.edf')[0]
            folderpath = filepath.split(firstname)[0]
            csv_file = f"{firstname}_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_features.csv"

            if os.path.exists(f"{folderpath}{csv_file}"):
                self.signal.emit([progress,f"--Feature file exists, skipped extracting features"])
            else:
                self.signal.emit([progress,f"--Started extracting features"])

                if self.model_name == "1_LightGBM-2EEG":
                    message1 = save_single_edf_to_csv_2eeg(edf_filepath=filepath, model_name = self.model_name, epoch_len=PARAMETERS['epoch_length'])
                if self.model_name == "2_LightGBM-1EEG":
                    message1 = save_single_edf_to_csv_1eeg(edf_filepath=filepath, model_name = self.model_name, epoch_len=PARAMETERS['epoch_length'])
                
                progress = (index+0.5)/self.num_files * 100
                if message1 is not None:
                    self.signal.emit([progress,f"--{message1}"])
                else:
                    self.signal.emit([progress,f"--Finished extracting features"])

            df = pd.read_csv(f"{folderpath}{csv_file}")
            features = df.columns[1:-3].tolist()
            print(features)
            df['score'] = df['score'].astype("float")
            X = df[features]   
            progress = (index+0.3)/self.num_files * 100
            self.signal.emit([progress,f"--Started predicting the scores"])

            ##
            y_prediction = self.model.predict(X)
            progress = (index+0.5)/self.num_files * 100
            self.signal.emit([progress,f"--Finished prediction"])

            ##
            df_output = pd.DataFrame({
                "Epoch No.": list(range(len(y_prediction))),
                "Stage_Code": y_prediction
            })
            df_output["Stage_Code"] = df_output["Stage_Code"].astype("int")
            df_output["Stage"] = df_output["Stage_Code"].map(STAGE_CODE)
            df_output.to_csv( f"{folderpath}{firstname}_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_scores.csv", index=False)

            progress = (index+0.6)/self.num_files * 100
            self.signal.emit([progress,f"--Saved the score file at {folderpath}{firstname}_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_scores.csv"])

            if variable.run_SHAP:
            ##
                if self.model_name == "1_LightGBM-2EEG":
                    self.signal.emit([progress,f"--Calculating SHAP values"])
                    explainer, shap_values_500samples, indices_500samples = get_shap(df, features, model = self.model)
                    progress = (index+0.9)/self.num_files * 100
                    self.signal.emit([progress,f"--Finished calculating SHAP values"])
                    with open(f"{folderpath}{firstname}_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_explainer.pickle", 'wb') as handle:
                        pickle.dump(explainer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    with open(f"{folderpath}{firstname}_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_shap_500samples.pickle", 'wb') as handle:
                        pickle.dump(shap_values_500samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    np.save(f"{folderpath}{firstname}_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_indicies_500samples.npy", indices_500samples)

                    self.signal.emit([progress,f"--Saved SHAP values at {folderpath}{firstname}_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_shap_500samples.pickle"])

                if self.model_name == "2_LightGBM-1EEG":
                    self.signal.emit([progress,f"--Calculating SHAP values"])
                    explainer, shap_values_500samples, indices_500samples = get_shap(df, features, model = self.model)
                    progress = (index+0.9)/self.num_files * 100
                    self.signal.emit([progress,f"--Finished calculating SHAP values"])
                    with open(f"{folderpath}{firstname}_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_explainer.pickle", 'wb') as handle:
                        pickle.dump(explainer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    with open(f"{folderpath}{firstname}_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_shap_500samples.pickle", 'wb') as handle:
                        pickle.dump(shap_values_500samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    np.save(f"{folderpath}{firstname}_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_indicies_500samples.npy", indices_500samples)

                    self.signal.emit([progress,f"--Saved SHAP values at {folderpath}{firstname}_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_shap_500samples.pickle"])
            

            progress = (index+1)/self.num_files * 100
            elapsed_time = time.time() - starttime
            print("elapsed_time: ", elapsed_time)
            self.signal.emit([progress,f"elapsed_time: {elapsed_time}"])

        progress = 100
        self.signal.emit([progress, "Done!"])



class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)

        self.input_base_dir = './'
        self.filepath_list = []
        self.filename_list = [0]
        self.num_files = 0

        self.thread_run = Thread_run_all_files()
        self.thread_run.signal.connect(self.update_progress)

        self.button_input_files.clicked.connect(self.update_file_list)
        self.button_clear_input.clicked.connect(self.clear)
        self.button_run.clicked.connect(self.run_all_files)
        
        # self.combobox_models.currentIndexChanged.connect(self.update_plot_event)

        self.button_plot.clicked.connect(self.plot_edf)
        
        self.checkbox_run_shap.stateChanged.connect(self.checkbox_for_running_SHAP)

        print(STAGE_CODE_REVERSE['Wake'])
        self.label_wake_code.setText( f"Wake: {STAGE_CODE_REVERSE['Wake']}" )
        self.label_nrem_code.setText( f"NREM: {STAGE_CODE_REVERSE['NREM']}" )
        self.label_rem_code.setText( f"REM: {STAGE_CODE_REVERSE['REM']}" )
        print("epoch length: ", str(PARAMETERS['epoch_length']))
        self.combobox_epoch_length.setCurrentText(str(PARAMETERS['epoch_length']))

        self.combobox_select_n_epochs.currentIndexChanged.connect(self.update_display_n_epochs)
        self.button_goto_epoch.clicked.connect(self.update_display_goto_epoch)
        self.button_previous.clicked.connect(self.update_display_previous)
        self.button_previous_more.clicked.connect(self.update_display_previous_more)
        self.button_next.clicked.connect(self.update_display_next)
        self.button_next_more.clicked.connect(self.update_display_next_more)
        self.combobox_epoch_length.currentIndexChanged.connect(self.update_epoch_length)
        self.combobox_selected_epoch_stage.currentIndexChanged.connect(self.user_edit_stage)
        self.combobox_models.currentIndexChanged.connect(self.update_model)
        self.listWidget_input.itemSelectionChanged.connect(self.selectionChanged)
        

        self.map_colors = {1: "orange", 2: "blue", 3: "red"}
        self.map_stages = {1:"Wake", 2:"NREM", 3:"REM"}

        self.edf_path = None
        self.feature_file_path = None
        self.eeg_100zh_path = None
        self.score_file_path = None
        
        self.eeg_100hz = None
        self.n_channels = None
        self.df_score = None
        self.scores = None
        self.stages = None
        self.color_epochs = None
        self.score_file_user_edit_path = None
        self.df_score_user_edit = None
        self.n_epochs = None
        self.ch_names = None

        models = os.listdir("./models/")
        models = sorted(models)
        for model in models:
            self.combobox_models.addItem(model)

    def checkbox_for_running_SHAP(self, state):
        if state == 2:  # state 2 corresponds to checked state
            variable.run_SHAP = True
        else:
            variable.run_SHAP = False  # Reset to default background color
        #(50,50)

    def update_file_list(self):
        added_filepath_list = QFileDialog.getOpenFileNames(
            self, 'open file', self.input_base_dir, "EDF/EDF+ Files (*.edf)"
        )[0]
        if len(added_filepath_list) > 0:
            self.filepath_list += added_filepath_list
            self.num_files = len(self.filepath_list)
            self.filename_list = self.filename_list * self.num_files

            self.listWidget_input.clear()
            for i in range(self.num_files):
                last_slash_index = self.filepath_list[i].rfind('/')
                self.input_base_dir = self.filepath_list[i][:last_slash_index]
                self.filename_list[i] = self.filepath_list[i][last_slash_index + 1:]
                self.listWidget_input.addItem(self.filepath_list[i])

        if self.num_files > 0:
            self.button_run.setEnabled(True)
    

    # def update_plot_event(self):
    #     if str(self.combobox_models.currentText()).split(".")[0] == "1_LightGBM-2EEG":
    #         try: self.button_plot.clicked.disconnect(self.plot_edf_1eeg)
    #         except Exception: pass
    #         try: self.button_plot.clicked.disconnect(self.plot_edf)
    #         except Exception: pass
    #         self.button_plot.clicked.connect(self.plot_edf)
    #         print("1_LightGBM-2EEG")
    #     if str(self.combobox_models.currentText()).split(".")[0] == "2_LightGBM-1EEG":
    #         try: self.button_plot.clicked.disconnect(self.plot_edf_1eeg)
    #         except Exception: pass
    #         try: self.button_plot.clicked.disconnect(self.plot_edf)
    #         except Exception: pass
    #         self.button_plot.clicked.connect(self.plot_edf_1eeg)
    #         print("2_LightGBM-1EEG")


    def clear(self):
        self.progressBar.setProperty("value", 0)
        self.label_status.setText("")
        self.filepath_list = []
        self.filename_list = [0]
        self.num_files = 0
        self.listWidget_input.clear()
        self.button_run.setEnabled(False)


    @pyqtSlot("PyQt_PyObject")
    def update_progress(self, emitted_signal):
        progress = emitted_signal[0]
        message = emitted_signal[1]
        self.progressBar.setProperty('value', '{:.0f}'.format(progress))
        self.textbox.appendPlainText(f"{message}\n")
        if progress == 100:
            self.button_run.setEnabled(True)
            self.button_input_files.setEnabled(True)
            self.button_clear_input.setEnabled(True)
            self.label_status.setText("Done")


    def run_all_files(self):
        self.button_run.setEnabled(False)
        self.button_input_files.setEnabled(False)
        self.button_clear_input.setEnabled(False)
        self.label_status.setText("Processing...")

        self.thread_run.filepath_list = self.filepath_list
        self.thread_run.model_name = self.model_name
        self.thread_run.start()


    def plot_edf(self):
        print("start plottinng lightgbm-2EEG")

        print(self.listWidget_input.selectedItems())
        print(self.model_name)
        self.edf_path = self.listWidget_input.selectedItems()[0].text()
        self.feature_file_path = self.edf_path.replace(".edf", f"_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_features.csv")
        self.eeg_100zh_path = self.edf_path.replace(".edf", f"_{self.model_name}_rs_100hz.npy")
        self.score_file_path = self.edf_path.replace(".edf", f"_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_scores.csv")
        self.ch_names_path = self.edf_path.replace(".edf", f"_ch_names.pickle")
        
        self.ListWidget_warning.setText("")

        if os.path.exists(self.eeg_100zh_path):
            self.eeg_100hz = np.load(self.eeg_100zh_path)
            self.n_channels = self.eeg_100hz.shape[0]
            print("n_channels: ", self.n_channels)
        else:
            self.ListWidget_warning.setText(f"{self.eeg_100zh_path} doesn't exist, cannot visualize")
            return 
        
        if os.path.exists(self.ch_names_path):
            with open (self.ch_names_path, 'rb') as fp:
                self.ch_names = pickle.load(fp)
        else:
            self.ListWidget_warning.setText(f"{self.ch_names_path} doesn't exist, cannot visualize")
            return 

        if os.path.exists(self.score_file_path):
            self.df_score = pd.read_csv(self.score_file_path)
            self.scores = self.df_score["Stage_Code"].values
            self.stages = self.df_score["Stage"].values
            self.n_epochs = len(self.scores)
            self.color_epochs = [self.map_colors[e] for e in self.scores]
        else:  
            # self.button_plot.setEnabled(True)
            self.ListWidget_warning.setText(f"{self.score_file_path} doesn't exist, cannot visualize")
            return 

        self.score_file_user_edit_path = self.edf_path.replace(".edf", f"_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_scores_user_edit.csv")
        self.df_score_user_edit = None
        if os.path.exists(self.score_file_user_edit_path):
            self.df_score_user_edit = pd.read_csv(self.score_file_user_edit_path)

        self.figure.clf()
        self.canvas.draw()
        self.figure_shap_global.clf()
        self.canvas_shap_global.draw()
        self.figure_shap_epoch.clf()
        self.canvas_shap_epoch.draw()
        
        n_epochs_display = self.get_n_epochs_display()

        self.max_time = int(len(self.eeg_100hz[0])/100)   # 100hz sampling frequency
        time_sec = np.arange(0, self.max_time, 0.01)
        time_epochs_start = np.arange(0, self.max_time, PARAMETERS['epoch_length'])
        time_epochs_end = time_epochs_start + PARAMETERS['epoch_length']
        
        df = pd.read_csv(self.feature_file_path)

        self.features = df.columns[1:-3].tolist()
        df['score'] = self.df_score['Stage_Code'].astype("float")
        X = df[self.features]
        
        self.n_axes = self.n_channels + 1

        # ax1 plots first channel in eeg file
        self.ax1 = self.figure.add_subplot(self.n_axes,1,2)
        self.ax1.plot(time_sec, self.eeg_100hz[0])
        self.ax1.get_xaxis().set_visible(False)
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.spines['bottom'].set_visible(False)
        self.epoch_start = 0
        self.ax1.set_xlim(self.epoch_start,PARAMETERS['epoch_length']*n_epochs_display)
        self.ax1.set_ylabel(self.ch_names[0],rotation=0,fontsize=8,ha="right")

        # ax2 plots 2nd channel in eeg file
        ax2 = self.figure.add_subplot(self.n_axes,1,3, sharex=self.ax1)
        ax2.plot(time_sec, self.eeg_100hz[1])
        ax2.get_xaxis().set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.set_ylabel(self.ch_names[1],rotation=0,fontsize=8,ha="right")

        # ax3 plots 3rd channel in eeg file
        if self.n_channels == 3:
            ax3 = self.figure.add_subplot(self.n_axes,1,4, sharex=self.ax1)
            ax3.plot(time_sec, self.eeg_100hz[2])
            ax3.set_xlabel("Time (sec)")
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['bottom'].set_visible(False)
            ax3.set_ylabel(self.ch_names[2],rotation=0,fontsize=8,ha="right")

        # ax_hypnogram
        ax_hypnogram = self.figure.add_subplot(self.n_axes,1,1, sharex=self.ax1)
        ax_hypnogram.plot(time_epochs_start, self.scores, "|", markersize=7, color="black")
        ax_hypnogram.hlines(self.scores, time_epochs_start, time_epochs_end, colors=self.color_epochs, linewidths=3)
        ax_hypnogram.set_yticks([1,2,3])
        ax_hypnogram.set_yticklabels(["Wake", "NREM", "REM"])
        ax_hypnogram.get_xaxis().set_visible(False)
        ax_hypnogram.spines['top'].set_visible(False)
        ax_hypnogram.spines['right'].set_visible(False)
        ax_hypnogram.spines['bottom'].set_visible(False)

        if self.df_score_user_edit is not None:

            difference_locations = np.where(self.df_score["Stage_Code"] != self.df_score_user_edit["Stage_Code"])
            scores_changed_to = self.df_score_user_edit["Stage_Code"].values[difference_locations]
            
            for i_epoch, i_score in zip(difference_locations[0], scores_changed_to):

                time_epoch_start = PARAMETERS['epoch_length'] * i_epoch
                time_epoch_end = time_epoch_start + PARAMETERS['epoch_length']

                ax_hypnogram.hlines(i_score, time_epoch_start, time_epoch_end, 
                                        colors=self.map_colors[i_score], linewidths=3,
                                        linestyles="dotted")

        def onclick_lightgbm(event):
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                ('double' if event.dblclick else 'single', event.button,
                event.x, event.y, event.xdata, event.ydata))
                
            self.epoch_index_shap = int(event.xdata//PARAMETERS['epoch_length'])
            print("self.epoch_index_shap: ", self.epoch_index_shap)

            x = X.loc[[self.epoch_index_shap],:]
            
            try:
                self.highlight_selection1.remove()
            except:
                print("highlight_selection1 not found")
            try:
                self.highlight_selection2.remove()
            except:
                print("highlight_selection2 not found")
            try:
                self.highlight_selection3.remove()
            except:
                print("highlight_selection3 not found")
            try:
                self.highlight_selection4.remove()
            except:
                print("highlight_selection4 not found")

            self.highlight_selection1 = self.figure.axes[0].hlines(
                0,
                self.epoch_index_shap*PARAMETERS['epoch_length'], (self.epoch_index_shap+1)*PARAMETERS['epoch_length'],
                colors="pink", linewidths=100, alpha=0.4)
            self.highlight_selection2 = self.figure.axes[1].hlines(
                0,
                self.epoch_index_shap*PARAMETERS['epoch_length'], (self.epoch_index_shap+1)*PARAMETERS['epoch_length'],
                colors="pink", linewidths=100, alpha=0.4)
            self.highlight_selection3 = self.figure.axes[2].hlines(
                0,
                self.epoch_index_shap*PARAMETERS['epoch_length'], (self.epoch_index_shap+1)*PARAMETERS['epoch_length'],
                colors="pink", linewidths=100, alpha=0.4)
            if self.n_channels == 3:
                self.highlight_selection4 = self.figure.axes[3].hlines(
                    0,
                    self.epoch_index_shap*PARAMETERS['epoch_length'], (self.epoch_index_shap+1)*PARAMETERS['epoch_length'],
                    colors="pink", linewidths=100, alpha=0.4)
                    
            if event.button==3:
                if variable.run_SHAP:
                    self.figure_shap_epoch.clf()
                    shap_values_x = explainer.shap_values(x)
                    self.plot_shap_epoch(shap_values_x)

            self.canvas.draw()

            # update combobox_selected_epoch_stage
            self.combobox_selected_epoch_stage.setEnabled(True)
            self.combobox_selected_epoch_stage.setCurrentText(self.stages[self.epoch_index_shap])
        
        cid = self.canvas.mpl_connect('button_press_event', onclick_lightgbm)
        self.canvas.draw()
        print("finished plotting traces")

        # Enable the buttons
        self.combobox_select_n_epochs.setEnabled(True)
        self.button_goto_epoch.setEnabled(True)
        self.button_previous.setEnabled(True)
        self.button_previous_more.setEnabled(True)
        self.button_next.setEnabled(True)
        self.button_next_more.setEnabled(True)

        if variable.run_SHAP:

            explaner_path = self.edf_path.replace(".edf", f"_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_explainer.pickle")
            shap_500samples_path = self.edf_path.replace(".edf", f"_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_shap_500samples.pickle")
            indicies_500samples_path = self.edf_path.replace(".edf", f"_{self.model_name}_epoch_length_{PARAMETERS['epoch_length']}_sec_indicies_500samples.npy")

            ## Plot global SHAP values

            with open(explaner_path, 'rb') as handle:
                explainer = pickle.load(handle)

            with open(shap_500samples_path, 'rb') as handle:
                shap_values_500samples = pickle.load(handle)

            indicies_500samples = np.load(indicies_500samples_path).tolist()

            df_500samples = df.loc[indicies_500samples,]
            self.plot_shap_global(shap_values_500samples, df_500samples, indicies_500samples)


    # def plot_edf_1eeg(self):
    #     print("start plotting lightgbm-1EEG")

    #     message = "--SHAP values are currently not implemented for LightGBM-1EEG model;"
    #     self.textbox.appendPlainText(f"{message}\n")
    #     message = "--Will not show SHAP values"
    #     self.textbox.appendPlainText(f"{message}\n")
            
    #     self.figure.clf()
    #     self.canvas.draw()
    #     self.figure_shap_global.clf()
    #     self.canvas_shap_global.draw()
    #     self.figure_shap_epoch.clf()
    #     self.canvas_shap_epoch.draw()

    #     self.map_stages = {1:"Wake", 2:"NREM", 3:"REM"}

    #     edf_path = self.listWidget_input.selectedItems()[0].text()
        
    #     eeg_100zh_path = edf_path.replace(".edf", f"_{self.model_name}_rs_100hz.npy")
    #     score_file_path = edf_path.replace(".edf", f"_{self.model_name}_scores.csv")
        
    #     eeg_100hz = np.load(eeg_100zh_path)
    #     n_channels = eeg_100hz.shape[0]
    #     print("n_channels: ", n_channels)
    #     df_score = pd.read_csv(score_file_path)
    #     scores = df_score["Stage_Code"].values
    #     map_colors = {1: "orange", 2: "blue", 3: "red"}
    #     color_epochs = [map_colors[e] for e in scores]

    #     self.n_epochs = len(scores)
    #     n_epochs_display = self.get_n_epochs_display()

    #     self.max_time = int(len(eeg_100hz[0])/100)   # 100hz sampling frequency
    #     time_sec = np.arange(0, self.max_time, 0.01)
    #     time_epochs_start = np.arange(0, self.max_time, PARAMETERS['epoch_length'])
    #     time_epochs_end = time_epochs_start + PARAMETERS['epoch_length']
    
    #     self.n_axes = n_channels + 1

    #     self.ax1 = self.figure.add_subplot(self.n_axes,1,2)
    #     self.ax1.plot(time_sec, eeg_100hz[0])
    #     self.ax1.get_xaxis().set_visible(False)
    #     self.ax1.spines['top'].set_visible(False)
    #     self.ax1.spines['right'].set_visible(False)
    #     self.ax1.spines['bottom'].set_visible(False)
    #     self.epoch_start = 0
    #     self.ax1.set_xlim(self.epoch_start,PARAMETERS['epoch_length']*n_epochs_display)
        
    #     self.ax1.set_ylabel('EEG',rotation=0,fontsize=8,ha="right")

    #     ax2 = self.figure.add_subplot(self.n_axes,1,3, sharex=self.ax1)
    #     ax2.plot(time_sec, eeg_100hz[1])
    #     ax2.get_xaxis().set_visible(False)
    #     ax2.spines['top'].set_visible(False)
    #     ax2.spines['right'].set_visible(False)
    #     ax2.spines['bottom'].set_visible(False)
    #     ax2.set_ylabel('EMG',rotation=0,fontsize=8,ha="right")

    #     ax_hypnogram = self.figure.add_subplot(self.n_axes,1,1, sharex=self.ax1)
    #     ax_hypnogram.plot(time_epochs_start, scores, "|", markersize=7, color="black")
    #     ax_hypnogram.hlines(scores, time_epochs_start, time_epochs_end, colors=color_epochs, linewidths=3)
    #     ax_hypnogram.set_yticks([1,2,3])
    #     ax_hypnogram.set_yticklabels(["Wake", "NREM", "REM"])
    #     ax_hypnogram.get_xaxis().set_visible(False)
    #     ax_hypnogram.spines['top'].set_visible(False)
    #     ax_hypnogram.spines['right'].set_visible(False)
    #     ax_hypnogram.spines['bottom'].set_visible(False)
        
    #     self.canvas.draw()
    #     print("finished plotting traces")

    #     # def onclick_lightgbm(event):
    #     #     print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #     #         ('double' if event.dblclick else 'single', event.button,
    #     #         event.x, event.y, event.xdata, event.ydata))
            
    #     #     # if event.dblclick:
    #     #     #     display_n_epochs_list = [100, 10, 5, 3, 2, 1]
    #     #     #     n_epochs_display_current = self.get_n_epochs_display()
    #     #     #     index_n_epochs_display = 1 + next(
    #     #     #         (i for i,x in enumerate(display_n_epochs_list) if x < n_epochs_display_current))
    #     #     #     self.epoch_start = int(event.xdata//PARAMETERS['epoch_length'])
    #     #     #     self.combobox_select_n_epochs.setCurrentIndex(index_n_epochs_display)

    #     #     if event.button==3:
    #     #         self.figure_shap_epoch.clf()
                
    #     #         self.epoch_index_shap = int(event.xdata//PARAMETERS['epoch_length'])
    #     #         print("self.epoch_index_shap: ", self.epoch_index_shap)

    #     #         x = X.loc[[self.epoch_index_shap],:]
    #     #         shap_values_x = explainer.shap_values(x)
    #     #         self.plot_shap_epoch(shap_values_x)

    #     #         try:
    #     #             self.highlight_selection1.remove()
    #     #         except:
    #     #             print("highlight_selection1 not found")
    #     #         try:
    #     #             self.highlight_selection2.remove()
    #     #         except:
    #     #             print("highlight_selection2 not found")
    #     #         try:
    #     #             self.highlight_selection3.remove()
    #     #         except:
    #     #             print("highlight_selection3 not found")
    #     #         try:
    #     #             self.highlight_selection4.remove()
    #     #         except:
    #     #             print("highlight_selection4 not found")

    #     #         self.highlight_selection1 = self.figure.axes[0].hlines(
    #     #             0,
    #     #             self.epoch_index_shap*PARAMETERS['epoch_length'], (self.epoch_index_shap+1)*PARAMETERS['epoch_length'],
    #     #             colors="pink", linewidths=100, alpha=0.4)
    #     #         self.highlight_selection2 = self.figure.axes[1].hlines(
    #     #             0,
    #     #             self.epoch_index_shap*PARAMETERS['epoch_length'], (self.epoch_index_shap+1)*PARAMETERS['epoch_length'],
    #     #             colors="pink", linewidths=100, alpha=0.4)
    #     #         self.highlight_selection3 = self.figure.axes[2].hlines(
    #     #             0,
    #     #             self.epoch_index_shap*PARAMETERS['epoch_length'], (self.epoch_index_shap+1)*PARAMETERS['epoch_length'],
    #     #             colors="pink", linewidths=100, alpha=0.4)
    #     #         if n_channels == 3:
    #     #             self.highlight_selection4 = self.figure.axes[3].hlines(
    #     #                 0,
    #     #                 self.epoch_index_shap*PARAMETERS['epoch_length'], (self.epoch_index_shap+1)*PARAMETERS['epoch_length'],
    #     #                 colors="pink", linewidths=100, alpha=0.4)
    #     #         self.canvas.draw()
        
    #     # cid = self.canvas.mpl_connect('button_press_event', onclick_lightgbm)
    #     # self.canvas.draw()
    #     # print("finished plotting traces")

    #     # ## Plot global SHAP values
    #     # df = pd.read_csv(feature_file_path)
    #     # df_score = pd.read_csv(score_file_path)

    #     # self.features = df.columns[1:-3].tolist()
    #     # df['score'] = df_score['Stage_Code'].astype("float")
    #     # X = df[self.features]

    #     # with open(explaner_path, 'rb') as handle:
    #     #     explainer = pickle.load(handle)

    #     # with open(shap_500samples_path, 'rb') as handle:
    #     #     shap_values_500samples = pickle.load(handle)

    #     # indicies_500samples = np.load(indicies_500samples_path).tolist()

    #     # df_500samples = df.loc[indicies_500samples,]
    #     # self.plot_shap_global(shap_values_500samples, df_500samples, indicies_500samples)

    #     # Enable the buttons
    #     self.combobox_select_n_epochs.setEnabled(True)
    #     self.combobox_selected_epoch_stage.setEnabled(True)
    #     self.button_goto_epoch.setEnabled(True)
    #     self.button_previous.setEnabled(True)
    #     self.button_previous_more.setEnabled(True)
    #     self.button_next.setEnabled(True)
    #     self.button_next_more.setEnabled(True)


    def plot_shap_global(self, shap_values_500samples, df_500samples, indicies_500samples):
        
        dict_shap = {
            "feature":[],
            "feature_value":[],
            "Epoch":[],
            "SHAP_Wake":[],
            "SHAP_NREM":[],
            "SHAP_REM":[],
        }

        for i in range(500):
            dict_shap["feature"] += self.features
            dict_shap["feature_value"] += df_500samples.loc[indicies_500samples[i], self.features].tolist()
            dict_shap["Epoch"] += [indicies_500samples[i]] * len(self.features)
            dict_shap["SHAP_Wake"] += shap_values_500samples[0][i].tolist()
            dict_shap["SHAP_NREM"] += shap_values_500samples[1][i].tolist()
            dict_shap["SHAP_REM"] += shap_values_500samples[2][i].tolist()
        df_shap_global = pd.DataFrame(dict_shap)
        map_feature_value_max = df_shap_global.groupby("feature").max()["feature_value"].to_dict()
        map_feature_value_min = df_shap_global.groupby("feature").min()["feature_value"].to_dict()
        print(map_feature_value_max)
        df_shap_global["feature_value_max"] = df_shap_global["feature"].map(map_feature_value_max)
        df_shap_global["feature_value_min"] = df_shap_global["feature"].map(map_feature_value_min)
        df_shap_global["feature_value_normalized"] = \
            (df_shap_global["feature_value"] - df_shap_global["feature_value_min"])\
                /(df_shap_global["feature_value_max"] - df_shap_global["feature_value_min"])
        print(df_shap_global)


        df_shap = pd.DataFrame({
            "feature": self.features,
            "average_SHAP_Wake": np.mean(shap_values_500samples[0], axis=0),
            "average_SHAP_NREM": np.mean(shap_values_500samples[1], axis=0),
            "average_SHAP_REM": np.mean(shap_values_500samples[2], axis=0),
        })

        df_shap["average_SHAP_Wake_abs"] = df_shap["average_SHAP_Wake"].abs()
        df_shap["average_SHAP_NREM_abs"] = df_shap["average_SHAP_NREM"].abs()
        df_shap["average_SHAP_REM_abs"] = df_shap["average_SHAP_REM"].abs()

        df_shap["average_SHAP_Wake_sign"] = np.sign(df_shap["average_SHAP_Wake"])
        df_shap["average_SHAP_NREM_sign"] = np.sign(df_shap["average_SHAP_NREM"])
        df_shap["average_SHAP_REM_sign"] = np.sign(df_shap["average_SHAP_REM"])

        map_color = {1:"red", -1:"blue"}
        df_shap["SHAP_Wake_color"] = df_shap["average_SHAP_Wake_sign"].map(map_color)
        df_shap["SHAP_NREM_color"] = df_shap["average_SHAP_NREM_sign"].map(map_color)
        df_shap["SHAP_REM_color"] = df_shap["average_SHAP_REM_sign"].map(map_color)

        df_shap_wake_top10 = \
            df_shap.sort_values("average_SHAP_Wake_abs").iloc[-10:,]
        top10_features_wake = df_shap_wake_top10["feature"]
        df_shap_global_wake_top10 = df_shap_global[df_shap_global["feature"].isin(top10_features_wake)]
        
        df_shap_nrem_top10 = \
            df_shap.sort_values("average_SHAP_NREM_abs").iloc[-10:,]
        top10_features_nrem = df_shap_nrem_top10["feature"]
        df_shap_global_nrem_top10 = df_shap_global[df_shap_global["feature"].isin(top10_features_nrem)]

        df_shap_rem_top10 = \
            df_shap.sort_values("average_SHAP_REM_abs").iloc[-10:,]
        top10_features_rem = df_shap_rem_top10["feature"]
        df_shap_global_rem_top10 = df_shap_global[df_shap_global["feature"].isin(top10_features_rem)]

        ax_shap_wake = self.figure_shap_global.add_subplot(1,3,1)
        ax_shap_wake.tick_params(axis='y', which='major', labelsize=8)
        g = sns.stripplot(data=df_shap_global_wake_top10, x="SHAP_Wake", y="feature", 
            hue="feature_value_normalized", size=2, palette=sns.color_palette("light:#ff0000", as_cmap=True), 
            ax=ax_shap_wake)
        g.legend_.remove()
        ax_shap_wake.vlines(0,-0.5,9.5, color="gray")
        ax_shap_wake.set_ylabel(None)
        ax_shap_wake.set_title(f"Global Wake SHAP Values", fontsize=10)
        
        ax_shap_nrem = self.figure_shap_global.add_subplot(1,3,2)
        ax_shap_nrem.tick_params(axis='y', which='major', labelsize=8)
        g = sns.stripplot(data=df_shap_global_nrem_top10, x="SHAP_NREM", y="feature", 
            hue="feature_value_normalized", size=2, palette=sns.color_palette("light:#ff0000", as_cmap=True), 
            ax=ax_shap_nrem)
        g.legend_.remove()
        ax_shap_nrem.vlines(0,-0.5,9.5, color="gray")
        ax_shap_nrem.set_ylabel(None)
        ax_shap_nrem.set_title(f"Global NREM SHAP Values", fontsize=10)
        
        ax_shap_rem = self.figure_shap_global.add_subplot(1,3,3)
        ax_shap_rem.tick_params(axis='y', which='major', labelsize=8)
        g = sns.stripplot(data=df_shap_global_rem_top10, x="SHAP_REM", y="feature", 
            hue="feature_value_normalized", size=2, palette=sns.color_palette("light:#ff0000", as_cmap=True), 
            ax=ax_shap_rem)
        g.legend_.remove()
        ax_shap_rem.vlines(0,-0.5,9.5, color="gray")
        ax_shap_rem.set_ylabel(None)
        ax_shap_rem.set_title(f"Global REM SHAP Values", fontsize=10)

        self.canvas_shap_global.draw()


    def plot_shap_epoch(self, shap_values_x):
        
        df_shap = pd.DataFrame({
            "feature": self.features,
            "SHAP_Wake": shap_values_x[0][0],
            "SHAP_NREM": shap_values_x[1][0],
            "SHAP_REM": shap_values_x[2][0],
        })

        df_shap["SHAP_Wake_abs"] = df_shap["SHAP_Wake"].abs()
        df_shap["SHAP_NREM_abs"] = df_shap["SHAP_NREM"].abs()
        df_shap["SHAP_REM_abs"] = df_shap["SHAP_REM"].abs()

        df_shap["SHAP_Wake_sign"] = np.sign(df_shap["SHAP_Wake"])
        df_shap["SHAP_NREM_sign"] = np.sign(df_shap["SHAP_NREM"])
        df_shap["SHAP_REM_sign"] = np.sign(df_shap["SHAP_REM"])

        map_color = {1:"red", -1:"blue"}
        df_shap["SHAP_Wake_color"] = df_shap["SHAP_Wake_sign"].map(map_color)
        df_shap["SHAP_NREM_color"] = df_shap["SHAP_NREM_sign"].map(map_color)
        df_shap["SHAP_REM_color"] = df_shap["SHAP_REM_sign"].map(map_color)

        df_shap_wake_top10 = \
            df_shap.sort_values("SHAP_Wake_abs").iloc[-10:,]
        
        df_shap_nrem_top10 = \
            df_shap.sort_values("SHAP_NREM_abs").iloc[-10:,]

        df_shap_rem_top10 = \
            df_shap.sort_values("SHAP_REM_abs").iloc[-10:,]

        ax_shap_wake = self.figure_shap_epoch.add_subplot(1,3,1)
        ax_shap_wake.barh(width=df_shap_wake_top10["SHAP_Wake"], 
                y=df_shap_wake_top10["feature"], 
                color=df_shap_wake_top10["SHAP_Wake_color"])
        ax_shap_wake.tick_params(axis='y', which='major', labelsize=8)
        ax_shap_wake.set_title(f"Wake SHAP Values\nEpoch: {self.epoch_index_shap}", fontsize=10)
        
        ax_shap_nrem = self.figure_shap_epoch.add_subplot(1,3,2)
        ax_shap_nrem.barh(width=df_shap_nrem_top10["SHAP_NREM"], 
                y=df_shap_nrem_top10["feature"], 
                color=df_shap_nrem_top10["SHAP_NREM_color"])
        ax_shap_nrem.tick_params(axis='y', which='major', labelsize=8)
        ax_shap_nrem.set_title(f"NREM SHAP Values\nEpoch: {self.epoch_index_shap}", fontsize=10)
        
        ax_shap_rem = self.figure_shap_epoch.add_subplot(1,3,3)
        ax_shap_rem.barh(width=df_shap_rem_top10["SHAP_REM"], 
                y=df_shap_rem_top10["feature"], 
                color=df_shap_rem_top10["SHAP_REM_color"])
        ax_shap_rem.tick_params(axis='y', which='major', labelsize=8)
        ax_shap_rem.set_title(f"REM SHAP Values\nEpoch: {self.epoch_index_shap}", fontsize=10)

        self.canvas_shap_epoch.draw()


    def get_n_epochs_display(self):
        n_epochs_display = 10
        selected_n_epochs = self.combobox_select_n_epochs.currentText()
        if selected_n_epochs == "All":
            n_epochs_display = self.n_epochs
        else:
            n_epochs_display = int(selected_n_epochs)
        return n_epochs_display


    def update_display_goto_epoch(self):
        n_epochs_display = self.get_n_epochs_display()

        self.epoch_start, done = QInputDialog.getInt(
           self, 'Input Dialog', 'Enter the Epoch You Want to View:')

        self.figure.axes[0].set_xlim(PARAMETERS['epoch_length']*self.epoch_start, PARAMETERS['epoch_length']*(self.epoch_start + n_epochs_display))
        self.update_epoch_labels(n_epochs_display)
        self.canvas.draw()


    def update_display_n_epochs(self):
        n_epochs_display = self.get_n_epochs_display()
        self.figure.axes[0].set_xlim(PARAMETERS['epoch_length']*self.epoch_start, PARAMETERS['epoch_length']*(self.epoch_start + n_epochs_display))
        self.update_epoch_labels(n_epochs_display)
        self.canvas.draw()


    def update_epoch_labels(self, n_epochs_display):
        if n_epochs_display < 100:
            try:
                for i in range(10):
                    self.epochlabels[i].remove()
            except:
                print("No Epoch labels")
            self.epochlabels = []
            for i in range(10):
                self.epochlabels.append(
                    self.figure.axes[self.n_axes-1].annotate(
                        str(self.epoch_start+i), 
                        ((self.epoch_start+i+0.5)*PARAMETERS['epoch_length'], 1), ha="center", va="bottom")
                )
        else:
            try:
                for i in range(10):
                    self.epochlabels[i].remove()
            except:
                print("No Epoch labels")


    def update_display_previous(self):
        n_epochs_display = self.get_n_epochs_display()
        self.epoch_start -= 1
        self.figure.axes[0].set_xlim(PARAMETERS['epoch_length']*self.epoch_start, PARAMETERS['epoch_length']*(self.epoch_start + n_epochs_display) )
        self.update_epoch_labels(n_epochs_display)
        self.canvas.draw()


    def update_display_previous_more(self):
        n_epochs_display = self.get_n_epochs_display()
        self.epoch_start -= n_epochs_display
        self.figure.axes[0].set_xlim(PARAMETERS['epoch_length']*self.epoch_start, PARAMETERS['epoch_length']*(self.epoch_start + n_epochs_display) )
        self.update_epoch_labels(n_epochs_display)
        self.canvas.draw()


    def update_display_next(self):
        n_epochs_display = self.get_n_epochs_display()
        self.epoch_start += 1
        self.figure.axes[0].set_xlim(PARAMETERS['epoch_length']*self.epoch_start, PARAMETERS['epoch_length']*(self.epoch_start + n_epochs_display) )
        self.update_epoch_labels(n_epochs_display)
        self.canvas.draw()


    def update_display_next_more(self):
        n_epochs_display = self.get_n_epochs_display()
        self.epoch_start += n_epochs_display
        self.figure.axes[0].set_xlim(PARAMETERS['epoch_length']*self.epoch_start, PARAMETERS['epoch_length']*(self.epoch_start + n_epochs_display) )
        self.update_epoch_labels(n_epochs_display)
        self.canvas.draw()


    def update_epoch_length(self):
        PARAMETERS['epoch_length'] = int(self.combobox_epoch_length.currentText())
        pd.DataFrame(PARAMETERS, index=[0]).T\
            .reset_index(drop=False)\
            .rename(columns={"index":"parameter", 0:"value"})\
            .to_csv("parameters.csv", index=False)


    def update_model(self):
        self.model_name = str(self.combobox_models.currentText()).split(".")[0]
        if self.model_name == "2_LightGBM-1EEG":
            self.checkbox_run_shap.setChecked(False)
            self.checkbox_run_shap.setEnabled(False)
        if self.model_name == "1_LightGBM-2EEG":
            self.checkbox_run_shap.setEnabled(True)


    def selectionChanged(self):
        if self.listWidget_input.count() > 0:
        #if int(self.combobox_epoch_length.currentText()) == 
            self.button_plot.setEnabled(True)        
            
        
    def user_edit_stage(self):
        ## write to csv file
        if self.df_score_user_edit is None:
            self.df_score_user_edit = self.df_score

        stage_user_edit = self.combobox_selected_epoch_stage.currentText()
        score_user_edit = STAGE_CODE_REVERSE[stage_user_edit]
        self.df_score_user_edit.loc[self.df_score_user_edit["Epoch No."]==self.epoch_index_shap, "Stage"] =  stage_user_edit
        self.df_score_user_edit.loc[self.df_score_user_edit["Epoch No."]==self.epoch_index_shap, "Stage_Code"] = score_user_edit
        
        self.df_score_user_edit.to_csv(self.score_file_user_edit_path, index=False)

        ## update plot

        ax_hypnogram_edit = self.figure.get_axes()[self.n_axes-1]
        print(ax_hypnogram_edit)

        time_epoch_start = PARAMETERS['epoch_length'] * self.epoch_index_shap
        print(time_epoch_start)
        time_epoch_end = time_epoch_start + PARAMETERS['epoch_length']

        ax_hypnogram_edit.hlines(score_user_edit, time_epoch_start, time_epoch_end, 
                                 colors=self.map_colors[score_user_edit], linewidths=3,
                                 linestyles="dotted")
        
        self.canvas.draw()

        # time_epochs_start = np.arange(0, self.max_time, PARAMETERS['epoch_length'])
        # time_epochs_end = time_epochs_start + PARAMETERS['epoch_length']
        # ax_hypnogram.hlines(scores, time_epochs_start, time_epochs_end, colors=color_epochs, linewidths=3)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
#    sys.stdout=myWin
    myWin.show()
    sys.exit(app.exec_())

print('Done')
