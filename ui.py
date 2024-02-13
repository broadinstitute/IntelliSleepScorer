# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ABFbot_ui.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5.QtWidgets import QWidget, QPushButton, QListWidget, \
    QProgressBar, QLabel, QComboBox, QPlainTextEdit, QSizePolicy,\
    QGridLayout, QCheckBox
from PyQt5.QtCore import QRect, QMetaObject
from PyQt5.QtGui import QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT  
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        # MainWindow.center()
        # qr = self.frameGeometry()
        # cp = QDesktopWidget().availableGeometry().center()
        # qr.moveCenter(cp)
        # MainWindow.move(qr.topLeft())
        MainWindow.setFixedSize(1600, 1000)
        MainWindow.setWindowIcon(QIcon('logo.png'))

        MainWindow.setWindowTitle("IntelliSleepScorer v1.1")

        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)

        self.button_input_files = QPushButton(self)
        self.button_input_files.setGeometry(QRect(40, 20, 240, 40))
        self.button_input_files.setStyleSheet("font: 10pt;")
        self.button_input_files.setObjectName("button_input_files")
        self.button_input_files.setText("Select EDF/EDF+ File(s)")

        self.button_clear_input = QPushButton(self)
        self.button_clear_input.setGeometry(QRect(300, 20, 80, 40))
        self.button_clear_input.setStyleSheet("font: 10pt;")
        self.button_clear_input.setObjectName("button_clear_input")
        self.button_clear_input.setText("Clear")

        # Stage Codes
        self.label_wake_code = QLabel(self)
        self.label_wake_code.setGeometry(QRect(40, 80, 50, 20))
        self.label_wake_code.setStyleSheet("font: 10pt;")
        self.label_wake_code.setText("Wake:")

        self.label_nrem_code = QLabel(self)
        self.label_nrem_code.setGeometry(QRect(100, 80, 50, 20))
        self.label_nrem_code.setStyleSheet("font: 10pt;")
        self.label_nrem_code.setText("NREM:")

        self.label_rem_code = QLabel(self)
        self.label_rem_code.setGeometry(QRect(160, 80, 50, 20))
        self.label_rem_code.setStyleSheet("font: 10pt;")
        self.label_rem_code.setText("REM:")

        self.label_epoch_length = QLabel(self)
        self.label_epoch_length.setGeometry(QRect(220, 80, 160, 20))
        self.label_epoch_length.setStyleSheet("font: 10pt;")
        self.label_epoch_length.setText("Epoch Length:")


        # Select Model
        self.label_select_model = QLabel(self)
        self.label_select_model.setGeometry(QRect(40, 120, 100, 40))
        self.label_select_model.setStyleSheet("font: 10pt;")
        self.label_select_model.setText("Select Model")

        self.combobox_models = QComboBox(self)
        self.combobox_models.setGeometry(QRect(140, 120, 240, 40))


        self.listWidget_input = QListWidget(self)
        self.listWidget_input.setGeometry(QRect(40, 180, 340, 270))
        self.listWidget_input.setObjectName("listWidget_input")

        self.progressBar = QProgressBar(self)
        self.progressBar.setGeometry(QRect(40, 500, 340, 20))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setProperty("value", 0)
        self.progressBar.setStyleSheet("QProgressBar {background-color : lightgray}")

        self.label_status = QLabel(self)
        self.label_status.setGeometry(QRect(200, 500, 160, 20))
        self.label_status.setStyleSheet("font: 10pt;")
        self.label_status.setText("")
#        self.label_status.setObjectName("label_status")

        self.button_run = QPushButton(self)
        self.button_run.setGeometry(QRect(40, 540, 340, 40))
        self.button_run.setStyleSheet("font: 10pt;")
        self.button_run.setObjectName("button_run")
        self.button_run.setText("Score All Files")
        self.button_run.setEnabled(False)

        self.button_plot = QPushButton(self)
        self.button_plot.setGeometry(QRect(40, 600, 340, 40))
        self.button_plot.setStyleSheet("font: 10pt;")
        self.button_plot.setObjectName("button_plot")
        self.button_plot.setText("Visualize the Selected File")
        self.button_plot.setEnabled(False)

        self.checkbox = QCheckBox('RunSHAP', self)
        self.checkbox.stateChanged.connect(self.checkbox_for_running_SHAP)
        self.checkbox.move(75,450)
        
        # log
        self.textbox = QPlainTextEdit(self)
        self.textbox.setStyleSheet("font: 10pt;")
        self.textbox.setGeometry(QRect(40, 660, 340, 320))


        # Set right pane
        layout_right_pane = QGridLayout()
        
        # plot
        self.label_plot = QLabel()
        self.label_plot.setStyleSheet("font: 10pt;")
        self.label_plot.setWordWrap(True)
        self.label_plot.setText("Right click on an epoch to plot its SHAP values (LighGBM-2EEG only). Be patient, it takes a few seconds to update the plots.")

        self.figure = plt.figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        
        # Select number of epochs to display
        self.label_select_number_epochs = QLabel()
        self.label_select_number_epochs.setGeometry(QRect(0, 0, 200, 40))
        self.label_select_number_epochs.setStyleSheet("font: 10pt;")
        self.label_select_number_epochs.setText("Select Number of Epochs to Display")
        self.combobox_select_n_epochs = QComboBox()
        self.combobox_select_n_epochs.setGeometry(QRect(200, 0, 200, 40))
        self.combobox_select_n_epochs.addItem("All")
        self.combobox_select_n_epochs.addItem("100")
        self.combobox_select_n_epochs.addItem("10")
        self.combobox_select_n_epochs.addItem("5")
        self.combobox_select_n_epochs.addItem("3")
        self.combobox_select_n_epochs.addItem("2")
        self.combobox_select_n_epochs.addItem("1")
        self.combobox_select_n_epochs.setEnabled(False)

        # go to epoch

        self.button_goto_epoch = QPushButton(self)
        # self.button_goto_epoch.setGeometry(QRect(220, 540, 160, 40))
        self.button_goto_epoch.setStyleSheet("font: 10pt;")
        self.button_goto_epoch.setObjectName("button_goto_epoch")
        self.button_goto_epoch.setText("Go to Epoch")
        self.button_goto_epoch.setEnabled(False)


        self.button_previous = QPushButton(self)
        # self.button_previous.setGeometry(QRect(220, 540, 160, 40))
        self.button_previous.setStyleSheet("font: 10pt;")
        self.button_previous.setObjectName("button_previous")
        self.button_previous.setText("<")
        self.button_previous.setEnabled(False)

        self.button_previous_more = QPushButton(self)
        # self.button_previous_more.setGeometry(QRect(220, 540, 160, 40))
        self.button_previous_more.setStyleSheet("font: 10pt;")
        self.button_previous_more.setObjectName("button_previous_more")
        self.button_previous_more.setText("<<")
        self.button_previous_more.setEnabled(False)

        self.button_next = QPushButton(self)
        # self.button_next.setGeometry(QRect(220, 540, 160, 40))
        self.button_next.setStyleSheet("font: 10pt;")
        self.button_next.setObjectName("button_next")
        self.button_next.setText(">")
        self.button_next.setEnabled(False)

        self.button_next_more = QPushButton(self)
        # self.button_next_more.setGeometry(QRect(220, 540, 160, 40))
        self.button_next_more.setStyleSheet("font: 10pt;")
        self.button_next_more.setObjectName("button_next_more")
        self.button_next_more.setText(">>")
        self.button_next_more.setEnabled(False)

        self.figure_shap_epoch = plt.figure(layout="constrained")
        self.canvas_shap_epoch = FigureCanvasQTAgg(self.figure_shap_epoch)

        self.label_shap_epoch = QLabel()
        self.label_shap_epoch.setStyleSheet("font: 9pt;")
        self.label_shap_epoch.setWordWrap(True)
        self.label_shap_epoch.setText("Top 10 features with the highest absolute SHAP values for the selected epoch. Positive SHAP values indicate positive contribution to the prediction, and vice versa. If in the WAKE SHAP plot, you see a positive SHAP_Wake value for the feature 'emg_abs_max', it indicates that the 'emg_abs_max' value from the selected epoch increases the likelihood of the selected epoch being scored as Wake. Note that SHAP value only explains why the model makes the decision, it doesn't evaluate whether the decision is correct or not.")

        self.figure_shap_global = plt.figure(layout="constrained")
        self.canvas_shap_global = FigureCanvasQTAgg(self.figure_shap_global)

        self.label_shap_global = QLabel()
        self.label_shap_global.setStyleSheet("font: 9pt;")
        self.label_shap_global.setWordWrap(True)
        self.label_shap_global.setText("Top 10 features with the highest absolute Global SHAP values (calculated from 500 randomly sampled epochs). SHAP value shows how much a feature affected the prediction. Positive SHAP values indicate positive contribution to the prediction, and vice versa. Samples with redder color have higher feature values. Here is an example on how to interpret the plots. If in the WAKE SHAP plot, you see more redder dots on the right side of feature 'emg_abs_max' (more positive SHAP), it indicates that in general higher 'emg_abs_max' increases the likelihood of being scored as Wake. Note that SHAP value only explains why the model makes the decision, it doesn't evaluate whether the decision is correct or not.")

        layout_right_pane.addWidget(self.label_select_number_epochs, 0, 0, 1, 1)
        layout_right_pane.addWidget(self.combobox_select_n_epochs, 0, 1, 1, 1)
        layout_right_pane.addWidget(self.button_goto_epoch, 0, 2, 1, 1)
        layout_right_pane.addWidget(self.button_previous_more, 0, 3, 1, 1)
        layout_right_pane.addWidget(self.button_previous, 0, 4, 1, 1)
        layout_right_pane.addWidget(self.button_next, 0, 5, 1, 1)
        layout_right_pane.addWidget(self.button_next_more, 0, 6, 1, 1)
        layout_right_pane.addWidget(self.label_plot, 1, 0, 1, 7)
        layout_right_pane.addWidget(self.canvas, 2, 0, 1, 7)
        layout_right_pane.addWidget(self.canvas_shap_epoch, 3, 0, 1, 7)
        layout_right_pane.addWidget(self.label_shap_epoch, 4, 0, 1, 7)
        layout_right_pane.addWidget(self.canvas_shap_global, 5, 0, 1, 7)
        layout_right_pane.addWidget(self.label_shap_global, 6, 0, 1, 7)


        self.right_pane = QWidget(self)
        self.right_pane.setObjectName("right_pane")
        self.right_pane.setGeometry(QRect(420, 10, 1160, 980))
        self.right_pane.setLayout(layout_right_pane)

        QMetaObject.connectSlotsByName(MainWindow)


    def write(self, txt):
        self.textbox.appendPlainText(str(txt))


