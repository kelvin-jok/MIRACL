#! /usr/bin/env python

# coding: utf-8

from pprint import pprint
import sys
import os
import matplotlib
from  matplotlib import colorbar
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
from math import nan

# heatmap GUI setup

class STATSHeatmapMenu(QtWidgets.QWidget):

    def __init__(self):
        # create GUI
        QtWidgets.QMainWindow.__init__(self)
        super(STATSHeatmapMenu, self).__init__()
        self.setWindowTitle('STATS Heatmap Analysis')

        # Set the window dimensions
        # self.resize(500,200)

        # Create labels which displays the path to our chosen file
        self.lbl1 = QtWidgets.QLabel('No folder selected for Group 1 input data')
        self.lbl2 = QtWidgets.QLabel('No folder selected for Group 2 input data')
        self.lbl3 = QtWidgets.QLabel('No folder selected for Output. Default: Current Working Directory')
        # Create labels for Notes/Text
        group2_note = QtWidgets.QLabel()
        group2_note.setText(
            'NOTE: If Group 2 folder is selected then will generate heatmaps for Group 1, Group 2, and the Difference of Groups (Group 2 - Group 1)')
        group2_note.setAlignment(QtCore.Qt.AlignLeft)
        group2_note.setFont(QtGui.QFont("Sans Serif 10", weight=QtGui.QFont.Black))
        title_cmap = QtWidgets.QLabel()
        title_cmap.setText('Use Website or Button Below to Explore Colourmap Options')
        title_cmap.setAlignment(QtCore.Qt.AlignRight)
        website = QtWidgets.QLabel()
        website.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        website.setText('https://matplotlib.org/stable/tutorials/colors/colormaps.html')
        website.setAlignment(QtCore.Qt.AlignRight)
        custom_slice = QtWidgets.QLabel()
        custom_slice.setText(
            'NOTE: If None of the custom slicing axes are enabled (sagittal, coronal, axial) then the default slicing parameters will be used')
        custom_slice.setAlignment(QtCore.Qt.AlignLeft)
        custom_slice.setFont(QtGui.QFont("Sans Serif 10", weight=QtGui.QFont.Black))
        output_note = QtWidgets.QLabel()
        output_note.setText('NOTE: Output Filenames MUST NOT uses spaces. Use underscore instead')
        output_note.setAlignment(QtCore.Qt.AlignLeft)
        output_note.setFont(QtGui.QFont("Sans Serif 10", weight=QtGui.QFont.Black))
        group2_selected = QtWidgets.QLabel()
        group2_selected.setAlignment(QtCore.Qt.AlignLeft)
        group2_selected.setText('NOTE: Must Specify All Output Filenames if Group 2 input data folder is selected')
        group2_selected.setFont(QtGui.QFont("Sans Serif 10", weight=QtGui.QFont.Black))

        # Create push buttons for Folder Selection
        btn1 = QtWidgets.QPushButton('Select Group 1 input data folder', self)
        btn2 = QtWidgets.QPushButton('Select Group 2 input data folder', self)
        btn3 = QtWidgets.QPushButton('Select Output folder', self)

        # colourmap push button option
        btn4 = QtWidgets.QPushButton('Matplotlib Ver. 3.4.2 Colourmap Options', self)

        # Enable/Disabled Checkbox Options
        self.sagittal_enable = QtWidgets.QCheckBox("Enable  Custom Sagittal Slicing")
        self.sagittal_status = self.sagittal_enable.isChecked()
        self.coronal_enable = QtWidgets.QCheckBox("Enable Custom Coronal Slicing")
        self.coronal_status = self.coronal_enable.isChecked()
        self.axial_enable = QtWidgets.QCheckBox("Enable Custom Axial Slicing")
        self.axial_status = self.axial_enable.isChecked()
        self.fig_enable = QtWidgets.QCheckBox("Enable Custom Figure Dimensions")
        self.fig_status = self.fig_enable.isChecked()
        # Run GUI Button
        run = QtWidgets.QPushButton('Run', self)

        # Create other inputs (Spinbox/line edits)
        self.gauss = QtWidgets.QDoubleSpinBox()
        self.gauss.setMinimum(0)
        self.gauss.setValue(4)
        self.gauss.setAlignment(QtCore.Qt.AlignRight)
        self.voxels = QtWidgets.QComboBox()
        self.voxels.setEditable(True)
        self.voxels.lineEdit().setReadOnly(True)
        self.voxels.showPopup = self.showPopupAndCheck
        self.voxels.addItems(["10", "25", "50"])
        self.voxels.setCurrentIndex(1)
        self.cmap_pos = QtWidgets.QLineEdit()
        self.cmap_pos.setText("Reds")
        self.cmap_pos.setAlignment(QtCore.Qt.AlignRight)
        self.cmap_neg = QtWidgets.QLineEdit()
        self.cmap_neg.setText("Blues")
        self.cmap_neg.setAlignment(QtCore.Qt.AlignRight)

        # custom slice axis input and layout
        self.axis = []
        self.slice_layout = []

        for i in range(3):
            self.slices = []
            self.slice_layout.append(QtWidgets.QHBoxLayout())
            for j in range(5):
                self.slices.append(QtWidgets.QSpinBox())
                self.slices[j].setMinimum(1)
                self.slices[j].setMaximum(10000)
                self.slices[j].setAlignment(QtCore.Qt.AlignLeft)
                self.slices[j].setDisabled(True)
                self.slice_layout[i].addWidget(self.slices[j])
            self.axis.append(self.slices)
            del self.slices

        # custom figure dimensions input and layout
        self.figure_dimensions_width = QtWidgets.QDoubleSpinBox()
        self.figure_dimensions_width.setMinimum(1)
        self.figure_dimensions_width.setMaximum(60.0)
        self.figure_dimensions_width.setValue(7.0)
        self.figure_dimensions_width.setAlignment(QtCore.Qt.AlignLeft)
        self.figure_dimensions_width.setDisabled(True)
        self.figure_dimensions_height = QtWidgets.QDoubleSpinBox()
        self.figure_dimensions_height.setMinimum(1)
        self.figure_dimensions_height.setMaximum(60.0)
        self.figure_dimensions_height.setValue(3.0)
        self.figure_dimensions_height.setAlignment(QtCore.Qt.AlignLeft)
        self.figure_dimensions_height.setDisabled(True)

        dim_layout = QtWidgets.QHBoxLayout()
        dim_layout.addWidget(self.figure_dimensions_width)
        dim_layout.addWidget(self.figure_dimensions_height)

        # Remaining inputs (Spinbox/line edits)
        self.outfiles_g1 = QtWidgets.QLineEdit()
        self.outfiles_g1.setText("Group_1")
        self.outfiles_g1.setAlignment(QtCore.Qt.AlignRight)
        self.outfiles_g2 = QtWidgets.QLineEdit()
        self.outfiles_g2.setAlignment(QtCore.Qt.AlignRight)
        self.outfiles_dif = QtWidgets.QLineEdit()
        self.outfiles_dif.setAlignment(QtCore.Qt.AlignRight)
        self.extensions = QtWidgets.QLineEdit()
        self.extensions.setText("tiff")
        self.extensions.setAlignment(QtCore.Qt.AlignRight)
        self.dots_per_inch = QtWidgets.QSpinBox()
        self.dots_per_inch.setMinimum(1)
        self.dots_per_inch.setMaximum(1200)
        self.dots_per_inch.setValue(500)
        self.dots_per_inch.setAlignment(QtCore.Qt.AlignRight)
        self.template = QtWidgets.QLineEdit()
        self.template.setText("_brainmask")
        self.template.setAlignment(QtCore.Qt.AlignRight)

        # layout for input widgets
        self.layout = QtWidgets.QFormLayout()
        self.layout.addRow(self.lbl1, btn1)
        self.layout.addRow("Voxel Size - Choose from 10, 25, or 50 um", self.voxels)
        self.layout.addRow(self.lbl2, btn2)
        self.layout.addRow(group2_note)
        self.layout.addRow("Gaussian smoothing sigma (Non-negative number. ex: 1, 2 or 2.5)", self.gauss)
        self.layout.addRow(title_cmap)
        self.layout.addRow(website)
        self.layout.addRow("", btn4)
        self.layout.addRow("Colourmap for Positive Values", self.cmap_pos)
        self.layout.addRow("Colourmap for Negative Values", self.cmap_neg)
        self.layout.addRow("Enable Custom Sagittal Slicing", self.sagittal_enable)
        self.layout.addRow(
            "sagittal slicing: start_slice_number, interval, number_of_slices, number_of_rows, number_of_columns",
            self.slice_layout[0])
        self.layout.addRow("Enable Custom Coronal Slicing", self.coronal_enable)
        self.layout.addRow(
            "coronal slicing: start_slice_number, interval, number_of_slices, number_of_rows, number_of_columns",
            self.slice_layout[1])
        self.layout.addRow("Enable Custom Axial Slicing", self.axial_enable)
        self.layout.addRow(
            "axial slicing: start_slice_number, interval, number_of_slices, number_of_rows, number_of_columns",
            self.slice_layout[2])
        self.layout.addRow(custom_slice)
        self.layout.addRow("Enable Custom Figure Dimensions", self.fig_enable)
        self.layout.addRow("Figure Width and Height", dim_layout)
        self.layout.addRow(self.lbl3, btn3)
        self.layout.addRow(output_note)
        self.layout.addRow("Output Filename Group 1", self.outfiles_g1)
        self.layout.addRow(group2_selected)
        self.layout.addRow("Output Filename Group 2", self.outfiles_g2)
        self.layout.addRow("Output Filename Difference of Groups (Group 2 - Group 1)", self.outfiles_dif)
        self.layout.addRow("Figure Extension", self.extensions)
        self.layout.addRow("Dots Per Inch", self.dots_per_inch)
        self.layout.addRow("template_suffix", self.template)
        self.layout.addRow(run)
        self.setLayout(self.layout)

        # Connect the clicked signal to the get_functions handlers
        btn1.clicked.connect(self.get_group1)
        btn2.clicked.connect(self.get_group2)
        btn3.clicked.connect(self.get_outdir)
        # Connect the clicked signal to message box to display Matplotlib Colour Options
        btn4.clicked.connect(lambda: self.msgbox(
            'Matplotlib Colourmap Options Ver. 3.4.2: \n Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, crest, crest_r, cubehelix, cubehelix_r, flag, flag_r, flare, flare_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, turbo, turbo_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r'
            ))
        # Connect the checked signal to the state_changed handlers
        self.sagittal_enable.stateChanged.connect(self.s_state_changed)
        self.coronal_enable.stateChanged.connect(self.c_state_changed)
        self.axial_enable.stateChanged.connect(self.a_state_changed)
        self.fig_enable.stateChanged.connect(self.fig_state_changed)
        # Connect the clicked signal tp the print_input handler
        run.clicked.connect(self.print_input)

        # store the results of the STATS flags in an obj similar to args
        self.inputs = type('', (), {})()

    # get folder directories
    def get_group1(self):
        input_folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Group 1 input data folder'))
        if input_folder:
            input_folder_str = "Group 1 input data folder: " + input_folder
            self.lbl1.setText(input_folder_str)
            self.inputs.group1 = input_folder
            print('Group 1 Input Data Folder :%s' % input_folder.lstrip())
        else:
            self.inputs.group1 = None
            self.lbl1.setText('No Folder selected')

    def get_group2(self):
        input_folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Group 2 input data folder'))
        if input_folder:
            input_folder_str = "Group 2 input data folder: " + input_folder
            self.lbl2.setText(input_folder_str)
            self.inputs.group2 = input_folder
            print('Group 2 Input Data Folder :%s' % input_folder.lstrip())
        else:
            self.inputs.group2 = None
            self.lbl2.setText('No Folder selected')

    def get_outdir(self):
        output_folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Output folder'))
        if output_folder:
            output_folder_str = "Output folder: " + output_folder
            self.lbl3.setText(output_folder_str)
            self.inputs.dir_outfile = output_folder
            print('Output Data Folder :%s' % output_folder.lstrip())
        else:
            self.inputs.get_outdir = os.getcwd()
            self.lbl3.setText('No Folder selected')

    # check state of checkboxes
    def s_state_changed(self, int):
        if self.sagittal_enable.isChecked():
            for i in range(5):
                self.axis[0][i].setDisabled(False)
            self.sagittal_status = self.sagittal_enable.isChecked()
        else:
            for i in range(5):
                self.axis[0][i].setDisabled(True)
            self.sagittal_status = self.sagittal_enable.isChecked()

    def c_state_changed(self, int):
        if self.coronal_enable.isChecked():
            for j in range(5):
                self.axis[1][j].setDisabled(False)
            self.coronal_status = self.coronal_enable.isChecked()
        else:
            for i in range(5):
                self.axis[1][i].setDisabled(True)
            self.coronal_status = self.coronal_enable.isChecked()

    def a_state_changed(self, int):
        if self.axial_enable.isChecked() == True:
            for j in range(5):
                self.axis[2][j].setDisabled(False)
            self.axial_status = self.axial_enable.isChecked()
        else:
            for i in range(5):
                self.axis[2][i].setDisabled(True)
            self.axial_status = self.axial_enable.isChecked()

    def fig_state_changed(self, int):
        if self.fig_enable.isChecked():
            self.figure_dimensions_width.setDisabled(False)
            self.figure_dimensions_height.setDisabled(False)
            self.fig_status = self.fig_enable.isChecked()
        else:
            self.figure_dimensions_width.setDisabled(True)
            self.figure_dimensions_height.setDisabled(True)
            self.fig_status = self.fig_enable.isChecked()

    # Make Combobox pop-up menu stationary
    def showPopupAndCheck(self):

        QtWidgets.QComboBox.showPopup(self.voxels)
        popup = self.findChild(QtWidgets.QFrame)
        rect = popup.geometry()
        if not rect.contains(self.voxels.mapToGlobal(self.voxels.rect().center())):
            return

    def msgbox(self, text):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(text)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()
        closing = False
        return closing

        # convert inputs to proper datatype, print and assign to self.inputs

    def print_input(self):
        # get folder directories
        self.inputs.group1 = check_attr_group1(self)
        self.inputs.group2 = check_attr_group2(self)
        self.inputs.dir_outfile = check_attr_dir_outfile(self)
        # retrieve other inputs
        gaussin = float(self.gauss.text())
        voxel = int(self.voxels.currentText())
        cmap_p = str(self.cmap_pos.text())
        cmap_n = str(self.cmap_neg.text())

        # Retrieve any custom axis values
        s_cut = []
        c_cut = []
        a_cut = []

        if self.sagittal_status:
            for i in range(5):
                s_cut.append(int(self.axis[0][i].text()))
        else:
            s_cut = nan
        if self.coronal_status:
            for i in range(5):
                c_cut.append(int(self.axis[1][i].text()))
        else:
            c_cut = nan
        if self.axial_status:
            for i in range(5):
                a_cut.append(int(self.axis[2][i].text()))
        else:
            a_cut = nan
        # retrieve figure dimensions
        figure_dimensions_height = float(self.figure_dimensions_height.text())
        figure_dimensions_width = float(self.figure_dimensions_width.text())

        if self.fig_status:
            figure_dimensions = [figure_dimensions_width, figure_dimensions_height]
        else:
            figure_dimensions = None

        # retrieve outfiles and validate name properties
        g1 = str(self.outfiles_g1.text())
        g2 = str(self.outfiles_g2.text())
        dif = str(self.outfiles_dif.text())

        outfiles = []

        for i in [g1, g2, dif]:
            outfiles.append(i)
            if isinstance(self.inputs.group2, type(None)):
                break

        extensions = str(self.extensions.text())
        dots_per_inch = int(self.dots_per_inch.text())
        template = str(self.template.text())
        # assign to self.input to return to heatmap script
        self.inputs.sigma = gaussin
        self.inputs.vox = voxel
        self.inputs.colourmap_pos = cmap_p
        self.inputs.colourmap_neg = cmap_n
        self.inputs.sagittal = s_cut
        self.inputs.coronal = c_cut
        self.inputs.axial = a_cut
        self.inputs.figure_dim = figure_dimensions
        self.inputs.outfile = outfiles
        self.inputs.extension = extensions
        self.inputs.dpi = dots_per_inch
        self.inputs.t = template

        print('gauss :%f' % gaussin)
        print('voxel :%i' % voxel)
        print('colourmap_pos :%s' % cmap_p.lstrip())
        print('colourmap_neg :%s' % cmap_n.lstrip())
        print('sagittal : %s' % s_cut)
        print('coronal : %s' % c_cut)
        print('axial : %s' % a_cut)
        print('figure dimensions %s' % figure_dimensions)
        print('Output Filename(s): %s' % outfiles)
        print('Figure Extension: %s' % extensions)
        print('DPI: %i' % dots_per_inch)
        print('Template: %s' % template)

        if self.closing == True:
            self.inputs.run = True
            self.close()

    def closeEvent(self, event):
        for window in QtWidgets.QApplication.topLevelWidgets():
            window.close()

def main():
    # Create an PyQT5 application object.
    global app_stats
    app_stats = QtWidgets.QApplication(sys.argv)
    stats_arg = STATSHeatmapMenu()
    stats_arg.show()
    app_stats.exec_()
    return stats_arg.inputs

if __name__ == "__main__":
    sys.exit(main())
    # sys.exit(app_stats.exec_())