#! /usr/bin/env python

# coding: utf-8 

from pprint import pprint
import sys
import os

from PyQt5 import QtWidgets, QtCore, QtGui


def check_attr_input_volume(self):
    if  hasattr(self.inputs, "in_csv"):
        return(self.inputs.in_csv)
    else:
        return(None)    
def check_attr_dir_outfile(self):
    if  hasattr(self.inputs, "dir_outfile"):
        return(self.inputs.dir_outfile)
    else:
        return(os.getcwd())            

class STATSPlotSingleSubjMenu(QtWidgets.QWidget):

    def __init__(self):
    # create GUI
        QtWidgets.QMainWindow.__init__(self)
    
        super(STATSPlotSingleSubjMenu, self).__init__()

        self.setWindowTitle('STATS Plot Single Subject')

        # Set the window dimensions
        # self.resize(500,200)

        # Create labels which displays the path to our chosen file
        self.lbl1 = QtWidgets.QLabel("No Input Volume file selected (Select CSV file generated from 'miracl lbls stats')")
        self.lbl2 = QtWidgets.QLabel('No folder selected for Output. Default: Current Working Directory')

        # Create push buttons for Folder Selection
        btn1 = QtWidgets.QPushButton("Select Input Volume file (CSV file generated from 'miracl lbls stats')", self)
        btn2 = QtWidgets.QPushButton('Select Output folder', self)
        #btn1.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    
        # Run GUI Button
        run = QtWidgets.QPushButton('Run', self)

        # Create other inputs (Spinbox/line edits)
        self.sorts = QtWidgets.QComboBox()
        self.sorts.setEditable(True)
        self.sorts.lineEdit().setReadOnly(True)
        self.sorts.showPopup = self.showPopupAndCheck
        self.sorts.addItems(["Mean", "StdD", "Max", "Min", "Count", "Vol"])

        self.thresholding = QtWidgets.QDoubleSpinBox()
        self.thresholding.setValue(0.75)
        self.thresholding.setAlignment(QtCore.Qt.AlignRight)

        # layout for input widgets        
        self.layout = QtWidgets.QFormLayout()
        self.layout.addRow(self.lbl1, btn1)
        self.layout.addRow("Sort Values by ...", self.sorts)
        self.layout.addRow(self.lbl2, btn2)         
        self.layout.addRow("Threshold (Non-negative number)", self.thresholding)

        self.layout.addRow(run)
        self.setLayout(self.layout)

        # Connect the clicked signal to the get_functions handlers
        btn1.clicked.connect(self.get_input_volume)
        btn2.clicked.connect(self.get_outdir)
       
        #Connect the clicked signal tp the print_input handler
        run.clicked.connect(self.print_input)
  
        # store the results of the STATS flags in an obj similar to args
        self.inputs = type('', (), {})()

    #get folder directories
    def get_input_volume(self):
        input_volume = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Input Volume file')[0]
        if input_volume:
            input_volume_str="Input Volume: " + input_volume
            self.lbl1.setText(input_volume_str)
            self.inputs.in_csv = input_volume
            print('Input Volume :%s' % input_volume.lstrip())
        else:
            self.inputs.in_csv = None
            self.lbl1.setText('No Input Volume selected')

    def get_outdir(self):
        output_folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Output folder'))
        if output_folder:
            output_folder_str="Output folder: " + output_folder
            self.lbl2.setText(output_folder_str)
            self.inputs.dir_outfile = output_folder
            print('Output Data Folder :%s' % output_folder.lstrip())
        else:
            self.inputs.dir_outfile = os.getcwd()
            self.lbl2.setText('No Folder selected')

    #Make Combobox pop-up menu stationary
    def showPopupAndCheck(self):
        
        QtWidgets.QComboBox.showPopup(self.sorts)
        popup=self.findChild(QtWidgets.QFrame)
        rect = popup.geometry()
        if not rect.contains(self.sorts.mapToGlobal(self.sorts.rect().center())):
            return      

    def msgbox(self, text):
        msg=QtWidgets.QMessageBox()   
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(text)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()
        closing=False
        return closing       

    #convert inputs to proper datatype, print and assign to self.inputs
    def print_input(self):             
        #will close GUI after checking if group folder and outfile arguments are valid (closing)
        closing=True    
        #validate files/folder directories
        self.inputs.in_csv=check_attr_input_volume(self)

        if isinstance(self.inputs.in_csv, type(None)):
            closing=self.msgbox("Must Select Valid Input Volume File. ( .CSV file)")
        elif ('.csv' in self.inputs.in_csv)==False:
            closing=self.msgbox("Selected file was not a .CSV")    

        self.inputs.dir_outfile=check_attr_dir_outfile(self)   
        #retrieve other inputs

        sorts = str(self.sorts.currentText())
        thresholds = float(self.thresholding.text())

        self.inputs.sort = sorts
        self.inputs.threshold = thresholds

        print('Sorted Values by :%s' % sorts)             
        print('threshold :%f' % thresholds)

        if closing==True:
            self.inputs.run==True
            stats_plot_single_subj.close()


def main():
    # Create an PyQT5 application object.

    global app_stats
    app_stats = QtWidgets.QApplication(sys.argv)
    global stats_plot_single_subj
    stats_plot_single_subj = STATSPlotSingleSubjMenu()
    stats_plot_single_subj.show()
    app_stats.exec_()

    return stats_plot_single_subj.inputs

if __name__ == "__main__":
    sys.exit(main())
    #sys.exit(app_stats.exec_())