from dynaphopy import functions as reading

__author__ = 'abel'
import sys, os
from PyQt4 import QtGui,QtCore
from GUI.main_window import Ui_MainWindow


class Dialog(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        # Set up the user interface from Designer.
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.loadstructure_button.clicked.connect(self.load_structure)
        self.ui.loaddynamic_button.clicked.connect(self.load_dynamic)
        self.ui.energy_button.clicked.connect(self.show_boltzmann)
        self.ui.list_structures.currentItemChanged.connect(self.structure_change)
        self.ui.list_dynamic.currentItemChanged.connect(self.dynamic_change)

        #Calculation related variables

        self.calculation = None

    def load_structure(self):
        file_name = QtGui.QFileDialog.getOpenFileName(self, 'Select VASP structure',os.path.expanduser("~"))
        if file_name:
            item = QtGui.QListWidgetItem(self.ui.list_structures)
            item.setText(file_name)
            structure = reading.read_from_file_structure_outcar(file_name)
            item.setData(QtCore.Qt.UserRole,structure)

    def load_dynamic(self):
        file_name = QtGui.QFileDialog.getOpenFileName(self, 'Select VASP MD calculation',os.path.expanduser("~"))
        if file_name:
            item = QtGui.QListWidgetItem(self.ui.list_dynamic)
            item.setText(file_name)
            structure_item = self.ui.list_structures.currentItem()
            structure = structure_item.data(QtCore.Qt.UserRole).toPyObject()
            print(structure.get_cell())
            dynamic = reading.read_from_file_trajectory(file_name,structure)

            item.setData(QtCore.Qt.UserRole,dynamic)

    def structure_change(self):
        self.ui.loaddynamic_button.setEnabled(True)

    def dynamic_change(self):
        dynamic_item = self.ui.list_dynamic.currentItem()
        dynamic = dynamic_item.data(QtCore.Qt.UserRole).toPyObject()
        self.calculation = controller.Calculation(dynamic)
        self.ui.energy_button.setEnabled(True)

    def show_boltzmann(self):
        self.calculation.show_bolzmann_distribution()




if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    window = Dialog()
    ui = Ui_MainWindow()

    #Initial position and size
    window.setGeometry(400,300,800,500)
    window.show()
    sys.exit(app.exec_())