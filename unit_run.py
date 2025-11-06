

from MSI_main import Window as C_WIN
from view_demo import Window as V_WIN
from console_ui import Ui_MainWindow as C_W_UI
import sys 

from PyQt5.QtWidgets import QWidget,QApplication,QMainWindow,QMessageBox
from multiprocessing import Process,Queue

def msi_process(*args):


    ChildGUI = C_WIN()
    ChildGUI.main(*args)

    #sys.exit(MainThred.exec_())

def view_process(*args):


    ChildGUI = V_WIN()
    ChildGUI.main(*args)

    #sys.exit(MainThred.exec_())


    
    
class Window(QMainWindow,C_W_UI):
    def __init__(self,parent=None,**kwargs):
        #from qt_material import apply_stylesheet
        
        self.app=QApplication(sys.argv)
        #apply_stylesheet(self.app, theme='dark_cyan.xml')
        QMainWindow.__init__(self,parent)
        self.setupUi(self)
        
        self.pipe1=Queue()
        self.pipe2=Queue()
        
        self.multiprcess1 = Process(target=msi_process,args=(self.pipe1,self.pipe2))
        self.multiprcess2 = Process(target=view_process,args=(self.pipe2,self.pipe1))

        
        
        
        self.sign_and_slot()
    
    def close_msi(self):
        self.multiprcess1.terminate()
    def close_view(self):
        self.multiprcess2.terminate()
    
    def start_msi(self):
        self.multiprcess1 = Process(target=msi_process,args=(self.pipe1,self.pipe2))
        self.multiprcess1.start()
    def start_view(self):
        self.multiprcess2 = Process(target=view_process,args=(self.pipe2,self.pipe1))
        self.multiprcess2.start()
    
    def sign_and_slot(self):
        self.actionView.triggered.connect(self.start_view)
        self.actionMSI.triggered.connect(self.start_msi)
        self.actionView_2.triggered.connect(self.close_view)
        self.actionMSI_2.triggered.connect(self.close_msi)
        
    
    def main(self): 
        self.show()
        
        self.multiprcess1.start()
        self.multiprcess2.start()
        
        sys.exit(self.app.exec_())
    def closeEvent(self, event):
        result = QMessageBox.question(self, "title", "I am the boss, you can't kill me！！！'_'", QMessageBox.Yes | QMessageBox.No)
        if(result == QMessageBox.Yes):
            event.accept()
            self.close_msi()
            self.close_view()
            super().closeEvent(event)
            # 通知服务器的代码省略，这里不是重点...
        else:
            event.ignore()

if __name__ == "__main__":
    window=Window()
    window.main()