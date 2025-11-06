

from PyQt5.Qt import QMutex
from PyQt5.QtCore import pyqtSignal,QThread,QRunnable,pyqtSlot

import re
import serial
import serial.tools.list_ports
import time

qmut_1=QMutex();
class Listener(QThread):
    LC_signal = pyqtSignal()

    finish_desi_row=pyqtSignal()
    call_for_desi_input=pyqtSignal()
    finish_lesa_point=pyqtSignal()
    finish_desi_scan_signal=pyqtSignal()
    finish_lesa_scan_signal=pyqtSignal()
    def __init__(self,ser):
        super(Listener,self).__init__()
        self.ser=ser
    def run(self):
        while(1):
            time.sleep(0.05)
            #display=""
            #qmut_1.lock()
            self.command=self.ser.readline().decode('utf-8','ignore')
            #qmut_1.unlock()
            
            if len(self.command)>0:
                print(f'Serial-> {self.command}')
                self.parse_command()
            '''
            while len(self.command)>0:
                if self.command!='\n':
                    display=str(self.command)
                
                print(f'Serial-> {display}')
                #self.signal.emit(display)#连接显示界面  
                self.parse_command()
                self.command=self.ser.readline().decode('utf-8','ignore')
                '''
                
    def parse_number(self,content,value):
        target=re.compile(r".*?{}([\d.]*).*?".format(content),re.S)
        result=target.findall(self.command)
        if len(result):
            #print result[0]
            try:
                data=float(result[0])
            except ValueError:
                return value
            else:
                return data;
        else:
            return value;
    
    def parse_command(self):
        code=self.parse_number("PYTHON",-1)
        #print(75,code);
        if code==0:
            pass;
        
        if code==77:
            print(80,code);
            self.LC_signal.emit();
        if code==200:
            print('disi finish row')
            self.finish_desi_row.emit()
        if code==205:
            print('lesa finish point')
            self.finish_lesa_point.emit()
        if code==202:
            print("call for desi input")
            self.call_for_desi_input.emit()
        

        
class Port():
    portConnectSignal=pyqtSignal();
    def __init__(self):
        self._port_sign_and_solt()
    def _detectPorts(self):
        port_list=list(serial.tools.list_ports.comports())
        my_connected=0;
        while len(port_list)<=0:
            print("the serial port can't find.\n")
            my_connected=QMessageBox.question(qmainWindow, "提问对话框", "你要继续搞测试吗？", QMessageBox.Yes | QMessageBox.No)
            print(my_connected)
            print(type(my_connected))
            if my_connected>20000:
                sys.exit();
        
        self.port_show_combobox.clear()
        port_items=[]
        for item in port_list:
            print(list(item) )
            port_items.append(str(list(item)))
        self.port_show_combobox.addItems(port_items);
    def _disconn_port(self):
        self.ser.close();
    def _connect_port(self):
        current_chose=self.port_show_combobox.currentText()
        print(current_chose)
        self.ser=serial.Serial( current_chose.split("\'")[1] , 115200, timeout=0.01)
        
        
        self.listener=Listener(self.ser)
        self.listener.start()
        self.portConnectSignal.emit();
    def _thread_delay(self,delay_time): #sleep secound
        enter_time=time.time()
        while time.time()-enter_time  < delay_time:
            QApplication.processEvents()
            time.sleep(0.01)
            
    def _write_c(self,command,timeout=0.2):
        self._thread_delay(timeout)
        #qmut_1.lock()
        self.ser.write((command+' \n').encode())#self.ser.write((command+'\n').encode())
        #qmut_1.unlock()
        print(127,"in write_c command is: ",command);
    
    def _port_sign_and_solt(self):
        self.connect_port_button.clicked.connect(self._connect_port)
        self.disconnect_port_button.clicked.connect(self._disconn_port)
        self.detect_port_button.clicked.connect(self._detectPorts)
        
class WorkerGUI(QRunnable):
    def __init__(self, InFunc,*args):
        super(WorkerGUI, self).__init__()
        self.Func = InFunc
        self.para=args

    @pyqtSlot()
    def run(self):
        self.Func(*self.para)
        return        
        
class MyQthread(QThread):
    def __init__(self,run_function,*object):
        super(MyQthread,self).__init__()
        self.run_function=run_function
        self.object=object
    def run(self):
        self.run_function(*self.object)