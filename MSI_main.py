from DESI import *
from LESA import *
from  window_UI import Ui_MainWindow as C_W_UI
from qt_base import Listener, Port,MyQthread,WorkerGUI

from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtWidgets import QWidget,QApplication,QMainWindow,QSizePolicy,QGridLayout,QDialog,QDialogButtonBox,QFormLayout,QDoubleSpinBox
from PyQt5.QtCore import pyqtSignal,QThread,QMetaType,QThreadPool
from PyQt5.QtWidgets import QFileDialog,QInputDialog,QMessageBox
from PyQt5.QtGui import QRegExpValidator, QIntValidator, QDoubleValidator # 输入类型验证器

import pathlib
import requests
import time
import re
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import sys 
import os
import _thread
import threading
import json
from multiprocessing import Process,Queue

class MyDialog(QDialog):
    def __init__(self):
        super().__init__();
        self.setWindowTitle('Plot Configs')
        formLayout = QFormLayout(self)
        
        self.X = QDoubleSpinBox()
        self.Y = QDoubleSpinBox()
        
        button = QDialogButtonBox(QDialogButtonBox.Ok)
        formLayout.addRow('Start', self.X)
        formLayout.addRow('Start', self.Y)
        formLayout.addRow(button)
        button.clicked.connect(self.accept)
    def get_input(self):
        pass
        self.show()
        if self.exec() == QDialog.Accepted:
            return True,self.X.value(),self.Y.value()
        else:
            return False,None,None 
            
class Window(QMainWindow,C_W_UI,Port):
    def __init__(self,parent=None,**kwargs):
        #from qt_material import apply_stylesheet
        
        self.app=QApplication(sys.argv)
        #apply_stylesheet(self.app, theme='dark_teal.xml')
        QMainWindow.__init__(self,parent)
        self.setupUi(self)
        
        
        Port.__init__(self)
        self._sign_and_solt()
        
        #self.enable_run(False);
        self.set_validator()
        
        #self.seq=pd.DataFrame();
        
        self.DESI=DESI();
        self.LESA=LESA();
        self.scan_para={};
        
        self.send_data_queue=Queue(1);
        self.recive_data_queue=Queue(1);
        self.threadpool = QThreadPool()
        self.WrkrGUI =[]
        
        
        self.suit_for_write=True;
        self.all_thread_enable=True;
        self.command_id=0;
        
        self.tasks=pd.DataFrame()
        self.task_id=0;
        
        self.lesa_sample_points=[];
        self.lesa_is_running=False;
        self.msi_is_running=False;
        self.host_online=False;
        
        
        self._events={f'E{x}':[] for x in range(100)}
        self._events_clicked={f'E{x}':0 for x in range(100)}
        
        
        #self.read_para()
    def get_task(self):
        if self.tasks.shape[0]==0:
            return False,None
        else:
            try:
            
                return True, self.tasks.loc[self.tasks['fid']==self.task_id,:].iloc[0,:]
            except Exception as e:
                print(96,e)
                return False,None
    def set_validator(self):
        doubleValidator = QDoubleValidator(0,30,2)
        self.LE_E0.setValidator(doubleValidator)
        self.LE_E1.setValidator(doubleValidator)
        self.LE_E2.setValidator(doubleValidator)
        self.LE_E3.setValidator(doubleValidator)
        self.LE_E4.setValidator(doubleValidator)
        self.LE_E5.setValidator(doubleValidator)
        self.LE_E6.setValidator(QDoubleValidator(0,10,2))
        self.LE_E7.setValidator(QDoubleValidator(0,10000,2))
        self.LE_E8.setValidator(QDoubleValidator(0,10,2))
        self.LE_E9.setValidator(QDoubleValidator(0,10000,2))
        
        
        self.lineEdit_2.setValidator(doubleValidator)
        self.lineEdit_3.setValidator(doubleValidator)
        self.lineEdit_4.setValidator(QIntValidator(-4095,4095))
        self.lineEdit_5.setValidator(QIntValidator(-4095,4095))
        
        mIntValidator=QIntValidator(0,100000)
        self.lineEdit_6.setValidator(mIntValidator)
        self.lineEdit_7.setValidator(mIntValidator)
        self.lineEdit_8.setValidator(mIntValidator)
        self.lineEdit_9.setValidator(mIntValidator)
        
        
        self.pushButton_8.setStyleSheet("background-color: yellow")
        
    def enable_run(self,enabled):
        self.pushButton.setEnabled(enabled)
        self.pushButton_2.setEnabled(enabled)
        self.pushButton_3.setEnabled(enabled)
        self.pushButton_4.setEnabled(enabled)
        self.pushButton_6.setEnabled(enabled)
        self.pushButton_7.setEnabled(enabled)
    def __main__(self):
        pass
        
    
    
    
    def _mainWindow_do_command(self,command):
        self.all_thread_enable=True;
        self.LE_current_command.setText(command)
        
        print(558,f"process the {self.command_id} command, and the command is {command}")
        self.command_id+=1;
        if 'W' in command:
            #中断
            if 'T' in command:
                hit=re.search(r'T[0-9]+(\.[0-9]+)?',command)
                #time.sleep(float(command[hit.span()[0]+1:hit.span()[1]]) )
                print(614,"wait for T{}".format(float(command[hit.span()[0]+1:hit.span()[1]])))
                self._thread_delay(float(command[hit.span()[0]+1:hit.span()[1]]))
                return
            
            if 'E' in command:
                key=re.search(r'E(\d+)',command).group()
                print(614,"wait for {}".format(key))
                while self._events_clicked[key]==0:
                    self._thread_delay(0.1)
                    if  self.isSTOP:
                        self.LE_current_command.setText("STOP!!!")
                        return
                self._events_clicked[key]=0;
                return

                
            if 'F' in command:
                self.output_file=command;
                print(627,self.output_file)
                return;
            return
        if '$E' in command: #执行组合函数
                
                key=re.search(r'E(\d+)',command).group()
                print("\n\n\n+++++++++++++++++++\n exec E{}\n".format(key))
                self._event_bt_functions(key);
                #for command in self._events[key]:
                #    self._mainWindow_do_command(command)
                return
                
        self._write_c(command);
    
    
    
    
    
    
    
    def read_para(self):
        
        
        
        
        
        
        self.scan_para['x_l']=float( self.LE_E0.text() ) 
        self.scan_para['x_r']=float( self.LE_E1.text() ) 
        self.scan_para['x_s']=float( self.LE_E2.text() )
        self.scan_para['x_e']=float( self.LE_E3.text() )
        self.scan_para['y_s']=float( self.LE_E4.text() )
        self.scan_para['y_e']=float( self.LE_E5.text() )
        self.scan_para['y_interval']=float( self.LE_E6.text() )
        self.scan_para['y_current']=self.scan_para['y_s']
        
        self.scan_para['wait_time']=float( self.LE_E8.text() )
        self.scan_para['msi_speed']=float( self.LE_E9.text() )
        self.scan_para['S_speed']=float( self.LE_E7.text() )
        
        self.scan_para['sample_hight']=float( self.lineEdit_2.text() )
        self.scan_para['ion_hight']=float( self.lineEdit_3.text() )
        self.scan_para['sample_position']=float( self.lineEdit_4.text() )
        self.scan_para['ion_position']=float( self.lineEdit_5.text() )
        self.scan_para['suck_time']=float( self.lineEdit_6.text() )
        self.scan_para['pump_time']=float( self.lineEdit_7.text() )
        self.scan_para['extract_time']=float( self.lineEdit_8.text() )
        self.scan_para['ion_time']=float( self.lineEdit_9.text() )
        self.scan_para["host_url"]= "http://"+self.lineEdit_10.text()
        self.scan_para['remote']= self.checkBox.checkState()
        
        self.scan_para['auto_standby']=self.actionauto_stanby.isChecked()
        self.scan_para['end_with_E99']=self.actionend_with_E99.isChecked()
        
        print(125,self.scan_para)
    
    def write_para(self):
        self.LE_E0.setText(str( self.scan_para['x_l'] ))
        self.LE_E1.setText(str( self.scan_para['x_r'] ))
        self.LE_E2.setText(str( self.scan_para['x_s'] ))
        self.LE_E3.setText(str( self.scan_para['x_e'] ))
        self.LE_E4.setText(str( self.scan_para['y_s'] ))
        self.LE_E5.setText(str( self.scan_para['y_e'] ))
        self.LE_E6.setText(str( self.scan_para['y_interval'] ))
        #self.LE_E9.setText(str( self.scan_para['msi_speed'] ))
        
        
        self.lineEdit_2.setText(str( self.scan_para['sample_hight'] ))
        self.lineEdit_3.setText(str( self.scan_para['ion_hight'] ))
        self.lineEdit_4.setText(str( self.scan_para['sample_position'] ))
        self.lineEdit_5.setText(str( self.scan_para['ion_position'] ))
        self.lineEdit_6.setText(str( self.scan_para['suck_time'] ))
        self.lineEdit_7.setText(str( self.scan_para['pump_time'] ))
        self.lineEdit_8.setText(str( self.scan_para['extract_time'] ))
        self.lineEdit_9.setText(str( self.scan_para['ion_time'] ))
        
        
        self.LE_E8.setText(str( self.scan_para.get('wait_time',0 )) )
        self.LE_E9.setText(str( self.scan_para.get('msi_speed',0 )) ) 
        self.LE_E7.setText(str( self.scan_para.get('S_speed',0 )) )  
        #self.scan_para["host_url"]= "http://"+self.lineEdit_10.text()
        self.lineEdit_10.setText(self.scan_para["host_url"].replace("http://",""))
        self.checkBox.setCheckState(  self.scan_para.get('remote',0) )  
        
    def start_desi(self):
        self.task_id=0;
        self.read_para()
        send_data={'type':'start'}
        send_data['data']=self.scan_para
        self.DESI.set_desi_para(self.scan_para)
        
        self.send_data_queue.put(send_data)
        
        
        self._mainWindow_do_command(f"G0 X{self.scan_para['msi_speed']} Y{self.scan_para['msi_speed']} a{self.scan_para['S_speed']} A{self.scan_para['S_speed']}")
        
        self.msi_is_running=True;
        self.run_MSI()
        print("start desi to view")
        
    def run_MSI(self):
        #self.start_desi()

        
        if self.msi_is_running:
            
            if self.DESI.run():
                while not self.DESI.task_queue.empty():
                    task=self.DESI.task_queue.get()
                    self._mainWindow_do_command(task.code)
                self._mainWindow_do_command("D40 C202 I0 P0") #self.add_task("call for desi","D40 C202 I0 P0")
            else:
                self.msi_is_running=False
                self._mainWindow_do_command("D40 C211")
                
                self.scan_para['auto_standby']=self.actionauto_stanby.isChecked()
                self.scan_para['end_with_E99']=self.actionend_with_E99.isChecked()
                
                if self.scan_para['auto_standby']:
                    self._mainWindow_do_command("D40 C29")# close LC Pump
                    self._mainWindow_do_command("D40 C61")# close pressure gas pump
                    self._mainWindow_do_command("D40 C20 T10000")
                    self._mainWindow_do_command("D40 C51")
                    self._mainWindow_do_command("D40 C40 I0")
                    self._mainWindow_do_command("D40 C1 I0 T0")
                    self._mainWindow_do_command("D40 C1 I1 T0")
                if self.scan_para['end_with_E99'] :
                    self._mainWindow_do_command("$E99")
                #self.add_task("lc pump off",f"D40 C29")
                #self.add_task("wait time",f"D40 C20 T10")
                #self.add_task("close GAS",f"D40 C51")
        
    def finish_desi(self):
        print(183,"finish the MSI");
    
    def start_lesa(self):
        self.read_para()
        if len(self.lesa_sample_points)==0:
            print("there is no sample points")
            return
        print(f"sample points contain: {self.lesa_sample_points}")
        #self.read_para()
        self.LESA.set_lesa_para(self.scan_para)
        self._mainWindow_do_command(f"G0 X{self.scan_para['msi_speed']} Y{self.scan_para['msi_speed']} a{self.scan_para['S_speed']} A{self.scan_para['S_speed']}")
        self.lesa_is_running=True
        self.run_LESA()
    
    def run_LESA(self):
    
        if self.lesa_is_running:
            if len(self.lesa_sample_points)>0:
                s_p=self.lesa_sample_points[0]#self.lesa_sample_points.pop()
                self.lesa_sample_points.remove(s_p)
                #print(f"the LESA sample point is :{s_p}")
                self.LESA.add_one_sample_point(s_p[0],s_p[1])
                #print(f"success add one sample")
                #self.LESA.report_task()
                while not self.LESA.task_queue.empty():
                    task=self.LESA.task_queue.get()
                    #print("msi_main",task.code);
                    self._mainWindow_do_command(task.code)
            else:
                self.lesa_is_running=False
                self._mainWindow_do_command("D40 C212")
        else:
            pass
      
        
    def finish_lesa(self):
        print(188,"finish the lesa");
    def desi_finish_a_row(self):
        pass
    
    
    def query_for_write(self):
        self._mainWindow_do_command("M210");
    
    def _thread_delay(self,delay_time): #sleep secound
        enter_time=time.time()
        while time.time()-enter_time  < delay_time:
            QApplication.processEvents()
            time.sleep(0.05)
    def _direct_exec_commamd(self):
        self.command_id=0;
        self._events_clicked={f'E{x}':0 for x in range(100)}
        self.isSTOP=False;
        self._mainWindow_do_command(self.LE_command.text())
    
    
    def load_code(self):
        import pickle
        file_name=QFileDialog.getOpenFileName(self,"open file dialog","./msi_meth/","para_file(*.msiPara)") 
        if file_name[0]:
            temp_file=open(file_name[0],'rb')
            self.scan_para.update(pickle.load(temp_file))
            temp_file.close()
            self.write_para()
    def save_code(self):
        import pickle
        parameters={}
        file_name=QFileDialog.getSaveFileName(self,"save file dialog","./msi_meth/","para_file(*.msiPara)") 
        
        if file_name[0]:
            self.read_para()
            #print(file_name)
            #self.parameters=json.dumps(self.parameters)
            #json.dump(data,open(file_name[0],'w'))
            temp_file=open(file_name[0],'wb')
            pickle.dump(self.scan_para,temp_file)
            temp_file.close()
    
    def load_seq(self):
        self.task_id=0;
        file_name=QFileDialog.getOpenFileName(self,"open file dialog","./seq/","excel_file(*.xlsx)") 
        if file_name[0]:
            self.tasks=pd.read_excel(file_name[0]);
        if len(self.tasks)>0:
            self.enable_run(True)
    
    def show_seq(self):
        print(self.tasks)
    
    
    
    def get_commands(self,text):
        #print(text.split('\n'))
        #去除注释
        output=[]
        for sentence in text.split('\n'):
            
            
            
            if '#' in sentence:
                sentence=sentence[:sentence.find('#')]
            if len(sentence.replace(' ',''))==0:
                continue    
                

            output.append(sentence)
                
        return output
        
    def load_command_text(self):
    
        file_name=QFileDialog.getOpenFileName(self,"open file dialog","./commands_list/","txt_file(*.txt)") 
        if file_name[0]:
            with open(file_name[0],"r") as f:
                file=f.read()
        else:
            return
        
        self._setup_commands=[]
        self._events={f'E{x}':[] for x in range(100)}
        self._events_clicked={f'E{x}':0 for x in range(100)}
        self._setup_commands=self.get_commands(file)
        

        for command in self._setup_commands:
            if '@' in command:
                sign=re.search(r'E(\d+)',command).group()
                self._events[sign].append(command[command.find('@')+1:] )
                continue
            self._mainWindow_do_command(command);
        print(534,self._events);
    
    def _event_bt_functions(self,key):
        
        print(key,' has been clicked');
        for command in self._events[key]:
            self._mainWindow_do_command(command)
        self._events_clicked[key]=1
        
        
                    
        
    
        
    
    def finish_desi_row(self):
        eable,row=self.get_task()
        if eable:
            raw_file=os.path.join(row['path'],row['file']+".raw")
            if self.scan_para['remote']>0:
                if pathlib.Path(raw_file).exists():
                    pass
                else:
                    try:
                        File=requests.get(self.scan_para["host_url"]+row['file']+".raw",stream=True)
                        if File.ok:
                            with open(os.path.join(row['path'],row['file'])+".raw",'wb') as f:
                                for chunk in File.iter_content(chunk_size=1024):
                                    if chunk:
                                        f.write(chunk)
                        else:
                            print("!!!!!!!!!!!Warning in net !!!!!!!!!")
                    except Exception as e:
                        print(389,"msi_main, finish_desi_row: ",e)
                
            data={'type':'massconvert','fid':self.task_id,'file':os.path.join(row['path'],row['file'])   }  
            self.send_data_queue.put(data)
            self.task_id+=1
    def report_pipe_size(self):
        print(f"the size of send_queue is {self.send_data_queue.qsize()} ")
        print(f"the size of recv_queue is {self.recive_data_queue.qsize()} ")
        
    def sent_para_to_view(self):
        #actionsend_parameters
        self.read_para()
        send_data={'type':'scan_para'}
        send_data['data']=self.scan_para
        self.send_data_queue.put(send_data)
    
    def set_task_id_to_zero(self):
        self.task_id=0;
    
    def set_task_id(self):
        input_id, ok = QInputDialog.getInt(self, 'set taskID', 'please input：',min=0,max=500,step=1)
        if ok:
            self.task_id=input_id    
    def report_taskID(self):
        print(f"taskID is {self.task_id}")
    def clear_send_pipe(self):
        while not self.send_data_queue.empty():
            self.send_data_queue.get()
    
    def test_LESA_pump(self):
        self.scan_para['pump_time']=float( self.lineEdit_7.text() )
        self._mainWindow_do_command(f"D40 C33 I0 T{self.scan_para['pump_time']}")
        
        
    def test_LESA_suck(self):
        self.scan_para['suck_time']=float( self.lineEdit_6.text() )
        self._mainWindow_do_command("D40 C35")
        self._mainWindow_do_command("D40 C31")
        self._mainWindow_do_command("D40 C20 T2000")
        self._mainWindow_do_command(f"D40 C34 T{self.scan_para['suck_time']}")
        self._mainWindow_do_command("D40 C36")
    def test_LESA_sample(self,target):
        if target==0:
            self._mainWindow_do_command("D40 C1 I2 T0")
        if target==1:
            self.scan_para['sample_hight']=float( self.lineEdit_2.text() )
            self._mainWindow_do_command(f"D40 C1 I2 T{self.scan_para['sample_hight']}")
        if target==2:
            self.scan_para['ion_hight']=float( self.lineEdit_3.text() )
            self._mainWindow_do_command(f"D40 C1 I2 T{self.scan_para['ion_hight']}")
    
    def test_LESA_excrete(self):
        self._mainWindow_do_command(f"D40 C37 I1 T500")
    def test_LESA_servo(self,target):
        
        
        if target==0:
            self.scan_para['sample_position']=float( self.lineEdit_4.text() )
            self._mainWindow_do_command(f"D40 C3 I1 T{self.scan_para['sample_position']}")
        if target==1:
            self.scan_para['ion_position']=float( self.lineEdit_5.text() )
            self._mainWindow_do_command(f"D40 C3 I1 T{self.scan_para['ion_position']}")
            
            
    def test_msi(self,target):
        
        
        
        
        
        
        
        if target=="x_l":
            self.scan_para['x_l']=float( self.LE_E0.text() ) 
            self._mainWindow_do_command(f"D40 C1 I0 T{self.scan_para[target]}")
        if target=="x_r":
            self.scan_para['x_r']=float( self.LE_E1.text() ) 
            self._mainWindow_do_command(f"D40 C1 I0 T{self.scan_para[target]}")
        if target=="x_s":
            self.scan_para['x_s']=float( self.LE_E2.text() )
            self._mainWindow_do_command(f"D40 C1 I0 T{self.scan_para[target]}")
        if target=="x_e":
            self.scan_para['x_e']=float( self.LE_E3.text() ) 
            self._mainWindow_do_command(f"D40 C1 I0 T{self.scan_para[target]}")
        if target=="y_s":
            self.scan_para['y_s']=float( self.LE_E4.text() ) 
            self._mainWindow_do_command(f"D40 C1 I1 T{self.scan_para[target]}")
        if target=="y_e":
            self.scan_para['y_e']=float( self.LE_E5.text() ) 
            self._mainWindow_do_command(f"D40 C1 I1 T{self.scan_para[target]}")
        
        if target=="speed":
            self.scan_para['msi_speed']=float( self.LE_E9.text() )
            self.scan_para['S_speed']=float( self.LE_E7.text() )
            self._mainWindow_do_command(f"G0 X{self.scan_para['msi_speed']} Y{self.scan_para['msi_speed']} a{self.scan_para['S_speed']} A{self.scan_para['S_speed']}")
        
            
    def report_prog(self):
        pass
    def _process_number(self,code,default=0):
        hit=re.search(r'{}[0-9]+(\.[0-9]+)?'.format(code),command)
        return float(command[hit.span()[0]+1:hit.span()[1]])
    #def set_suit_for_write(self,cando):
    #    self.suit_for_write=cando
    
        
        
    
    def recive_data_func(self):
        while 1:
            if self.recive_data_queue.empty():
                time.sleep(0.1)
                
                continue
            else:
                recive_data=self.recive_data_queue.get()
                print("MSI_main have recive\t",recive_data)
                if recive_data['type']=="LESA_sample_point":
                    self.lesa_sample_points=recive_data["data"]
                    
    def stop_msi(self):
        pass
        self.lesa_is_running=False;
        self.msi_is_running=False;
        self._mainWindow_do_command("D2")
        self.DESI.stop()
        self.LESA.stop()
        self.task_id=0;
    
    def stop_lesa(self):
        pass
        self.lesa_is_running=False;
        self.msi_is_running=False;
        self._mainWindow_do_command("D2")
        self.DESI.stop()
        self.LESA.stop()
        self.task_id=0;    
        
    def add_LESA_point(self):
        is_g,x,y=MyDialog().get_input()
        if is_g:
            print(f"success get [{x} , {y}] point")
            self.lesa_sample_points.append([x,y])
    def move_to_point(self):
        is_g,x,y=MyDialog().get_input()
        if is_g:
            print(f"to the [{x} , {y}] point")
            #self.lesa_sample_points.append([x,y])
            self._mainWindow_do_command(f"D40 C1 I0 T{x}")
            self._mainWindow_do_command(f"D40 C1 I1 T{y}")
    
    def show_LESA_point(self):
        print(f"self.lesa_sample_points is {self.lesa_sample_points}")
    def ping_url(self):
        try:
            self.scan_para["host_url"]= "http://"+self.lineEdit_10.text()
            result=requests.get(self.scan_para["host_url"])
            if result.ok:
                self.pushButton_8.setStyleSheet("background-color: green")
                self.host_online=True
            else:
                self.host_online=False
                self.pushButton_8.setStyleSheet("background-color: red")
        except Exception as e:
            self.pushButton_8.setStyleSheet("background-color: red")
            print("url with error:",e)
    def _connect_signal(self):
        #self.listener.SUIT_FOR_WRITE_signal.connect(self.set_suit_for_write)
        self.listener.call_for_desi_input.connect(self.run_MSI)
        self.listener.finish_desi_scan_signal.connect(self.finish_desi)
        self.listener.finish_lesa_scan_signal.connect(self.finish_lesa)
        self.listener.finish_desi_row.connect(self.finish_desi_row)
        self.listener.finish_lesa_point.connect(self.run_LESA)

    def _sign_and_solt(self):
        self.actionload.triggered.connect(self.load_code)
        self.actionsave.triggered.connect(self.save_code)
        self.actionadd_LESA_point.triggered.connect(self.add_LESA_point)
        self.actionshow_LESA_point.triggered.connect(self.show_LESA_point)
        self.actionmove_to_point.triggered.connect(self.move_to_point)
        self.actionload_command_text.triggered.connect(self.load_command_text)
        
        
        self.actionsend_parameters.triggered.connect(self.sent_para_to_view)
        self.actionreport_pipe_size.triggered.connect(self.report_pipe_size)
        self.actionclear_send_pipe.triggered.connect(self.clear_send_pipe)
        self.actionset_tastid_to_zero.triggered.connect(self.set_task_id_to_zero)        
        self.actionset_taskID.triggered.connect(self.set_task_id)
        self.actionreport_taskID.triggered.connect(self.report_taskID)
        
        
        self.pushButton_5.clicked.connect(self.load_seq)
        self.pushButton_6.clicked.connect(self.show_seq)
        self.pushButton_7.clicked.connect(self.report_prog)
        self.PB_write_c.clicked.connect(self._direct_exec_commamd);
        self.pushButton.clicked.connect(self.start_desi);
        self.pushButton_2.clicked.connect(self.start_lesa);
        self.pushButton_8.clicked.connect(self.ping_url);
        
        
        self.pushButton_3.clicked.connect(self.stop_lesa);
        self.pushButton_4.clicked.connect(self.stop_msi);
        
        self.pushButton_9.clicked.connect(self.test_LESA_pump);
        self.pushButton_10.clicked.connect(self.test_LESA_suck);
        self.pushButton_15.clicked.connect(self.test_LESA_excrete);
        self.pushButton_12.clicked.connect(lambda x: self.test_LESA_sample(1));
        self.pushButton_11.clicked.connect(lambda x: self.test_LESA_sample(0));
        self.pushButton_14.clicked.connect(lambda x: self.test_LESA_sample(2));
        self.pushButton_13.clicked.connect(lambda x: self.test_LESA_sample(0));
        
        self.pushButton_16.clicked.connect(lambda x: self.test_LESA_servo(0));
        self.pushButton_17.clicked.connect(lambda x: self.test_LESA_servo(1));
        
        self.pushButton_18.clicked.connect(lambda x: self.test_msi("x_l"));
        self.pushButton_19.clicked.connect(lambda x: self.test_msi("x_r"));
        self.pushButton_20.clicked.connect(lambda x: self.test_msi("x_s"));
        self.pushButton_21.clicked.connect(lambda x: self.test_msi("x_e"));
        self.pushButton_22.clicked.connect(lambda x: self.test_msi("y_s"));
        self.pushButton_23.clicked.connect(lambda x: self.test_msi("y_e"));
        self.pushButton_24.clicked.connect(lambda x: self.test_msi("speed"));
        self.pushButton_25.clicked.connect(lambda x: self.test_msi("speed"));
        
        
        
        
        
        self.portConnectSignal.connect(self._connect_signal)            
    def main(self,send_data_queue,recive_data_queue): 
        self.show()
        self.send_data_queue=send_data_queue;
        self.recive_data_queue=recive_data_queue;
        self.threadpool.start( WorkerGUI(self.recive_data_func) )
        sys.exit(self.app.exec_())
    
    def __del__(self):
        self.ser.close();
        print("******************************\n\
            *********************************\n\
            *****     MSI have been del ****\n\
            *********************************")
    
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        """
        重写QWidget类的closrEvent方法，在窗口被关闭的时候自动触发
        """
        super().closeEvent(a0)  # 先添加父类的方法，以免导致覆盖父类方法（这是重点！！！）
        self.listener.terminate()
        self.__del__();
        
if __name__=='__main__':
    pipe1=Queue()
    #pipe2=Queue()
    window=Window()
    window.main(pipe1,pipe1)