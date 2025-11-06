
from  view_UI import Ui_MainWindow as V_W_UI
from qt_base import WorkerGUI
from mcanvas import MyProcessCanvas,MyClusterCanvas,MyMsiCanvas,MyPcaloadingCanvas,DRCanvas,MyMassSpectrumCanvas
from file_convert import convert_f
from pre_processing import sub_process_mzml_by_dict3,subproces_do_post_compensation,ManagerSpect2, TargetArray_Mem




from dimension_reduction import DR_FUNC
from cluster_functions import CLUSTER_FUNC

from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtWidgets import QWidget,QApplication,QMainWindow,QSizePolicy,QGridLayout,QDoubleSpinBox,QVBoxLayout,QFormLayout,QDialog,QDialogButtonBox,QMessageBox,QLabel,QDockWidget
from PyQt5.QtCore import pyqtSignal,QThread,QMetaType,QThreadPool
from PyQt5.QtWidgets import QFileDialog,QInputDialog,QMessageBox
from PyQt5.QtGui import QRegExpValidator, QIntValidator, QDoubleValidator # 输入类型验证器
from PyQt5.Qt import QMutex

import psutil
import sys 
import os
from multiprocessing import Process,Queue,Manager,Pool,cpu_count,set_start_method
from multiprocessing.managers import BaseManager
import pathlib
import pandas as pd
import numpy as np
import copy
import time
import pickle
from functools import wraps
import json
import signal

POOL_NUM=10

import atexit
class ProcessManager:
    """专业的进程管理器"""
    
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or max(1, cpu_count() - 1)
        self.pool = None
        self._child_processes = set()
        self._is_cleaning = False
        
        # 注册退出清理函数
        atexit.register(self.cleanup_all_processes)
        
    def create_pool(self):
        """创建进程池"""
        if self.pool is not None:
            self.cleanup_pool()
            
        # 设置启动方法
        set_start_method('spawn', force=True)
        print(f"68 in demo, there are {self.max_workers} workers in ProcessManager pool")
        self.pool = Pool(
            processes=self.max_workers,
            initializer=self._process_initializer
        )
        return self.pool
        
    def _process_initializer(self):
        """子进程初始化"""
        # 忽略中断信号，由主进程统一处理
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
    def apply_async(self, func, args=(), kwargs={}, callback=None):
        """异步执行任务"""
        if self.pool is None:
            self.create_pool()
            
        result = self.pool.apply_async(func, args, kwargs, callback)
        
        # 记录子进程ID
        self._update_child_processes()
        return result
        
    def _update_child_processes(self):
        """更新子进程列表"""
        if self.pool is None:
            return
            
        try:
            # 获取所有活跃的工作进程
            for worker in self.pool._pool:
                if worker.is_alive():
                    self._child_processes.add(worker.pid)
        except:
            pass
            
    def cleanup_pool(self):
        """清理进程池"""
        if self.pool is None:
            return
            
        if not self._is_cleaning:
            self._is_cleaning = True
            
            try:
                print("正在清理进程池...")
                
                # 1. 终止池中的任务
                self.pool.terminate()
                
                # 2. 等待进程结束
                self.pool.join()
                
                # 3. 关闭池
                self.pool.close()
                
            except Exception as e:
                print(f"清理进程池时出错: {e}")
                
            finally:
                self.pool = None
                self._is_cleaning = False
                print("进程池清理完成")
                
    def cleanup_all_processes(self):
        """清理所有相关进程"""
        self.cleanup_pool()
        self._kill_zombie_processes()
        
    def _kill_zombie_processes(self):
        """杀死僵尸进程"""
        try:
            current_pid = os.getpid()
            parent = psutil.Process(current_pid)
            
            # 获取所有子进程
            children = parent.children(recursive=True)
            
            for child in children:
                try:
                    if child.is_running():
                        print(f"终止僵尸进程: PID {child.pid}, 名称: {child.name()}")
                        child.terminate()
                        child.wait(timeout=5)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    pass
                    
        except Exception as e:
            print(f"清理僵尸进程时出错: {e}")
            
    def __del__(self):
        """析构函数"""
        self.cleanup_all_processes()
        


import numpy as np
from typing import Union, List, Tuple
class FunctionClass():
    def resolve_label_conflicts(L1: Union[List[int], np.ndarray], 
                               L2: Union[List[int], np.ndarray]) -> Union[List[int], np.ndarray]:
        """
        解决L1和L2之间的标签冲突问题，保持输入输出类型一致
        
        参数:
        L1: list或np.array, 完整的类别标签数组
        L2: list或np.array, 部分样本的新标签数组，None表示未标记
        
        返回:
        与输入类型一致的解决冲突后的L2标签数组
        """
        
        # 保存输入类型
        input_type = type(L1)
        is_numpy_array = isinstance(L1, np.ndarray)
        
        # 转换为list进行处理
        L1_list = L1.tolist() if is_numpy_array else L1
        L2_list = L2.tolist() if isinstance(L2, np.ndarray) else L2
        
        # 输入验证
        #if len(L2_list) > len(L1_list):
        #    raise ValueError("L2的长度不能大于L1")
        
        # 创建L2的副本以避免修改原数组
        L2_resolved = L2_list.copy()
        
        # 获取L1中所有存在的标签（去重）
        existing_labels = set(L1_list)
        
        # 获取所有可用的标签值
        max_label = max(max(L1_list) if L1_list else 0, 
                       max([x for x in L2_list if x is not None]) if L2_list else 0)
        all_possible_labels = set(range(1, max_label + 10))  # 多预留一些标签
        
        # 找出可用的新标签（不在L1中的标签）
        available_labels = list(all_possible_labels - existing_labels)
        available_labels.sort()  # 排序以确保一致性
        
        # 记录L2中每个原始标签映射到的新标签
        label_mapping = {}
        
        # 第一次遍历：识别所有需要修改的标签
        labels_to_modify = set()
        for label in L2_resolved:
            if label is not None and label in existing_labels:
                labels_to_modify.add(label)
        
        # 为需要修改的标签分配新的标签值
        for original_label in labels_to_modify:
            if original_label not in label_mapping:
                # 分配最小的可用标签
                if available_labels:
                    new_label = available_labels.pop(0)
                    label_mapping[original_label] = new_label
                else:
                    # 如果没有可用标签，使用当前最大标签+1
                    current_max = max(set(L1_list) | set([x for x in L2_resolved if x is not None]) | set(available_labels))
                    new_label = current_max + 1
                    label_mapping[original_label] = new_label
        
        # 第二次遍历：应用标签映射
        for i in range(len(L2_resolved)):
            if L2_resolved[i] is not None and L2_resolved[i] in label_mapping:
                L2_resolved[i] = label_mapping[L2_resolved[i]]
        
        # 根据输入类型返回相应格式
        if is_numpy_array:
            return np.array(L2_resolved)
        else:
            return L2_resolved












class MyDialog(QDialog):
    def __init__(self):
        super().__init__();
        self.setWindowTitle('Plot Configs')
        formLayout = QFormLayout(self)
        
        def get_dsBox():
            temp_dsB=QDoubleSpinBox()
            temp_dsB.setMaximum(100)
            temp_dsB.setMinimum(0)
            temp_dsB.setDecimals(2)            
            temp_dsB.setSingleStep(1)
            return temp_dsB
        self.left= get_dsBox()
        self.top = get_dsBox()
        self.right = get_dsBox()
        self.bottom = get_dsBox()
        self.label=QLabel("fig parameters")
        button = QDialogButtonBox(QDialogButtonBox.Ok)
        formLayout.addRow('figure size:', self.label)
        formLayout.addRow('left', self.left)
        formLayout.addRow('top', self.top)
        formLayout.addRow('right', self.right)
        formLayout.addRow('bottom', self.bottom)
        formLayout.addRow(button)
        button.clicked.connect(self.accept)
    def get_input(self,figure_size:tuple=(400,300),raw_corp_para:tuple=(0,0,400,300)):
        pass
        self.show()
        self.label.setText(str(figure_size))
        
        

        
        self.left.setValue(raw_corp_para[0]/figure_size[0]*100)
        self.top.setValue(raw_corp_para[1]/figure_size[1]*100)
        

        self.right.setValue(raw_corp_para[2]/figure_size[0]*100)

        self.bottom.setValue(raw_corp_para[3]/figure_size[1]*100)
        
        if self.exec() == QDialog.Accepted:
            L,U,R,B=self.left.value(),self.top.value(),self.right.value(),self.bottom.value()
            L,U,R,B=int(L*figure_size[0]/100),int(U*figure_size[1]/100),int(R*figure_size[0]/100),int(B*figure_size[1]/100)
            if U>B:U=B
            if L>R:L=R
            return True,L,U,R,B
        else:
            return False,None,None ,None,None
            
class MyDialog2(QDialog):
    def __init__(self):
        super().__init__();
        self.setWindowTitle('Plot Configs')
        formLayout = QFormLayout(self)
        
        def get_dsBox():
            temp_dsB=QDoubleSpinBox()
            temp_dsB.setMaximum(4000)
            temp_dsB.setMinimum(0)
            temp_dsB.setDecimals(5)            
            temp_dsB.setSingleStep(1)
            return temp_dsB
        self.startmz = get_dsBox()
        self.endmz = get_dsBox()
        
        button = QDialogButtonBox(QDialogButtonBox.Ok)
        formLayout.addRow('Start mz', self.startmz)
        formLayout.addRow('end mz', self.endmz)
        formLayout.addRow(button)
        button.clicked.connect(self.accept)
    def get_input(self,mz_range:tuple=(50,2000)):
        pass
        self.startmz.setValue(mz_range[0])
        self.endmz.setValue(mz_range[1])
        self.show()
        if self.exec() == QDialog.Accepted:
            s,e=self.startmz.value(),self.endmz.value()
            if e<s:e=s
            return True,s,e
        else:
            return False,None,None 


#from PyQt5 import uic
class Window(QMainWindow,V_W_UI):
    #masscovert_finished_signal=pyqtSignal()
    show_sample_point_signal=pyqtSignal(dict)
    show_ms_signl=pyqtSignal(float,float)
    #threadpool_end_signal=pyqtSignal()
    def __init__(self,parent=None,**kwargs):
        #from qt_material import apply_stylesheet
        
        self.app=QApplication(sys.argv)
        #apply_stylesheet(self.app, theme='dark_cyan.xml')
        QMainWindow.__init__(self,parent)
        self.setupUi(self)
        #uic.loadUi('View2.ui', self)
        
        self._init_thread()
        self._init_processing()
        self._init_canvas()
        self.init_button()
        
        self.para={};
        self.read_para()
        self.msi_scan_para={"x_s":1,"x_e":10,"y_s":2,"y_e":8,'y_interval':1}
        self.copy_all_info_df=pd.DataFrame()
        self.copy_all_inten_df=pd.DataFrame()
        
        self.all_inten_DR_na=np.array([[]]);
        
        self.process_mzml_info_files=[] # {'fid':0,'mzml_file':'path'} 用来重新处理数据的
        
        self.display_MS_df=pd.DataFrame([[]])
        
        print(self.para)
        self.sign_and_slot()
        
        self.pinned_cluser=[]
        self.temp_dpi=-1;
        self.re_caculate_matrix=False
        self.is_show_sample_point=True
        self.is_blinking_sample_point=False
        
        self.running_post_compensation=False
        self.colormap_list=['Blues','BrBG','BuGn','BuPu','CMRmap','GnBu','Greens','Greys','OrRd',
            'Oranges','PRGn','PiYG','PuBu','PuBuGn','PuOr','PuRd','Purples','RdBu',
            'RdGy','RdPu','RdYlBu','RdYlGn','Reds','Spectral','Wistia','YlGn','YlGnBu',
            'YlOrBr','YlOrRd','afmhot','autumn','binary','bone','brg','bwr','cool','coolwarm',
            'copper','cubehelix','flag','gist_earth','gist_gray','gist_heat','gist_ncar','gist_rainbow',
            'gist_stern','gist_yarg','gnuplot','gnuplot2','gray','hot','hsv','jet','nipy_spectral',
            'ocean','pink','prism','rainbow','seismic','spring','summer','terrain','winter','Accent',
            'Dark2','Paired','Pastel1','Pastel2','Set1','Set2','Set3','tab10','tab20','tab20b','tab20c',]


        
        
    def init_button(self) :
        self.PB_E6.setEnabled(False)
        self.PB_DR.setEnabled(False)
        self.actionclassify_collection.setEnabled(False)
    def _init_thread(self):
        self.set_validator()
        
        self.send_data_queue=Queue(1);
        self.recive_data_queue=Queue(1);
        self.threadpool = QThreadPool()
        self.threadpool_is_alive=True;
        self.WrkrGUI =[]
        
    def _init_canvas(self):
        self._process_plot=MyProcessCanvas(width=3,height=6)
        self.gridLayout_myPlot1=QGridLayout(self.groupBox_3)
        self.gridLayout_myPlot1.addWidget(self._process_plot)
        
        self._dr_plot=DRCanvas(width=3,height=6)
        self.gridLayout_myPlot4=QVBoxLayout(self.groupBox_5)
        self.gridLayout_myPlot4.addWidget(self._dr_plot)
        
        self._pcaL_plot=MyPcaloadingCanvas(width=3,height=6)
        
        self._mass_spectrum_plot=MyMassSpectrumCanvas(width=4,height=4,UI=self)
        self.gridLayout_myPlot5=QVBoxLayout(self.groupBox_6)
        self.gridLayout_myPlot5.addWidget(self._mass_spectrum_plot)
        self.gridLayout_myPlot5.addWidget(self._mass_spectrum_plot.get_toolbar(self))
        
        self._cluster_plot=MyClusterCanvas(width=4.5,height=6,show_ms_signl=self.show_ms_signl)
        self._cluster_extent=None
        
        self.gridLayout_myPlot2=QVBoxLayout(self.groupBox_7)
        self.cockwidget=QDockWidget("cluster")
        
        self.gridLayout_myPlot2.addWidget(self._cluster_plot)
        self.gridLayout_myPlot2.addWidget(self._cluster_plot.get_toolbar(self))
        
        self._msi_plot=MyMsiCanvas(width=4.5,height=6,show_ms_signl=self.show_ms_signl)
        self.gridLayout_myPlot3=QVBoxLayout(self.groupBox_10)
        self.gridLayout_myPlot3.addWidget(self._msi_plot)
        self.gridLayout_myPlot3.addWidget(self._msi_plot.get_toolbar(self))
        
        
        
        
        
    
    def _init_processing(self):
        '''
        self.manager = BaseManager()
        self.manager.register('ManagerSpect', ManagerSpect2)
        self.manager.register('TargetArray_Mem', TargetArray_Mem)
        self.manager.start()
        
        self.mz_obj=self.manager.TargetArray_Mem(precision=1e-3)
        self.inten_obj=self.manager.ManagerSpect(1e-3)
        
        self.pre_pro_pool=Pool(POOL_NUM)
        self.mz_queue=Manager().Queue()
        self.mz_queue.put(1)
        self.inten_queue=Manager().Queue()
        self.inten_queue.put(1)
        
        self.res_l=[]
        
        self.mzml_list=[];#[{"fid":0,"file":r"****\***"},{...}]
        '''
        
        """改进的进程初始化"""
        self.process_manager = ProcessManager(max_workers=POOL_NUM)
        
        self.manager = BaseManager()
        self.manager.register('ManagerSpect', ManagerSpect2)
        self.manager.register('TargetArray_Mem', TargetArray_Mem)
        self.manager.register('Queue', Queue)
        self.manager.start()
        
        self.mz_obj = self.manager.TargetArray_Mem(precision=1e-3)
        self.inten_obj = self.manager.ManagerSpect(1e-3)
        
        # 使用Manager的Queue而不是multiprocessing.Queue
        self.mz_queue = self.manager.Queue()
        self.mz_queue.put(1)
        self.inten_queue = self.manager.Queue()
        self.inten_queue.put(1)
        
        self.res_l = []
        self.mzml_list = []
        
    def re_init_processing(self):
        '''
        self.para['pre_process_error']=float( self.lineEdit_10.text() )
        self.para['pre_process_threshold']=self.doubleSpinBox_6.value()
        self.mz_obj.re_init(self.para['pre_process_error'])
        self.inten_obj.re_init(self.para['pre_process_error'])
        self.pre_pro_pool.terminate()
        self.pre_pro_pool=Pool(POOL_NUM)
        try:
            temp_data=self._gen_message('start',self.msi_scan_para)
            self._process_plot.process(temp_data)
        except Exception as e:
            print(125,e)
        '''
        """重新初始化处理"""
        # 先清理旧的
        self.cleanup_processes()
        
        # 重新初始化
        self._init_processing()
        
        # 其他初始化代码...
        self.para['pre_process_error'] = float(self.lineEdit_10.text())
        self.para['pre_process_threshold'] = self.doubleSpinBox_6.value()
        self.mz_obj.re_init(self.para['pre_process_error'])
        self.inten_obj.re_init(self.para['pre_process_error'])
        
        try:
            temp_data = self._gen_message('start', self.msi_scan_para)
            self._process_plot.process(temp_data)
        except Exception as e:
            print(125, e)
        
        
    def exception_catching(title):
        def deco1(func):
            @wraps(func)
            def deco2(self,*args,**kwargs):
                print(116,args)
                try:
                    result=func(self,*args,**kwargs)
                    return result
                except Exception as e:
                    QMessageBox.warning(self, title , str(e))
                    print(256,e)
            return deco2
        return deco1 
        
    
        
    def trigger_processing_one_mzml(self,mzml_file="",fid=0):
        if pathlib.Path(mzml_file).exists():
            self.process_one_mzml(fid,mzml_file)
        else:
            print(f"not input a file, the input is:{mzml_file}")
            return
        
    
    def process_one_mzml(self,fid,mzml_file):
        #self.pre_pro_pool.apply_async(sub_process_mzml,args=(fid,mzml_file,self.obj))
        #self.pre_pro_pool.close()
        #self.pre_pro_pool.join()
        #self.inten_obj.set_time_stamp(begin_time)
        temp_time=time.time()
        res=self.process_manager.apply_async(sub_process_mzml_by_dict3,args=(self.inten_queue,self.mz_queue,fid,mzml_file,self.mz_obj,self.inten_obj,self.para['pre_process_threshold']))
        #self.res_l.append(res)
        res.get()
        mystr=f' 83, finished the deal {fid} mzml!!!, cost {round(time.time()-temp_time,2)}s'
        print('\r',mystr,end='\n',flush=True)
        
        send_data={"type":"finish_process_one_mzml","fid":fid}
        self.recive_data_queue.put(send_data)
    
    def do_post_compensation(self):
        if self.inten_obj.get_inten_len()==0:
            print("there is no data in inten_df")
            return
        
        if len(self.process_mzml_info_files)==0:
            print("there is no data in process_mzml_info_files")
            return
        if self.running_post_compensation: return
        
        self.running_post_compensation=True
        self.pushButton_8.setEnabled(False)
        
        if self.inten_queue.empty():
            print("post_compensation_waitting for unlock")
        print("post_compensation_waitting is unlocked")
        
        
        self.inten_queue.get()
        all_row_column_list_copy=copy.deepcopy(self.inten_obj.output_row_cloumn_list())
        self.inten_queue.put(1)
        
        target_mz_list=all_row_column_list_copy[-1]['mz_column']
        
        temp_handle=[]
        for item in all_row_column_list_copy:
            fid=item['fid']
            res=self.process_manager.apply_async(subproces_do_post_compensation,args=(self.inten_queue,fid,item['mz_column'],target_mz_list,self.inten_obj))
            temp_handle.append(res)
            
        def sub_thread_spy(*args):
            for item in temp_handle:
                item.get()
            
            
        
            self.running_post_compensation=False
            self.pushButton_8.setEnabled(True)
            print("########  finish post compensation###########")
            
        self.threadpool.start( WorkerGUI(sub_thread_spy,1) )
    
    def do_post_compensation_old(self):
        if self.inten_obj.get_inten_len()==0:
            print("there is no data in inten_df")
            return
        
        if len(self.process_mzml_info_files)==0:
            print("there is no data in process_mzml_info_files")
            return
        if self.inten_queue.empty():
            print("post_compensation_waitting for unlock")
        print("post_compensation_waitting is unlocked")
        
        self.inten_queue.get()
        try:
            self.inten_obj.post_compensation()
        except Exception as e:
            print("184 in view_demo, error happen in post_compensation. You'd better re_processing_inten_df again!!!\n", e)
        self.inten_queue.put(1)
        
    def re_processing_inten_df(self):
        pass
        self.re_init_processing();
        for mzml_file_info in self.process_mzml_info_files:
            self.threadpool.start( WorkerGUI(self.trigger_processing_one_mzml,mzml_file_info['file'],mzml_file_info['fid']  ) )
    
    
    def write_para(self):
        self.lineEdit_10.setText(str(self.para['pre_process_error'] ))
        if self.para['matrix_relative_inten']:
            self.radioButton.setChecked(1)
            self.radioButton_2.setChecked(0)
        else:
            self.radioButton.setChecked(0)
            self.radioButton_2.setChecked(1)
            
            
        self.checkBox_3.setCheckState(  self.para['matric_IS_active'] )   
        
        self.lineEdit_11.setText(str( self.para['matric_IS_mz']))
        self.checkBox_5.setCheckState(  self.para['matric_duplexing'] )   
        
        self.doubleSpinBox.setValue( self.para['matric_resolution_value'])
        self.spinBox.setValue( self.para['matric_Cluster_n'])
        
        self.cluster_meth_combobox_2.setCurrentText(self.para['matrix_RE_func'])
        self.spinBox_5.setValue( self.para['matrix_RE_components'])
        self.cluster_meth_combobox.setCurrentText(self.para['matrix_Cluster_func'])
        self.spinBox.setValue( self.para['matrix_Cluster_components'])
        
        self.spinBox_4.setValue( self.para['matric_pcaL_components'])
        self.spinBox_2.setValue( self.para['matric_Metablism_n'])
        self.spinBox_3.setValue( self.para['matric_pcal_s_n'])
        
        
        self.doubleSpinBox_2.setValue( self.para['matric_pick_alpha'])
        self.doubleSpinBox_3.setValue( self.para['matric_pick_size'])
        
        self.checkBox_6.setCheckState(  self.para.get('matric_real_R_time_active',0) )  
        self.checkBox_8.setCheckState(  self.para.get('matric_real_delay_active',0) )  
        
        self.doubleSpinBox_4.setValue( self.para.get('matric_real_R_time',0))
        self.doubleSpinBox_5.setValue( self.para.get('matric_real_delay',0))
        self.doubleSpinBox_6.setValue( self.para.get('pre_process_threshold',0))
        
        
        self.doubleSpinBox_7.setValue( self.para.get('matric_extract_mz_TH',2))
        
        
        self.lineEdit.setText(str(self.para['MSI_target_mz']))
        self.lineEdit_2.setText(str(self.para['MSI_step_mz']))
        self.lineEdit_3.setText(str(self.para['MSI_step_error']))
        
        
        
        self.para['msi_colormap']=self.para.get('msi_colormap','gnuplot2')
        self.para['clustr_colormap']=self.para.get('clustr_colormap','tab20b')
        
        self._msi_plot.set_colormap(self.para['msi_colormap'])
        self._cluster_plot.set_colormap(self.para['clustr_colormap'])
        
        self.para['msi_crop_para']=self.para.get('msi_crop_para',(0,0,600,400))
        
        self.checkBox_9.setCheckState(  self.para.get("detection_in_collection",0) ) 
        self.spinBox_6.setValue( self.para.get('loop_control_num',20))
        self.spinBox_7.setValue( self.para.get('rank_num',2))
        self.spinBox_8.setValue( self.para.get('area_threshold',3))
        
        #self.para['matric_point_blink']= self.checkBox_3.checkState()
        self.checkBox_10.setCheckState(  self.para.get("matric_point_blink",2) ) 
        
    def read_para(self):
        self.para={};
        self.para['smoth_Gaussian_active']= self.checkBox_2.checkState()
        self.para['smoth_Gaussian_sigma'] = int( self.lineEdit_7.text() )
        self.para['smoth_Gaussian_W'] = int( self.lineEdit_6.text() )
        self.para['smoth_Gaussian_H'] = int( self.lineEdit_7.text() )
        
        self.para['smoth_interpolation_active'] = self.checkBox_2.checkState()
        self.para['smoth_interpolation_x'] = int( self.lineEdit_4.text() )
        self.para['smoth_interpolation_y'] = int( self.lineEdit_5.text() )
        

        self.para['pre_process_error']=float( self.lineEdit_10.text() )
        self.para['pre_process_threshold']=self.doubleSpinBox_6.value()
        
        self.para['matrix_relative_inten']=self.radioButton.isChecked()
        self.para['matric_IS_active']= self.checkBox_3.checkState()
        self.para['matric_IS_mz']=float( self.lineEdit_11.text() )
        self.para['matric_duplexing']=self.checkBox_5.checkState()
        
        self.para['matric_real_R_time_active'] = self.checkBox_6.checkState()
        self.para['matric_real_delay_active'] = self.checkBox_8.checkState()
        self.para['matric_real_R_time'] = self.doubleSpinBox_4.value()
        self.para['matric_real_delay'] = self.doubleSpinBox_5.value()
        
        self.para['matric_extract_mz_TH'] = self.doubleSpinBox_7.value()
        
        self.para['matric_resolution_value']= self.doubleSpinBox.value()
        self.para['matric_Cluster_n']= self.spinBox.value()
        
        
        self.para['matrix_RE_func']=self.cluster_meth_combobox_2.currentText()
        self.para['matrix_RE_components']=self.spinBox_5.value()
        self.para['matrix_Cluster_func']=self.cluster_meth_combobox.currentText()
        self.para['matrix_Cluster_components']=self.spinBox.value()
        
        
        self.para['matric_pcaL_components']=self.spinBox_4.value()
        self.para['matric_Metablism_n']= self.spinBox_2.value()
        self.para['matric_pcal_s_n']= self.spinBox_3.value()
        
        self.para['matric_pick_alpha']=self.doubleSpinBox_2.value()
        self.para['matric_pick_size']= self.doubleSpinBox_3.value()
        self.para['matric_point_blink']= self.checkBox_10.checkState()
        
        self.para['MSI_V_max_times']=float( self.lineEdit_18.text() )
        self.para['MSI_V_min_times']=float( self.lineEdit_22.text() )
        

        self.para['MSI_target_mz']=float( self.lineEdit.text() )
        self.para['MSI_step_mz']=float( self.lineEdit_2.text() )
        self.para['MSI_step_error']=float( self.lineEdit_3.text() )
        
        
        
        self.para['msi_colormap']='gnuplot2'
        self.para['clustr_colormap']='tab20b'
        
        self.para['msi_crop_para']=(0,0,600,400)
        
        
        self.para['detection_in_collection']=self.checkBox_9.checkState()
        self.para['loop_control_num']=self.spinBox_6.value()
        self.para['rank_num']=self.spinBox_7.value()
        self.para['area_threshold']=self.spinBox_8.value()
        
        
        
    def set_validator(self):
        doubleValidator = QDoubleValidator(0,1000,2)
        intValidator=QIntValidator(0,99)
        
        self.lineEdit_7.setValidator(intValidator)
        self.lineEdit_6.setValidator(intValidator)
        self.lineEdit_9.setValidator(intValidator)
        self.lineEdit_4.setValidator(intValidator)
        self.lineEdit_5.setValidator(intValidator)
        
        
        
        self.lineEdit_10.setValidator(doubleValidator)
        self.lineEdit_18.setValidator(doubleValidator)
        self.lineEdit_22.setValidator(doubleValidator)
        
        doubleValidator5 = QDoubleValidator(0,10000,5)
        #self.lineEdit_8.setValidator(doubleValidator5)
        self.lineEdit.setValidator(doubleValidator5)
        self.lineEdit_2.setValidator(doubleValidator5)
        self.lineEdit_3.setValidator(doubleValidator5)
        self.lineEdit_11.setValidator(doubleValidator5)
        
    
    def set_msi_crop_para(self):
        w,h=self._msi_plot.get_canvas_size()
        is_ok,left,top,right,bottom=MyDialog().get_input((w,h),self.para['msi_crop_para'])
        if is_ok:
            print(f"set the crop para to:{( left,top,right,bottom  )}")
            self._msi_plot.set_crop_para(( left,top,right,bottom  ))
            self.para['msi_crop_para']=(left,top,right,bottom)
    
    def _is_file_exit(self,file_path):
        if pathlib.Path(file_path).exists():
            return True
        else:
            print(f"the {file_path} is not exit")
            return False
    
    def file_convert_function(self,file_path,fid):
        if not self._is_file_exit(file_path):return
        
        convert_f(file_path)
        
        send_data={'type':"pro_processing","file":file_path.replace('.raw','.mzML'),'fid':fid}
        self.recive_data_queue.put(send_data)
    
    def read_show_rows_func(self):
        
        if self.actionshow_odd_rows.isChecked():return 1
        if self.actionshow_even_rows.isChecked():return 2
        return 0
    def show_rows_func(self,code=0):
        if code==0:
            self.actionshow_all_rows.setChecked(True)
            self.actionshow_odd_rows.setChecked(False)
            self.actionshow_even_rows.setChecked(False)
        elif code==1:
            self.actionshow_all_rows.setChecked(False)
            self.actionshow_odd_rows.setChecked(True)
            self.actionshow_even_rows.setChecked(False)
        elif code==2:
            self.actionshow_all_rows.setChecked(False)
            self.actionshow_odd_rows.setChecked(False)
            self.actionshow_even_rows.setChecked(True)
    
    def show_info_inten_df(self):
        print("wait for unlock")
        self.inten_queue.get()
        
        #self.copy_all_info_df,self.copy_all_inten_df=self.inten_obj.output()
        
        temp_info_file_abpath,temp_inten_file_abpath=self.inten_obj.output()
        if temp_info_file_abpath :
            self.copy_all_info_df=pd.read_pickle(temp_info_file_abpath)
            self.copy_all_inten_df=pd.read_pickle(temp_inten_file_abpath)
        
        print(self.copy_all_info_df)
        print(self.copy_all_inten_df)
        self.inten_queue.put(1)
    
    def get_Matrix_from_raw_data(self):
        self.inten_queue.get()
        
        temp_info_file_abpath,temp_inten_file_abpath=self.inten_obj.output()
        if temp_info_file_abpath :
            self.raw_all_info_df=pd.read_pickle(temp_info_file_abpath)
            self.raw_all_inten_df=pd.read_pickle(temp_inten_file_abpath)
        self.inten_queue.put(1)
        print("successful to get matrix from raw data")
    def calculate_Matrix(self):
        
        
        self.para['matrix_relative_inten']=self.radioButton.isChecked()
        self.para['matric_IS_active']= self.checkBox_3.checkState()
        self.para['matric_IS_mz']=float( self.lineEdit_11.text() )
        self.para['matric_duplexing']=self.checkBox_5.checkState()        
        self.para['matric_real_R_time_active'] = self.checkBox_6.checkState()
        self.para['matric_real_delay_active'] = self.checkBox_8.checkState()
        self.para['matric_real_R_time'] = self.doubleSpinBox_4.value()
        self.para['matric_real_delay'] = self.doubleSpinBox_5.value()
        self.para['show_rows_func']=self.read_show_rows_func()
        #print(self.para)
        #return
        #self.para['matric_Cluster_n']= self.spinBox.value()
        #self.para['matric_Metablism_n']= self.spinBox_2.value()
        #self.para['matric_pcal_s_n']= self.spinBox_3.value()
        
        self.re_caculate_matrix=True
        try:
            self.copy_all_info_df=self.raw_all_info_df.copy(deep=True)
            self.copy_all_inten_df=self.raw_all_inten_df.copy(deep=True)
        except Exception as e:
            print(f"859 in view demo, error has occord in calculate_Matrix, and the error is {e}")
            return
        
        if len(self.copy_all_info_df) != len(self.copy_all_inten_df):
            print("error in caculate Matrix, the length isn't match between info_df and inten df. You'd better re_processing again");
            return
        if len(self.copy_all_info_df)==0:
            print(f"there is no data in all_inten_df")
            return 
        
        def to_relative_inten(ser):#横向
            return ser/ser.max()
        
        def map_info_df_to_point_df(ser,odd_equetion,even_equetion,row_therhold,fid_equation=[1,0]):
            output_map=[0,0,0]
            if ser['fid'] % 2==0:
                if ser['R_time']>row_therhold[1]:
                    output_map[2]=1
                    return output_map
                else:
                    output_map[0]=ser['R_time']*odd_equetion[0]+odd_equetion[1]
                    output_map[1]=ser['fid']*fid_equation[0]+fid_equation[1]
                    
                    return output_map
            else:
                if ser['R_time']<row_therhold[0]:
                    output_map[2]=1
                    return output_map
                else:
                    output_map[0]=ser['R_time']*even_equetion[0]+even_equetion[1]
                    output_map[1]=ser['fid']*fid_equation[0]+fid_equation[1]
                    return output_map
            return output_map
        
        def map_to_IS(ser,IS_inten):
            return ser/IS_inten
        
        if self.para['matrix_relative_inten']:
            self.copy_all_inten_df=self.copy_all_inten_df.apply(to_relative_inten,axis=1)
        
        if self.para['matric_IS_active']>0:
            upper=self.para['matric_IS_mz']*( 1 + self.para['pre_process_error']/2 )
            lower=self.para['matric_IS_mz']*( 1 - self.para['pre_process_error']/2 )
            numeric_columns= self.copy_all_inten_df.columns.astype('float')
            chosen_columns=(numeric_columns < upper ) & (numeric_columns >= lower )
            #self.copy_all_info_df['x']=self.copy_all_info_df['x'].apply(lambda x: resolution(x,self.para['matric_resolution_value']))
            IS_inten=self.copy_all_inten_df.loc[:,chosen_columns].mean(axis=1)
            IS_inten.fillna(0,inplace=True)
            IS_inten=IS_inten+1
            self.copy_all_inten_df=self.copy_all_inten_df.apply(lambda x: map_to_IS(x,IS_inten),axis=0)
            print(400,"IS is actived")
            #self.copy_all_info_df['target_inten']=self.copy_all_inten_df.loc[:,chosen_columns].mean(axis=1)    #np.mean( self.copy_all_inten_df.loc[:,chosen_columns], axis=1)
            #self.copy_all_info_df.fillna(0,inplace=True);
        ####
        row_therhold=[self.copy_all_info_df['R_time'].min(),self.copy_all_info_df['R_time'].max()]
        if "x_s" not  in self.msi_scan_para:
            print('no_scan para input')
            return 
            
        if self.para['matric_duplexing']>0:
            #odd_therhold=[0,0]
            #odd_row
            if self.para["matric_real_R_time_active"]==0 :
                self.para['matric_real_R_time']=self.copy_all_info_df['R_time'].max()
            
            if self.para["matric_real_delay_active"]==0:
                self.para['matric_real_delay']=0
            
            
            delay_time=self.para['matric_real_delay']/60
            RT_max=self.para['matric_real_R_time']/60 + delay_time
            RT_min=0+ delay_time
            #if self.para["matric_Cut_forward"]:
            #    RT_max=self.copy_all_info_df['R_time'].max()
            #    RT_min=self.copy_all_info_df['R_time'].min()+RT_max*self.para['matric_Cut_tail_value']
            #else:
            #    RT_min=self.copy_all_info_df['R_time'].min()
            #    RT_max=self.copy_all_info_df['R_time'].max()*(1-self.para['matric_Cut_tail_value'])
            odd_equetion=[1,0]
            odd_equetion[0]=(self.msi_scan_para['x_e']-self.msi_scan_para['x_s'])/(RT_max-RT_min)
            odd_equetion[1]=self.msi_scan_para['x_s']-odd_equetion[0]*RT_min
            row_therhold[1]=RT_max
            
            #even_row
            #RT_min=self.copy_all_info_df['R_time'].min()+ self.copy_all_info_df['R_time'].max() * self.para['matric_Cut_tail_value']
            #RT_max=self.copy_all_info_df['R_time'].max()
            even_equetion=[1,0]
            
            even_equetion[0]=(self.msi_scan_para['x_s']-self.msi_scan_para['x_e'])/(RT_max-RT_min)
            even_equetion[1]=self.msi_scan_para['x_e']-even_equetion[0]*RT_min
            row_therhold[0]=RT_min
        else:
            RT_min=self.copy_all_info_df['R_time'].min()
            RT_max=self.copy_all_info_df['R_time'].max()
            
            K_x=(self.msi_scan_para['x_e']-self.msi_scan_para['x_s'])/(RT_max-RT_min)
            bx=self.msi_scan_para['x_s']-K_x*RT_min
            odd_equetion=[K_x,bx]
            even_equetion=odd_equetion
            
        fid_max=int(   (self.msi_scan_para['y_e']-self.msi_scan_para['y_s'])/ self.msi_scan_para['y_interval']           ) + 1 #self.copy_all_info_df['fid'].max()
        fid_min=0#self.copy_all_info_df['fid'].min()
        
        
        
        K_y=(self.msi_scan_para['y_e']-self.msi_scan_para['y_s'])/ (fid_max-fid_min)
        by=self.msi_scan_para['y_s']-K_y*fid_min
        y_equetion=[K_y,by]
        
        y_ee=self.copy_all_info_df['fid'].max()*K_y + by
        self._cluster_extent=(self.msi_scan_para['x_e'],self.msi_scan_para['x_s'],y_ee,self.msi_scan_para['y_s'])
        self._cluster_plot.set_extent(self._cluster_extent)
        self._msi_plot.set_extent(self._cluster_extent)
        print(self._cluster_extent)
        
        if 'x' not in self.copy_all_info_df.columns:
            self.copy_all_info_df['x']=0;
            self.copy_all_info_df['y']=0;
        if 'cut' not in self.copy_all_info_df.columns:
            self.copy_all_info_df['cut']=0;
        
        if 'labels' not in self.copy_all_info_df.columns:
            self.copy_all_info_df['labels']=0;
            
        if 'pinned' not in self.copy_all_info_df.columns:
            self.copy_all_info_df['pinned']=0;
        if 'collected' not in self.copy_all_info_df.columns:
            self.copy_all_info_df['collected']=0;
            
        print(f"321,therhold:{row_therhold}")
        self.copy_all_info_df[['x','y','cut']]=self.copy_all_info_df[['R_time','fid']].apply(lambda x: map_info_df_to_point_df(x,odd_equetion,even_equetion,row_therhold,y_equetion),axis=1,result_type="expand")
        cut_remain=self.copy_all_info_df['cut']==0
        
        self.copy_all_info_df['row_type']=self.copy_all_info_df['fid']&2+1
        
        if self.para['show_rows_func']!=0:
            temp_rows_hit=self.copy_all_info_df['row_type']!=self.para['show_rows_func']
            self.copy_all_info_df.loc[temp_rows_hit,'cut']=1
        
        
        print(f"326, there still remain {cut_remain.sum()} point ")
        self.copy_all_info_df=self.copy_all_info_df.loc[cut_remain,:]
        
        
        #重新把x映射到 MSI的物理x扫描范围
        #self.msi_scan_para['x_e'],self.msi_scan_para['x_s']
        table_max,table_min=self.copy_all_info_df['x'].max(),self.copy_all_info_df['x'].min()
        real_max,real_min=self.msi_scan_para['x_e'],self.msi_scan_para['x_s']
        tt_k=(real_max-real_min)/(table_max-table_min)
        tt_b=real_max-tt_k*table_max
        self.copy_all_info_df['x']=self.copy_all_info_df['x'].apply(lambda x:tt_k*x+tt_b)
        
        self.copy_all_info_df_temp=copy.deepcopy(self.copy_all_info_df) # 为什么要这个,因为 setdpi 可以使用原始的x,y数据（未被 set_dpi的）
        self.set_dpi()
        
        self.copy_all_inten_df=self.copy_all_inten_df.loc[cut_remain,:]
        
        self.copy_all_inten_df.fillna(value=0,inplace=True);
        self.temp_copy_all_inten_df=copy.deepcopy( self.copy_all_inten_df )
        self.extract_mz_by_TH()
        
        
        self.PB_DR.setEnabled(True)    
        self.PB_E6.setEnabled(False) 
        
        print("finish the calculate_Matrix ")
    def extract_mz_by_TH(self):
        self.para['matric_extract_mz_TH'] = self.doubleSpinBox_7.value()
        
        try:
            inten_np=self.temp_copy_all_inten_df.values
            base_peek_inten=np.max(inten_np,axis=1)
            thresholds_value=(base_peek_inten* self.para['matric_extract_mz_TH']/100 ).reshape(-1,1)
            
            hits_all=inten_np>thresholds_value
            
            hits=np.sum(hits_all,axis=0)
            hit_columns=hits>0
            
            self.copy_all_inten_df=self.temp_copy_all_inten_df.loc[:,hit_columns]
            print(f"extract_mz_by  {self.para['matric_extract_mz_TH']}% of base_peak")
        except Exception as e:
            print(542," error in extract_mz_list : " ,e )
        
        
        
    def set_dpi(self):
    
        self.para['matric_resolution_value']= self.doubleSpinBox.value()
        if self.temp_dpi== self.para['matric_resolution_value'] and self.re_caculate_matrix==False:
            print("the dpi have no change")
            return
        def resolution(value,res=0.2,decimal=4):
            return round(( ( value/res).astype('int32') )* res , decimal)
        try:
        
            self.copy_all_info_df[['x','y']]=self.copy_all_info_df_temp[['x','y']].apply( lambda x: resolution(x,self.para['matric_resolution_value']), axis=1,result_type="expand")#np.round(all_info_df['x'],1)    
            #self.copy_all_info_df[['x','y']]=self.copy_all_info_df[['x','y']].apply( lambda x: resolution(x,self.para['matric_resolution_value']), axis=1,result_type="expand")#np.round(all_info_df['x'],1)   
            print(f"set the dpi to {self.para['matric_resolution_value']}")
            self.temp_dpi= self.para['matric_resolution_value']
            self.re_caculate_matrix=True
        except Exception as e:
            print(512,e)    
        
    
    
    
    def pca_plot_func(self):
        self.para['matric_pcaL_components']=self.spinBox_4.value()
        self.para['matric_Metablism_n']= self.spinBox_2.value()
        
        self._pcaL_plot.myPlot(self.copy_all_inten_df,self.para['matric_pcaL_components'],self.para['matric_Metablism_n'])
    
    
        
        
        
    def dimension_reduction_func(self,inten_df):
        if self.copy_all_inten_df.shape[0]==0:
            print("copy_all_inten_df is null")
            return
        self.para['matrix_RE_func']=self.cluster_meth_combobox_2.currentText()
        self.para['matrix_RE_components']=self.spinBox_5.value()
        print("972 in demo, begin to DR")
        hit=self.copy_all_info_df['pinned']==0
        
        self.all_inten_DR_na=DR_FUNC[self.para['matrix_RE_func']]( self.copy_all_inten_df.loc[hit,:],self.para['matrix_RE_components'] )
        print("1094 in demo, filish to DR")
        self._dr_plot.my_plot(self.all_inten_DR_na)
        self.PB_E6.setEnabled(True)
        print("979 in demo, finished to DR")
        
    def cluster_func(self):
        self.para['matrix_Cluster_func']=self.cluster_meth_combobox.currentText()
        self.para['matrix_Cluster_components']=self.spinBox.value()
        self.para['matric_resolution_value']= self.doubleSpinBox.value()
        
        self.para['smoth_Gaussian_active']= self.checkBox_2.checkState()
        self.para['smoth_Gaussian_sigma'] = int( self.lineEdit_7.text() )
        self.para['smoth_Gaussian_W'] = int( self.lineEdit_6.text() )
        self.para['smoth_Gaussian_H'] = int( self.lineEdit_7.text() )
        
        self.para['smoth_interpolation_active'] = self.checkBox_2.checkState()
        self.para['smoth_interpolation_x'] = int( self.lineEdit_4.text() )
        self.para['smoth_interpolation_y'] = int( self.lineEdit_5.text() )
        
        
        
        if self.all_inten_DR_na.shape[0]==0:
            print(f"all_inten_DR_na is null")
            return
        
        print("/n/n1001 in demo, begin to Cluster")
        hit=self.copy_all_info_df['pinned']==0
        not_hit=~hit
        
        labels=CLUSTER_FUNC[self.para['matrix_Cluster_func']](self.all_inten_DR_na,  self.para['matrix_Cluster_components']   )
        
        #labels=labels+len(self.pinned_cluser)+1
        
        #把少量pix 类 当作噪点
        labels=labels+1
        raw_labels=labels.copy()
        all_hit_len=len(labels)
        for label in np.unique(raw_labels):
            np_hit=labels==label
            if np.sum(np_hit)<all_hit_len/(3*self.para['matrix_Cluster_components']):
                labels[np_hit]=0
        labels=labels+1#小于0.01的 统一归为other
        
        #解决L1和L2之间的标签冲突问题，保持输入输出类型一致
        labels=FunctionClass.resolve_label_conflicts(self.copy_all_info_df.loc[not_hit,:]["labels"].values,labels)
        
        
        
        
        
        self.copy_all_info_df.loc[hit,'labels']=labels
        
        self._dr_plot.set_labels(self.copy_all_info_df['labels'])
        
        self._cluster_plot.myPlot(self.copy_all_info_df,self.para['matric_resolution_value'],self.para)
        self._cluster_plot.is_show_MS=True
        print(f"there are {len(self.copy_all_info_df['labels'].unique())} labels in all_info_df, and the unique labels is\n:",self.copy_all_info_df['labels'].unique() )
        
        print("972 in demo, finished to Cluster")
    
    def include_cluster_in_collection_func(self):
        collect_target=self._cluster_plot.get_cluster_pick_num()
        if collect_target==-1:
            print("have not pick a point")
             
        else:
            hit=self.copy_all_info_df['labels']==collect_target
            if hit.sum()==0:
                print("nope! the hit cluster points is O!")
            else:
                print("/n/n1045 in demo, begin to DR")
                self.copy_all_info_df.loc[hit,'collected']=1
                
                print(f"success collect the {collect_target} cluster points,, the collection_shape is {np.sum(self.copy_all_info_df['collected']==1)}")
                self.actionclassify_collection.setEnabled(False)
                print("1050 in demo, finished to DR")
                
    def exclude_cluster_in_collection_func(self):
        collect_target=self._cluster_plot.get_cluster_pick_num()
        if collect_target==-1:
            print("have not pick a point")
             
        else:
            
            hit=self.copy_all_info_df['labels']==collect_target
            if hit.sum()==0:
                print("nope! the cluster have not been pinned yet")
            else:
                #self.copy_all_info_df.loc[hit,'labels']=len(self.pinned_cluser)
                self.copy_all_info_df.loc[hit,'collected']=0
                
                print(f"success exclude the {collect_target} cluster from collections, the collection_shape is {np.sum(self.copy_all_info_df['collected']==1)}")
                self.actionclassify_collection.setEnabled(False)
    
    def auto_DR_and_cluster(self):
        self.auto_methoud_is_recursion=True #"loop" False # "recursion" True
        self.para["loop_control_num"]=self.spinBox_6.value()
        self.para["detection_in_collection"]=self.checkBox_9.checkState()
        
        if self.copy_all_inten_df.shape[0]==0:
            print("copy_all_inten_df is null")
            return
            
            
        if "labels" not in self.copy_all_info_df.columns:
            print("please pin background frist")
            return
        
        hit_text="pinned"
        
        if self.para["detection_in_collection"]>0:
            hit_text="collected"
        
        hit=self.copy_all_info_df[hit_text]==0
        
        target_label_rank=0
        if self.auto_methoud_is_recursion:
            counts_label_series=self.copy_all_info_df.loc[hit,:]['labels'].value_counts()
            loop_control_num=self.para["loop_control_num"]
            while counts_label_series.iat[target_label_rank]>hit.sum()/loop_control_num:
                #cluster_label=counts_label_series.index[counts_label_series==counts_label_series.max()][0]
                cluster_label=counts_label_series.index[target_label_rank]
                state=self.auto_DR_and_cluster_func(cluster_label)
                if not state:
                    target_label_rank+=1
                    continue
                    
                    
                counts_label_series=self.copy_all_info_df.loc[hit,:]['labels'].value_counts()
                #if counts_label_series.max()>hit.sum()/10
                
                loop_control_num-=1
                if loop_control_num<0:break;
        
        
    
    def auto_DR_and_cluster_func(self,cluster_label):
        hit=self.copy_all_info_df['labels']==cluster_label
        if hit.sum()<10:
            print("1044, no enough pix to auto df and cluster")
            
        self.para['matrix_RE_func']=self.cluster_meth_combobox_2.currentText()
        self.para['matrix_RE_components']=self.spinBox_5.value()
        self.para['matrix_Cluster_func']=self.cluster_meth_combobox.currentText()
        self.para['matrix_Cluster_components']=self.spinBox.value()
        self.para['matric_resolution_value']= self.doubleSpinBox.value()
        
        self.para['smoth_Gaussian_active']= self.checkBox_2.checkState()
        self.para['smoth_Gaussian_sigma'] = int( self.lineEdit_7.text() )
        self.para['smoth_Gaussian_W'] = int( self.lineEdit_6.text() )
        self.para['smoth_Gaussian_H'] = int( self.lineEdit_7.text() )
        
        self.para['smoth_interpolation_active'] = self.checkBox_2.checkState()
        self.para['smoth_interpolation_x'] = int( self.lineEdit_4.text() )
        self.para['smoth_interpolation_y'] = int( self.lineEdit_5.text() )
        
        print(f"/n/n/n1131 in demo, begin to auto DR and cluster in label {cluster_label}")
        
        self.all_inten_DR_na=DR_FUNC[self.para['matrix_RE_func']]( self.copy_all_inten_df.loc[hit,:],self.para['matrix_RE_components'] )
        
        
        not_hit=~hit
        
        labels=CLUSTER_FUNC[self.para['matrix_Cluster_func']](self.all_inten_DR_na,  self.para['matrix_Cluster_components']   )
        
        #把少量pix 类 当作噪点
        labels=labels+1
        raw_labels=labels.copy()
        all_hit_len=len(labels)
        for label in np.unique(raw_labels):
            np_hit=labels==label
            if np.sum(np_hit)<all_hit_len/(3*self.para['matrix_Cluster_components']):
                labels[np_hit]=0
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        if np.max(counts)>len(labels)*0.95:
            print(f"class {cluster_label} clustring failure!!")
            return False
        
        
        labels=labels+1#小于0.01的 统一归为other
        
        #解决L1和L2之间的标签冲突问题，保持输入输出类型一致
        labels=FunctionClass.resolve_label_conflicts(self.copy_all_info_df.loc[not_hit,:]["labels"].values,labels)
        
        
        
        
        
        self.copy_all_info_df.loc[hit,'labels']=labels
        
        self._cluster_plot.myPlot(self.copy_all_info_df,self.para['matric_resolution_value'],self.para)
        self._cluster_plot.is_show_MS=True
        print(f"there are {len(self.copy_all_info_df['labels'].unique())}")
        print(f"1168 in demo, Finished to auto DR and cluster in label:{cluster_label}/n")
        return True
    
    def auto_detect_cluster_func(self):
        self.para['rank_num']=self.spinBox_7.value()
        self.para['area_threshold']=self.spinBox_8.value()
        self._cluster_plot.auto_detect_cluster(self.para['rank_num'],self.para['area_threshold'])
    
    def DR_collection_func(self):
        
        hit= self.copy_all_info_df["collected"]==1
        print(f"there are {np.sum(hit)} pix need to be DR ->",end='')
        self.collection_for_DR_df=self.copy_all_inten_df.loc[hit,:]
        if self.collection_for_DR_df.shape[0]==0:
            print("collection_for_DR_df is null")
            return
        self.para['matrix_RE_func']=self.cluster_meth_combobox_2.currentText()
        self.para['matrix_RE_components']=self.spinBox_5.value()
        
        #hit=self.copy_all_info_df['pinned']==0
        self.collection_for_DR_df_na=DR_FUNC[self.para['matrix_RE_func']]( self.collection_for_DR_df,self.para['matrix_RE_components'] )
        self.actionclassify_collection.setEnabled(True)
        print(f" #######<-finish DR ")
    def classify_collection_func(self):
        if len(self.collection_for_DR_df_na)==0:
            print("no data in collection for DR")
            return 
        
        self.para['matrix_Cluster_func']=self.cluster_meth_combobox.currentText()
        self.para['matrix_Cluster_components']=self.spinBox.value()
        self.para['matric_resolution_value']= self.doubleSpinBox.value()
        
        self.para['smoth_Gaussian_active']= self.checkBox_2.checkState()
        self.para['smoth_Gaussian_sigma'] = int( self.lineEdit_7.text() )
        self.para['smoth_Gaussian_W'] = int( self.lineEdit_6.text() )
        self.para['smoth_Gaussian_H'] = int( self.lineEdit_7.text() )
        
        self.para['smoth_interpolation_active'] = self.checkBox_2.checkState()
        self.para['smoth_interpolation_x'] = int( self.lineEdit_4.text() )
        self.para['smoth_interpolation_y'] = int( self.lineEdit_5.text() )
        
        
        
        
        hit=self.copy_all_info_df['collected']==1
        
        #labels=CLUSTER_FUNC[self.para['matrix_Cluster_func']](self.collection_for_DR_df_na,  self.para['matrix_Cluster_components']   )
        
        #labels=labels+len(self.copy_all_info_df['labels'].unique())+3
        not_hit=~hit
        
        labels=CLUSTER_FUNC[self.para['matrix_Cluster_func']](self.collection_for_DR_df_na,  self.para['matrix_Cluster_components']   )
        
        #labels=labels+len(self.pinned_cluser)+1
        
        
        #把少量pix 类 当作噪点
        labels=labels+1
        raw_labels=labels.copy()
        all_hit_len=len(labels)
        for label in np.unique(raw_labels):
            np_hit=labels==label
            if np.sum(np_hit)<all_hit_len/(3*self.para['matrix_Cluster_components']):
                labels[np_hit]=0
        labels=labels+1#小于0.01的 统一归为other     
        
        
        
        labels=FunctionClass.resolve_label_conflicts(self.copy_all_info_df.loc[not_hit,:]["labels"].values,labels)
        
        
        
        self.copy_all_info_df.loc[hit,'labels']=labels
        
        #self._dr_plot.set_labels(self.copy_all_info_df['labels'])
        self._cluster_plot.myPlot(self.copy_all_info_df,self.para['matric_resolution_value'],self.para)
        self._cluster_plot.is_show_MS=True
        print(f"there are {len(self.copy_all_info_df['labels'].unique())} labels in all_info_df, and the unique labels is\n:",self.copy_all_info_df['labels'].unique() )
    
    
    def clear_cluster_collection(self):
        self.copy_all_info_df['collected']=0
        
        
        
        
        
    

    
    def pin_cluser_fun(self):
        pin_target=self._cluster_plot.get_cluster_pick_num()
        if pin_target==-1:
            print("have not pick a point")
             
        else:
            hit=self.copy_all_info_df['labels']==pin_target
            if hit.sum()==0:
                print("nope! the hit cluster points is O!")
            else:
                self.copy_all_info_df.loc[hit,'labels']=len(self.pinned_cluser)
                self.copy_all_info_df.loc[hit,'pinned']=1
                self.pinned_cluser.append(1)
                self.PB_E6.setEnabled(False)
                print(f"success pin the {pin_target} cluster points, the pinned number is {np.sum(self.copy_all_info_df['pinned']==1)}")
    
    def dpin_cluser_fun(self):
        pin_target=self._cluster_plot.get_cluster_pick_num()
        if pin_target==-1:
            print("have not pick a point")
             
        else:
            
            hit=self.copy_all_info_df['labels']==pin_target
            if hit.sum()==0:
                print("nope! the cluster have not been pinned yet")
            else:
                #self.copy_all_info_df.loc[hit,'labels']=len(self.pinned_cluser)
                self.copy_all_info_df.loc[hit,'pinned']=0
                self.pinned_cluser.pop()
                self.PB_E6.setEnabled(False)
                print(f"success dpin the {pin_target} cluster points, the pinned number is {np.sum(self.copy_all_info_df['pinned']==1)}")
    def clear_pin_cluster(self):
        self.copy_all_info_df['pinned']=0
    def display_mass_spectrum_func(self,x,y):
        print(f"in view, get ms location in ({x},{y})")
        
        sub_y=abs(self.copy_all_info_df['y']-y)
        hit_y=  sub_y==sub_y.min()
        
        sub_x=abs(self.copy_all_info_df.loc[hit_y,:]['x']-x)
        hit_x=  sub_x==sub_x.min()
        
        hit_y[hit_y]=hit_x
        
        
        ms_df=self.copy_all_inten_df.loc[hit_y,:]
        self.display_MS_df=ms_df
        print("the point MS is :")
        print(ms_df)
        mz_list=ms_df.columns.to_list()
        inten_list=ms_df.iloc[0,:].to_list()
        n2_peaks=list(zip(mz_list,inten_list))
        self._mass_spectrum_plot.myPlot(n2_peaks)
    
    def msi_plot_func(self):
        self.para['smoth_Gaussian_active']= self.checkBox_2.checkState()
        self.para['smoth_Gaussian_sigma'] = int( self.lineEdit_7.text() )
        self.para['smoth_Gaussian_W'] = int( self.lineEdit_6.text() )
        self.para['smoth_Gaussian_H'] = int( self.lineEdit_7.text() )
        
        self.para['smoth_interpolation_active'] = self.checkBox_2.checkState()
        self.para['smoth_interpolation_x'] = int( self.lineEdit_4.text() )
        self.para['smoth_interpolation_y'] = int( self.lineEdit_5.text() )
        
        
        
        self.para['MSI_target_mz']=float( self.lineEdit.text() )
        #self.para['MSI_step_mz']=float( self.lineEdit_2.text() )
        self.para['MSI_step_error']=float( self.lineEdit_3.text() )
        
        upper=self.para['MSI_target_mz']*( 1 + self.para['MSI_step_error'] )
        lower=self.para['MSI_target_mz']*( 1 - self.para['MSI_step_error'] )
        numeric_columns= self.copy_all_inten_df.columns.astype('float')
        chosen_columns=(numeric_columns < upper ) & (numeric_columns >= lower )
        
        if len(self.copy_all_info_df)<10:
            print("no Matrix data")
            return
        self.copy_all_info_df['target_inten']=self.copy_all_inten_df.loc[:,chosen_columns].mean(axis=1)    #np.mean( self.copy_all_inten_df.loc[:,chosen_columns], axis=1)
        self.copy_all_info_df.fillna(0,inplace=True);
        try:
            pix_df=self.copy_all_info_df.groupby(by=['y','x'])['target_inten'].mean().unstack()#.agg(lambda x: x.value_counts().index[0]).unstack()
            pix_df.fillna(method='ffill',axis=1,inplace=True)
            pix_df.fillna(method='bfill',axis=1,inplace=True)
            self._msi_plot.myPlot(pix_df,self.para,plot_title=round(self.para['MSI_target_mz'],4))
            self._msi_plot.is_show_MS=True
        except Exception as e:
            print(self.copy_all_info_df)
            print(e)
            
    def msi_func_show_mz_list(self):
        numeric_columns= self.copy_all_inten_df.columns.astype('float')
        print(580,"###############MZ_LIST###############\n\n",sorted(numeric_columns),f"\n### the length is {len(numeric_columns)}########END_LIST#########\n")
    
    
        
    
    def msi_func_up_mz_list(self):
        numeric_columns= self.copy_all_inten_df.columns.astype('float')
        sorted_numeric_columns=np.array( sorted(numeric_columns) )
        current_mz=float( self.lineEdit.text() )
        min_id= abs(sorted_numeric_columns-current_mz).argmin()
        try:
            up_id=min_id-1
            self.lineEdit.setText(str( round(sorted_numeric_columns[up_id],5) ))
            self.msi_plot_func()
        except Exception as e:
            print(592,e)
    def msi_func_down_mz_list(self):
        numeric_columns= self.copy_all_inten_df.columns.astype('float')
        sorted_numeric_columns=np.array( sorted(numeric_columns) )
        current_mz=float( self.lineEdit.text() )
        min_id=abs(sorted_numeric_columns-current_mz).argmin()
        try:
            up_id=min_id+1
            self.lineEdit.setText(str( round(sorted_numeric_columns[up_id],5) ))
            self.msi_plot_func()
            
        except Exception as e:
            print(592,e)
            
    def next_mz_but_f(self):
        self.para['MSI_target_mz']=float( self.lineEdit.text() )
        self.para['MSI_step_mz']=float( self.lineEdit_2.text() )
        #self.para['MSI_step_error']=float( self.lineEdit_3.text() )
        
        self.para['MSI_target_mz']=round( self.para['MSI_target_mz'] + self.para['MSI_step_mz'],5)
        self.lineEdit.setText(str(self.para['MSI_target_mz']))
        self.msi_plot_func()
    def prev_mz_but_f(self):
        self.para['MSI_target_mz']=float( self.lineEdit.text() )
        self.para['MSI_step_mz']=float( self.lineEdit_2.text() )
        #self.para['MSI_step_error']=float( self.lineEdit_3.text() )
        
        self.para['MSI_target_mz']=round( self.para['MSI_target_mz'] - self.para['MSI_step_mz'],5)
        self.lineEdit.setText(str(self.para['MSI_target_mz']))
        self.msi_plot_func()
    def _gen_message(self,m_type,m_data):
        data={'type':m_type}
        data.update(m_data)
        return data
    
    def read_inten_info(self):
        try:
            temp_all_info_df=pd.read_csv(r'E:\document\Experimental_data\通用机械模型\Dual_PESI\MSI_CIESI_1106\python\info_df.csv',index_col=0)
            temp_all_inten_df=pd.read_csv(r'E:\document\Experimental_data\通用机械模型\Dual_PESI\MSI_CIESI_1106\python\inten_df.csv',index_col=0)
            #self.inten_obj.input_all_info_inten_df(temp_all_info_df,temp_all_inten_df)
        except Exception as e:
            print('view_demo,436,\n',e)
            
        
        
        self.msi_scan_para={"x_s":1,"x_e":10,"y_s":2,"y_e":8,'y_interval':1,
            }
        
        
        
        #self.process_plot_fun(self._gen_message('start',self.msi_scan_para)          )
        
        #print(self.inten_obj.output())
    
    
    
    
    
    
    
    def aquiring_row(self,fid):
        pass
        
    def process_plot_fun(self,data):##{'type':'start','data':data}
        if data['type']=='start':
            #data.update(self.msi_scan_para)
            
            self._process_plot.process(data)
        
        else:
            self._process_plot.process(data)
    
    
    
    def finish_process_one_mzml(self):
        pass
        

    
    def recive_data_func(self):
        while 1 and self.threadpool_is_alive:
            if self.recive_data_queue.empty():
                time.sleep(0.2)
                continue
            else:
                recive_data=self.recive_data_queue.get()
                print("have recive\t",recive_data)
                
                if recive_data['type']=='start':
                    self.process_mzml_info_files=[]
                    self.msi_scan_para=recive_data['data']
                    #temp_data=self._gen_message('start',self.msi_scan_para)
                    self.re_init_processing();
                    #self._process_plot.process(temp_data)
                
                if recive_data['type']=="massconvert":
                    self.threadpool.start( WorkerGUI(self.file_convert_function,recive_data['file']+'.raw',recive_data['fid']) )
                    self.process_plot_fun(self._gen_message('massconvert',recive_data)          )
                if recive_data['type']=="pro_processing":
                    mzml_file_dict={"file":recive_data['file'],'fid':recive_data['fid']} 
                    if mzml_file_dict in self.process_mzml_info_files: 
                        print("this mzml file have been pro_processing")
                        return
                    self.process_mzml_info_files.append(mzml_file_dict)
                    self.threadpool.start( WorkerGUI(self.trigger_processing_one_mzml,recive_data['file'],recive_data['fid']  ) )
                    self.process_plot_fun(self._gen_message('pro_processing',recive_data) )
                
                if recive_data['type']=="msi_stop":
                    self.process_mzml_info_files=[]
                
                if recive_data['type']=="finish_process_one_mzml":
                    
                    self.process_plot_fun(self._gen_message('finish_process_one_mzml',recive_data)          )
                
                if recive_data['type']=="scan_para":
                    self.msi_scan_para=recive_data['data']
                    print(self.msi_scan_para)
    
    
    def send_sample_points(self):
        samples=self._cluster_plot._output_sample_points()
        print("send to msi_main",samples)
        self.send_data_queue.put({"type":"LESA_sample_point","data":samples})
        
    
    def get_pcal_compounds(self):
        self.para['matric_pcal_s_n']= self.spinBox_3.value()
        self.pcal_compounds_list,top_p_id=self._pcaL_plot.output_compound_list(self.para['matric_pcal_s_n'])
        #self._cluster_plot.draw_pcal_s_points(top_p_id)
    
    
        
    def enable_pick(self,enable):
        self.para['matric_pick_alpha']=self.doubleSpinBox_2.value()
        self.para['matric_pick_size']= self.doubleSpinBox_3.value()
        if enable:self.actionactive.setChecked(0)
        self._cluster_plot.enable_pick(enable,self.para['matric_pick_alpha'],self.para['matric_pick_size'])
    
    def enable_cluster_active(self):
        self.para['matric_pick_alpha']=self.doubleSpinBox_2.value()
        self.para['matric_pick_size']= self.doubleSpinBox_3.value()
        if self.actionactive.isChecked():
            self._cluster_plot.enable_cluster_pick(True,self.para['matric_pick_alpha'],self.para['matric_pick_size'])
            self.checkBox_7.setCheckState(0)
        else:
            self._cluster_plot.enable_cluster_pick(False,self.para['matric_pick_alpha'],self.para['matric_pick_size'])
    
    def _preprocessing_one_f(self):
        pass
        def temp_preprocessing():
            pass
        
        self.threadpool.start( WorkerGUI(temp_preprocessing) )
    def _mass_convert_one_f(self):
        pass
        def temp_convert():
            convert_f(r"E:\document\Experimental_data\通用机械模型\Dual_PESI\test_data\20220408_desi_ghous_02\data02.raw")
            print("v_demo,end_convert")
        
        self.threadpool.start( WorkerGUI(temp_convert) )
    
    def msi_func_output_mz_list(self):
        numeric_columns= self.copy_all_inten_df.columns.astype('float')
        sorted_mz=sorted(numeric_columns)
        file_name=QFileDialog.getSaveFileName(self,"output_mz_list","./mz_list/","excel(*.xlsx)") 
        if file_name[0]:
            pd.DataFrame(sorted_mz,columns=["mz"]).to_excel(file_name[0])
    def save_all_df_para(self,*args):
        file_name=QFileDialog.getSaveFileName(self,"save file dialog","./output/","output_file(*.all_output)") 
        if file_name[0]:
            self.read_para()
            all_output={}
            all_output["msi_scan_para"]=self.msi_scan_para
            all_output["view_para"]=self.para
            # 缺了seq df。 导致载入后不能再processing
            self.inten_queue.get()
            all_output["all_info_df"]=self.raw_all_info_df
            all_output["all_inten_df"]=self.raw_all_inten_df
            self.inten_queue.put(1)
            
            
            all_output["copy_all_info_df"]=self.raw_all_info_df
            all_output["copy_all_inten_df"]=self.raw_all_inten_df
            all_output["_cluster_extent"]=self._cluster_extent
            #self.msi_scan_para={"x_s":1,"x_e":10,"y_s":2,"y_e":8,'y_interval':1}
            with open(file_name[0],'wb') as f:
                pickle.dump(all_output,f)
    
    
    
    def load_all_df_para(self,*args):
        file_name=QFileDialog.getOpenFileName(self,"open file dialog","./output/","output_file(*.all_output)") 
        if file_name[0]:
            with open(file_name[0],'rb') as f:
                all_input=pickle.load(f)
            self.msi_scan_para.update(all_input['msi_scan_para'])
            
            
            self._cluster_extent=all_input.get("_cluster_extent",None)
            if self._cluster_extent!=None:
                pass
            else:
                self._cluster_extent=(self.msi_scan_para['x_e'],self.msi_scan_para['x_s'],self.msi_scan_para['y_e'],self.msi_scan_para['y_s'])
            self._msi_plot.set_extent(self._cluster_extent)
            self._cluster_plot.set_extent(self._cluster_extent)
            
            
            self.para.update(all_input['view_para'])
            '''
            with open("temp_info.pkl","wb") as f:
                temp_info_file_abpath=os.path.abspath(f.name)
                all_input["all_info_df"].to_pickle(f)
            with open("temp_inten.pkl","wb") as f:
                temp_inten_file_abpath=os.path.abspath(f.name)
                all_input["all_inten_df"].to_pickle(f)  
            
            self.inten_obj.input_all_info_inten_df(temp_info_file_abpath,temp_inten_file_abpath)
            '''
            
            self.raw_all_info_df=all_input["all_info_df"]
            self.raw_all_inten_df=all_input["all_inten_df"]
            
            self.copy_all_info_df=all_input["copy_all_info_df"]
            self.copy_all_inten_df=all_input["copy_all_inten_df"]
            
            self.write_para()
            
        if len( self.copy_all_info_df)>0:
            self.PB_DR.setEnabled(True)
            self.PB_E6.setEnabled(True)
    
    @exception_catching("load_mzml_info")  
    def load_mzml_info(self,*args):
        QMessageBox.about(self,"file type","please input a mzml info excel, witch contain the 'fid', 'path', and'file' columns\n(like seq uesed for NanoDESI,no .raw in file)")
        file_name=QFileDialog.getOpenFileName(self,"open file dialog","./mzml_info/","excel file(*.xlsx)") 
        
        if file_name[0]:
            mz_info_df=pd.read_excel(file_name[0])
            def path_join(ser):
                return os.path.join(ser['path'],ser['file']+".mzml")
            mz_info_df['file']=mz_info_df.apply(path_join,axis=1)
            print(mz_info_df)
            self.process_mzml_info_files=mz_info_df[['fid','file']].to_dict('record')
            print(self.process_mzml_info_files)
    
    @exception_catching("load_msi_scan_para")            
    def load_msi_scan_para(self,*args):
        QMessageBox.about(self,"file type","please load a msi scan para json file, witch contain the 'x_s','x_e','y_s','y_e','y_interval'")
        file_name=QFileDialog.getOpenFileName(self,"open file dialog","./msi_para/","para file(*.json)") 
        if file_name[0]:
            with open(file_name[0],"r") as f:
                self.msi_scan_para.update( json.load(f) )
            print(self.msi_scan_para)
    
    
    
    
    
    def save_para(self,*args):
        file_name=QFileDialog.getSaveFileName(self,"save file dialog","./view_meth/","view_file(*.viewPara)") 
        if file_name[0]:
            self.read_para()
            with open(file_name[0],'wb') as f:
                pickle.dump(self.para,f)
                
   
    def load_para(self,*args):
        file_name=QFileDialog.getOpenFileName(self,"open file dialog","./view_meth/","view_file(*.viewPara)") 
        if file_name[0]:
            with open(file_name[0],'rb') as f:
                self.para.update( pickle.load(f) )
                print(self.para)
                
                self.write_para()
    
    
    def output_calculated_info(self):
        pass
        file_name=QFileDialog.getSaveFileName(self,"save file dialog","./df_data/","csv(*.csv)") 
        if file_name[0]:
            self.copy_all_info_df[["x",'y','cut','R_time','fid']].to_csv(file_name[0],index=False)
        
    def output_calculated_inten(self):
        pass
        file_name=QFileDialog.getSaveFileName(self,"save file dialog","./df_data/","csv(*.csv)") 
        if file_name[0]:
            self.copy_all_inten_df.to_csv(file_name[0],index=False)
        
    
    def output_msi_pixmap(self):
        file_name=QFileDialog.getSaveFileName(self,"save file dialog","./out_pixmap/","excel(*.xlsx)") 
        if file_name[0]:
            pixmap_2dlist=self._msi_plot.output_pixmap()
            pd.DataFrame(pixmap_2dlist).to_excel(file_name[0])
    
    def output_cluster_pixmap(self):
        file_name=QFileDialog.getSaveFileName(self,"save file dialog","./out_pixmap/","excel(*.xlsx)") 
        if file_name[0]:
            pixmap_2dlist=self._cluster_plot.output_pixmap()
            pd.DataFrame(pixmap_2dlist).to_excel(file_name[0])        
    
    def output_all_msi_in_mzList(self):
    

        
        output_path = QFileDialog.getExistingDirectory(self, "select a floder")
        if not output_path:return
        
        self.para['output_mz_msi_range']=self.para.get('output_mz_msi_range',(0,3000))
        
        
        def deal_func(mz_range:tuple,all_mz_list,the_dir='./'):
            
            for mz in all_mz_list:
                if mz<mz_range[0] or mz>mz_range[1]:continue
                self.lineEdit.setText(str( round(mz,5) ))
                self.msi_plot_func()
                self._msi_plot.output_msi_img(the_dir)
        
        
        is_ok,startmz,endmz=MyDialog2().get_input(self.para['output_mz_msi_range'])
        if is_ok:
            self.para['output_mz_msi_range']=(startmz,endmz)
            numeric_columns= self.copy_all_inten_df.columns.astype('float')
            sorted_mzlist=sorted(numeric_columns)
            result = QMessageBox.question(self, "Attention", f"there are {len(sorted_mzlist)} in list, do you still want exec it????'_'", QMessageBox.Yes | QMessageBox.No)
            if(result == QMessageBox.Yes):
                self.threadpool.start( WorkerGUI(deal_func,self.para['output_mz_msi_range'],sorted_mzlist,output_path  ) )
    
    def output_MS_data_df(self):
        
        file_name=QFileDialog.getSaveFileName(self,"save file dialog","./MS1_data/","excel(*.xlsx)") 
        if file_name[0]:
            self.display_MS_df.to_excel(file_name[0])
    
    def show_or_hide_sample_points_func(self,is_blink=False):
        if is_blink:
            self.para['matric_point_blink']= self.checkBox_10.checkState()
            
        if not is_blink:
            self.is_show_sample_point=~self.is_show_sample_point
        
        
        print(1608,self.is_show_sample_point,self.para['matric_point_blink'])
        
        def _inner_func(*args):
            self.para['matric_pick_alpha']=self.doubleSpinBox_2.value()
            self.para['matric_pick_size']= self.doubleSpinBox_3.value()
            self.para['matric_point_blink']= self.checkBox_10.checkState()
            temp_is_show=self.is_show_sample_point
            while self.para['matric_point_blink'] and self.is_show_sample_point>0 and self.threadpool_is_alive:
                self.para['matric_point_blink']= self.checkBox_10.checkState()
                self.is_blinking_sample_point=True
                time.sleep(0.5)
                self.show_sample_point_signal.emit({"size":self.para['matric_pick_size'],
                    "alpha":self.para['matric_pick_alpha'],
                    "is_show":temp_is_show
                    })
                temp_is_show=~temp_is_show
            self.is_blinking_sample_point=False
        
        if self.is_show_sample_point>0 and not self.is_blinking_sample_point and self.para['matric_point_blink']>0:
            self.threadpool.start( WorkerGUI(_inner_func,1) )
        if self.is_show_sample_point<0:
            self.show_sample_point_signal.emit({"size":self.para['matric_pick_size'],
                    "alpha":self.para['matric_pick_alpha'],
                    "is_show":False
                    })
        if self.is_show_sample_point>0 and self.para['matric_point_blink']<1:
            self.show_sample_point_signal.emit({"size":self.para['matric_pick_size'],
                    "alpha":self.para['matric_pick_alpha'],
                    "is_show":True
                    })
        
        
    
    def show_msi_para(self):
        print("msi para is -> \n",self.msi_scan_para)
    
    def show_view_para(self):
        print("View para is -> \n",self.para)
        
    def show_calculate_Matrix(self):
        print("the calculated Matrix -> \n",self.copy_all_info_df)
    
    def set_cluster_colormap(self):
        item,ok=QInputDialog.getItem(self,"select Input dialog","编程语言列表",self.colormap_list,0,False)
        if ok and item:
            self._cluster_plot.set_colormap(item)
        
    def set_msi_colormap(self):
        item,ok=QInputDialog.getItem(self,"select Input dialog","编程语言列表",self.colormap_list,0,False)
        if ok and item:
            self._msi_plot.set_colormap(item)
    def out_msi_img_to_clip(self):
        from io import BytesIO
        import win32clipboard
        image=self._msi_plot.out_img()
        
        output = BytesIO()
        image.save(output, 'BMP')
        data = output.getvalue()[14:]
        output.close()
        
        
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        win32clipboard.CloseClipboard()
    #def threadpool_end_solt(self):
    #    self.threadpool_is_alive=False
    def sign_and_slot(self):
        self.actionload_para.triggered.connect(self.load_para)
        self.actionsave_para.triggered.connect(self.save_para)
        
        self.actionauto_DR_Classify.triggered.connect(self.auto_DR_and_cluster)
        
        self.actionoutput_mz_list_excel.triggered.connect(self.msi_func_output_mz_list)
        self.actionoutput_all.triggered.connect(self.save_all_df_para)
        self.actionloading_all.triggered.connect(self.load_all_df_para)
        
        self.actionshow_msi_para.triggered.connect(self.show_msi_para)
        self.actionshow_view_para.triggered.connect(self.show_view_para)
        self.actionshow_calculated_Matrix.triggered.connect(self.show_calculate_Matrix)
        
        self.actionms_convert_one.triggered.connect(self._mass_convert_one_f);
        self.actionpreprocessing_one.triggered.connect(self._preprocessing_one_f);
        self.actionread_info_and_inten.triggered.connect(self.read_inten_info);
        self.actionshow_info_inten.triggered.connect(self.show_info_inten_df)
        
        self.actionload_mzml_info.triggered.connect(self.load_mzml_info)
        self.actionload_msi_para.triggered.connect(self.load_msi_scan_para)
        
        self.actionoutput_calculated_info_df.triggered.connect( self.output_calculated_info  )
        self.actionoutput_calculated_inten_df.triggered.connect( self.output_calculated_inten  )
        
        self.actionshow_all_rows.triggered.connect(lambda x:self.show_rows_func(0))
        self.actionshow_odd_rows.triggered.connect(lambda x:self.show_rows_func(1))
        self.actionshow_even_rows.triggered.connect(lambda x:self.show_rows_func(2))
        
        self.actionactive.triggered.connect(self.enable_cluster_active)
        self.actionpin_cluster.triggered.connect(self.pin_cluser_fun)
        self.actiondpin_cluster.triggered.connect(self.dpin_cluser_fun)
        self.actionclear.triggered.connect(self.clear_pin_cluster)
        
        
        self.actioncollect_for_DR.triggered.connect(self.include_cluster_in_collection_func)
        self.actionexclude_cluster_in_collection.triggered.connect(self.exclude_cluster_in_collection_func)
        self.actionDR_collection.triggered.connect(self.DR_collection_func)
        self.actionclassify_collection.triggered.connect(self.classify_collection_func)
        self.actionclear_collect.triggered.connect(self.clear_cluster_collection)
        
        self.actionoutput_MSI_pixMap.triggered.connect(self.output_msi_pixmap)
        self.actionoutput_cluster_pixMap.triggered.connect(self.output_cluster_pixmap)
                
        self.actionoutput_MS1_data_df.triggered.connect(self.output_MS_data_df)
        
        self.actioncrop_img.triggered.connect(self.set_msi_crop_para)
        self.actionoutput_all_msi_in_mzList.triggered.connect(self.output_all_msi_in_mzList)
        
        self.actionset_cluster_colormap.triggered.connect(self.set_cluster_colormap)
        self.actionset_msi_colormap.triggered.connect(self.set_msi_colormap)
        
        self.pushButton_16.clicked.connect(self.out_msi_img_to_clip)
        
        self.pushButton_7.clicked.connect(self.re_processing_inten_df)
        self.pushButton_8.clicked.connect(self.do_post_compensation)
        
        self.pushButton.clicked.connect(self.calculate_Matrix)
        self.pushButton_2.clicked.connect(self.msi_plot_func)
        self.pushButton_3.clicked.connect(self.prev_mz_but_f)
        self.pushButton_4.clicked.connect(self.next_mz_but_f)
        self.PB_E8.clicked.connect(self.pca_plot_func)
        
        self.pushButton_6.clicked.connect(self.send_sample_points)
        self.PB_E9.clicked.connect(self.get_pcal_compounds)
        self.PB_DR.clicked.connect(self.dimension_reduction_func)
        self.PB_E6.clicked.connect(self.cluster_func)
        
        self.checkBox_7.stateChanged.connect(self.enable_pick)
        self.checkBox_10.stateChanged.connect(lambda:self.show_or_hide_sample_points_func(True))
        
        self.pushButton_5.clicked.connect(self._cluster_plot.clear_sample_points)
        self.pushButton_9.clicked.connect(lambda:self.show_or_hide_sample_points_func(False))
        self.pushButton_10.clicked.connect(self._cluster_plot.clear_sample_points)#(self._cluster_plot.del_pcal_s_points)
        
        self.pushButton_11.clicked.connect(self.msi_func_show_mz_list)
        self.pushButton_12.clicked.connect(self.msi_func_up_mz_list)
        self.pushButton_13.clicked.connect(self.msi_func_down_mz_list)
        
        self.pushButton_15.clicked.connect(self.extract_mz_by_TH)
        self.pushButton_14.clicked.connect(self.set_dpi)
        
        self.pushButton_17.clicked.connect(self.auto_DR_and_cluster)
        self.pushButton_19.clicked.connect(self.auto_detect_cluster_func)
        
        self.pushButton_20.clicked.connect(self.get_Matrix_from_raw_data)
        
        self.show_sample_point_signal.connect(self._cluster_plot.show_or_hide_sample_points)
        #self.threadpool_end_signal.connect(self.threadpool_end_solt)
        self.show_ms_signl.connect(self.display_mass_spectrum_func)

    
    def closeEvent(self, event):
        """重写关闭事件，确保优雅退出"""
        super().closeEvent(event)  # 先添加父类的方法，以免导致覆盖父类方法（这是重点！！！）
        print("窗口关闭事件触发...")
        
        # 设置退出标志
        self.is_exiting = True
        
        # 执行清理
        self.cleanup_all_resources()
        
        
        #QApplication.quit()
        
        # 接受关闭事件
        event.accept()
        
        print("窗口关闭完成")
        
    def __del__(self):
        try:
            print("Window __del__ - 开始子类清理")
            
            # 1. 先执行子类的清理逻辑
            self.cleanup_all_resources()
            
            # 2. 然后调用父类的 __del__ 方法
            # 注意：这里使用 super() 而不是直接调用父类方法
            super().__del__()
            
        except Exception as e:
            print(f"析构过程中出错: {e}")
        finally:
            print("MyWindow __del__ - 清理完成")
    def cleanup_processes(self):
        """清理所有进程"""
        print("开始清理所有进程...")
        
        # 清理进程池
        if hasattr(self, 'process_manager'):
            self.process_manager.cleanup_all_processes()
            
        # 清理可能的僵尸进程
        self._cleanup_zombie_processes()
        
        print("进程清理完成")
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        import signal
        
        def signal_handler(sig, frame):
            print(f"接收到终止信号 {sig}, 执行清理并退出")
            self.cleanup_all_resources()
            # 优雅退出
            QApplication.quit()
        
        # 注册常见信号
        signals = [signal.SIGINT, signal.SIGTERM, signal.SIGABRT]
        for sig in signals:
            try:
                signal.signal(sig, signal_handler)
            except (ValueError, AttributeError):
                # 在某些平台上可能不支持某些信号
                pass
    
    
    
    def cleanup_all_resources(self):
        """清理所有资源"""
        print("开始清理所有资源...")
        
        
        # 2. 清理线程池
        self._cleanup_threadpool()
        print("in demo, 线程清理完成")
        
        # 1. 清理进程
        self.cleanup_processes()
        
        
        # 3. 清理管理器
        self._cleanup_manager()
        print("in demo, manager清理完成")
        # 4. 清理队列
        self._cleanup_queues()
        print("in demo, queues清理完成")
        print("所有资源清理完成")
        
        #raise ValueError("success to cleanup all")
    
    def _cleanup_threadpool(self):
        """清理线程池"""
        if hasattr(self, 'threadpool'):
            try:
                # 等待所有线程完成或超时
                #self.threadpool.waitForDone(5000)  # 5秒超时
                self.threadpool_is_alive=False
                self.threadpool.clear();
                self.threadpool=""
            except Exception as e:
                print(f"清理线程池时出错: {e}")
    def _cleanup_manager(self):
        """清理管理器"""
        if hasattr(self, 'manager'):
            try:
                self.manager.shutdown()
            except Exception as e:
                print(f"关闭管理器时出错: {e}")
    def _cleanup_queues(self):
        """清理队列"""
        # 清空队列以避免阻塞
        if hasattr(self, 'send_data_queue'):
            try:
                while not self.send_data_queue.empty():
                    try:
                        self.send_data_queue.get_nowait()
                    except:
                        break
            except:
                pass
        
        if hasattr(self, 'recive_data_queue'):
            try:
                while not self.recive_data_queue.empty():
                    try:
                        self.recive_data_queue.get_nowait()
                    except:
                        break
            except:
                pass
    def _cleanup_zombie_processes(self):
        """清理僵尸进程"""
        '''
        try:
            import psutil
            current_pid = os.getpid()
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid']):
                try:
                    # 检查是否是我们的子进程
                    if (proc.info['ppid'] == current_pid and 
                        proc.info['name'] and 
                        'python' in proc.info['name'].lower()):
                        
                        print(f"终止子进程: PID {proc.info['pid']}")
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except psutil.TimeoutExpired:
                            proc.kill()
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            print(f"清理僵尸进程时出错: {e}")
            
        '''
        try:
            current_pid = os.getpid()
            parent = psutil.Process(current_pid)
            
            # 获取所有子进程
            children = parent.children(recursive=True)
            
            for child in children:
                try:
                    if child.is_running():
                        print(f"终止僵尸进程: PID {child.pid}, 名称: {child.name()}")
                        child.terminate()
                        child.wait(timeout=5)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    pass
                    
        except Exception as e:
            print(f"清理僵尸进程时出错: {e}")
    def main(self,send_data_queue,recive_data_queue): 
    
        # 2. 初始化进程管理器（如果尚未初始化）
        if not hasattr(self, 'process_manager'):
            self._init_processing_improved()
            # 3. 设置信号处理
        self._setup_signal_handlers()
        
        # 4. 注册退出清理函数
        #atexit.register(self.cleanup_processes)
        
        self.show()
        self.send_data_queue=send_data_queue;
        self.recive_data_queue=recive_data_queue;
        self.threadpool.start( WorkerGUI(self.recive_data_func) )
        sys.exit(self.app.exec_())



      
if __name__=='__main__':
    pipe1=Queue(1)
    #pipe2=Queue(1)
    # 设置信号处理
    #setup_signal_handlers()
    
    #app = QApplication(sys.argv)
    
    window=Window()
    window.main(pipe1,pipe1)
    print("hellow view.py")