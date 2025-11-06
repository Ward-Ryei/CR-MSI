from PyQt5.QtCore import pyqtSignal, QTimer
from PyQt5.QtWidgets import QWidget,QApplication,QMainWindow,QSizePolicy,QGridLayout
import PIL.Image as Image


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas,  NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import matplotlib.pyplot as plt
import copy
import os
from smoth import smoth_image,gaussBlur


from ClusterAnalyzer import ClusterAnalyzer
from collections import defaultdict


class CustomNavigationToolbar(NavigationToolbar):
    # 信号定义

    
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self._my_is_home=True
        
    def home(self, *args):
        """重写home方法 - 捕获home按钮点击"""
        print("Home button clicked!")
        self._my_is_home=True
        result = super().home(*args)
        return result
        
    def zoom(self, *args):
        """重写缩放方法"""
        print("Zoom button clicked!")
        result = super().zoom(*args)
        self._my_is_home=False
        return result

class MyCanvas(FigureCanvas):
    update_signal=pyqtSignal(int)
    def __init__(self,parent=None,width=6,height=3,dpi=100,*args,**kwargs):
        self.fig=Figure(figsize=(width,height),dpi=dpi)
        self.axes=self.fig.add_subplot(111)
        
        
        
        #self.axes.autoscale(False)
        
        #self.axes.hold(False)#每次调用plot时，原来的坐标轴被清除，已经被弃用
        #可以显示调用 axes.clear()进行清除数据
        
        #self.axes.grid('off')
        
        FigureCanvas.__init__(self,self.fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self,QSizePolicy.Expanding,QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        #self.fig.canvas.mpl_connect('button_press_event',self._zoom_f)
        #self.fig.canvas.mpl_connect('button_release_event',self._zoom_f)
        
        #self.fig.canvas.mpl_connect('button_press_event',self._draw_rect)
        #self.fig.canvas.mpl_connect('motion_notify_event',self._draw_rect)
       
    def get_toolbar(self,UI):
        self.figtoolbar=NavigationToolbar(self, UI)
        return self.figtoolbar
            
    def _plot_f(self):
        self.axes.clear() #清除坐标轴


        self.draw_img()
        #print(49,"canvas draw")
        self.fig.canvas.draw() #画布绘制
        self.fig.canvas.flush_events() # 画布刷新
        
    def draw_img(self):
        pass
    
    def __del__(self):
        pass
        #self.fig.close()
class MyMassSpectrumCanvas(MyCanvas):
    def __init__(self,UI=None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._cc_init()
        self.figtoolbar=CustomNavigationToolbar(self, UI)
    
    def _cc_init(self):
        self.mslines=self.axes.vlines([100,300,600], 0, [10,300,40],colors='black', linewidth=1, alpha=1)
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['bottom'].set_position('zero')
        self.axes.set_xlabel('m/z')
        self.axes.set_ylabel('Intensity') 
        
    def myPlot(self,n2_data=[[1,2],[3,4],[5,6]]):
        self.mz_values,self.intensity_values = zip(* n2_data)
        self._plot_f()
        
    def get_toolbar(self,UI):
        #self.figtoolbar=CustomNavigationToolbar(self, UI)
        return self.figtoolbar
        
    def draw_img2(self):
        
        self.axes.vlines(self.mz_values, 0, self.intensity_values, 
                            colors='black', linewidth=1, alpha=1)
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['bottom'].set_position('zero')
        self.axes.set_xlabel('m/z')
        self.axes.set_ylabel('Intensity')
    def draw_img(self):
    
        self.mslines=self.axes.vlines(self.mz_values, 0, self.intensity_values, 
                            colors='black', linewidth=1, alpha=1)

    def _plot_f(self):
        if self.figtoolbar._my_is_home:
            self.axes.clear() #清除坐标轴
            self.draw_img2()
        else:
        
            self.mslines.remove()
            self.draw_img()
        #print(49,"canvas draw")
        self.fig.canvas.draw() #画布绘制
        self.fig.canvas.flush_events() # 画布刷新
    

class MyCurveCanvas(MyCanvas):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.data=data
        self._width=width;
        if self._width:
            if len(self.data)>self._width:
                self.data=self.data.iloc[-self._width:,:]
                
        self.X=np.array(self.data['T']).flatten()
        self.Y=np.array(self.data['V']).flatten()
        self.axes.set_xlim(np.min(self.X),np.max(self.X))
        self.axes.set_ylim(np.min(self.Y)-10,np.max(self.Y)*1.1)
        
        self._plot_f()
        
    def draw_img(self):
        
        self.axes.plot(self.data['T'],self.data['V'],linewidth=0.8,label="V")   #########

class MyProcessCanvas(MyCanvas):
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.schedule=pd.DataFrame(columns=['fid','acquire','convert','process'])
    
    def process(self,data):
        try:
            if data['type']=='start':
                y_n=int( (data['y_e']-data['y_s'])/data['y_interval'] ) +1
                temp_d=np.zeros((y_n,4))
                temp_d[:,0]=np.arange(y_n)
                self.schedule=pd.DataFrame(temp_d,columns=['fid','acquire','convert','process'])
            if data['type']=='massconvert':
                self.schedule.loc[ self.schedule['fid']==data['fid'], 'acquire']=1
            
            if data['type']=='pro_processing':
                self.schedule.loc[ self.schedule['fid']==data['fid'], 'convert']=1
            if data['type']=='finish_process_one_mzml':
                self.schedule.loc[ self.schedule['fid']==data['fid'], 'process']=1
        except Exception as e:
            print('mc,94,',e)
        self._plot_f()
    
    def myPlot(self): # x=[0,10], y=[0,10]
        pass
        
        
            
        
        
    def draw_img(self):
        for idx,content in self.schedule.iterrows():
            y=content['fid']
            for item,key in zip(content[1:],self.schedule.columns[1:]):
                c='yellow'
                if item==0:
                    c='yellow'
                if item==1:
                    c='green'
                self.axes.scatter(key,idx,c=c)
        
        
from sklearn.cluster import KMeans     
class MyClusterCanvas(MyCanvas):
    def __init__(self,width=4.5,height=6,show_ms_signl=None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        self.data=np.array([[0,0],[0,0]])
        self.cmap=plt.get_cmap("tab20b")
        
        self.pick_active=False
        self.cluster_pick_active=False
        self.cluster_pick_num=-1;
        self.sample_points=[]
        self.pcal_sample_points=[]
        self.fig.canvas.mpl_connect('button_press_event',self.pick_sample_points)
        self.alpha=0.5
        self.size=2
        self.all_info_df=pd.DataFrame()
        self.pix_df=pd.DataFrame()
        self.is_show_smaple_points=True;
        self.extent=(10,0,10,0)
        
        self.cluster_analyzer = ClusterAnalyzer()
        
        self.is_show_MS=False
        self.show_ms_signl=show_ms_signl
        
        self.plot_para={'smoth_Gaussian_active':False,
            "smoth_Gaussian_sigma":5,
            "smoth_Gaussian_W":3,
            "smoth_Gaussian_H":3,
            "smoth_interpolation_active":False,
            "smoth_interpolation_x":0,
            "smoth_interpolation_y":0,
            }
            
            
        self.rank_colors = {
            1: 'red',      # 最大 - 金色
            2: 'gold',       # 第二大 - 红色
            3: 'blue',      # 第三大 - 蓝色
            4: 'green',     # 第四大 - 绿色
            5: 'purple'     # 第五大 - 紫色
            }
        
        self.rank_markers = {
            1: '*',         # 最大 - 星号
            2: 'o',         # 第二大 - 圆圈
            3: 's',         # 第三大 - 正方形
            4: '^',         # 第四大 - 三角形
            5: 'D'          # 第五大 - 菱形
            }
        
        self.cluster_points_ddict=defaultdict(list)
            
    def set_colormap(self,colormap:str="tab20b"):
        try:
            self.cmap=plt.get_cmap(colormap)
            self._plot_f()
        except Exception as e:
            print(f"162, set colormap have meet error: ",e)
    def enable_cluster_pick(self,enable,alpha,size):
        self.cluster_pick_active=enable
        if enable:self.pick_active=False
        self.alpha=alpha
        self.size=size
    
    def enable_pick(self, enable,alpha,size):
        self.pick_active=enable
        if enable:self.cluster_pick_active=False
        self.alpha=alpha
        self.size=size
    def pick_sample_points(self,event):
        
        
        if self.is_show_MS:
            if (event.name=='button_press_event') and (event.button==1):
                x,y=event.xdata,event.ydata
                #print(f" xy  ({x,y})-> show_MS_location in mycanvas")
                self.show_ms_signl.emit(x,y)
        
        self.cluster_pick_num=-1    
        if self.cluster_pick_active:
            try:
                if (event.name=='button_press_event') and (event.button==1):
                    x,y=event.xdata,event.ydata
                    self.cluster_pick_num=self.get_value_by_xy(x,y)
                    
                    print(f" xy  ({x,y})-> {self.cluster_pick_num}")
                    
            except Exception as e:
                print("mycanvas,184,",e)
        
        if self.pick_active:
            try:
                if (event.name=='button_press_event') and (event.button==1):
                    print(f"mycluster plot, clicked xy is  {event.xdata,event.ydata}")
                    pass
                    #self.sample_points.append([round(event.xdata,2),round(event.ydata,2)])
                    #print(self.sample_points)
                    ##得到的是 x，y 坐标
                    point={}
                    point["center"]=(round(event.xdata,2),round(event.ydata,2))
                    point["area"]=10
                    point["method"]="manual_pick"
                    point["is_selected"]=True
                    
                    if point not in self.cluster_points_ddict[self.cluster_pick_num]:
                        self.cluster_points_ddict[self.cluster_pick_num].append(point)
                        self._plot_f()
                        
            except Exception as e:
                print("mycanvas,181,",e)
        
    def get_cluster_pick_num(self):
        return self.cluster_pick_num
        
    def clear_sample_points(self):
        self.cluster_points_ddict=defaultdict(list)
        #self.sample_points=[]
        self._plot_f()
    
    def myPlot2(self,all_info_df,all_inten_df,cluster_num=5,dpi=0.1):
        def resolution(value,res=0.2,decimal=1):
            return round(round( value/res,0)*res , decimal)
        self.all_info_df=all_info_df
        self.all_info_df[['x','y']]=self.all_info_df[['x','y']].apply( lambda x: resolution(x,dpi), axis=1,result_type="expand")#np.round(all_info_df['x'],1)
        kmeans=KMeans(init='k-means++',n_clusters=cluster_num,n_init=10,random_state=1)
        kmeans.fit(all_inten_df)
        self.all_info_df['labels']=kmeans.labels_
        self.pix_df=self.all_info_df.groupby(by=['y','x'])['labels'].agg(lambda x: x.value_counts().index[0]).unstack()
        #print(pix_df)
        self.pix_df.fillna(method='ffill',axis=1,inplace=True)
        self.pix_df.fillna(method='bfill',axis=1,inplace=True)
        self.data=self.pix_df
        self._plot_f()
    
    
    def get_value_by_xy(self,x,y):
        
        x_na=self.pix_df.columns.values
        y_na=self.pix_df.index.values
        def get_near_id(target,na):
            abs_sub=abs(target-na)
            min_v=abs_sub.min()
            hit= (min_v==abs_sub)
            return np.where(hit>0)[0][0]
        
        idx=get_near_id(x,x_na)
        idy=get_near_id(y,y_na)
        
        
        return self.pix_df.iloc[idy,idx]
        
    
    def myPlot(self,all_info_df,dpi=0.1,plot_para={}):
        self.plot_para.update(plot_para)
        #pass
        #def resolution(value,res=0.2,decimal=1):
        #    return round(round( value/res,0)*res , decimal)
        self.all_info_df=all_info_df
        #self.all_info_df[['x','y']]=self.all_info_df[['x','y']].apply( lambda x: resolution(x,dpi), axis=1,result_type="expand")#np.round(all_info_df['x'],1)
        self.pix_df=self.all_info_df.groupby(by=['y','x'])['labels'].agg(lambda x: x.value_counts().index[0]).unstack()
        self.pix_df.fillna(method='ffill',axis=1,inplace=True)
        self.pix_df.fillna(method='bfill',axis=1,inplace=True)
        self.data=self.pix_df.values
        
        if self.plot_para.get("smoth_interpolation_active",0)>0:
            self.data=smoth_image(self.data,self.plot_para['smoth_interpolation_x'],self.plot_para['smoth_interpolation_y'])
        print(f"mycanvas 198, cluster success")
        self._plot_f()
    
    def draw_pcal_s_points(self,top_p_id):
        index_ids=np.arange(self.pix_df.shape[0])
        columns_ids=np.arange(self.pix_df.shape[1])
        def position_to_pix_ip(ser):
            ip_x=( (self.pix_df.columns==ser['x'])*columns_ids ).max()
            ip_y=( (self.pix_df.index  ==ser['y'])*index_ids ).max()
            return [ip_x,ip_y]
        if len(top_p_id)==0:
            print("no pcal_sample_points input")
            return
        try:
            top_info=copy.deepcopy( self.all_info_df.iloc[top_p_id,:] )
            #top_info['pix_ip_x']=0;
            #top_info['pix_ip_y']=0;
            #top_info[['ip_x','ip_y']]=top_info[['x','y']].apply(position_to_pix_ip,axis=1,result_type="expand")
            self.pcal_sample_points=top_info[['x','y']].values.tolist()
            
            self._plot_f()
        except Exception as e:
            print(f"mycanvas,221,draw_pcal_s_points:",e)
    def del_pcal_s_points(self):
        self.pcal_sample_points=[]
        self._plot_f()
    
    def _pix_iloc_to_position(self,pix_iloc):#pix_iloc_list=(row_id:y,col_id:x)
        #self.extent
        #self.data.shape
        col_k=(self.extent[1]-self.extent[0])/self.data.shape[1]  # X
        row_k=(self.extent[2]-self.extent[3])/self.data.shape[0]   #Y
        
        return round(pix_iloc[1]*col_k+self.extent[0],2),round(pix_iloc[0]*row_k+self.extent[3],2)
        
    def auto_detect_cluster(self,rank_num=1,area_threshold=3):
        if self.data.shape[1]<5:
            print("no pix data supply")
            return
        
        
        rank_centers = self.cluster_analyzer.find_centers_by_area_rank(
            np.flip(self.data,axis=1), min_area=area_threshold, rank_threshold=rank_num, include_all_ranks=True
            )
        
        valid_rank_centers = self.cluster_analyzer.verify_centers_on_blocks(
            np.flip(self.data,axis=1), rank_centers
            )
        self.cluster_analyzer.print_centers_info(rank_centers,"top-ranked")
        
        print(f"/n/nThe valid_rank_centers is: \n",valid_rank_centers)
        for class_key in valid_rank_centers:
            for block in  valid_rank_centers[class_key]:
                block["center"]=self._pix_iloc_to_position(  block["center"] ) # iloc to position
                if block not in self.cluster_points_ddict[class_key]:
                    
                    self.cluster_points_ddict[class_key].append(block)
        
        
        
        print(f"/n/nThe points is: \n",self.cluster_points_ddict)
        
        
        self._plot_f()



    
    def _pix_ip_to_position(self,sample_points):
        p_y=self.pix_df.index
        p_x=self.pix_df.columns
        
        return [[p_x[x],p_y[y]]  for x,y in sample_points ]
    
    def _output_sample_points(self):
        k_p_list=[]
        p_p_list=[]
        
        if len(self.cluster_points_ddict)>0:
            for cluster_point_key in self.cluster_points_ddict:
                for point in self.cluster_points_ddict[cluster_point_key]:
                    k_p_list.append(point["center"])
            
            return k_p_list
            
            
        else:
            print("mycanvas 364, no points")
            return [[]]
        
        '''
        if len( self.sample_points )>0:
            #k_p_list=self._pix_ip_to_position(self.sample_points)
            k_p_list=copy.deepcopy(self.sample_points)
        if len(self.pcal_sample_points) >0:
            #p_p_list=self._pix_ip_to_position(self.pcal_sample_points)
            p_p_list=self.pcal_sample_points
        k_p_list.extend(p_p_list)
        return k_p_list
        '''
        
    def output_pixmap(self):
        return np.flip(self.data,axis=1)
        
    def set_extent(self,extent):
        self.extent=extent
    
    def set_smoth(self,is_smoth=False,smoth_para=[0,0],is_guass=False,guass_para=[5,3,3]):
        self.is_smoth=is_smoth
        self.smoth_para=smoth_para
        self.is_guass=is_guass
        self.guass_para=guass_para
    
    def draw_img(self):
        #self.axes.axis('off')
        #rows,cols=self.data.shape
        #print("myclusterplot 222 \n",self.data)
        if self.extent:
            self.axes.imshow(np.flip(self.data,axis=1),cmap=self.cmap,extent=self.extent)
        else:
            self.axes.imshow(np.flip(self.data,axis=1),cmap=self.cmap)
        #
        if self.is_show_smaple_points==1:
            if len(self.cluster_points_ddict)>0:
                for class_id, blocks in self.cluster_points_ddict.items():
                    for block in blocks:
                        x, y = block['center']
                        rank = block.get('rank', 1)
                        color = self.rank_colors.get(rank, 'gray')
                        marker = self.rank_markers.get(rank, 'x')
                        if block['method']=="manual_pick":
                            color = 'red'
                            marker = '1'
                        self.axes.scatter(x, y, 
                           color=color, alpha=self.alpha,s=self.size,
                           marker=marker, 
                           linewidth=2
                           )
            
            
            
            #if len(self.sample_points)>0:            
            #    self.axes.scatter( * zip(*self.sample_points) ,c='red',alpha=self.alpha,s=self.size)
            
            #if len(self.pcal_sample_points)>0:            
            #    self.axes.scatter( * zip(*self.pcal_sample_points) ,c='red',alpha=self.alpha,s=self.size)
    def show_or_hide_sample_points(self,input_para={}):#is_show=True,size=None,alpha=None):
        
        if input_para.get('size',None) ==None:input_para['size']=self.size
        if input_para.get('alpha',None) ==None:input_para['alpha']=self.alpha
        #if input_para.get('is_show',None) ==None:input_para['is_show']=self.alpha
        self.size=input_para.get('size',None)
        self.alpha=input_para.get('alpha',None)
        self.is_show_smaple_points=input_para.get('is_show',False)  
        #print(f"is_show_smaple_points is : {self.is_show_smaple_points}")
        self._plot_f()
    
    


class MyMsiCanvas(MyCanvas):
    def __init__(self,width=4.5,height=6,show_ms_signl=None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        #self.axes.axis('off')
        self.cmap=plt.get_cmap("gnuplot2")
        self.plot_title='msi'
        self.data=np.array([[0,0],[0,0]])
        self.extent=(10,0,10,0)
        self.is_show_MS=False
        self.show_ms_signl=show_ms_signl
        self.fig.canvas.mpl_connect('button_press_event',self.show_MS_location)
        self.plot_para={'smoth_Gaussian_active':False,
            "smoth_Gaussian_sigma":5,
            "smoth_Gaussian_W":3,
            "smoth_Gaussian_H":3,
            "smoth_interpolation_active":False,
            "smoth_interpolation_x":0,
            "smoth_interpolation_y":0,
            }
        self.crop_para=(0,0,*(self.get_canvas_size()))
    
    
    def show_MS_location(self,event):
        if self.is_show_MS:
            if (event.name=='button_press_event') and (event.button==1):
                x,y=event.xdata,event.ydata
                #print(f" xy  ({x,y})-> show_MS_location in mycanvas")
                self.show_ms_signl.emit(x,y)
            
    def get_canvas_size(self):
        return self.fig.canvas.get_width_height()
    def set_extent(self,extent):
        self.extent=extent
    def set_crop_para(self,corp_para:tuple=(0,0,400,300)):
        self.crop_para=corp_para
    def myPlot(self,data_pd,plot_para={},plot_title="ms"):
        print(f"mycanvas 285, msi success,the pix shape is {data_pd.shape}")
        self.plot_para.update(plot_para)
        self.data=data_pd.values
        
        self.plot_title=plot_title
        if self.plot_para.get("smoth_interpolation_active",0)>0:
            self.data=smoth_image(self.data,self.plot_para['smoth_interpolation_x'],self.plot_para['smoth_interpolation_y'])
        if self.plot_para.get("smoth_Gaussian_active",0)>0:
            self.data=gaussBlur(self.data,self.plot_para['smoth_Gaussian_sigma'],self.plot_para['smoth_Gaussian_W'],self.plot_para['smoth_Gaussian_H'],)
        
        self._plot_f()
        
        pass
    def output_pixmap(self):
        return np.flip(self.data,axis=1)
    def set_colormap(self,colormap:str="gnuplot2"):
        try:
            self.cmap=plt.get_cmap(colormap)
            self._plot_f()
        except Exception as e:
            print(f"162, set colormap have meet error: ",e)    

    def out_img(self)->Image.Image:
        w, h = self.fig.canvas.get_width_height()
        
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        
        buf.shape = (w, h, 3)
        
        buf = np.roll(buf, 3, axis=2)
        
        image = Image.frombytes("RGB", (w, h), buf.tobytes())#"RGBA"
        print(369,f"the crop para is{self.crop_para} in {(0,0,w,h)} ")
        return image.crop(self.crop_para)
    
    def draw_img(self):
        print(f"msi 401,the pix shape is {self.data.shape}")
        if self.extent:
            self.axes.imshow(np.flip(self.data,axis=1),cmap=self.cmap,extent=self.extent)
        else:
            self.axes.imshow(np.flip(self.data,axis=1),cmap=self.cmap)
        self.axes.set_title(self.plot_title,fontsize=12,color='r')
    def output_msi_img(self,file_path):
        output_file=os.path.join(file_path,str(self.plot_title)+".png")  
        self.fig.savefig(output_file)
        
from sklearn.decomposition import PCA 
#from adjustText import adjust_text
import copy
class MyPcaloadingCanvas(MyCanvas):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        #self.axes.axis('off')
        self.data=pd.DataFrame([[0,0],[0,0]])
        
        self.loading=pd.DataFrame()
        self.threhold_distance=0;
        self.distance=pd.DataFrame()
        self.hit_mask=pd.DataFrame()
        self.compound_list=[]
        self.all_inten_df=pd.DataFrame()
        
    def myPlot(self,all_inten_df,n_components=10,metablism_n=50):
        
        pca=PCA(n_components=10,whiten=True)
        self.all_inten_df=all_inten_df
        pca.fit(self.all_inten_df.values )
        out=pca.transform( self.all_inten_df.values )
        self.loading = pd.DataFrame(pca.components_.T,index=self.all_inten_df.columns)
        self.distance= self.loading.apply(lambda x: np.power(x,2)).iloc[:,0:2].sum(axis=1)
        try:
            self.threhold_distance=self.distance.sort_values(ascending = False).iloc[metablism_n]
        except Exception as e:
            print('mycanvas 254',e)
            print(self.distance,type(self.distance))
            self.threhold_distance=self.distance.sort_values(ascending = False).iloc[-1]
        
        self.hit_mask=self.distance>=self.threhold_distance
        #print(loadings,distance)
        #loadings_m = pca.components_.T * np.sqrt(pca.explained_variance_)
        #loading_matrix = pd.DataFrame(loadings_m)
        
        
        
        
        
        
        self._plot_f()
        pass
    
    
    
    def draw_img(self):
        if self.loading.shape[0]==0:return
        
        text=[]
        self.compound_list=[]
        ##for item in self.hit_mask.unique():
        hit_data=self.loading.loc[self.hit_mask,:]
        self.compound_list=hit_data.index.to_list()
        self.axes.scatter(hit_data.loc[:,0],hit_data.loc[:,1],c="red",s=4)
        hit_data=self.loading.loc[~self.hit_mask,:]
        self.axes.scatter(hit_data.loc[:,0],hit_data.loc[:,1],c="black",s=4)
        
        
        self.axes.set_xlabel(f'p1',fontsize=6)
        self.axes.set_ylabel(f'p2',fontsize=6)
    
    def output_compound_list(self,top_sample_point=10):
        print("the all_inten_df is\n",self.all_inten_df)
        all_compounds_df=copy.deepcopy( self.all_inten_df.loc[:,self.compound_list] )
        all_compounds_df=all_compounds_df.apply(lambda x: x/x.max(),axis=0 ) # 纵向归一化
        print("to normalization, \n",all_compounds_df)
        all_compounds_df=all_compounds_df.apply(lambda x: np.power( x-x.mean(),2),axis=0) #纵向求距离
        dis_all_compounds_df=all_compounds_df.sum(axis=1) # 横向求每个数据帧的距离和
        print("myc,263,dis_all_compounds_df\n",dis_all_compounds_df)
        top=dis_all_compounds_df.nlargest(top_sample_point)
        top_index=top.index
        print("myc,263,top\n",top)
        print(self.compound_list)
        
        return self.compound_list,top_index

class MS_Spectrum_Plot(MyCanvas):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        self.data=np.array([[1,1],[2,2]])
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['bottom'].set_position(('data',0))
        #ax.spines['left'].set_position(('data',0))
        # 去掉网格
        self.axes.grid(False)
    def my_plot(self,np2d_peak):
        self.data = np2d_peak
        
        self._plot_f()
        
    
        
    def draw_img(self):
        if self.data.shape[0]==0:
            return
        
        self.axes.vlines(self.data[:,0], 0, self.data[:,1], colors="black", linewidth=0.75)
        
        

class DRCanvas(MyCanvas):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        self.data=np.array([[]])
        self.labels=np.array([])
        self.cmap=plt.get_cmap("tab20b")
    def my_plot(self,dr_data):
        self.data = dr_data
        
        self._plot_f()
    def set_labels(self,labels):
        self.labels=labels
        self._plot_f()
    def draw_img(self):
        if self.data.shape[0]==0:
            return
        
        if self.labels.shape[0]==self.data.shape[0]:
            self.axes.scatter(self.data[:,0],self.data[:,1],c=self.labels,cmap=self.cmap,s=2)
        else:
            self.axes.scatter(self.data[:,0],self.data[:,1],s=2)