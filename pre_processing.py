


from collections import defaultdict as ddict
from multiprocessing import Process, Queue,Pool, Manager
from multiprocessing.managers import BaseManager
import pymzml
import time
import sys
import os
import re
import fnmatch
import pickle
import pandas as pd
import numpy as np
import copy

def concat_columns(clumns,error=0.05):
    if not isinstance(clumns,np.ndarray):clumns=np.array(clumns)
    clumns=clumns.reshape(1,-1)
    #print("the clumns is :",clumns)
    m_ones=np.ones(clumns.shape).T
    #print("the m_ones is:", m_ones)
    #print("the clumns - m_ones:\n", clumns-clumns.T)
    id_na=np.arange(clumns.shape[1])
    #print("\n\nid_na:\n",id_na)
    #M1=m_ones@clumns
    #M2=M1.T
    #print("\n\M1:##########\n",M1)
    #print(M2)
    #M3=np.abs( M1-M2 )
    M3=np.abs(clumns-clumns.T)
    hit= M3 <= error
    #print("\n\M3:##########\n",M3*1,M3.shape) 
    #print("\n\hit:##########\n",hit*1,hit.shape)
    #np.savetxt('hit.txt',hit*1,fmt='%i', delimiter=",")
    #print(f"papare the hit cost {time.time()-begin_time}")
    
    
    id_hit=hit*id_na 
    #print(id_hit)
    
    match_1=np.max(id_hit,axis=1)
    #np.savetxt('nmatch_1.txt',match_1*1,fmt='%i', delimiter=",")
    #print("\n\nmatch_1:##########\n",match_1)
    match_2=(match_1==id_na.reshape(-1,1)) # 矩阵等式判断
    
    
    #print("\n\nmatch_2:##########\n",match_2*1,match_2.shape)
    #np.savetxt('nmatch_2.txt',match_2*1,fmt='%i', delimiter=",")
    
    
    return match_2[np.sum(match_2,axis=1)>0,:]
    

def concat_df_columns_with_in_error(input_df,error=5e-3): #error  5mDa
    #print(input_df)
    na_concated_columns=concat_columns(input_df.columns,error)
    #print("\n\n\nna_concated_columns########\n",na_concated_columns*1,na_concated_columns.shape)
    
    
    df_concated_columns=pd.DataFrame(na_concated_columns.T)
    #print("\n\n\df_concated_columns########\n",df_concated_columns*1,df_concated_columns.shape)
    #print(df_concated_columns)
    #print(f"papare the df_concated_columns cost {time.time()-begin_time}")
    df_concated_columns_mean=[]
    #last_ser=pd.core.series.Series()
    def deal_ser(ser):
        d_hit_df=input_df.loc[:,ser.values]
        output_ser=pd.core.series.Series(d_hit_df.max(axis=1))
        df_concated_columns_mean.append(np.mean(d_hit_df.columns))
        
        return output_ser
    output_df=df_concated_columns.apply(deal_ser,axis=0)
    #print(f"papare the df_concated_columns.apply cost {time.time()-begin_time}")
    output_df.columns=df_concated_columns_mean
    #output_df.drop_duplicates(inplace=True,axis=1)
    #output_df=output_df.T.drop_duplicates().T
    #print(f"papare the reset output_df.columns cost {time.time()-begin_time}")
    
    return output_df

class ManagerData():
    def __init__(self):
        self.df=pd.DataFrame(columns=[-1,-2])
        self.temp_df_list=[];
        self.measured_precision={"LOW":[5e-3,0.2],"HIGH":[5e-4,0.02]}
    
    def get_precision(self,precision="LOW",idx=0):
        return self.measured_precision[precision][idx]
    
    def add_row_df(self,pd_raw):
        #self.df=self.df.append(pd_raw)
        self.df=pd.concat([self.df,pd_raw],axis=0);
    def add_mzml_df(self,mzml_df):
        self.temp_df_list.append(mzml_df)
    
    def add_align_one_df_row(self,pd_row):
        self.df=pd.concat([self.df,pd_row],axis=0);
        self.df=concat_df_columns_with_in_error(self.df,0.005)
    
    def output_df(self):
        self.df.sort_values(by=[-1,-2],ascending=[True,True],inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        return self.df
    def output(self):
        self.df=pd.concat(self.temp_df_list)
        self.df.reset_index(drop=True, inplace=True)
        return self.df
        
    
def sub_process(queue,file_id,R_time,d2_list,obj): #一帧谱图一个进程
    temp=np.array(d2_list)
    
    if temp.shape[0]>0 and temp.shape[1]>1:
        try:
            temp[:,0]=np.around(temp[:,0],1)
            tdict=dict(temp)
        except Exception as e:
            print("in sub_process3 try,",e,flush=True)
            tdict={}
    else:
        print("in sub_process3 else, the mzinten shape is ",temp.shape,flush=True)
        tdict={}
    tdict[-2]=R_time
    tdict[-1]=file_id
    
    queue.get()
    obj.add_pd_raw(tdict)
    i=1
    queue.put(i)
    return 1;

def sub_process_mzml_by_dict(file_id,mzml_file,obj,precision="LOW",begin_time=0): #一个mzml 文件一个谱图
    TIC=pymzml.run.Reader(mzml_file)
    ddlist=[]
    for spectrum in TIC:
        spectrum.measured_precision=obj.get_precision(precision,0)
        spectrum.peaks('reprofiled')
        d2_list=spectrum.peaks('centroided')
        R_time=spectrum.scan_time_in_minutes()
        if not isinstance(d2_list,np.ndarray):d2_list=np.array(d2_list)
        if d2_list.shape[0]>0 and d2_list.shape[1]>1:
            try:            
                tdict=dict(d2_list)
            except Exception as e:
                print("in sub_process_mzml try,",e,flush=True)
                tdict={}
        else:
            print("in sub_process_mzml else, the mzinten shape is ",d2_list.shape,flush=True)
            tdict={}
        tdict[-2]=R_time
        tdict[-1]=file_id
        ddlist.append(tdict)
    print(f"Timestamp: {time.time()-begin_time},\t colect the sp_dicts from mzml.")
    temp_df=pd.DataFrame(ddlist)#temp_df 的列太多,先根据分辨率进行一次合并
    print(f"Timestamp: {time.time()-begin_time},\t generate the df from sp_dicts.")
    
    temp_df=concat_df_columns_with_in_error(temp_df,obj.get_precision(precision,1))
    print(f"Timestamp: {time.time()-begin_time},\t concat the nearly columns")
    
    
    
    
    
    obj.add_mzml_df(temp_df)
    return 1
def sub_process_mzml_by_pd(file_id,mzml_file,obj,precision="LOW",begin_time=0): #一个mzml 文件一个谱图
    TIC=pymzml.run.Reader(mzml_file)
    #ddlist=[]
    for spectrum in TIC:
        spectrum.measured_precision=obj.measured_precision[precision][0]
        spectrum.peaks('reprofiled')
        d2_list=spectrum.peaks('centroided')
        R_time=spectrum.scan_time_in_minutes()
        if not isinstance(d2_list,np.ndarray):d2_list=np.array(d2_list)
        if d2_list.shape[0]>0 and d2_list.shape[1]>1:
            try:            
                tdict=dict(d2_list)
            except Exception as e:
                print("in sub_process_mzml try,",e,flush=True)
                return 
                tdict={}
        else:
            print("in sub_process_mzml else, the mzinten shape is ",d2_list.shape,flush=True)
            return
        
        tdict[-2]=R_time
        tdict[-1]=file_id
        df_one_spect=pd.DataFrame([tdict]);
        obj.add_align_one_df_row(df_one_spect)
        
        #ddlist.append(tdict)
    #print(f"Timestamp: {time.time()-begin_time},\t colect the sp_dicts from mzml.")
    #temp_df=pd.DataFrame(ddlist)#temp_df 的列太多,先根据分辨率进行一次合并
    #print(f"Timestamp: {time.time()-begin_time},\t generate the df from sp_dicts.")
    
    #temp_df=concat_df_columns_with_in_error(temp_df,0.005)
    #print(f"Timestamp: {time.time()-begin_time},\t concat the nearly columns")
    
    
    
    
    
    #obj.add_align_one_df_row(temp_df)
    return 1   

class Test():
    def __init__(self):
        self.manager = BaseManager()
        self.manager.register('ManagerData', ManagerData)
        self.manager.start()
        self.obj=self.manager.ManagerData()
        self.pre_pro_pool=Pool(2)
        
        
    
    def run_one_mzml(self,fid,mzml_file,begin_time):
        #self.pre_pro_pool.apply_async(sub_process_mzml,args=(fid,mzml_file,self.obj))
        #self.pre_pro_pool.close()
        #self.pre_pro_pool.join()
        
        self.pre_pro_pool.apply(sub_process_mzml_by_dict,args=(fid,mzml_file,self.obj,"LOW",begin_time))
        result=self.obj.output()
        print(result)





class TargetArray():
    def __init__(self,size_unit=10000,precision=5e-6):
        self.size_unit=size_unit
        self._d2_array=np.ones([2,self.size_unit])
        self.size=size_unit
        self.aid=0
        self.precision=precision;
        
        
        self.temp_list=[];
        self.temp_num=0;
    def re_init(self):
        self._d2_array=np.ones([2,self.size_unit])
        self.size=size_unit
        self.aid=0
    def _expand_array_size(self):
        self._d2_array=np.hstack([self._d2_array,np.ones([2,self.size_unit])  ]  )
        self.size+=self.size_unit
    
    def get_mz(self):
        return self._d2_array[1,(self._d2_array[0,:]==0)]
    
    def get_temp_mz(self):
        return self.temp_list
    
    def reset_temp(self):
        self.temp_list=[];
        self.temp_num=0;
        
    def append(self,mz):
        if self.aid>=self.size:self._expand_array_size();
        mzs=self.get_mz()
        #print(mzs)
        #print(f"\r deal the {self.temp_num}",end="",flush=True)
        self.temp_num+=1
        if   (np.abs(mzs-mz)< mz*self.precision).any()  :
            return
        else:
            self._d2_array[:,self.aid]=[0,mz]
            self.temp_list.append(mz)
            self.aid+=1

class TT():
    
        pass

class ManagerMatrix():
    def __init__(self,precision=5e-3):
        
        

        self.all_info_df=pd.DataFrame();
        self.all_inten_df=pd.DataFrame()
        
        self.row_info_df=pd.DataFrame();
        self.row_inten_df=pd.DataFrame()
        
        self.row_spec_list=[];
        
        #self.instrument="LOW"
        #self.measured_precision={"LOW":[5e-3,0.2],"HIGH":[5e-4,0.02]}
        self.precision=precision
        
        self.mz_manager=TargetArray(10000,self.precision)
        
        
        self.time_stamp=0;
    
    def set_time_stamp(self,value):
        self.time_stamp=value
    
    def add_spec_dict(self,s_dict):
        self.row_spec_list.append(s_dict)
    
        
    def finish_a_row(self):
        pass
        
        
        ##row
        self.row_info_df=pd.DataFrame(self.row_spec_list)
        
        #self.row_mz_list=[]
        def get_row_mz(item):
            #self.row_mz_list.expend(item[:,0].tolist())
            for mz in item[:,0]:self.mz_manager.append(mz)
            
        self.row_info_df["peaks"].apply(get_row_mz)
        print(f"Timestamp: {time.time()-self.time_stamp},\t finish row mz")
        print(f"len(all_mz) is {len(self.mz_manager.get_mz())}")
        def get_target_inten(peaks,target):
            #print(322,peaks,target)
            sub_abs=np.abs(peaks[:,0]-target)
            #print(sub_abs)
            #print(sub_abs <= target*error,target*error) 
            hit= sub_abs <= target*self.precision
            #print(hit)
            if np.sum(hit)==0:
                return 0
            else:
                return np.sum(peaks[hit,1])
        
        
        def get_inten_from_all_info_df(ser):
            
            return self.all_info_df['peaks'].apply(lambda x:get_target_inten(x, ser.name) )
        
        
        all_inten_df_len=len( self.all_inten_df )
        if all_inten_df_len==0:
            pass
            
        else:# 向右添加新列
            columns=self.mz_manager.get_temp_mz()
            if len(columns)==0:
                pass ##不用向右叠加新列
            else:
                temp_inten_df=pd.DataFrame(np.zeros([all_inten_df_len,len(columns)]),columns=columns)
                temp_inten_df=temp_inten_df.apply(get_inten_from_all_info_df,axis=0)
                self.all_inten_df=pd.concat([self.all_inten_df,temp_inten_df],axis=1)
            
        print(f"Timestamp: {time.time()-self.time_stamp},\t finish join inten_pd to right")
        ## generate row_inten_df from row_info_df
        all_columns=self.mz_manager.get_mz()
        self.row_inten_df= pd.DataFrame( np.zeros( [len( self.row_info_df), len(all_columns)]),columns=all_columns)
        #print(355,self.row_inten_df)
        
        def get_inten_from_row_info_df(ser):
            #print(358,ser)
            #target_mz=ser.name
            #print( self.row_info_df['peaks'] )
            
            column_inten=self.row_info_df['peaks'].apply(lambda x:get_target_inten(x, ser.name) )
            column_inten.name=ser.name
            #print(365,column_inten)
            return  column_inten
        self.row_inten_df=self.row_inten_df.apply(get_inten_from_row_info_df,axis=0)
        
        #print(self.row_inten_df)
        

        print(f"Timestamp: {time.time()-self.time_stamp},\t finish join inten_pd to bottom")
        
        
        ### update row to all
        self.all_info_df=pd.concat([self.all_info_df,self.row_info_df],axis=0)
        self.all_inten_df=pd.concat( [self.all_inten_df,self.row_inten_df])
        
        print(f"Timestamp: {time.time()-self.time_stamp},\t finish update row to all")
        
        
        
        
        self.mz_manager.reset_temp()
        self.reset_row_data()
        
        
        
        
        
        
    def reset_row_data(self):
        self.row_info_df=pd.DataFrame()
        self.row_inten_df=pd.DataFrame()
        self.row_spec_list=[];
        
    
    def get_precision(self):
        
        return self.precision
    
    
    def output(self):
        self.all_info_df.reset_index(drop=True, inplace=True)
        self.all_inten_df.reset_index(drop=True, inplace=True)
        return self.all_info_df,self.all_inten_df


def sub_process_mzml_by_dict2(file_id,mzml_file,obj,precision="LOW",begin_time=0): #一个mzml 文件一个谱图
    TIC=pymzml.run.Reader(mzml_file)
    ddlist=[]
    for spectrum in TIC:
        spectrum.measured_precision=obj.get_precision()

        spectrum.peaks('reprofiled')
        d2_list=spectrum.peaks('centroided')
        if not isinstance(d2_list,np.ndarray):d2_list=np.array(d2_list)
        if d2_list.shape[0]==0: continue
        tdict={}
        tdict['peaks']=d2_list
        tdict['R_time']=spectrum.scan_time_in_minutes()
        tdict['f_id']=file_id
        obj.add_spec_dict(tdict)
    print(f"Timestamp: {time.time()-begin_time},\t finish collect the spect dicts")
    
    obj.finish_a_row()
    
    
    
      

class Test2():
    def __init__(self):
        self.manager = BaseManager()
        self.manager.register('ManagerMatrix', ManagerMatrix)
        self.manager.start()
        self.obj=self.manager.ManagerMatrix(1e-3)
        self.pre_pro_pool=Pool(10)
        
        
    
    def run_one_mzml(self,fid,mzml_file,begin_time):
        #self.pre_pro_pool.apply_async(sub_process_mzml,args=(fid,mzml_file,self.obj))
        #self.pre_pro_pool.close()
        #self.pre_pro_pool.join()
        self.obj.set_time_stamp(begin_time)
        self.pre_pro_pool.apply(sub_process_mzml_by_dict2,args=(fid,mzml_file,self.obj,"LOW",begin_time))
        #sub_process_mzml_by_dict2( fid,mzml_file,self.obj,"LOW",begin_time)
        
        #result=self.obj.output()
        #print(result)

    def run_one_mzml_multhread(self,fid,mzml_file,begin_time):
        
        self.obj.set_time_stamp(begin_time)
        self.pre_pro_pool.apply_async(sub_process_mzml_by_dict2,args=(fid,mzml_file,self.obj,"LOW",begin_time))
    
    def wait_for_end(self):
        self.pre_pro_pool.close()
        self.pre_pro_pool.join()
        print("all task have been done")




if __name__=="__main__111":
    begin_time=time.time();
    t=Test2()
    
    mzml_file=r"E:\document\Experimental_data\通用机械模型\Dual_PESI\test_data\20231010_FR_NanoESI_LXP_1ppm_2KV_27.mzML"
    #t.run_one_mzml(1,mzml_file,begin_time);
    
    
    
    t.run_one_mzml(1,r"E:\document\Experimental_data\通用机械模型\Dual_PESI\test_data\20220408_desi_ghous_02\data01.mzML",begin_time)
    print(f"Timestamp: {time.time()-begin_time},\t finished one")
    t.run_one_mzml(2,r"E:\document\Experimental_data\通用机械模型\Dual_PESI\test_data\20220408_desi_ghous_02\data02.mzML",begin_time)
    print(f"Timestamp: {time.time()-begin_time},\t finished two")
    
    t.run_one_mzml(3,r"E:\document\Experimental_data\通用机械模型\Dual_PESI\test_data\20220408_desi_ghous_02\data03.mzML",begin_time)
    print(f"Timestamp: {time.time()-begin_time},\t finished three")
    
    all_info_df,all_inten_df=t.obj.output()
    print(all_info_df)
    print("\n\n################\n\n")
    print(all_inten_df)
    
    
    all_info_df.to_excel('info_df.xlsx')
    all_inten_df.to_excel('inten_df.xlsx')

if __name__=="__main__2":
    begin_time=time.time();
    t=Test2() 
    
    
    files_dir=r"E:\document\Experimental_data\通用机械模型\Dual_PESI\test_data\20220408_desi_ghous_02\mzml"
    
    files=os.listdir(files_dir)
    import re
    pattern1=re.compile(r'\d+',re.S)
    files.sort(key= ( lambda x: float(pattern1.findall(x)[-1]) if pattern1.findall(x) else 0) ) # file sorted
    mzml_files=[ os.path.join(files_dir,txt) for txt in files if fnmatch.fnmatch(txt,'*.mzML')]
    #print(mzml_files,sep='\n')
    
    [print(x) for x in mzml_files]
    
    total_spectrum_num=0
    for idx,content in enumerate(mzml_files):
        file=content
        t.run_one_mzml(idx,content,begin_time)
        print(f"Timestamp: {time.time()-begin_time},\t finished {idx}th mzml\n\n\n")
    
        #if idx>1:break;
    #t.wait_for_end();
    
    all_info_df,all_inten_df=t.obj.output()
    print(f"############# all_inten_df.shape is {all_inten_df.shape}")
    all_info_df.to_csv('info_df.csv')
    all_inten_df.to_csv('inten_df.csv')
    
    



 



import pickle
def sub_process_mzml_by_dict3(inten_queue,mz_queue,file_id,mzml_file,mz_obj,inten_obj,threshold=0,mslevel=1,subBlank_ISms_input=ddict(None)): #一个mzml 文件一个谱图
    TIC=pymzml.run.Reader(mzml_file)
    ddlist=[]
    row_spect_dict_list=[]
    subBlank_ISms=ddict(None)
    subBlank_ISms.update(subBlank_ISms_input)
    precision=inten_obj.get_precision();
    print(f"pro_proce 538, begin to deal the{mzml_file}, the mass level is {mslevel}, the subBlank_ISms is {subBlank_ISms}")
    for spectrum in TIC:
        if spectrum["MS:1000511"]!=mslevel:continue
        
        if subBlank_ISms.get("is_subBlank",None):
            key=(spectrum["MS:1000511"],spectrum["MS:1000501"],spectrum["MS:1000500"])  #mslevel , 谱图扫描上、下限
            with open("./temp_blank_stander.temp","rb") as f:
                blank_sps=pickle.load(f)
            if subBlank_ISms.get(f"blank_ms{mslevel}_is_IS_active",False):
                IS_inten=(np.array(spectrum.has_peak( subBlank_ISms[f'blank_ms{mslevel}_matric_IS_mz']  ))[:,1].max())
                spectrum/IS_inten
            standerd_Blank_spec=blank_sps[key]
            spectrum-standerd_Blank_spec
        
        
        spectrum.measured_precision=precision

        spectrum.peaks('reprofiled')
        d2_list=spectrum.peaks('centroided')
        if not isinstance(d2_list,np.ndarray):d2_list=np.array(d2_list)
        if d2_list.shape[0]==0: 
            d2_list=np.array([[500,0]])
        else:
            threhold_value=np.max( d2_list[:,1] ) * (threshold/100)
            d2_list=d2_list[d2_list[:,1]>threhold_value,:]
        tdict={'x':0,'y':0,'cut':0}
        tdict['peaks']=d2_list
        tdict['R_time']=spectrum.scan_time_in_minutes()
        tdict['fid']=file_id
        tdict['scan_range']=(spectrum["MS:1000501"],spectrum["MS:1000500"])

        
        row_spect_dict_list.append(tdict)
    
    row_info_df=pd.DataFrame(row_spect_dict_list)
    #print(f"pro_proce 571, the row_info_df is {row_info_df}")
    def get_row_mz(item):
        #self.row_mz_list.expend(item[:,0].tolist())
        for mz in item[:,0]:mz_obj.append(mz,file_id)
    
    
    mz_queue.get()
    
    row_info_df["peaks"].apply(get_row_mz)
    all_mz=copy.deepcopy( mz_obj.get_mz() )
    #temp_mz=copy.deepcopy( mz_obj.get_temp_mz(file_id) )
    #print(f"Timestamp: {time.time()-begin_time},\t finish {file_id} collect the spect dicts, and the len(all_mz) is {len(all_mz)}")
    mz_queue.put(1)
    
    
    
    
    def get_target_inten(peaks,target):
        #print(322,peaks,target)
        sub_abs=np.abs(peaks[:,0]-target)
        #print(sub_abs)
        #print(sub_abs <= target*error,target*error) 
        hit= sub_abs <= target*precision
        #print(hit)
        if np.sum(hit)==0:
            return 0
        else:
            return np.sum(peaks[hit,1])
    
    
    row_inten_df= pd.DataFrame( np.zeros( [len( row_info_df), len(all_mz)]),columns=all_mz)
    def get_inten_from_row_info_df(ser):
        #print(358,ser)
        #target_mz=ser.name
        #print( self.row_info_df['peaks'] )
        
        column_inten=row_info_df['peaks'].apply(lambda x:get_target_inten(x, ser.name) )
        column_inten.name=ser.name
        #print(365,column_inten)
        return  column_inten
    row_inten_df=row_inten_df.apply(get_inten_from_row_info_df,axis=0)
    
    
    inten_queue.get()
    inten_obj.add_spect_data(row_info_df,row_inten_df,{"fid":file_id,"mz_column":all_mz})
    #print(f"Timestamp: {time.time()-begin_time},\t finish {file_id} collect the inten df")
    inten_queue.put(1)

    return 1;

def subproces_do_post_compensation(inten_queue,fid,current_mz,all_mz,inten_obj):
    hit_mz= ~ np.in1d(all_mz,current_mz)
    if(np.sum(hit_mz)==0):return 
    print(f"there is {np.sum(hit_mz)} mz from {fid} have  to be compensted, that include {all_mz[hit_mz]}",)
    
    time_stamp=time.time()
    
    inten_queue.get()
    info_df=inten_obj.output_info_df_by_fid(fid)
    compenstion_df=inten_obj.output_inten_df_by_index_and_mz(info_df.index,all_mz[hit_mz])
    precision=inten_obj.get_precision()
    print(f"@@@@@@ subprocess_post_compensation {fid } success get data")
    inten_queue.put(1)
    
    def get_target_inten(peaks,target):
        #print(322,peaks,target)
        sub_abs=np.abs(peaks[:,0]-target)
        #print(sub_abs)
        #print(sub_abs <= target*error,target*error) 
        hit= sub_abs <= target*precision
        #print(hit)
        if np.sum(hit)==0:
            return 0
        else:
            return np.sum(peaks[hit,1])
    
    
    
    def get_inten_from_row_info_df(ser):
        
        
        column_inten=info_df['peaks'].apply(lambda x:get_target_inten(x, ser.name) )
        column_inten.name=ser.name
        #print(365,column_inten)
        return  column_inten
    
    
    
    
    
    compenstion_df=compenstion_df.apply(lambda x:get_inten_from_row_info_df(x),axis=0)
    
    inten_queue.get()
    print(f"@@@@@@@ subprocess_post_compensation {fid } start update data")
    inten_obj.update_all_inten_df(compenstion_df)
    inten_obj.update_row_cloumn_list(fid,all_mz)
    print(f"@@@@@@@ subprocess_post_compensation {fid } success update data")
    inten_queue.put(1)
    print(f"Timestamp: {time.time()-time_stamp},\t finish the {fid}th post_compensation")
    return 1
    

class TargetArray_Mem():
    def __init__(self,size_unit=10000,precision=5e-6):
        self.size_unit=size_unit
        self._d2_array=np.ones([2,self.size_unit])
        self.size=self.size_unit
        self.aid=0
        self.precision=precision;
        
        self.temp_dict=ddict(list)
    
    def re_init(self,precision=5e-3):
        self._d2_array=np.ones([2,self.size_unit])
        self.size=self.size_unit
        self.aid=0
        self.precision=precision;
    
    
        
    
    def get_mz(self):
        return self._d2_array[1,(self._d2_array[0,:]==0)]    
    
    def get_temp_mz(self,temp_id):
        return self.temp_dict[temp_id]
        
    def reset_temp(self):
        pass
    def _expand_array_size(self):
        self._d2_array=np.hstack([self._d2_array,np.ones([2,self.size_unit])  ]  )
        self.size+=self.size_unit
    def append(self,mz,temp_id):
        if self.aid>=self.size:self._expand_array_size();
        mzs=self.get_mz()
        #print(mzs)
        #print(f"\r deal the {self.temp_num}",end="",flush=True)
        #self.temp_num+=1
        if   (np.abs(mzs-mz)< mz*self.precision).any()  :
            return
        else:
            self._d2_array[:,self.aid]=[0,mz]
            #self.temp_dict[temp_id].append(mz)
            self.aid+=1
    



class ManagerSpect():
    def __init__(self,precision=5e-3,threhold_value=2):
        
        

        self.all_info_df=pd.DataFrame();
        self.all_inten_df=pd.DataFrame()
        
        self.row_cloumn_list=[]
        
        
        self.time_stamp=0;
        self.precision=precision
        self.threhold_value=threhold_value
        
    def set_time_stamp(self,value):
        self.time_stamp=value
    
    
    def input_all_info_inten_df(self,all_info_df,all_inten_df):
        self.all_info_df=all_info_df
        self.all_inten_df=all_inten_df
    
    
    def re_init(self,precision=5e-3):
        self.all_info_df=pd.DataFrame();
        self.all_inten_df=pd.DataFrame()
        
        self.row_cloumn_list=[]
        
        
        self.time_stamp=0;
        self.precision=precision
        
    
    def add_spec_dict(self,s_dict):
        self.row_spec_list.append(s_dict)
    
    
    
    def add_spect_data(self,row_info_df,row_inten_df,row_cloumn={"fid":0,"mz_column":[]}):
        try:
            self.all_inten_df=pd.concat([self.all_inten_df,row_inten_df],axis=0)
            self.all_info_df=pd.concat([self.all_info_df,row_info_df])
            self.all_info_df.reset_index(inplace=True,drop=True)
            self.all_inten_df.reset_index(inplace=True,drop=True)
            
            #self.all_inten_df=pd.concat([self.all_inten_df,right_inten_df],axis=1)
            
            self.row_cloumn_list.append(row_cloumn)
        except Exception as e:
            print("699 in pre_processing have error: ",e)
    
    def get_inten_columns(self):
        return self.all_inten_df.columns
    

    def get_precision(self):
        
        return self.precision
    
    def update_row_cloumn_list(self,fid,new_columns):#for process
        idx=0
        for item_dict in self.row_cloumn_list:
            if item_dict['fid']==fid:
                self.row_cloumn_list[idx]={'fid':fid,'mz_column':new_columns}
            idx+=1    
    def update_all_inten_df(self,compenstion_df):     #for process   
        self.all_inten_df.update(compenstion_df,overwrite=True)               
    
    def output_row_cloumn_list(self): # for view demo
        return self.row_cloumn_list
    def output_info_df_by_fid(self,fid): #for process
        return self.all_info_df.loc[self.all_info_df['fid']==fid,:]
    def output_inten_df_by_index_and_mz(self,index,mz_column): #for process
        return self.all_inten_df.loc[index,mz_column]
    
    
    def post_compensation(self):
        
        #inten_queue.get()
        if self.all_inten_df.shape[0]==0:return
        if len(self.row_cloumn_list)==0:
            
            return
        
        copy_row_cloumn_list=copy.deepcopy( self.row_cloumn_list )
        self.all_info_df.reset_index(drop=True, inplace=True)
        self.all_inten_df.reset_index(drop=True, inplace=True)
        
        #inten_queue.put(1)
        
        row_num=len(copy_row_cloumn_list)
        if row_num==0:return
        
        
        def get_target_inten(peaks,target):
            #print(322,peaks,target)
            sub_abs=np.abs(peaks[:,0]-target)
            #print(sub_abs)
            #print(sub_abs <= target*error,target*error) 
            hit= sub_abs <= target*self.precision
            #print(hit)
            if np.sum(hit)==0:
                return 0
            else:
                return np.sum(peaks[hit,1])
        
        
        
        def get_inten_from_row_info_df(ser,fid):
            #print(358,ser)
            #target_mz=ser.name
            #print( self.row_info_df['peaks'] )
            
            column_inten=self.all_info_df.loc[self.all_info_df['fid']==fid,'peaks'].apply(lambda x:get_target_inten(x, ser.name) )
            column_inten.name=ser.name
            #print(365,column_inten)
            return  column_inten
        
        #row_inten_df=row_inten_df.apply(get_inten_from_row_info_df,axis=0)
        
        all_mz=copy_row_cloumn_list[-1]['mz_column']
        last_fid=copy_row_cloumn_list[-1]['fid']
        
        for idx,item in enumerate(copy_row_cloumn_list):
            #if item['fid']==last_fid:break;
            
            
            hit= ~ np.in1d(all_mz,item['mz_column'])
            if(np.sum(hit)==0):continue
            print(f"there is {np.sum(hit)} mz from {item['fid']} have  to be compensted, that include {all_mz[hit]}",)
            row_hit=self.all_info_df['fid']==item['fid']
            #self.all_inten_df.loc[ row_hit ,all_mz[hit]] = self.all_inten_df.loc[ row_hit ,hit].apply(lambda x:get_inten_from_row_info_df(x,item['fid']))
            compenstion_df=self.all_inten_df.loc[ row_hit ,all_mz[hit] ].apply(lambda x:get_inten_from_row_info_df(x,item['fid']))
            
            #print(compenstion_df)
            self.all_inten_df.update(compenstion_df,overwrite=True)
            self.row_cloumn_list[idx]={'fid':item['fid'],'mz_column':all_mz}
        print(f"Timestamp: {time.time()-self.time_stamp},\t finish the post_compensation")
    
    def get_inten_len(self):
        return self.all_inten_df.shape[0]
    
    def output(self):
        self.all_info_df.reset_index(drop=True, inplace=True)
        self.all_inten_df.reset_index(drop=True, inplace=True)
        return copy.deepcopy(self.all_info_df),copy.deepcopy(self.all_inten_df)


class ManagerSpect2():
    def __init__(self,precision=5e-3,threhold_value=2):
        
        

        self.all_info_df=pd.DataFrame();
        self.all_inten_df=pd.DataFrame()
        
        self.row_cloumn_list=[]
        
        
        self.time_stamp=0;
        self.precision=precision
        self.threhold_value=threhold_value
        
    def set_time_stamp(self,value):
        self.time_stamp=value
    
    
    
    
    
    def re_init(self,precision=5e-3):
        self.all_info_df=pd.DataFrame();
        self.all_inten_df=pd.DataFrame()
        
        self.row_cloumn_list=[]
        
        
        self.time_stamp=0;
        self.precision=precision
        
    
    def add_spec_dict(self,s_dict):
        self.row_spec_list.append(s_dict)
    
    
    
    def add_spect_data(self,row_info_df,row_inten_df,row_cloumn={"fid":0,"mz_column":[]}):
        try:
            self.all_inten_df=pd.concat([self.all_inten_df,row_inten_df],axis=0)
            self.all_info_df=pd.concat([self.all_info_df,row_info_df])
            self.all_info_df.reset_index(inplace=True,drop=True)
            self.all_inten_df.reset_index(inplace=True,drop=True)
            
            #self.all_inten_df=pd.concat([self.all_inten_df,right_inten_df],axis=1)
            
            self.row_cloumn_list.append(row_cloumn)
        except Exception as e:
            print("699 in pre_processing have error: ",e)
    
    def get_inten_columns(self):
        return self.all_inten_df.columns
    

    def get_precision(self):
        
        return self.precision
    
    def update_row_cloumn_list(self,fid,new_columns):#for process
        idx=0
        for item_dict in self.row_cloumn_list:
            if item_dict['fid']==fid:
                self.row_cloumn_list[idx]={'fid':fid,'mz_column':new_columns}
            idx+=1    
    def update_all_inten_df(self,compenstion_df):     #for process   
        self.all_inten_df.update(compenstion_df,overwrite=True)               
    
    def output_row_cloumn_list(self): # for view demo
        return self.row_cloumn_list
    def output_info_df_by_fid(self,fid): #for process
        return self.all_info_df.loc[self.all_info_df['fid']==fid,:]
    def output_inten_df_by_index_and_mz(self,index,mz_column): #for process
        return self.all_inten_df.loc[index,mz_column]
    
    
    def post_compensation(self):
        
        #inten_queue.get()
        if self.all_inten_df.shape[0]==0:return
        if len(self.row_cloumn_list)==0:
            
            return
        
        copy_row_cloumn_list=copy.deepcopy( self.row_cloumn_list )
        self.all_info_df.reset_index(drop=True, inplace=True)
        self.all_inten_df.reset_index(drop=True, inplace=True)
        
        #inten_queue.put(1)
        
        row_num=len(copy_row_cloumn_list)
        if row_num==0:return
        
        
        def get_target_inten(peaks,target):
            #print(322,peaks,target)
            sub_abs=np.abs(peaks[:,0]-target)
            #print(sub_abs)
            #print(sub_abs <= target*error,target*error) 
            hit= sub_abs <= target*self.precision
            #print(hit)
            if np.sum(hit)==0:
                return 0
            else:
                return np.sum(peaks[hit,1])
        
        
        
        def get_inten_from_row_info_df(ser,fid):
            #print(358,ser)
            #target_mz=ser.name
            #print( self.row_info_df['peaks'] )
            
            column_inten=self.all_info_df.loc[self.all_info_df['fid']==fid,'peaks'].apply(lambda x:get_target_inten(x, ser.name) )
            column_inten.name=ser.name
            #print(365,column_inten)
            return  column_inten
        
        #row_inten_df=row_inten_df.apply(get_inten_from_row_info_df,axis=0)
        
        all_mz=copy_row_cloumn_list[-1]['mz_column']
        last_fid=copy_row_cloumn_list[-1]['fid']
        
        for idx,item in enumerate(copy_row_cloumn_list):
            #if item['fid']==last_fid:break;
            
            
            hit= ~ np.in1d(all_mz,item['mz_column'])
            if(np.sum(hit)==0):continue
            print(f"there is {np.sum(hit)} mz from {item['fid']} have  to be compensted, that include {all_mz[hit]}",)
            row_hit=self.all_info_df['fid']==item['fid']
            #self.all_inten_df.loc[ row_hit ,all_mz[hit]] = self.all_inten_df.loc[ row_hit ,hit].apply(lambda x:get_inten_from_row_info_df(x,item['fid']))
            compenstion_df=self.all_inten_df.loc[ row_hit ,all_mz[hit] ].apply(lambda x:get_inten_from_row_info_df(x,item['fid']))
            
            #print(compenstion_df)
            self.all_inten_df.update(compenstion_df,overwrite=True)
            self.row_cloumn_list[idx]={'fid':item['fid'],'mz_column':all_mz}
        print(f"Timestamp: {time.time()-self.time_stamp},\t finish the post_compensation")
    
    def get_inten_len(self):
        return self.all_inten_df.shape[0]
    
    def output(self):
        self.all_info_df.reset_index(drop=True, inplace=True)
        self.all_inten_df.reset_index(drop=True, inplace=True)
        if len(self.all_info_df)<10:
            print("no raw data in manager obj")
            return None,None
        with open("temp_info.pkl","wb") as f:
            temp_info_file_abpath=os.path.abspath(f.name)
            self.all_info_df.to_pickle(f)
        with open("temp_inten.pkl","wb") as f:
            temp_inten_file_abpath=os.path.abspath(f.name)
            self.all_inten_df.to_pickle(f)    
        
        
        
        
        return temp_info_file_abpath, temp_inten_file_abpath
        #return copy.deepcopy(self.all_info_df),copy.deepcopy(self.all_inten_df)

    def input_all_info_inten_df(self,temp_info_file_abpath,temp_inten_file_abpath):
        
        self.all_info_df=pd.read_pickle(temp_info_file_abpath)
        self.all_inten_df=pd.read_pickle(temp_inten_file_abpath)
        
        







class Test3():
    def __init__(self):
        self.manager = BaseManager()
        self.manager.register('ManagerSpect', ManagerSpect)
        self.manager.register('TargetArray_Mem', TargetArray_Mem)
        self.manager.start()
        
        self.mz_obj=self.manager.TargetArray_Mem(precision=1e-3)
        self.inten_obj=self.manager.ManagerSpect(1e-3)
        
        self.pre_pro_pool=Pool(11)
        
        
        self.mz_queue=Manager().Queue()
        self.mz_queue.put(1)
        self.inten_queue=Manager().Queue()
        self.inten_queue.put(1)
        
        self.res_l=[]
        
    
    def run_one_mzml(self,fid,mzml_file,begin_time):
        #self.pre_pro_pool.apply_async(sub_process_mzml,args=(fid,mzml_file,self.obj))
        #self.pre_pro_pool.close()
        #self.pre_pro_pool.join()
        self.inten_obj.set_time_stamp(begin_time)
        res=self.pre_pro_pool.apply_async(sub_process_mzml_by_dict3,args=(self.inten_queue,self.mz_queue,fid,mzml_file,self.mz_obj,self.inten_obj,begin_time))
        self.res_l.append(res)
        #sub_process_mzml_by_dict2( fid,mzml_file,self.obj,"LOW",begin_time)
        
        #result=self.obj.output()
        #print(result)


    def wait_for_end(self):
        
        num=0
        task_num=len(self.res_l)
        for res in self.res_l:
            num+=res.get()
            mystr=f'the deal {num}/{task_num} spectrum!!!'
            print('\r',mystr,end='\n',flush=True)
        
        print()
        #self.pre_pro_pool.close()
        #self.pre_pro_pool.join()
        print("all task have been done")




if __name__=="__main__44":
    begin_time=time.time();
    t=Test3()
    
    mzml_file=r"E:\document\Experimental_data\通用机械模型\Dual_PESI\test_data\20231010_FR_NanoESI_LXP_1ppm_2KV_27.mzML"
    #t.run_one_mzml(1,mzml_file,begin_time);
    
    
    
    t.run_one_mzml(1,r"E:\document\Experimental_data\通用机械模型\Dual_PESI\test_data\20220408_desi_ghous_02\data01.mzML",begin_time)
    print(f"Timestamp: {time.time()-begin_time},\t finished one")
    t.run_one_mzml(2,r"E:\document\Experimental_data\通用机械模型\Dual_PESI\test_data\20220408_desi_ghous_02\data02.mzML",begin_time)
    print(f"Timestamp: {time.time()-begin_time},\t finished two")
    
    t.run_one_mzml(3,r"E:\document\Experimental_data\通用机械模型\Dual_PESI\test_data\20220408_desi_ghous_02\data03.mzML",begin_time)
    print(f"Timestamp: {time.time()-begin_time},\t finished three")
    
    
    t.wait_for_end()
    t.inten_obj.post_compensation()
    
    all_info_df,all_inten_df=t.inten_obj.output()
    print(all_info_df)
    print("\n\n################\n\n")
    print(all_inten_df)
    
    
    all_info_df.to_excel('info_df_mulpro1.xlsx')
    all_inten_df.to_excel('inten_df_mulpro1.xlsx')




if __name__=="__main__":
    begin_time=time.time();
    t=Test3() 
    
    
    files_dir=r"E:\document\Experimental_data\通用机械模型\Dual_PESI\test_data\20220408_desi_ghous_02\mzml"
    
    files=os.listdir(files_dir)
    import re
    pattern1=re.compile(r'\d+',re.S)
    files.sort(key= ( lambda x: float(pattern1.findall(x)[-1]) if pattern1.findall(x) else 0) ) # file sorted
    mzml_files=[ os.path.join(files_dir,txt) for txt in files if fnmatch.fnmatch(txt,'*.mzML')]
    #print(mzml_files,sep='\n')
    
    [print(x) for x in mzml_files]
    
    total_spectrum_num=0
    for idx,content in enumerate(mzml_files):
        file=content
        t.run_one_mzml(idx,content,begin_time)
        #print(f"Timestamp: {time.time()-begin_time},\t finished {idx}th mzml\n\n\n")
    
        if idx>2:break;
    #t.wait_for_end();
    t.wait_for_end()
    t.inten_obj.post_compensation()
    all_info_df,all_inten_df=t.inten_obj.output()
    print(f"############# all_inten_df.shape is {all_inten_df.shape}")
    all_info_df.to_csv('info_df_mulpro.csv')
    all_inten_df.to_csv('inten_df_malpro.csv')



 