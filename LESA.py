from basement import *


class LESA(Acquisition):
    def __init__(self):
        super().__init__()
        self.sample_hight=0;
        self.ion_hight=0;
        self.sample_position=2550;
        self.ion_position=3550;
        self.suck_time=100;
        self.pump_time=100;
        self.ion_time=10;
        
        self.LESA_para={"sample_hight":0,"ion_hight":0,
            "sample_position":2550,"ion_position":3550,
            "suck_time":100,"pump_time":100,"extract_time":1000,"ion_time":10}
        
    def add_one_sample_point(self,x,y):
        #sample
        self.add_task("move x",f"D40 C1 I0 T{x}")
        self.add_task("move y",f"D40 C1 I1 T{y}")
        
        #self.add_task("suck M on ",f"D40 C35")
        
        self.add_task("move to sample",f"D40 C3 I1 T{self.LESA_para['sample_position']}")
        self.add_task("move z",f"D40 C1 I2 T{self.LESA_para['sample_hight']}")
        
        
        self.add_task("lc pump on",f"D40 C28")
        self.add_task("pump time",f"D40 C20 I0 T{self.LESA_para['pump_time']}")
        self.add_task("lc pump off",f"D40 C29")
        
        #self.add_task("prepare_liquid_relay for suck ",f"D40 C31 I1")
        
        self.add_task("wait for suck",f"D40 C20 I0 T{self.LESA_para['extract_time']}")
        
        #self.add_task("suck ",f"D40 C34 T{self.LESA_para['suck_time']}")
        
        #self.add_task("suck M off ",f"D40 C36")
        
        #self.add_task("wait",f"D40 C20 I0 T1000")
        self.add_task("move z",f"D40 C1 I2 T0")
        
        
        self.add_task("move to ion",f"D40 C3 I1 T{self.LESA_para['ion_position']}")
        self.add_task("move z",f"D40 C1 I2 T{self.LESA_para['ion_hight']}")
        self.add_task("call hv",f"D40 C40 I1 T0")
        self.add_task("wait",f"D40 C20 I0 T1000")
        self.add_task("active mass",f"D40 C5 I0")
        self.add_task("wait",f"D40 C20 I0 T{self.LESA_para['ion_time']*1000}")
        self.add_task("discall hv",f"D40 C40 I0 T0")
        self.add_task("move z",f"D40 C1 I2 T{-0.1}")
        self.add_task("zero z",f"D40 C38 I2 T0")
        
        self.add_task("move to ion",f"D40 C3 I1 T{(self.LESA_para['ion_position']+ self.LESA_para['sample_position'] )/2}")
        self.add_task("prepare_liquid_relay for pull",f"D40 C30 I1 T0")
        self.add_task("wait",f"D40 C20 I0 T1000")
        self.add_task("gas pull",f"D40 C32 I0 T2000")
        
        self.add_task("wait",f"D40 C20 I0 T2000")
        
        
        self.add_task("finished a LESA point",f"D40 C205 I0 T0")
        
        
        
    def set_lesa_para(self,new_dict):
        self.LESA_para.update(new_dict)
    
    