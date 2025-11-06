from basement import *

class DESI(Acquisition):
    def __init__(self):
        super().__init__()
        
        self.desi_para={"x_l":0,"x_r":0,"x_s":0,"x_e":0,
            "y_s":0,"y_e":0,"y_interval":0,
            "y_current":0,"msi_speed":0,}
    
        self.current_row=0;
        self.total_row0=0;
    
    def add_two_scan_row(self):
        #self.y_current=y_s
        exec=False
        if self.desi_para['y_current']<=self.desi_para['y_e']:
            self.add_task("move x",f"D40 C1 I0 T{self.desi_para['x_l']}")
            self.add_task("move y",f"D40 C1 I1 T{self.desi_para['y_current']}")
            #self.add_task("init row time",f"D20")
            self.add_task("move x",f"D40 C1 I0 T{self.desi_para['x_s']}")
            self.add_task("active ms",f"D40 C5 I0")
            self.add_task("move x",f"D40 C1 I0 T{self.desi_para['x_e']}")
            self.add_task("report row time",f"D40 C6")
            self.add_task("move x",f"D40 C1 I0 T{self.desi_para['x_r']}")
            
            
            self.add_task("wait time",f"D40 C20 T{round( self.desi_para.get('wait_time',5) * 1000,2)}")
            self.add_task("finished desi rows","D40 C200 I0 P0")
            self.desi_para['y_current']=round(self.desi_para['y_current']+self.desi_para['y_interval'],2);
            exec=True
            
            
        if self.desi_para['y_current']<=self.desi_para['y_e']:
            self.add_task("move x",f"D40 C1 I0 T{self.desi_para['x_r']}")
            self.add_task("move y",f"D40 C1 I1 T{self.desi_para['y_current']}")
            #self.add_task("init row time",f"D20")
            self.add_task("move x",f"D40 C1 I0 T{self.desi_para['x_e']}")
            self.add_task("active ms",f"D40 C5")
            self.add_task("move x",f"D40 C1 I0 T{self.desi_para['x_s']}")
            self.add_task("report row time",f"D40 C6")
            self.add_task("move x",f"D40 C1 I0 T{self.desi_para['x_l']}")
            
            
            self.add_task("wait time",f"D40 C20 T{self.desi_para.get('wait_time',5) * 1000}")
            self.add_task("finished desi rows","D40 C200 I0 P0")
            self.desi_para['y_current']=round(self.desi_para['y_current']+self.desi_para['y_interval'],2);
            exec=True
        
        
        else:
            pass
            #self.add_task("finish desi","D40 C211 I0 P0")
            #self.add_task("call for desi","D40 C202 I0 P0")
            #self.add_task("lc pump off",f"D40 C29")
            #self.add_task("wait time",f"D40 C20 T10")
            #self.add_task("close GAS",f"D40 C51")
        return exec
    
    def run(self):
        return self.add_two_scan_row()
        #if self.desi_para['y_current']<=self.desi_para['y_e']:
        #    self.add_two_scan_row()
        #    return 1
        #else:
        #    return 0
    
    def set_desi_para(self,new_dict):
        self.desi_para.update(new_dict)
    
