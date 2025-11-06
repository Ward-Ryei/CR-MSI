import subprocess
import time
import os

def convert_f(file_path):
    folder_path, file_name = os.path.split(file_path)
    command_in=f'msconvert "{file_path}" --outdir "{folder_path}" --mz64 --inten64' # 二级过滤 ：--filter "msLevel 2"
    print(command_in)
    process = subprocess.Popen(command_in, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while(1):
        time.sleep(0.2)
        command_output = process.stdout.readline().decode('utf-8')
        #print(command_output)
        process.poll() #拉取状态码
        if process.returncode == 0:break;
    
    process.wait()
    print("in file_convert:massconvert finished!!")
