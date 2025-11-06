
import queue
class Plan():
    def __init__(self,note="",code="",active_code="",return_code=""):
        self.note=note
        self.code=code
        self.active_code=active_code
        self.return_code=return_code
    def report_plan(self):
        print("the plan is: ",self.code)
        


class Acquisition():
    def __init__(self):
        pass
        self.task_queue=queue.SimpleQueue()
        
    def add_task(self,note="",code="",active_code="",return_code=""):
        self.task_queue.put(Plan(note,code,active_code,return_code))

    def report_task(self):
        print("task_len is:", self.task_queue.qsize())
        temp=[]
        while not self.task_queue.empty():
            plan=self.task_queue.get()
            plan.report_plan()
            temp.append(plan)
        for task in temp:
            self.task_queue.put(task)
    def stop(self):
        while not self.task_queue.empty():
            self.task_queue.get()

if __name__=="__main__":
    a=Acquisition()
    a.add_task("yoyo", "D1 P5");
    a.add_task("yoyo", "D2 P8");
    a.report_task()

