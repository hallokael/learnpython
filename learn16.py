from multiprocessing import Process,Pool
import os,time,random
def run_proc(name):
    print('Run child %s(%s)'%(name,os.getpid()))
def task(name):
    print('Run task %s'%name)
if __name__=='__test__':
    print('Parent '+str(os.getpid()))
    p=Process(target=run_proc,args=('test',))
    print('child start')
    p.start()
    p.join()
    print("child end")
if __name__=='__main__':
    p=Pool(4)
    for i in range(5):
        p.apply_async(task,args=(i,))
    p.close()
    p.join()
    print("all done")