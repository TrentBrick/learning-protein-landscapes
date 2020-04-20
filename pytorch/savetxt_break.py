from multiprocessing import Process, Queue, cpu_count
import numpy as np 
wut = open('test_break.txt', 'ab')
for i in range(10):
    np.savetxt(wut , np.asarray([6,7,8,9,10]))
wut = wut.close()
print(wut)
wut = open(wut, 'ab')
for i in range(10):
    np.savetxt(wut , np.asarray([6,7,8,9,10]))
            