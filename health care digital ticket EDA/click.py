#%%
import win32api, win32con
def click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)



import time
import math



for j in range(10000):
    if j <=10000:
        for i in range(500):
            x = int(500+math.sin(math.pi*i/100)*500)
            y = int(500+math.cos(i)*100)
            win32api.SetCursorPos((x,y))
            time.sleep(.01)
        click(x,y)  
# %%


