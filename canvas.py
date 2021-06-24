from tkinter import *

import mss as mss
import pyautogui
import numpy
import random
import tkcap
import PIL
import win32api

x, y = 200, 200 # size

def __init__():

    app_title = "Handwriting Detection AI"

    def paint(event):

        color = 'black'
        x_l = event.x-1 # lower
        y_l = event.y-1
        x_u = event.x+1 # upper
        y_u = event.y+1
        w.create_oval(x_l, y_l, x_u, y_u, fill=color, outline=color)


    def save():

        x1 = canvas.winfo_x()
        y1 = canvas.winfo_y()

        monitor = {"top": x1, "left": y1, "width": x, "height": y}
        sct = mss.mss()
        sct_img = sct.grab(monitor)
        mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
        print(output)
        image = numpy.array(sct_img)

    def save_function():
        cap = tkcap.CAP(canvas)
        cap.capture( "img_" + app_title + ' ' + str(random.random()) + ".png")

    canvas = Tk()
    canvas.title(app_title)
    w = Canvas(canvas, width=x, height=y, bg = 'white')

    w.pack(expand=YES, fill=BOTH)
    w.bind("<B1-Motion>", paint)

    message = Label(canvas, text="Please enter your digit (from 0 to 9)")
    message.pack(side=BOTTOM)


    btn = Button(canvas, text='Save your drawing!', bd='5', command=save_function)
    btn.pack(side='top')
    canvas.bind(btn)
    canvas.resizable("false", "false")

    win32api.MessageBox(0, ' > Please draw a digit & save, and do not move the window position', 'Hi! One quick thing')
    canvas.mainloop()



__init__()