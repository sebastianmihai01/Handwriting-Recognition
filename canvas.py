from tkinter import *

import mss as mss
import pyautogui
import numpy
import random
import tkcap
import pylab as pl
import PIL
import win32api
import pyautogui as pyag

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

    def nn_output():

        x1 = canvas.winfo_x()
        y1 = canvas.winfo_y()
        sct = mss.mss()
        monitor = sct.monitors[1]

        top = monitor["top"];
        left = monitor["left"]
        width = monitor["width"]
        height = monitor["height"]

        monitor = {"top": top, "left": left, "width": width+x, "height": height+y}
        formatted_image = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)

        sct_img = sct.grab(monitor)
        mss.tools.to_png(sct_img.rgb, sct_img.size, output=formatted_image)
        print("image saved")
        image = numpy.array(sct_img)

    def screenshot_canvas():
        cap = tkcap.CAP(canvas)
        img_name = "img_" + app_title + ' ' + str(random.random()) + ".png"
        image = cap.capture(img_name)

    canvas = Tk()
    canvas.title(app_title)
    w = Canvas(canvas, width=x, height=y, bg = 'white')

    w.pack(expand=YES, fill=BOTH)
    w.bind("<B1-Motion>", paint)

    message = Label(canvas, text="Please enter your digit (from 0 to 9)")
    message.pack(side=BOTTOM)

    btn = Button(canvas, text='Save your drawing!', bd='5', command = screenshot_canvas)
    btn.pack(side='top')
    canvas.bind(btn)
    canvas.resizable("false", "false")

    win32api.MessageBox(0, ' > Please draw a digit & save, and do not move the window position', 'Hi! One quick thing')
    canvas.mainloop()



__init__()