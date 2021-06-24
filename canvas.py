from tkinter import *

canvas = Tk()
canvas.title("Handwriting Recognition AI")

canvas_width = 400
canvas_width = 450

def draw(event):
    color ='black'
    x1,y1 = (event.x-1),(event.y-1)
    x2,y2 = (event.x+1), (event.y+1)

