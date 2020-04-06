import tkinter as tk
from tkinter.filedialog import askopenfilename


def import_csv_data():
    global v
    csv_file_path = askopenfilename()
    print(csv_file_path)
    v.set(csv_file_path)
    #df = pd.read_csv(csv_file_path)

class App(object):
    def def __init__(self, master, **kwargs):
        self.master = master
        self.create_text()

win = tk.Tk()
canvas1 = tk.Canvas(win, width = 400, height = 300)
canvas1.pack()

win.title("LauzHack Project")

label1 = tk.Label(win, text='Import .csv file with drug data')
label1.config(font=('helvetica', 14))
canvas1.create_window(200, 25, window=label1)

entry1 = tk.Entry(win)
canvas1.create_window(200, 140, window=entry1)


def analyzeData():
    print("hi")


button1 = tk.Button(win, text='Browse Data Set',command=import_csv_data)
canvas1.create_window(200, 180, window=button1)

# Import csv file

v = tk.StringVar()
entry = tk.Entry(win, textvariable=v).grid(row=0, column=1)


win.mainloop()
