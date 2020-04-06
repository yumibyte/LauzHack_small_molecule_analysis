
# Creating GUI
import tkinter as tk
from tkinter import filedialog
import pandas as pd

root = tk.Tk()

label1 = tk.Label(root, text="Import .csv file with drug data")

canvas1 = tk.Canvas(root, width=300, height=300, bg='lightsteelblue2', relief='raised')
canvas1.pack()

root.title("LauzHack Project")
label1.config(font=('helvetica', 14))
canvas1.create_window(200, 25, window=label1)

def getCSV():
    global df

    import_file_path = filedialog.askopenfilename()
    df = pd.read_csv(import_file_path)
    print(df)


browseButton_CSV = tk.Button(root, text="      Import CSV File     ", command=getCSV, bg='green', fg='white',
                             font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 150, window=browseButton_CSV)

root.mainloop()