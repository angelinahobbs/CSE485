import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageTk
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Frame, Label, ttk, Toplevel
import cv2
import subprocess

# root frame
root = tk.Tk()

# container for video frame and bottom frame
left_container = Frame(root, width=1440, height=1080)
left_container.pack(side='left', expand=True, fill='both')

# video frame
video_frame = Frame(left_container, bg="black", width=1440, height=810)
video_frame.pack_propagate(False)
video_frame.pack(side='top')
video_label = Label(video_frame)
video_label.pack(fill='both', expand=True)

# bottom left frame
bottom_frame = Frame(left_container, bg="black", width=1440, height=270)
bottom_frame.pack_propagate(False)
bottom_frame.pack(side='top')

# graph frame with settings button
right_frame = Frame(root, bg='black', width=480)
right_frame.pack(side="right", fill="y")
right_frame.pack_propagate(False)

# dropdown label
dropdown_label = Label(bottom_frame, text="Select an Animal to Track:", bg='black', fg='white')
dropdown_label.pack(side='top', pady=10)

# dropdown list of animals
animals = ["Bear", "Wolf", "Coyote", "Deer", "Fox", "Squirrel", "Human", "Dog", "Cat"]
animal_checks = {}
for animal in animals:
    animal_checks[animal] = tk.IntVar(value=1)  # sets all animals to being tracked

selected_animal = tk.StringVar()
animal_dropdown = ttk.Combobox(bottom_frame, textvariable=selected_animal, values=animals)
animal_dropdown.pack(side='top', pady=10)
animal_counts = {}

# Label for displaying counts
counts_label = Label(bottom_frame, text="Counts:\n", justify=tk.LEFT, bg='black', fg='white', width=33)
counts_label.pack(side='right', padx=10, pady=10)

# graph Frame
graph_frame = Frame(right_frame)
graph_frame.pack(side='top', fill='both', expand=True, pady=20)

# actual graph
fig, ax = plt.subplots(figsize=(5, 2), tight_layout=True)
ax.xaxis.set_major_locator(mpl.dates.AutoDateLocator())
ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d %H:%M'))
plt.xticks(rotation=45)
plt.ylabel("Counts")

# set canvas for graph and add to tkinter
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side='top', fill='both', expand=True)

# notification variables
movement_notif = tk.BooleanVar(value=False)
battery_notif = tk.BooleanVar(value=False)

cap = None


# custom settings switch
class ToggleSwitch(tk.Frame):
    def __init__(self, master, text="", width=50, height=20, variable=None, **kwargs):
        super().__init__(master, bg="black", **kwargs)
        # attach to variable
        self.is_on = variable
        if self.is_on is None:
            self.is_on = tk.BooleanVar(value=False)

        # create canvas
        self.canvas = tk.Canvas(self, width=width, height=height, background="black")
        self.canvas.pack(side="right")

        # if the value is true, draw on
        if self.is_on.get():
            self.canvas.create_rectangle(0, 0, width, height, outline="", fill="blue", tags="rect")
            self.canvas.create_oval(-height / 2, 0, height / 2, height, outline="black", fill="white", tags="circle")
            self.canvas.move("circle", width, 0)
        else:  # if value false, draw off
            self.canvas.create_rectangle(0, 0, width, height, outline="", fill="grey", tags="rect")
            self.canvas.create_oval(-height / 2, 0, height / 2, height, outline="black", fill="white", tags="circle")

        # bind button to toggle func
        self.canvas.bind("<Button-1>", self.toggle)

        # label for switch
        self.label = tk.Label(self, text=text, foreground="white", background="black")
        self.label.pack(side="left", padx=10)

    # toggle func when clicked
    def toggle(self, event=None):
        self.is_on.set(not self.is_on.get())
        x1, y1, x2, y2 = self.canvas.coords("rect")
        # if True, move circle to right and update color
        if self.is_on.get():
            self.canvas.move("circle", x2, 0)
            self.canvas.itemconfig("rect", fill="blue")
        else:  # otherwise move back and update color
            self.canvas.move("circle", -x2, 0)
            self.canvas.itemconfig("rect", fill="grey")


# yolo function for identification and tracking
def run_yolo():
    return


# for raspberry pi
def live_video():
    return


# selecting a video to play
def select_video():
    # get capture and check if running
    global cap
    # if its running, stop running so a new video can be selected
    if cap is not None:
        cap.release()
    # get the video file path with file explorer
    file_path = tk.filedialog.askopenfilename(title="Select a Video", filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")))
    # if the path exists, set the capture to the new video and begin updating
    if file_path:
        cap = cv2.VideoCapture(file_path)
        update_video()


# updates the video frame by frame
def update_video():
    # get capture and check if opened
    global cap
    if cap is None or not cap.isOpened():
        return

    # if opened, try and get next frame
    ret, frame = cap.read()
    if ret:
        # if running, convert frame to RGB image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        # resize the image to match video frame size
        img_resized = img.resize((1440, 810), Image.Resampling.LANCZOS)
        # convert to tk image and add into gui
        imgtk = ImageTk.PhotoImage(image=img_resized)
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)
        # after 33 ms (for 30 fps) update to next frame
        root.after(33, update_video)
    else:
        cap.release()


def open_settings():
    # base window
    settings_window = Toplevel(root, bg="black")
    settings_window.title("Settings")
    settings_window.geometry('1920x1080')

    # notification side
    notif_frame = Frame(settings_window, width=960, bg="black")
    Label(notif_frame, text="Notification Settings", foreground="white", background="black", font=("Helvetica", 20)).pack(pady=10)
    Label(notif_frame, text="Be Notified When:", foreground="white", background="black").pack(pady=10)
    ToggleSwitch(notif_frame, text="Movement Detected", variable=movement_notif).pack(pady=10)
    ToggleSwitch(notif_frame, text="Low Battery", variable=battery_notif).pack(pady=10)
    notif_frame.pack_propagate(False)
    notif_frame.pack(side="right", fill="y")

    # animal settings side
    animal_frame = Frame(settings_window, width=960, bg="black")
    Label(animal_frame, text="Animal Settings", foreground="white", background="black", font=("Helvetica", 20)).pack(pady=10)
    Label(animal_frame, text="Select Animal(s) to Track", foreground="white", background="black").pack(pady=10)
    animal_frame.pack_propagate(False)
    animal_frame.pack(side="left", fill="y")

    for animal in animals:
        tk.Checkbutton(animal_frame, text=animal, variable=animal_checks[animal], fg="black").pack(pady=5)


# draw function
def draw_graph():
    # clear the graph and set x axis format
    ax.clear()
    ax.xaxis.set_major_locator(mpl.dates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%m-%d %H:%M'))
    # set axis settings
    plt.xticks(rotation=45)
    plt.xlabel("Timestamp")
    plt.ylabel("Counts")
    # max var for y range
    max = 0

    # for each animal
    for animal, info in animal_counts.items():
        # if animal is selected to track
        if animal_checks[animal].get() == 1:
            # if count of animal is bigger than max, update
            if animal_counts[animal]['count'] > max:
                max = animal_counts[animal]['count']

            # get the dates in matplotlib format, range of dates, then graph
            dates = [datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in info['times']]
            counts = range(1, len(dates) + 1)
            ax.plot_date(dates, counts, '-o', label=animal)

    # update y range with step = 1
    plt.yticks(np.arange(0, max+1, step=1))

    # add legend to plot, format the dates and draw to canvas
    ax.legend()
    fig.autofmt_xdate()
    canvas.draw()


# adds animal to counter // ADD FUNCTIONALITY WITH TRACKER
def add_animal():
    animal = selected_animal.get()
    if animal:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if animal in animal_counts:
            animal_counts[animal]['count'] += 1
            animal_counts[animal]['times'].append(timestamp)
        else:
            animal_counts[animal] = {'count': 1, 'times': [timestamp]}
        update_counts()


# update counter label
def update_counts():
    draw_graph()
    display_text = "Counts:\n"
    for animal, info in animal_counts.items():
        display_text += f"{animal}: {info['count']} - Last added: {info['times'][-1]}\n"
    counts_label.config(text=display_text)


# upload video button
upload_button = tk.Button(bottom_frame, text="Upload Video", command=select_video)
upload_button.pack(side='left', padx=10, pady=10)

# live feed button
live_button = tk.Button(bottom_frame, text="Live Feed", command=live_video)
live_button.pack(side='left', padx=10, pady=10)

# add a count to graph
add_button = tk.Button(bottom_frame, text="Add Count", command=add_animal)
add_button.pack(side='top', pady=10)

# settings button
settings_icon = tk.PhotoImage(file="settings_button.png")
settings_button = tk.Button(right_frame, image=settings_icon, command=open_settings, bg="black")
settings_button.pack(side="right", pady=10)

root.title("CAPSTONE")
root.geometry('1920x1080')
root.mainloop()
