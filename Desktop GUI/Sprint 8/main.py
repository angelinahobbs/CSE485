import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from PIL import ImageTk
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Frame, Label, ttk, Toplevel
import cv2
import torch
from ultralytics import YOLO

# root frame
root = tk.Tk()

# container for video frame and bottom frame
left_container = Frame(root, width=960, height=720)
left_container.pack(side='left', expand=True, fill='both')

# video frame
video_frame = Frame(left_container, bg="black", width=960, height=540)
video_frame.pack_propagate(False)
video_frame.pack(side='top')
video_label = Label(video_frame)
video_label.pack(fill='both', expand=True)

# bottom left frame
bottom_frame = Frame(left_container, bg="black", width=960, height=180)
bottom_frame.pack_propagate(False)
bottom_frame.pack(side='top')

# graph frame with settings button
right_frame = Frame(root, bg='black', width=320)
right_frame.pack(side="right", fill="y")
right_frame.pack_propagate(False)

# counting animal variables
animal_counts = {}
animals = ["HUMAN", "DOG", "CAT", "DEER", "DUCK", "EAGLE", "HORSE", "RABBIT", "SNAKE", "FOX", "SQUIRREL", "BEAR",
           "HEDGEHOG", "LYNX", "MOUSE", "TURTLE"]
animal_checks = {}
for animal in animals:
    animal_checks[animal] = tk.IntVar(value=1)  # sets all animals to being tracked

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

# Different model paths
pets_path = 'weights/best.onnx'
wildlife_path = 'WildLifeModel/weights/last.pt'
current_path = pets_path

# new frame for model controls
model_switch_frame = Frame(bottom_frame, bg='black')
model_switch_frame.pack(side='top', fill='x', pady=25)

# label for model
model_label = Label(model_switch_frame, text="Current Model: Pets", bg='black', fg='white')
model_label.pack(side='top', fill='x')

# YOLO STUFF, creates both models and sends to gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pet_model = YOLO(pets_path)
# pet_model = pet_model.to(device)
wildlife_model = YOLO(wildlife_path)
wildlife_model = wildlife_model.to(device)
model = pet_model
threshold = 0.6

executor = ProcessPoolExecutor(max_workers=4)
results = None
file_path = None


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


# for raspberry pi
def live_video():
    return


# selecting a video to play
def select_video():
    # get capture and check if running
    global cap, results, model, device, file_path
    # if its running, stop running so a new video can be selected
    if cap is not None:
        cap.release()
    # get the video file path with file explorer
    file_path = tk.filedialog.askopenfilename(title="Select a Video", filetypes=(
        ("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")))
    # if the path exists, set the capture to the new video and begin updating
    if file_path:
        results = model(file_path, stream=True)
        cap = cv2.VideoCapture(file_path)
        update_video()


# updates the video frame by frame
def update_video():
    global cap, results, animal_counts
    if cap is None or not cap.isOpened():
        return

    # get next frame and check if it exists
    ret, frame = cap.read()
    if not ret:
        return

    if ret:
        detections = next(results, None)  # get next set of results

        # check if there are boxes to draw
        if detections.boxes.xyxy.numel() != 0:
            count = {}
            # for each box that exists
            for box in detections.boxes:
                # if the confidence is above 0.8 draw the box and add to graph
                if box.conf[0] > threshold:
                    # get animal type and box coords
                    class_id = box.data.tolist()[0][5]
                    bbox = box.xyxy
                    print(box.conf[0])
                    x1 = int(bbox[0, 0].item())
                    y1 = int(bbox[0, 1].item())
                    x2 = int(bbox[0, 2].item())
                    y2 = int(bbox[0, 3].item())

                    if count.keys().__contains__(str(detections.names[int(class_id)].upper())) is False:
                        count[str(detections.names[int(class_id)].upper())] = 1
                    else:
                        count[str(detections.names[int(class_id)].upper())] += 1

                    # draw the box on the frame and put the label
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, detections.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

            add_animal(count)

        # convert frame to tkinter image, resize and put onto gui
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_resized = img.resize((960, 540), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img_resized)
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

        root.after(1, update_video)  # call update video after 33 ms for 30 fps?
    else:
        cap.release()


# opens settings page
def open_settings():
    # base window
    settings_window = Toplevel(root, bg="black")
    settings_window.title("Settings")
    settings_window.geometry('1280x720')

    # notification side
    notif_frame = Frame(settings_window, width=960, bg="black")
    Label(notif_frame, text="Notification Settings", foreground="white", background="black",
          font=("Helvetica", 20)).pack(pady=10)
    Label(notif_frame, text="Be Notified When:", foreground="white", background="black").pack(pady=10)
    ToggleSwitch(notif_frame, text="Movement Detected", variable=movement_notif).pack(pady=10)
    ToggleSwitch(notif_frame, text="Low Battery", variable=battery_notif).pack(pady=10)
    notif_frame.pack_propagate(False)
    notif_frame.pack(side="right", fill="y")

    # animal settings side
    animal_frame = Frame(settings_window, width=960, bg="black")
    Label(animal_frame, text="Animal Settings", foreground="white", background="black", font=("Helvetica", 20)).pack(
        pady=10)
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

    max_count = 0

    # for each animal
    for animal, info in animal_counts.items():
        if animal_checks[animal].get() == 1:
            # get timestamps and counts
            times, counts = zip(*info['times']) if info['times'] else ([], [])
            dates = [datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in times]

            # update max count for y-axis limit
            max_count = max(max_count, max(counts, default=0))

            # plot counts for timestamps
            ax.plot_date(dates, counts, '-o', label=animal)

    # set y-axis limits
    plt.ylim((0, max_count + 1))

    # add legend to plot, format the dates and draw to canvas
    ax.legend()
    fig.autofmt_xdate()
    canvas.draw()


# adds animal to counter // ADD FUNCTIONALITY WITH TRACKER
def add_animal(detected_animals):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # reset counts to 0 for all animals initially
    for animal in animal_counts:
        animal_counts[animal]['count'] = 0

    # update counts for each detected animal
    for animal in detected_animals:
        if animal in animal_counts:
            animal_counts[animal]['count'] = detected_animals[animal]
        else:
            animal_counts[animal] = {'count': detected_animals[animal], 'times': []}

    # add timestamp for each animal
    for animal in animal_counts:
        animal_counts[animal]['times'].append((timestamp, animal_counts[animal]['count']))

    # re draw graph
    draw_graph()


def switch_models():
    global current_path, model, results, device, file_path, cap

    if current_path == pets_path:
        current_path = wildlife_path
        model_label.config(text="Current Model: Wildlife")
        model = wildlife_model
    else:
        current_path = pets_path
        model_label.config(text="Current Model: Pets")
        model = pet_model

    if cap is not None:
        cap.release()
        cap = None

    # prevents video from not playing
    root.after(500, resume)


def resume():
    global results, cap, model
    results = model(file_path, stream=True)
    cap = cv2.VideoCapture(file_path)
    root.after(10, update_video)


# Make sure to shut down the executor when the application closes
def on_close():
    executor.shutdown(wait=True)
    root.destroy()


# upload video button
upload_button = tk.Button(bottom_frame, text="Upload Video", command=select_video)
upload_button.pack(side='left', padx=10, pady=10)

# live feed button
live_button = tk.Button(bottom_frame, text="Live Feed", command=live_video)
live_button.pack(side='left', padx=10, pady=10)

# model button
model_button = tk.Button(model_switch_frame, text="Switch Models", command=switch_models)
model_button.pack(pady=10)

# settings button
settings_icon = tk.PhotoImage(file="settings_button.png")
settings_button = tk.Button(right_frame, image=settings_icon, command=open_settings, bg="black")
settings_button.pack(side="right", pady=10)

root.title("CAPSTONE")
root.geometry('1280x720')
root.mainloop()
