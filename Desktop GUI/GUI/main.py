import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import ImageTk
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Frame, Label, ttk, Toplevel
import cv2

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

# bottomn left frame
unknown_frame = Frame(left_container, bg="black", width=960, height=180)
unknown_frame.pack_propagate(False)
unknown_frame.pack(side='top')

# counter frame
infoFrame = Frame(root, bg='green', width=320)
infoFrame.pack(side="right", fill="y")
infoFrame.pack_propagate(False)

# dropdown label
dropdown_label = Label(infoFrame, text="Select an Animal to Track:", bg='green', fg='white')
dropdown_label.pack(pady=10)

# dropdown list of animals
animals = ["Bear", "Wolf", "Coyote", "Deer", "Fox", "Squirrel", "Human"]
animal_checks = {}
for animal in animals:
    animal_checks[animal] = tk.IntVar()

selected_animal = tk.StringVar()
animal_dropdown = ttk.Combobox(infoFrame, textvariable=selected_animal, values=animals)
animal_dropdown.pack(pady=10)
animal_counts = {}

# Label for displaying counts
counts_label = Label(unknown_frame, text="Counts:\n", justify=tk.LEFT)
counts_label.pack(side='right', padx=10, pady=10)

# graph Frame
graph_frame = Frame(infoFrame)
graph_frame.pack(side='bottom', fill='both', expand=True, pady=20)

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

cap = None


def select_video():
    global cap
    file_path = tk.filedialog.askopenfilename(title="Select a Video", filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")))
    if file_path:
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(file_path)
        update_video()


def update_video():
    if cap is None or not cap.isOpened():
        return
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_resized = img.resize((960, 540), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img_resized)
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)
        root.after(2, update_video)
    else:
        cap.release()


def open_settings():
    # base window
    settings_window = Toplevel(root, bg="black")
    settings_window.title("Settings")
    settings_window.geometry('1280x720')

    # notification side
    notif_frame = Frame(settings_window, width=640, bg="red")
    Label(notif_frame, text="Notification Settings").pack(pady=10)
    notif_frame.pack_propagate(False)
    notif_frame.pack(side="right", fill="y")

    # animal settings side
    animal_frame = Frame(settings_window, width=640, bg="blue")
    Label(animal_frame, text="Animal Settings").pack(pady=10)
    animal_frame.pack_propagate(False)
    animal_frame.pack(side="left", fill="y")

    for animal in animals:
        tk.Checkbutton(animal_frame, text=animal, variable=animal_checks[animal], fg="blue").pack(pady=5)


# draw function
def draw_graph():
    ax.clear()
    ax.xaxis.set_major_locator(mpl.dates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.ylabel("Counts")

    for animal, info in animal_counts.items():
        if animal_checks[animal].get() == 1:
            dates = [datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in info['times']]
            counts = range(1, len(dates) + 1)  # Create a list of 1 to N for each timestamp
            ax.plot_date(dates, counts, '-o', label=animal)

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
upload_button = tk.Button(unknown_frame, text="Upload Video", command=select_video)
upload_button.pack(side='left', padx=10, pady=10)

add_button = tk.Button(infoFrame, text="Add Count", command=add_animal)
add_button.pack(pady=10)

settings_icon = tk.PhotoImage(file="settings_button.png")
settings_button = tk.Button(infoFrame, image=settings_icon, command=open_settings, bg="green")
settings_button.pack(side="bottom", pady=10)

root.title("CAPSTONE")
root.geometry('1280x720')
root.mainloop()
