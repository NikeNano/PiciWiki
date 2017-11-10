import tkinter as tk
import threading
import imageio
from PIL import Image, ImageTk

video_name = "TestVideo/SampleVideo.mp4" #This is your video file path
video = imageio.get_reader(video_name)
print("start")
def stream(label):

    frame = 0
    for image in video.iter_data():
        frame += 1                                    #counter to save new frame number
        image_frame = Image.fromarray(image)          
        image_frame.save('FRAMES/frame_%d.png' % frame)      #if you need the frame you can save each frame to hd
        frame_image = ImageTk.PhotoImage(image_frame)
        label.config(image=frame_image)
        label.image = frame_image
        print("here")
        #if frame == 40: break                         #after 40 frames stop, or remove this line for the entire video

if __name__ == "__main__":

    root = tk.Tk()
    my_label = tk.Label(root)
    my_label.pack()
    thread = threading.Thread(target=stream, args=(my_label,))
    thread.daemon = 1
    thread.start()
    root.mainloop()