import numpy as np
import cv2
import datetime
import queue
from threading import Thread

# global variables
stop_thread = False             # controls thread execution


def start_capture_thread(cap, queue):
    # global stop_thread
	i=0
    # continuously read frames from the camera
	while True:
		ret, img = cap.read()
        # if not ret:
        #     break 
		queue.put(img)
		# cv2.imwrite('Images/frame{:d}.jpg'.format(i), img)
		i +=1
        # if (stop_thread):
        #     break


def get_queue():
    images=[]
    global stop_thread

    # create display window
    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)

    # initialize webcam capture object
    cap = cv2.VideoCapture('test.mp4')
    # cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

    # retrieve properties of the capture object
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    print('* Capture width:', cap_width)
    print('* Capture height:', cap_height)
    print('* Capture FPS:', cap_fps)

    # create a queue
    frames_queue = queue.Queue(maxsize=328)#328
    
    # start the capture thread: reads frames from the camera (non-stop) and stores the result in img
    t = Thread(target=start_capture_thread, args=(cap, frames_queue,), daemon=True) # a deamon thread is killed when the application exits
    t.start()

    # initialize time and frame count variables
    last_time = datetime.datetime.now()
    frames = 0
    cur_fps = 0
    i=0
    print(frames_queue.qsize())


    while(True):
        if (frames_queue.empty()):
            continue

        print(frames_queue.qsize())
        print(frames_queue)
        
        # if i%5 !=0 :
        #     _ = frames_queue.get()
        #     i+=1
        #     continue

        # # blocks until the entire frame is read
        # frames += 1

        # # # measure runtime: current_time - last_time
        # delta_time = datetime.datetime.now() - last_time
        # elapsed_time = delta_time.total_seconds()

        # # compute fps but avoid division by zero
        # if (elapsed_time != 0):
        #     cur_fps = np.around(frames / elapsed_time, 1)

        # retrieve an image from the queue
        img = frames_queue.get()
        images.append(img)
        # cv2.imwrite('Images/frame{:d}.jpg'.format(i), img)
        i+=1
        frames += 1
       
        # draw FPS text and display image
        if (img is not None):
            cv2.putText(img, 'FPS: ' + str(cur_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("webcam", img)

        # wait 1ms for ESC to be pressed
        key = cv2.waitKey(1)
        if (key == 27):
            stop_thread = True
            break
        if frames==328:#328
            print(len(images))
            cv2.destroyAllWindows()
            cap.release()
            # print(images)
            print('end trial')
            return images


# get_queue()