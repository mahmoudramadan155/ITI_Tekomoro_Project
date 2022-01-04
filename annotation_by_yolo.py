# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# limit the number of cpus used by high performance libraries
import sys
sys.path.insert(0, './yolov5')

import numpy as np
from yolov5.utils.augmentations import letterbox
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import cv2
import torch
from segmentation import Segmentation
from read_data import get_queue
from get_prepared_data_multi import do_All_Prepared

def get_data_prepared(captured_images):
    annotation=[]
    target_resolution = (1920, 1080)
    images_lst = []
    yolo_weights= 'yolov5s.pt'
    config_deepsort= 'deep_sort_pytorch/configs/deep_sort.yaml'
    imgsz= [640]
    imgsz *= 2 if len(imgsz) == 1 else 1
    half = False
    dnn=False
    device=''
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    # attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn)
    stride, names, pt, jit = model.stride, model.names, model.pt, model.jit
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()
        

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    for frame_idx, image_try in enumerate(captured_images):

        # Padded resize
        frame_img = letterbox(image_try, 640, 32, True)[0]

        # Convert
        frame_img = frame_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        frame_img = np.ascontiguousarray(frame_img)


        img, im0s, vid_cap, s = frame_img, image_try, None, ''
        # for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        #     # LOGGER.info(f"{vid_cap}")

        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(img, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred,.4,.5, classes=[0,2],agnostic=False,max_det=1000)
        dt[2] += time_sync() - t3
        
        # LOGGER.info(f"DIDN'T DETECT {pred} !!!")
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            
            s += '%gx%g ' % img.shape[2:]  # print string
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                
                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)): 
                        
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        x1 = output[0]
                        y1 = output[1]
                        x2 = output[2] - x1
                        y2 = output[3] - y1
                        
                        list_box = [x1,y1,x2,y2]
                        if names[c] == 'person' or names[c] == 'car':
                            is_agent = 1
                        else:
                            is_agent = 0
                        output_text = {'class_name':str(names[c]).capitalize(),'is_x_agent':is_agent,'bbox':list_box,'frame_id':frame_idx,'track_id':id}
                        
                        annotation.append(output_text) 
                        
                        
            else:
                deepsort.increment_ages()
           
            h,w = im0s.shape[0:2]
            rotate_90_clockwise = False
            
            if h > w:
                rotate_90_clockwise = True

            if rotate_90_clockwise:
                im0s = cv2.transpose(im0s)
            images_lst.append(cv2.resize(im0s, target_resolution)) 
    traj_data, person_box_data, other_box_data, multifuture_data = do_All_Prepared(annotation)
    #get segmented images
    seg = Segmentation(model_path='deeplabv3_xception_ade20k_train/frozen_inference_graph.pb')
    seg_imgs = seg.run_model(captured_images) 
    print('end trial2')
    return annotation, traj_data, person_box_data, other_box_data, multifuture_data, seg_imgs

images_try = get_queue()
get_data_prepared(images_try)