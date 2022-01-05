from annotation_by_yolo import get_data_prepared
from read_data import get_queue
from segmentation import Segmentation
from get_prepared_data_multi import do_All_Prepared
import matplotlib.pyplot as plt

images_try = get_queue()
annotation = get_data_prepared(images_try)
traj_data, person_box_data, other_box_data, multifuture_data = do_All_Prepared(annotation)    
seg = Segmentation(model_path='deeplabv3_xception_ade20k_train/frozen_inference_graph.pb')
seg_imgs = seg.run_model(images_try) 

