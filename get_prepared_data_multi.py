# coding=utf-8
"""Given the final dataset or the anchor dataset, compile prepared data."""

import argparse
import json
import os
import operator
import pickle
import numpy as np
from tqdm import tqdm

def do_All_Prepared(o_bboxes,drop_frame=12):#12 or 10
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def convert_bbox(bbox):
        x, y, w, h = bbox
        return [x, y, x + w, y + h]

    def get_feet(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, y2)


    def get_frame_data(o_bboxes):
        # bboxes = filter_neg_boxes(o_bboxes)
        bboxes = o_bboxes
        frame_data = {}  # frame_idx -> data
        for one in bboxes:
            if one["frame_id"] not in frame_data:
                frame_data[one["frame_id"]] = []
            frame_data[one["frame_id"]].append(one)

        return frame_data

    def filter_neg_boxes(bboxes):
        new_bboxes = []
        for bbox in bboxes:
            x, y, w, h = bbox["bbox"]
            coords = x, y, x + w, y + h
            bad = False
        for o in coords:
            if o < 0:
                bad = True
        if not bad:
            new_bboxes.append(bbox)
        return new_bboxes
    
    obs_length = 1
    class2classid = {"Person": 0,"Car": 1,}
    videoname = 1
    # traj_path = os.path.join('.', f'./traj_2.5fps/{videoname}')
    # person_box_path = os.path.join('.', f'./anno_person_box/{videoname}')
    # other_box_path = os.path.join('.', f'./anno_other_box/{videoname}')
    
    drop_frame = drop_frame
    
    # multi-future pred starts at 124/102
    # we want the obs to be 3.2 sec long
    if drop_frame == 12:
        frame_range = (40, 125) # range(40, 125, 12)
        start_frame, end_frame = 40, 125
    else:
        frame_range = (32, 103),  # range(32, 103, 10)
        start_frame, end_frame = 32, 103
    
    # 1. first pass, get the needed frames
    frame_data = get_frame_data(o_bboxes)
    frame_idxs = sorted(frame_data.keys())
    # print(frame_idxs)
    # assert frame_idxs[0] == 0
    needed_frame_idxs = frame_idxs[start_frame::drop_frame]
    
    assert len(needed_frame_idxs) > obs_length, (needed_frame_idxs, start_frame)
    obs_frame_idxs = needed_frame_idxs[:obs_length]
    
    # 2. gather data for each frame_idx, each person_idx
    traj_data = []  # [frame_idx, person_idx, x, y]
    person_box_data = {}  # (frame_idx, person_id) -> boxes
    other_box_data = {}  # (frame_idx, person_id) -> other boxes + boxclasids
    obs_x_agent_traj = []
    for frame_idx in obs_frame_idxs:
        box_list = frame_data[frame_idx]
        # filter out negative boxes
        box_list.sort(key=operator.itemgetter("track_id"))
        for i, box in enumerate(box_list):
            class_name = box["class_name"]
            track_id = box["track_id"]
            is_x_agent = box["is_x_agent"]
            bbox = convert_bbox(box["bbox"])
            if class_name == "Person":
                new_frame_idx = frame_idx - start_frame
                person_key = "%d_%d" % (new_frame_idx, track_id)

                x, y = get_feet(bbox)
                traj_data.append((new_frame_idx, float(track_id), x, y))
                if int(is_x_agent) == 1:
                    obs_x_agent_traj.append((new_frame_idx, float(track_id), x, y))

                person_box_data[person_key] = bbox

                all_other_boxes = [convert_bbox(box_list[j]["bbox"])
                                    for j in range(len(box_list)) if j != i]
                all_other_boxclassids = [class2classid[box_list[j]["class_name"]]
                                            for j in range(len(box_list)) if j != i]

                other_box_data[person_key] = (all_other_boxes, all_other_boxclassids)

    # now we save all the multi future paths for all agent.
    multifuture_data = {}  # videoname -> {"x_agent_traj", "all_boxes"}
    frame_data = get_frame_data(o_bboxes)
    frame_idxs = sorted(frame_data.keys())
    # assert frame_idxs[0] == 2
    # 1. first pass, get the needed frames
    needed_frame_idxs = frame_idxs[start_frame::drop_frame]

    assert len(needed_frame_idxs) > obs_length, (needed_frame_idxs, start_frame)
    pred_frame_idxs = needed_frame_idxs[obs_length:]

    x_agent_traj = []
    all_boxes = []
    for frame_idx in pred_frame_idxs:
        box_list = frame_data[frame_idx]
        box_list.sort(key=operator.itemgetter("track_id"))
        for i, box in enumerate(box_list):
            class_name = box["class_name"]
            track_id = box["track_id"]
            is_x_agent = box["is_x_agent"]
            bbox = convert_bbox(box["bbox"])

            new_frame_idx = frame_idx - start_frame
            if int(is_x_agent) == 1:
                x, y = get_feet(bbox)
                x_agent_traj.append((new_frame_idx, track_id, x, y))

            all_boxes.append((new_frame_idx, class_name, is_x_agent,
                            track_id, bbox))
    multifuture_data[videoname] = {
        "x_agent_traj": x_agent_traj, # future
        "all_boxes": all_boxes,
        "obs_traj": obs_x_agent_traj,
    }
    print(multifuture_data)

    return traj_data, person_box_data, other_box_data, multifuture_data

# o_bboxes = [{'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 2, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [483, 416, 41, 81], 'frame_id': 2, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1154, 155, 19, 70], 'frame_id': 2, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [516, 438, 43, 83], 'frame_id': 3, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [484, 416, 40, 81], 'frame_id': 3, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1153, 154, 21, 67], 'frame_id': 3, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 4, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [484, 416, 40, 81], 'frame_id': 4, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1153, 153, 21, 66], 'frame_id': 4, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 5, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [484, 416, 40, 81], 'frame_id': 5, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1155, 151, 20, 67], 'frame_id': 5, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 6, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [483, 416, 41, 81], 'frame_id': 6, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1155, 150, 21, 66], 'frame_id': 6, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 7, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [483, 416, 41, 81], 'frame_id': 7, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1156, 148, 20, 65], 'frame_id': 7, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1386, 87, 20, 46], 'frame_id': 7, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 8, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [483, 416, 41, 81], 'frame_id': 8, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1157, 145, 20, 65], 'frame_id': 8, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1386, 88, 18, 49], 'frame_id': 8, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 9, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 42, 81], 'frame_id': 9, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1158, 144, 20, 62], 'frame_id': 9, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1384, 89, 22, 52], 'frame_id': 9, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 10, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 10, 'track_id': 2}, 
# {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1161, 140, 18, 65], 'frame_id': 10, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1386, 93, 21, 50], 'frame_id': 10, 'track_id': 
# 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 11, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 11, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1162, 139, 18, 62], 'frame_id': 11, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1385, 96, 21, 49], 'frame_id': 11, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 12, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 12, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1162, 137, 21, 62], 'frame_id': 12, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1384, 98, 19, 48], 'frame_id': 12, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 13, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 13, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1164, 136, 19, 60], 'frame_id': 13, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1381, 99, 21, 52], 'frame_id': 13, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 14, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 14, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1165, 134, 19, 59], 'frame_id': 14, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1378, 103, 21, 52], 'frame_id': 14, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 15, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 15, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1165, 130, 20, 61], 'frame_id': 15, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1376, 105, 
# 21, 52], 'frame_id': 15, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 16, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 16, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1166, 128, 19, 61], 'frame_id': 16, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 17, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 17, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1167, 127, 19, 56], 'frame_id': 17, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 18, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 18, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1168, 124, 20, 58], 'frame_id': 18, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1372, 109, 22, 53], 'frame_id': 18, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 19, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 
# 1, 'bbox': [482, 416, 41, 81], 'frame_id': 19, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1170, 121, 18, 59], 'frame_id': 19, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1372, 111, 22, 52], 'frame_id': 19, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 20, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 20, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1170, 117, 19, 61], 'frame_id': 20, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 21, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 21, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 22, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 22, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 23, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 23, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 24, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 24, 'track_id': 2}, {'class_name': 
# 'Person', 'is_x_agent': 1, 'bbox': [1173, 109, 20, 60], 'frame_id': 24, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 25, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 25, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1174, 106, 20, 61], 'frame_id': 25, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1368, 130, 21, 56], 'frame_id': 25, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 26, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 26, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1175, 103, 20, 59], 'frame_id': 26, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1368, 132, 23, 57], 'frame_id': 26, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 27, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 27, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1173, 99, 23, 61], 'frame_id': 27, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1370, 136, 22, 56], 'frame_id': 27, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 28, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 28, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1174, 96, 22, 60], 'frame_id': 28, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1370, 140, 22, 57], 'frame_id': 28, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 29, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 29, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1174, 95, 23, 59], 'frame_id': 29, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1370, 145, 23, 58], 'frame_id': 29, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 30, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 30, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1172, 92, 25, 61], 'frame_id': 30, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1371, 150, 22, 56], 'frame_id': 30, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 31, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 31, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1173, 90, 25, 61], 'frame_id': 31, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1366, 156, 26, 55], 'frame_id': 31, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 32, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 32, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1175, 89, 23, 59], 'frame_id': 32, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1364, 159, 24, 56], 'frame_id': 32, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 33, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 33, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1178, 87, 21, 58], 'frame_id': 33, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1356, 163, 27, 58], 'frame_id': 33, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 34, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 34, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1180, 84, 20, 58], 'frame_id': 34, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1350, 166, 24, 58], 'frame_id': 34, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 35, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 35, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1181, 83, 20, 57], 'frame_id': 35, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1339, 169, 25, 57], 'frame_id': 35, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 36, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 36, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1181, 81, 20, 57], 'frame_id': 36, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1328, 169, 22, 58], 'frame_id': 36, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 37, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 37, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1183, 80, 17, 56], 'frame_id': 37, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1308, 170, 29, 60], 'frame_id': 37, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 38, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 38, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1184, 78, 17, 55], 'frame_id': 38, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1299, 170, 21, 62], 'frame_id': 38, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 39, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 39, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1185, 77, 17, 53], 'frame_id': 39, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1277, 169, 30, 62], 'frame_id': 39, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 40, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 40, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1186, 74, 17, 54], 'frame_id': 40, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1264, 166, 25, 66], 'frame_id': 40, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 41, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 41, 'track_id': 
# 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1187, 72, 16, 54], 'frame_id': 41, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1244, 165, 32, 64], 'frame_id': 41, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 42, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 42, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1188, 70, 16, 53], 'frame_id': 42, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1232, 164, 24, 65], 'frame_id': 42, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 43, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 43, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1188, 68, 17, 53], 'frame_id': 43, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1216, 161, 22, 65], 'frame_id': 43, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 44, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 44, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1188, 66, 17, 53], 'frame_id': 44, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1193, 160, 30, 66], 'frame_id': 44, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 45, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 
# 'frame_id': 45, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1188, 64, 16, 53], 'frame_id': 45, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1177, 158, 24, 63], 'frame_id': 45, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 46, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 41, 81], 'frame_id': 46, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1188, 62, 17, 54], 'frame_id': 46, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1160, 155, 21, 63], 'frame_id': 46, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 47, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 
# 416, 41, 81], 'frame_id': 47, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1189, 60, 16, 55], 'frame_id': 47, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1144, 155, 23, 64], 'frame_id': 47, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 48, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 416, 42, 80], 'frame_id': 48, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1189, 59, 17, 54], 'frame_id': 48, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1128, 154, 23, 63], 'frame_id': 48, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 44, 83], 'frame_id': 49, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 
# 'bbox': [483, 415, 40, 81], 'frame_id': 49, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1188, 57, 17, 54], 'frame_id': 49, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 
# 1, 'bbox': [1115, 151, 20, 63], 'frame_id': 49, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 45, 83], 'frame_id': 50, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [483, 415, 40, 81], 'frame_id': 50, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1189, 56, 15, 54], 'frame_id': 50, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1096, 150, 25, 62], 'frame_id': 50, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 45, 83], 'frame_id': 51, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 415, 42, 81], 'frame_id': 51, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1189, 54, 15, 52], 'frame_id': 51, 'track_id': 3}, {'class_name': 'Person', 
# 'is_x_agent': 1, 'bbox': [1081, 149, 21, 62], 'frame_id': 51, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [515, 438, 46, 83], 'frame_id': 52, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [483, 415, 40, 80], 'frame_id': 52, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1189, 53, 16, 52], 'frame_id': 52, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1065, 148, 23, 62], 'frame_id': 52, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [516, 438, 45, 83], 'frame_id': 53, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [483, 415, 40, 80], 'frame_id': 53, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1189, 51, 16, 52], 'frame_id': 53, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1050, 146, 23, 62], 'frame_id': 53, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [486, 430, 78, 86], 'frame_id': 53, 'track_id': 6}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [516, 438, 45, 83], 'frame_id': 54, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [483, 414, 40, 81], 'frame_id': 54, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1190, 49, 16, 52], 'frame_id': 54, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1037, 145, 23, 60], 'frame_id': 54, 'track_id': 4}, 
# {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [486, 430, 78, 86], 'frame_id': 54, 'track_id': 6}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [516, 438, 45, 83], 'frame_id': 55, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 415, 42, 80], 'frame_id': 55, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1190, 48, 16, 51], 'frame_id': 55, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1022, 145, 24, 60], 'frame_id': 55, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [486, 430, 77, 86], 'frame_id': 55, 'track_id': 6}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [516, 438, 45, 83], 'frame_id': 56, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 415, 42, 80], 'frame_id': 56, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1189, 47, 18, 50], 'frame_id': 56, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1012, 142, 20, 61], 'frame_id': 56, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [516, 438, 45, 83], 'frame_id': 57, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [482, 415, 42, 80], 'frame_id': 
# 57, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1189, 47, 18, 50], 'frame_id': 57, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [996, 141, 24, 59], 'frame_id': 57, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [516, 438, 45, 83], 'frame_id': 58, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [483, 415, 40, 80], 'frame_id': 58, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1190, 46, 18, 51], 'frame_id': 58, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [985, 142, 22, 57], 'frame_id': 58, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [485, 429, 79, 88], 'frame_id': 58, 'track_id': 6}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [516, 438, 45, 83], 
# 'frame_id': 59, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [483, 416, 40, 79], 'frame_id': 59, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1191, 44, 17, 51], 'frame_id': 59, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [971, 139, 24, 60], 'frame_id': 59, 'track_id': 4}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [485, 429, 79, 88], 'frame_id': 59, 'track_id': 6}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [516, 438, 45, 83], 'frame_id': 60, 'track_id': 1}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [483, 416, 
# 40, 79], 'frame_id': 60, 'track_id': 2}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [1191, 42, 18, 51], 'frame_id': 60, 'track_id': 3}, {'class_name': 'Person', 'is_x_agent': 1, 'bbox': [959, 137, 21, 59], 'frame_id': 60, 'track_id': 4}]

# do_All_Prepared(o_bboxes)