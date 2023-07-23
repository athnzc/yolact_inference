import json 
from pycocotools import mask
import matplotlib.pyplot as plt 
import numpy as np
import cv2 
import os 

# mask_file which contains the mask annotations extracted from yolact
mask_file = '/mnt/266A47CE6A479A07/athena/yolo/results_2023_07_17/sc03_ch01/mask2.json'
# mask_file exported from VGG in COCO format, which contains the filenames of the images 
# used for inference with their id
dataset_file = '/mnt/266A47CE6A479A07/athena/yolo/results_2023_07_17/sc03_ch01/sc03_ch01_coco.json'
img_save_folder = '/mnt/266A47CE6A479A07/athena/yolo/results_2023_07_17/sc03_ch01/figures'
results_save_folder = '/mnt/266A47CE6A479A07/athena/yolo/results_2023_07_17/sc03_ch01'
with open(mask_file, 'r') as f:
    data = json.load(f)

with open(dataset_file, 'r') as f:
    data2 = json.load(f)

# DATA SHOULD BE SAVED AS SUCH: 
# obj = {
#               "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
#               "bbox_mode": BoxMode.XYXY_ABS,
#               "segmentation": [poly],
#               "category_id": 0
#           }
#           objs.append(obj)
#print(data)
#print(data[0]['segmentation'])
#new_data = data
new_data = []
count = 0
previous_id = data[0]["image_id"]

for i in range(len(data)):
    decoded = mask.decode(data[i]['segmentation'])
    decoded = decoded.astype('uint8') * 255
    #print(decoded.shape)
    contours, hierarchy = cv2.findContours(decoded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #hierarchy: array with as many rows as the detected contours
    # each row is the 0-based index of: next_in_same_hierarchy previous_in_same_hierarchy first_child parent
    
    #print(hierarchy)
    polygons = []

    for object in contours:
        coords = []
        #print(object)
        # calculate bbox 
        #[np.min(px), np.min(py), np.max(px), np.max(py)] 
        for point in object:
            
            coords.append(int(point[0][0]))
            coords.append(int(point[0][1]))

        polygons.append(coords)

    img_id = data[i]["image_id"]

    #find corresponding filename 
    for item in data2["images"]:
        if item["id"] == img_id:
            file_name = item["file_name"]

    obj = { "image_id": img_id,
           "filename": file_name,
           "category_id": data[i]["category_id"],
              "segmentation": polygons,
              }
    
    new_data.append(obj)

    img = np.dstack([decoded, decoded, decoded])
    img = np.ascontiguousarray(img, dtype=np.uint8)
    img2 = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    
    print(previous_id, img_id)
    if previous_id == img_id:
        count = count + 1 
        #print('&^%$#@#$%^&*( SAME)')
    else: 
        count = 1
        previous_id = img_id

    mask_path = os.path.join(img_save_folder, os.path.splitext(os.path.basename(file_name))[0] +'_segm'+str(count)+'.png')
    print('Saving mask with contours as', mask_path)
    cv2.imwrite(mask_path,img2)
    

with open(os.path.join(results_save_folder, 'results.json'), 'w') as f:
    f.write(json.dumps(new_data, indent = 4))

# x, y = np.where(decoded == 1)
# print(x, y)
# x = np.reshape(x, (x.shape[0], 1))
# y = np.reshape(y, (y.shape[0], 1))
# coords = np.hstack((x, y))
# print(coords)
# coords = np.reshape(coords, (coords.shape[0]* coords.shape[1], 1))
# print(coords)
