import cv2
import joblib
import numpy as np
import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

PATH = "pred_annotation_area.pkl"
IMG_DIR = "/data/coco/val2017"
area_types = ["all", "small", "medium", "large"]
img_ids, topk, correct_annotations, all_annotations_area, annotation_map, pred_bboxs, imgs, imgToAnns = joblib.load(PATH)
print("IDS: ", img_ids[:100])
COLOR_DICT = {1: (255,0,0), 2: (0,255,0), 3:(0,0,255)}
COLOR_NAMES = {'green': (0,255,0), 'red': (0, 0, 255)}
MAX_RES = 640
def get_insert_dict(d,k, v={}):
    try:
        v = d[k]
    except:
        d[k] = v
    return v
def convert_annotation_to_map(annotations, annotation_map):
    imgToAnnotations = {}
    for i, tp in enumerate(area_types):
        annsOfType = annotations[i]
        for anno in annsOfType:
            imgId = annotation_map[anno]['image_id']
            ss = get_insert_dict(imgToAnnotations, imgId, {})
            # print(ss,tp)
            ss2 = get_insert_dict(ss, tp, [])
            ss2.append(anno)
        
    return imgToAnnotations
def convert_pred_bboxs_to_map(correct_annotations, annotation_map, pred_bboxs):
    imgToPredBBoxs = {}
    for i, tp in enumerate(area_types):
        annsOfType = correct_annotations[i]
        bboxsOfType = pred_bboxs[i]
        for k, anno in enumerate(annsOfType):
            imgId = annotation_map[anno]['image_id']
            ss = get_insert_dict(imgToPredBBoxs, imgId, {})
            # print(ss,tp)
            ss2 = get_insert_dict(ss, tp, [])
            ss2.append(bboxsOfType[k])
    return imgToPredBBoxs   
def convert_topkinds(ar, max_wh=640):
    ar = ar.numpy()
    v = [6400, 1600, 400]
    dims = [80, 40, 20]
    sz = [8, 16, 32] 

    out = np.zeros(ar.shape + (2,))
    dx = out[..., 0]
    dy = out[..., 1]
    print(out.shape, dx.shape, dy.shape)
    mark = np.ones(ar.shape, dtype=int)
    arx = ar
    colors = np.zeros(ar.shape, dtype=int)
    for i, vi in enumerate(v):
        ai = arx // vi
        si = arx - ai * vi
        pi = np.logical_and(ai == 0, mark == 1)
        mark[pi] = 0
        colors[pi] = i+1
        dx[pi] = si[pi] % dims[i] * sz[i] + sz[i] / 2
        dy[pi] = si[pi] // dims[i] * sz[i] + sz[i] / 2
        arx = arx - vi
        arx[arx < 0] = 0
    out *= max_wh * 1.0/MAX_RES
    
    return out, colors
correct_annotations_img_map = convert_annotation_to_map(correct_annotations, annotation_map)
all_annotatino_area_map = convert_annotation_to_map(all_annotations_area,annotation_map)
pred_bboxs_img_map = convert_pred_bboxs_to_map(correct_annotations, annotation_map, pred_bboxs)
# def visualize(imgId):
#     path = "%s/%s" % (IMG_DIR, imgs[imgId]['file_name'])
#     imgMat = cv2.imread(path)
#     annoIds = all_annotatino_area_map[imgId]['all']
#     for annoId in annoIds:
#         annoInfo = annotation_map[annoId]
#         bbox = annoInfo['bbox']
#         x,y, w, h = bbox
#         x, y, w, h = int(x), int(y), int(w), int(h)
#         cv2.rectangle(imgMat, (x, y), (x+w, y + h), (0,255,0), 1)   
#     cv2.imwrite("%d.png" % imgId, imgMat)
def get_reshape_size(x, y, target=640):
 
    if x < y:
        r = target * 1.0 / y
        x = int (r * x)
        y = target
    else:
        r = target * 1.0 / x
        x = target
        y = int (r * y)
    return x, y
        
def visualize2(imgId):
    path = "%s/%s" % (IMG_DIR, imgs[imgId]['file_name'])
    imgMat = cv2.imread(path)
    y0, x0,_ = imgMat.shape
    print("W, H", x0,y0)
    x, y = get_reshape_size(x0,y0) 
    #imgMat = cv2.resize(imgMat, (x, y)) 
    #y2, x2 , _ = imgMat.shape
    #print("W2, H2: ", x2, y2)
    annos = imgToAnns[imgId]
    for annoInfo in annos:
        bbox = annoInfo['bbox']
        x,y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(imgMat, (x, y), (x+w, y + h), COLOR_NAMES['red'], 1)   


    pred_bboxs = pred_bboxs_img_map[imgId]['all']
    for bbox in pred_bboxs:
        x,y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(imgMat, (x, y), (x+w, y + h), COLOR_NAMES['green'], 1)   
    # print("TopK", len(topk[imgId]))
    centers, colors = convert_topkinds(topk[imgId], max(x0,y0))
    # print(colors)
    for ii, cen in enumerate(centers):
        x, y = cen
        x, y = int(x), int(y)
        c = colors[ii]
        cv2.circle(imgMat, (x, y), c, COLOR_DICT[c], 1)   

    cv2.imwrite("%d.png" % imgId, imgMat)

if __name__ == "__main__":
    while 1:
        id = int(input())
        if id == -1:
            exit(-1)
        visualize2(id)



