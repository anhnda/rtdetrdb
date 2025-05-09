import joblib
import numpy as np

PATH = "pred_annotation_area.pkl"
img_ids, topk, correct_annotations, all_annotations_area, annotations_map, _, _, _= joblib.load(PATH)
print(img_ids)
AREA_NAME_MAP = {0: "all", 1: "small", 2: "medium", 3: "large"}
# Extract not predicted annotations
N_AREA = len(correct_annotations)
missing_annotations = [ [] for i in range(N_AREA)]
for i in range(N_AREA):
    predicted_area = correct_annotations[i]
    ground_truth_area = all_annotations_area[i]
    for g in ground_truth_area:
        if g not in predicted_area:
            missing_annotations[i].append(g)
    
def convert_topkind_to_pos(ar):
    v = [[0,80*80], [80*80, 80*80 + 40*40], [80*80 + 40*40, 10000000]]
    dims = [80,40,20]
    sz = [8,16,32]
    pos = []
    ii = 0
    for ind in ar:
        ii += 1
        # print(ind)
        for i, (start, end) in enumerate(v):
            if ind >= start and ind < end:
                ind = ind - start
                pos.append([ind%dims[i] * sz[i] + sz[i]/2, ind//dims[i] * sz[i] + sz[i]/2])
                # print(i, sz[i], [ind%dims[i] * sz[i], ind//dims[i] * sz[i]])
                break
       
    assert len(ar) == len(pos)
    pos = np.asarray(pos)
    return pos
def convert_topkinds(ar):
    # if type(ar) not np.ndarray:
    # ar = ar.numpy()
    v = [6400, 1600, 400]
    dims = [80, 40, 20]
    sz = [8, 16, 32]
    out = np.zeros(ar.shape + (2,))
    dx = out[..., 0]
    dy = out[..., 1]
    print(out.shape, dx.shape, dy.shape)
    mark = np.ones(ar.shape, dtype=int)
    arx = ar
    for i, vi in enumerate(v):
        ai = arx // vi
        si = arx - ai * vi
        pi = np.logical_and(ai == 0, mark == 1)
        mark[pi] = 0
        dx[pi] = si[pi] % dims[i] * sz[i] + sz[i] / 2
        dy[pi] = si[pi] // dims[i] * sz[i] + sz[i] / 2
        arx = arx - vi
        arx[arx < 0] = 0
    return out

print("Get img keypoints:...")
# img_keypoits = {k : convert_topkind_to_pos(v) for k, v in topk.items()}
def convert_img_keypoits():
    ids = []
    vtopk = []
    for k, v in topk.items():
        ids.append(k)
        vtopk.append(v)
    print(len(vtopk), len(vtopk[0]))
    vtopks = np.vstack(vtopk)
    # print(vtopks.shape)
    # exit(-1)
    converted_pos = convert_topkinds(vtopks)

    img_keypoits = {}
    for i, idx in enumerate(ids):
        img_keypoits[idx] = converted_pos[i]
    return img_keypoits
img_keypoits = convert_img_keypoits()
def get_points_in_rect(points, bbox):
    x_min, y_min, w, h = bbox
    x_max = x_min + w
    y_max = y_min + h
    points = np.array(points)
    mask = (
        (x_min <= points[:, 0]) & (points[:, 0] <= x_max) &
        (y_min <= points[:, 1]) & (points[:, 1] <= y_max)
    )
    return np.any(points[mask])

def stats_matching():
    print("For annotations can not be predicted...")
    # img_keypoits = {k : convert_topkind_to_pos(v) for k, v in topk.items()}
    print("Loop...")
    for i, missing_annotation in enumerate(missing_annotations):
        print("For area: ", AREA_NAME_MAP[i])
        if len(missing_annotation) == 0:
            print("Skip...")
            continue
        cc = 0
        
        for m in missing_annotation:
            ann = annotations_map[m]
            bbox = ann['bbox']
            area = ann['area']
            img_id = ann['image_id']
            print(img_id, area, bbox)
            keypoints = img_keypoits[img_id]
            # print(keypoints.shape, keypoints[:100])
            if (get_points_in_rect(keypoints, bbox)):
                cc += 1
        print("Ratio inside: ", cc, len(missing_annotation), cc * 1.0/ len(missing_annotation))
def stats_matching_correct():
    print("For annotation that can be predicted...")
    # img_keypoits = {k : convert_topkind_to_pos(v) for k, v in topk.items()}
    for i, missing_annotation in enumerate(correct_annotations):
        print("For area: ", AREA_NAME_MAP[i])
        if len(missing_annotation) == 0:
            print("Skip...")
            continue
        cc = 0
        
        for m in missing_annotation:
            ann = annotations_map[m]
            bbox = ann['bbox']
            area = ann['area']
            img_id = ann['image_id']
            # print(area, img_id, bbox)
            keypoints = img_keypoits[img_id]
            # print(keypoints.shape)
            if (get_points_in_rect(keypoints, bbox)):
                cc += 1
        print("Ratio inside: ", cc, len(missing_annotation), cc * 1.0/ len(missing_annotation))

def stats_matching_all_gt():
    print("For all annotations...")
    for i, missing_annotation in enumerate(all_annotations_area):
        print("For area: ", AREA_NAME_MAP[i])
        if len(missing_annotation) == 0:
            print("Skip...")
            continue
        cc = 0
        
        for m in missing_annotation:
            ann = annotations_map[m]
            bbox = ann['bbox']
            area = ann['area']
            img_id = ann['image_id']
            # print(area, img_id, bbox)
            keypoints = img_keypoits[img_id]
            # print(keypoints.shape)
            if (get_points_in_rect(keypoints, bbox)):
                cc += 1
        print("Ratio inside: ", cc, len(missing_annotation), cc * 1.0/ len(missing_annotation))
if __name__=="__main__":
    stats_matching()
    stats_matching_correct()
    stats_matching_all_gt()
            
            


