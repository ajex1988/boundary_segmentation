import cv2
import numpy as np
import math
import os
from scipy import interpolate
import glob
import json
import time

def filter_contours(contours):
    '''
    Remove noisy contours
    '''
    threshold = 25
    contour_num = len(contours)
    if contour_num == 1:
        return contours
    filtered_contours = []
    for contour in contours:
        if len(contour) >= threshold:
            filtered_contours.append(contour)
    return filtered_contours

def cal_local_coordinates(contour):
    '''
    Calculate the Cartesian and Polar coordinate
    '''
    # 1. Calculate the mass center using image moments
    local_coordinate = {"center":[],
                        "original_coordinate":[],
                        "polar_coordinate":[],
                        "cartesian_coordinate":[]}

    mo = cv2.moments(contour)
    x = mo['m10'] / (mo['m00'] + 1e-5)
    y = mo['m01'] / (mo['m00'] + 1e-5)

    local_coordinate["center"] = [x, y]

    pts_num = len(contour)
    for i in range(pts_num):
        ori_x = contour[i][0][0]
        ori_y = contour[i][0][1]

        card_x = ori_x - x
        card_y = ori_y - y

        angle = np.angle(card_x+card_y*1j)
        dist = np.linalg.norm((card_x,card_y))

        local_coordinate["original_coordinate"].append([ori_x,ori_y])
        local_coordinate["cartesian_coordinate"].append([card_x, card_y])
        local_coordinate["polar_coordinate"].append([angle, dist])

    return local_coordinate


def sort_coordinate(coordinate):
    '''
    Sort the coordinate so that the points start from 0 degree
    '''
    index = 0
    min_val = 10000
    for i in range(len(coordinate["polar_coordinate"])):
        cur = coordinate["polar_coordinate"][i][0]
        if 0 < cur < min_val:
            min_val = cur
            index = i

    increase = 0
    decrease = 0
    for i in range(5):
        if coordinate["polar_coordinate"][i+1][0] > coordinate["polar_coordinate"][i][0]:
            increase += 1
        else:
            decrease += 1
    bAscend = (increase > decrease)

    if bAscend:
        coordinate["polar_coordinate"][:] = coordinate["polar_coordinate"][index:]+coordinate["polar_coordinate"][0:index]
        coordinate["cartesian_coordinate"][:] = coordinate["cartesian_coordinate"][index:] + coordinate["cartesian_coordinate"][0:index]
        coordinate["original_coordinate"][:] = coordinate["original_coordinate"][index:] + coordinate["original_coordinate"][0:index]
    else:
        if index!=len(coordinate["polar_coordinate"])-1:
            coordinate["polar_coordinate"][:] = coordinate["polar_coordinate"][index+1:] + coordinate["polar_coordinate"][0:index+1]
            coordinate["cartesian_coordinate"][:] = coordinate["cartesian_coordinate"][index+1:] + coordinate["cartesian_coordinate"][0:index+1]
            coordinate["original_coordinate"][:] = coordinate["original_coordinate"][index + 1:] + coordinate["original_coordinate"][0:index + 1]
            coordinate["polar_coordinate"].reverse()
            coordinate["cartesian_coordinate"].reverse()
            coordinate["original_coordinate"].reverse()

    return coordinate


def angle_to_pos(coordinate):
    '''
    Cvt the angle which is negative to positive ones
    '''
    pts_num = len(coordinate["polar_coordinate"])
    for i in range(pts_num):
        if coordinate["polar_coordinate"][i][0] < 0:
            coordinate["polar_coordinate"][i][0] = 2*math.pi + coordinate["polar_coordinate"][i][0]
    return coordinate


def sample_bdry_pts(coordinate,sample_num):
    ori_sample_num = len(coordinate["polar_coordinate"])
    if ori_sample_num < sample_num:
        return coordinate
    # sample polar coordinate pts
    sampled_polar_coordinate = [[]]*sample_num
    step = 2*math.pi / sample_num
    # use a stupid search alg first, can be optimized
    for i in range(sample_num):
        r = i*step
        min_val = 1000
        min_idx = 0
        for j in range(ori_sample_num):
            diff = abs(coordinate["polar_coordinate"][j][0] - r)
            if diff < min_val:
                min_val = diff
                min_idx = j
        sampled_polar_coordinate[i] = coordinate["polar_coordinate"][min_idx]

    # sample cartesian coordinate pts
    sampled_cartesian_coordinate = [[]]*sample_num
    step = math.floor(ori_sample_num/sample_num)
    for i in range(sample_num):
        idx = i*step
        sampled_cartesian_coordinate[i] = coordinate["cartesian_coordinate"][idx]

    sampled_coordinate = {"center": coordinate["center"],
                          "polar_coordinate": sampled_polar_coordinate,
                          "cartesian_coordinate": sampled_cartesian_coordinate,
                          "original_coordinate": coordinate["original_coordinate"]}

    return sampled_coordinate


def experiment(mask_file_list, sample_num, output_dir, save_png=False):
    cnt = 0
    print(f"There are {len(mask_file_list)} files to process")
    pts_dir = os.path.join(output_dir,"pts")
    png_dir = os.path.join(output_dir,"png")
    if save_png and not os.path.exists(png_dir):
        os.makedirs(png_dir)
    if not os.path.exists(pts_dir):
        os.makedirs(pts_dir)
    tic = time.time()
    for mask_file in mask_file_list:
        file_name = os.path.basename(mask_file)
        names = mask_file.split('/')
        if (cnt+1) % 1000 == 0:
            toc = time.time()
            print(f"{cnt} processed, {toc-tic}s elapsed")
            tic = time.time()
        output_file = os.path.join(png_dir, file_name)
        mask = cv2.imread(mask_file, 0)
        mask3c = cv2.imread(mask_file,1)
        ret, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = filter_contours(contours)
        coordinates_dict = {"coordinates": []}

        if len(contours) > 1:
            print(f"{file_name} has {len(contours)} contours")

        for contour in contours:
            coordinate = cal_local_coordinates(contour)
            coordinate = sort_coordinate(coordinate)
            coordinate = angle_to_pos(coordinate)
            sampled_coordinate = sample_bdry_pts(coordinate,sample_num)
            coordinates_dict["coordinates"].append(sampled_coordinate)


            # save png
            if save_png:
                x_center = sampled_coordinate["center"][0]
                y_center = sampled_coordinate["center"][1]
                # polar coordinate
                p_angle = [sampled_coordinate["polar_coordinate"][i][0] for i in range(sample_num)]
                p_dist = [sampled_coordinate["polar_coordinate"][i][1] for i in range(sample_num)]

                px = [math.cos(p_angle[i]) * p_dist[i] + x_center for i in range(sample_num)]
                py = [math.sin(p_angle[i]) * p_dist[i] + y_center for i in range(sample_num)]

                px.append(px[0])
                py.append(py[0])

                tck, u = interpolate.splprep([px, py], s=0, per=True)

                pxi, pyi = interpolate.splev(np.linspace(0, 1, 2500), tck)

                pxi = np.expand_dims(pxi, 1)
                pyi = np.expand_dims(pyi, 1)
                s_contour = np.concatenate((np.array(pxi), np.array(pyi)), axis=1)
                s_contour = np.expand_dims(s_contour, 1)
                s_contour = s_contour.astype(np.int32)
                cv2.drawContours(mask3c, [s_contour], -1, (0, 0, 255), 3)
                cv2.imwrite(output_file,mask3c)
        # save extracted coordinates
        json_file = os.path.join(pts_dir,names[-3]+"_"+names[-2]+"_"+file_name[:-4]+".json")
        for i in range(len(coordinates_dict["coordinates"])):
            coordinates_dict["coordinates"][i]["center"][0] = float(coordinates_dict["coordinates"][i]["center"][0])
            coordinates_dict["coordinates"][i]["center"][1] = float(coordinates_dict["coordinates"][i]["center"][1])

            ori_pts_num = len(coordinates_dict["coordinates"][i]["original_coordinate"])
            for j in range(ori_pts_num):
                coordinates_dict["coordinates"][i]["original_coordinate"][j][0] = float(coordinates_dict["coordinates"][i]["original_coordinate"][j][0])
                coordinates_dict["coordinates"][i]["original_coordinate"][j][1] = float(coordinates_dict["coordinates"][i]["original_coordinate"][j][1])

            sampled_pts_num = len(coordinates_dict["coordinates"][i]["polar_coordinate"])
            for j in range(sampled_pts_num):
                coordinates_dict["coordinates"][i]["polar_coordinate"][j][0] = float(coordinates_dict["coordinates"][i]["polar_coordinate"][j][0])
                coordinates_dict["coordinates"][i]["polar_coordinate"][j][1] = float(coordinates_dict["coordinates"][i]["polar_coordinate"][j][1])
                coordinates_dict["coordinates"][i]["cartesian_coordinate"][j][0] = float(coordinates_dict["coordinates"][i]["cartesian_coordinate"][j][0])
                coordinates_dict["coordinates"][i]["cartesian_coordinate"][j][1] = float(coordinates_dict["coordinates"][i]["cartesian_coordinate"][j][1])

        if len(coordinates_dict["coordinates"]) > 0:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(coordinates_dict, f, ensure_ascii=False, indent=4)
            cnt += 1
    print(f"Total {cnt} masks saved")



def test_1():
    img_file = "/home/data/duke_liver/dataset/mask/arterial/lrml_0143_ct/0025.png"
    img = cv2.imread(img_file,0)
    ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = filter_contours(contours)
    for contour in contours:
        coordinate = cal_local_coordinates(contour)
        coordinate = sort_coordinate(coordinate)
        print("Done")


def test_2():
    mask_dir = "/home/data/duke_liver/dataset/mask/arterial/lrml_0143_ct"
    mask_file_list = glob.glob(mask_dir+"/*")
    sample_num = 16
    output_dir = "/home/data/duke_liver/dataset/experiment/bdry_pts/16"
    experiment(mask_file_list,sample_num,output_dir)
    print("1st exp done")


def test_3():
    '''
    Run the whole dataset
    '''
    mask_dir = "/home/data/duke_liver/dataset/mask"
    series_dir_list = glob.glob(mask_dir+"/*")
    mask_file_list = []
    for series_dir in series_dir_list:
        patient_dir_list = glob.glob(series_dir+"/*")
        for patient_dir in patient_dir_list:
            file_list = glob.glob(patient_dir+"/*")
            mask_file_list.extend(file_list)
    sample_num = 16
    output_dir = "/home/data/duke_liver/dataset/experiment/bdry_pts/16"
    save_png = False
    experiment(mask_file_list, sample_num, output_dir,save_png)
    print("whole dataset exp done")


def test_4():
    '''
    Draw contours based on interpolation
    '''
    img_dir = '/home/data/duke_liver/dataset/img'
    anno_dir = '/home/data/duke_liver/dataset/experiment/bdry_pts/16/pts'
    output_dir = '/home/data/duke_liver/intermediate/interpolation/bdry_16'

    sample_num = 16
    anno_file_list = glob.glob(anno_dir+"/*")

    for anno_file in anno_file_list:
        try:
            with open(anno_file,'r') as f:
                anno_dict = json.load(f)
                samples = anno_dict["coordinates"]
                for sample in samples:
                    center = sample["center"]
                    x_center = center[0]
                    y_center = center[1]
                    p_angle = [sample["polar_coordinate"][i][0] for i in range(sample_num)]
                    p_dist = [sample["polar_coordinate"][i][1] for i in range(sample_num)]

                    px = [math.cos(p_angle[i]) * p_dist[i] + x_center for i in range(sample_num)]
                    py = [math.sin(p_angle[i]) * p_dist[i] + y_center for i in range(sample_num)]

                    px.append(px[0])
                    py.append(py[0])

                    tck, u = interpolate.splprep([px, py], s=0, per=True)

                    pxi, pyi = interpolate.splev(np.linspace(0, 1, 2500), tck)

                    pxi = np.expand_dims(pxi, 1)
                    pyi = np.expand_dims(pyi, 1)
                    s_contour = np.concatenate((np.array(pxi), np.array(pyi)), axis=1)
                    s_contour = np.expand_dims(s_contour, 1)
                    s_contour = s_contour.astype(np.int32)

                    file_name = os.path.basename(anno_file)
                    series_name, pid_1,pid_2, img_no = file_name[:-5].split('_')
                    img_file = os.path.join(img_dir, series_name, f"{pid_1}_{pid_2}", f"{img_no}.png")
                    img = cv2.imread(img_file, 1)
                    cv2.drawContours(img, [s_contour], -1, (255, 0, 0), 3)

                output_file = os.path.join(output_dir,f"{series_name}_{pid_1}_{pid_2}_{img_no}.png")
                cv2.imwrite(output_file, img)


        except:
            print(f"###{anno_file}###")
def main():
    test_4()


if __name__ == "__main__":
    main()