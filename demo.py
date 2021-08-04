import torch, torchvision
from torch.utils.data import DataLoader
from util.dataset import DukeLiverDataset
import util
from model import SingleObjRegressor
from model import Loss
from torchvision import transforms
import glob
import json
import os
import cv2
import numpy as np
import math
from scipy import interpolate
from util.data_process import boundary
import shutil


def demo_1():
    '''
    Test some basic funcs of pytorch
    '''
    a = torch.tensor([8., 9.], requires_grad=True)
    b = torch.tensor([6., 4.], requires_grad=True)
    Q = 3 * a ** 3 - b ** 2
    external_grad = torch.tensor([1., 1.])
    Q.sum().backward()
    print(a.grad)
    print(9 * a ** 2)
    print(b.grad)
    print(2 * b)


def demo_2():
    '''
    Debug DukeLiverDataset class
    '''
    img_dir = '/home/data/duke_liver/duke_liver_coco/imgs/ctpre'
    annotation_file = '/home/data/duke_liver/duke_liver_coco/annotations_train_val/ctpre_val.json'
    dataset_ctarterial_val = DukeLiverDataset(img_dir=img_dir, annotation_file=annotation_file)
    print(len(dataset_ctarterial_val))


def demo_3():
    '''
    Single obj regression exp
    '''
    print("Demo 3, training single obj regression")
    img_dir = "/home/data/duke_liver/dataset/fmt/imgs/ctdelay"
    annotation_file_train = "/home/data/duke_liver/dataset/fmt/annotation_16pts_single_obj/ctdelay_train.json"
    annotation_file_val = "/home/data/duke_liver/dataset/fmt/annotation_16pts_single_obj/ctdelay_val.json"

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_train = util.dataset.DukeLiverDataset(img_dir=img_dir,
                                                  annotation_file=annotation_file_train)
    dataset_val = util.dataset.DukeLiverDataset(img_dir=img_dir,
                                                annotation_file=annotation_file_val)
    data_loader_train = DataLoader(dataset_train,batch_size=16)
    data_loader_val = DataLoader(dataset_val,batch_size=16)

    model = SingleObjRegressor.SingleObjRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    polar_loss_fn = Loss.PolarIoULoss()
    mse_loss_fn = torch.nn.MSELoss()
    size = len(dataset_train)
    for batch, (imgs, labels) in enumerate(data_loader_train):
        pred_center, pred_coordinate = model(imgs)
        labels = torch.squeeze(labels, 1)
        gt_center = labels[:,0:2]
        gt_coordinate = labels[:,2:]
        mse_loss = mse_loss_fn(pred_center, gt_center)
        polar_loss = polar_loss_fn(pred_coordinate, gt_coordinate)
        loss = mse_loss + polar_loss

        # bp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(imgs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    print("Done")


def demo_4():
    '''
    read the single obj anno, and extract the bdry pts and draw smooth results
    '''
    anno_dir = "/home/data/duke_liver/dataset/fmt/annotation_16pts_single_obj/"
    ori_mask_dir = "/home/data/duke_liver/results/20210802"
    tgt_mask_dir = "/home/data/duke_liver/results/20210803_single_obj"
    val_anno_list = glob.glob(anno_dir+"/*_val.json")
    sample_num = 16
    for val_anno_file in val_anno_list:
        series_name = os.path.basename(val_anno_file).split("_")[0]
        print(f"Processing series {series_name}")
        tgt_series_dir = os.path.join(tgt_mask_dir,series_name)
        if not os.path.exists(tgt_series_dir):
            os.makedirs(tgt_series_dir)
        with open(val_anno_file,'r') as f:
            anno = json.load(f)
        anno_list = anno["annotations"]
        for anno_dict in anno_list:
            img_name = anno_dict["file_name"]
            img_path = os.path.join(ori_mask_dir,series_name,img_name)
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) != 1:
                print(f"{img_name} has {len(contours)} contours")
                continue
            mask = np.zeros(img.shape, dtype=np.uint8)
            contour = contours[0]
            # cvt poly 2 curve
            coordinate = boundary.cal_local_coordinates(contour)
            coordinate = boundary.sort_coordinate(coordinate)
            coordinate = boundary.angle_to_pos(coordinate)
            sampled_coordinate = boundary.sample_bdry_pts(coordinate, sample_num)

            x_center = sampled_coordinate["center"][0]
            y_center = sampled_coordinate["center"][1]
            # polar coordinate
            p_angle = [sampled_coordinate["polar_coordinate"][i][0] for i in range(sample_num)]
            p_dist = [sampled_coordinate["polar_coordinate"][i][1] for i in range(sample_num)]

            px = [math.cos(p_angle[i]) * p_dist[i] + x_center for i in range(sample_num)]
            py = [math.sin(p_angle[i]) * p_dist[i] + y_center for i in range(sample_num)]

            px.append(px[0])
            py.append(py[0])

            try:
                tck, u = interpolate.splprep([px, py], s=0, per=True)
                pxi, pyi = interpolate.splev(np.linspace(0, 1, 2500), tck)

                pxi = np.expand_dims(pxi, 1)
                pyi = np.expand_dims(pyi, 1)
                s_contour = np.concatenate((np.array(pxi), np.array(pyi)), axis=1)
                s_contour = np.expand_dims(s_contour, 1)
                s_contour = s_contour.astype(np.int32)
                cv2.drawContours(mask, [s_contour], -1, 255, -1)

                tgt_mask_path = os.path.join(tgt_series_dir, img_name)
                cv2.imwrite(tgt_mask_path, mask)
            except:
                print(f"Interpolation Error")


def demo_5():
    '''
    Generate masks
    '''
    print("generate masks")
    src_mask_dir = "/home/data/duke_liver/dataset/mask"
    tgt_mask_dir = "/home/data/duke_liver/duke_liver_coco/masks"

    series_list = glob.glob(src_mask_dir+"/*")
    for series_dir in series_list:
        series_name = os.path.basename(series_dir)
        output_series_dir = os.path.join(tgt_mask_dir,series_name)
        if not os.path.exists(output_series_dir):
            os.makedirs(output_series_dir)
        patient_list = glob.glob(series_dir+"/*")
        for patient_dir in patient_list:
            patient_id = os.path.basename(patient_dir)
            slices_list = glob.glob(patient_dir+"/*")
            for slice_file in slices_list:
                slice_name = os.path.basename(slice_file)
                output_slice_name = "_".join([series_name,patient_id, slice_name])
                output_slice_path = os.path.join(output_series_dir, output_slice_name)
                shutil.copy(slice_file,output_slice_path)
    print("Done")


def demo_6():
    '''
    Calculate the performance of each series
    '''
    print("Calculate the performance of the polar mask")
    pred_dir = "/home/data/duke_liver/results/20210803_single_obj"
    gt_dir = "/home/data/duke_liver/duke_liver_coco/masks"
    img_dir = "/home/data/duke_liver/duke_liver_coco/imgs"
    result_dir = "/home/data/duke_liver/results/20210803_draw"

    series_list = glob.glob(pred_dir+"/*")
    for series_dir in series_list:
        series_name = os.path.basename(series_dir)
        output_dir = os.path.join(result_dir,series_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        dice = 0.0
        cnt = 0
        slice_list = glob.glob(series_dir+"/*")
        for slice_file in slice_list:
            slice_name = os.path.basename(slice_file)
            pred = cv2.imread(slice_file,cv2.IMREAD_GRAYSCALE)
            gt_file = os.path.join(gt_dir,series_name,slice_name)
            gt = cv2.imread(gt_file,cv2.IMREAD_GRAYSCALE)
            img_file = os.path.join(img_dir,series_name,slice_name)
            img = cv2.imread(img_file)
            dice += np.sum(pred[gt == 255]) * 2.0 / (np.sum(pred) + np.sum(gt))
            cnt += 1


            pred_color = np.zeros(img.shape,dtype=np.uint8)
            pred_color[:,:,:] = (0,255,0)

            blended = np.copy(img)
            alpha = 0.5
            h, w = pred.shape
            for i in range(h):
                for j in range(w):
                    if (pred[i, j] != 0):
                        blended[i, j] = img[i, j] * alpha + np.array([255, 0, 0]) * (1 - alpha)
            # save img
            output_file = os.path.join(output_dir,slice_name)
            cv2.imwrite(output_file, blended)

        print(f"{series_name}: {dice/cnt}")




def main():
    '''
    Main function for demo
    '''
    demo_6()

if __name__ == "__main__":
    main()