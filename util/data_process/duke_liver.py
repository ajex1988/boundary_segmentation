import nibabel as nib
import os
import glob
import cv2
import numpy as np
import json
from pycocotools import mask
import shutil
from itertools import groupby
from skimage import measure

def filename_img2mask(img_name):
    mask_name = img_name[:-12]+'.nii.gz'
    return mask_name


def get_modality(name):
    parts = name.split('_')
    modality = parts[-1]
    return modality


def get_exam_name(name):
    parts = name.split('_')
    exam_name = ""
    for i in range(len(parts)-1):
        exam_name += parts[i]+"_"
    return exam_name[:-1]


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def blend(img, mask, alpha):
    blended = np.copy(img)
    h, w, nc = mask.shape
    for i in range(h):
        for j in range(w):
            if (mask[i,j,0]!=0):
                blended[i,j] = img[i,j]*alpha+np.array([0,0,255])*(1-alpha)
    return blended

def draw_save_gt(img_dir, mask_dir, output_dir):
    '''
    Draw gt and save the img
    '''
    img_file_list = glob.glob(img_dir+"/*")
    for img_file in img_file_list:
        img_name = os.path.basename(img_file)
        name_list = img_name.split('_')
        vol_dir = img_name[len(name_list[0])+1:-len(name_list[-1])-1]
        mask_file = os.path.join(mask_dir,vol_dir,img_name[-8:])
        output_file = os.path.join(output_dir,img_name)

        img = cv2.imread(img_file)
        mask = cv2.imread(mask_file)
        blended = blend(img,mask,0.5)

        cv2.imwrite(output_file,blended)


def test_1():
    nib_file = 'lrml_0111_dynpre_0000.nii.gz'
    nib_dir = '/home/data/duke_liver/everything_scans'
    img = nib.load(os.path.join(nib_dir,nib_file))
    print(img.shape)

def test_2():
    '''
    Check if the image & mask are matched
    '''
    img_dir = '/home/data/duke_liver/everything_scans'
    mask_dir = '/home/data/duke_liver/everything_masks'
    img_names = glob.glob(img_dir+'/*')
    for img_name in img_names:
        mask_name = filename_img2mask(os.path.basename(img_name))
        if not os.path.exists(os.path.join(mask_dir, mask_name)):
            print("Not Match")
            return
    print("Match")



def task_1():
    '''
    Convert to 2D images, in coco format
    dataset
    -img
    --modality
    ---exam(patient)
    ----volume
    -----slice
    -mask
    --modality
    ---exam(patient)
    ----volume
    -----slice
    '''
    img_dir_src = '/home/data/duke_liver/everything_scans'
    mask_dir_src = '/home/data/duke_liver/everything_masks'
    img_dir_tgt = '/home/data/duke_liver/dataset/img'
    mask_dir_tgt = '/home/data/duke_liver/dataset/mask'

    img_names = glob.glob(img_dir_src + '/*')
    for img_name in img_names:
        name = filename_img2mask(os.path.basename(img_name))[:-7]
        mask_name = os.path.join(mask_dir_src,filename_img2mask(os.path.basename(img_name)))
        modality = get_modality(name)
        exam_name = get_exam_name(name)

        img_mod_subdir = os.path.join(img_dir_tgt,modality)
        if not os.path.exists(img_mod_subdir):
            os.makedirs(img_mod_subdir)
        img_exam_subdir = os.path.join(img_mod_subdir,exam_name)
        if not os.path.exists(img_exam_subdir):
            os.makedirs(img_exam_subdir)
            # process img
        nibimg = nib.load(img_name)
        img = nibimg.get_fdata()
        h, w, nc = img.shape
        for i in range(nc):
            slice_name = '{:04d}.png'.format(i)
            slice_path = os.path.join(img_exam_subdir,slice_name)
            slice = img[:,:,i]
            max_val = slice.max()
            min_val = slice.min()
            if (max_val!=min_val):
                slice = (slice-min_val)/(max_val-min_val)*255.0
            else:
                slice = (slice - min_val) / (max_val - min_val+1e-8) * 255.0
            slice.astype(np.uint8)
            cv2.imwrite(slice_path,slice)

        mask_mod_subdir = os.path.join(mask_dir_tgt,modality)
        if not os.path.exists(mask_mod_subdir):
            os.makedirs(mask_mod_subdir)
        mask_exam_subdir = os.path.join(mask_mod_subdir,exam_name)
        if not os.path.exists(mask_exam_subdir):
            os.makedirs(mask_exam_subdir)
        # process mask
        nibmasks = nib.load(mask_name)
        masks = nibmasks.get_fdata()
        h, w, nc = masks.shape
        for i in range(nc):
            slice_name = '{:04d}.png'.format(i)
            slice_path = os.path.join(mask_exam_subdir, slice_name)
            slice = masks[:, :, i]
            max_val = slice.max()
            min_val = slice.min()
            if (max_val!=min_val):
                slice = (slice - min_val) / (max_val - min_val) * 255.0
            slice.astype(np.uint8)
            cv2.imwrite(slice_path, slice)
    print("Done")
def get_category_map():
    category_map = {"liver": 1,
                     "arterial": 2,
                     "ctarterial": 3,
                     "ctdelay": 4,
                     "ctportal": 5,
                     "ctpre": 6,
                     "dynarterial": 7,
                     "dyndelay": 8,
                     "dyndelay1": 9,
                     "dyndelay2": 10,
                     "dynhbp1": 11,
                     "dynhbp2": 12,
                     "dynhbp3": 13,
                     "dynhbp4": 14,
                     "dynportal": 15,
                     "dynportal2": 16,
                     "dynpre": 17,
                     "dynpre1": 18,
                     "dynpre2": 19,
                     "dyntransitional": 20,
                     "dyntransitional1": 21,
                     "dyntransitional2": 22,
                     "dyntransitional3": 23,
                     "fat": 24,
                     "fat1": 25,
                     "fat2": 26,
                     "fat3": 27,
                     "haste": 28,
                     "opposed": 29,
                     "pdwf": 30,
                     "ssfse": 31,
                     "ssfsefs": 32,
                     "t1nfs": 33,
                     "t2fse": 34,
                     "t2fse1": 35,
                     "t2fse2": 36
                     }
    return category_map


def task_2():
    '''
    Convert the dataset processed in step 1 into coco format, RLE format, iscrowd = 1
    Zhe Zhu, 20210517
    '''
    print("Task 2, convert to coco format")
    img_id_start = 100000
    img_cnt = 0
    ann_id_start = 500000
    ann_cnt = 0
    p_folder = '/home/data/duke_liver'
    input_dataset_folder = os.path.join(p_folder,'dataset')
    modalities = [os.path.basename(n) for n in glob.glob(input_dataset_folder+"/img/*")]

    output_folder = '/home/data/duke_liver/duke_liver_coco'
    annotation_output_folder = os.path.join(output_folder,'annotations')
    img_output_folder = os.path.join(output_folder,'imgs')

    #category_map = get_category_map()
    emp_cnt = 0
    category_list = [{"supercategory": "liver",
                      "name": "liver",
                      "id": 1}]

    for modality in modalities:
        json_dict = {"images": [],
                     "info": "Duke liver segmentation dataset",
                     "license": "Duke internal use only",
                     "annotations": [],
                     "categories": category_list}
        output_json_file = os.path.join(annotation_output_folder,f"{modality}.json")


        imgs_output_dir = os.path.join(img_output_folder,modality)
        if not os.path.exists(imgs_output_dir):
            os.makedirs(imgs_output_dir)

        modality_dir = os.path.join(input_dataset_folder,"img",modality)
        exam_dir_list = glob.glob(modality_dir+"/*")

        for exam_dir in exam_dir_list:
            exam_id = os.path.basename(exam_dir)
            mask_dir = os.path.join(input_dataset_folder, "mask", modality, exam_id)
            img_file_list = glob.glob(exam_dir+"/*")
            for img_file in img_file_list:
                img_name = os.path.basename(img_file)
                new_img_name = f"{modality}_{exam_id}_{img_name}"
                new_img_id = img_id_start + img_cnt

                mask_file = os.path.join(mask_dir,img_name)
                bimask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
                height, width = bimask.shape[0], bimask.shape[1]



                bimask = bimask / 255
                bimask = np.expand_dims(bimask, axis=2)
                bimask = bimask.astype('uint8')
                bimask = np.asfortranarray(bimask)
                Rs = mask.encode(bimask)
                assert len(Rs) == 1
                Rs = Rs[0]
                area = int(mask.area(Rs))
                if area == 0:
                    emp_cnt += 1
                    continue  # skip the "empty" image
                bbox = mask.toBbox(Rs).tolist()

                rle = binary_mask_to_rle(bimask[:,:,0])
                ann_id = ann_id_start + ann_cnt
                ann_cnt += 1
                annotation_dict = {}
                annotation_dict['image_id'] = new_img_id
                annotation_dict['category_id'] = 1 #category_map[modality]
                annotation_dict['segmentation'] = rle
                annotation_dict['iscrowd'] = 1
                annotation_dict['id'] = ann_id
                annotation_dict['area'] = area
                annotation_dict['bbox'] = bbox

                json_dict['annotations'].append(annotation_dict)


                img_cnt += 1
                new_img_file = os.path.join(imgs_output_dir, new_img_name)
                shutil.copy(img_file, new_img_file)
                image_dict = {
                    "file_name": new_img_name,
                    "height": height,
                    "width": width,
                    "id": new_img_id,
                }
                json_dict["images"].append(image_dict)
        with open(output_json_file,'w') as writer:
            json.dump(json_dict,writer)
        print(f"{modality} Done")
    print(f"{emp_cnt} images skipped")
    print("All Done")


def task_3():
    '''
    Train & val split
    train : val = 4:1. Split by volume
    '''
    print("Task 3, split the dataset into train & test")
    src_drc = '/home/data/duke_liver/duke_liver_coco/imgs'
    tgt_drc = '/home/data/duke_liver/duke_liver_coco/train_val'

    modality_dir_list = glob.glob(src_drc+"/*")
    for modality_dir in modality_dir_list:
        modality_name = os.path.basename(modality_dir)
        tgt_dir_train = os.path.join(tgt_drc,modality_name+"_train")
        if not os.path.exists(tgt_dir_train):
            os.makedirs(tgt_dir_train)
        tgt_dir_val = os.path.join(tgt_drc,modality_name+"_val")
        if not os.path.exists(tgt_dir_val):
            os.makedirs(tgt_dir_val)
        img_file_list = glob.glob(modality_dir+"/*")
        vol_set = set()
        for img_file in img_file_list:
            vol_name = os.path.basename(img_file)[:-9]
            vol_set.add(vol_name)
        vol_list = list(vol_set)
        mid = int(len(vol_list)*0.8)
        vol_list_train = vol_list[:mid]
        vol_set_train = set(vol_list_train)
        print(f"{modality_name} splitted, {len(vol_list_train)} volume in train, {len(vol_set)-len(vol_set_train)} volume in val")
        n_train = 0
        n_val = 0
        for img_file in img_file_list:
            vol_name = os.path.basename(img_file)[:-9]
            if vol_name in vol_set_train:
                shutil.copy(img_file,os.path.join(tgt_dir_train,os.path.basename(img_file)))
                n_train += 1
            else:
                shutil.copy(img_file,os.path.join(tgt_dir_val,os.path.basename(img_file)))
                n_val += 1
        print(f"{n_train} training images, {n_val} validation images")


def task_4():
    '''
    Split train & val of the json
    '''
    print("Task 4, split train & val")
    img_dir = '/home/data/duke_liver/duke_liver_coco/train_val'
    json_dir = '/home/data/duke_liver/duke_liver_coco/annotations'
    output_dir = '/home/data/duke_liver/duke_liver_coco/annotations_train_val'

    val_set = {}
    mod_dir_list = glob.glob(img_dir+"/*_val")
    for mod_dir in mod_dir_list:
        mod_name = os.path.basename(mod_dir)[:-4]
        val_set[mod_name] = set()
        img_file_list = glob.glob(mod_dir+"/*")
        for img_file in img_file_list:
            img_name = os.path.basename(img_file)
            val_set[mod_name].add(img_name)

    json_file_list = glob.glob(json_dir+"/*")
    for json_file in json_file_list:
        mod_name = os.path.basename(json_file)[:-5]
        output_file_train = os.path.join(output_dir,mod_name+"_train.json")
        output_file_val = os.path.join(output_dir,mod_name+"_val.json")
        with open(json_file,'r') as reader:
            json_dict = json.load(reader)
        json_dict_train = {"images": [],
                          "info": json_dict["info"],
                          "license": json_dict["license"],
                          "annotations": [],
                          "categories": json_dict["categories"]}
        json_dict_val = {"images": [],
                          "info": json_dict["info"],
                          "license": json_dict["license"],
                          "annotations": [],
                          "categories": json_dict["categories"]}
        val_id = set()
        for img_dict in json_dict["images"]:
            if (img_dict["file_name"] in val_set[mod_name]):
                val_id.add(img_dict["id"])
                json_dict_val["images"].append(img_dict)
            else:
                json_dict_train["images"].append(img_dict)

        for annotation_dict in json_dict["annotations"]:
            if (annotation_dict["image_id"] in val_id):
                json_dict_val["annotations"].append(annotation_dict)
            else:
                json_dict_train["annotations"].append(annotation_dict)

        with open(output_file_train,'w') as f:
            json.dump(json_dict_train,f)
        with open(output_file_val,'w') as f:
            json.dump(json_dict_val,f)
    print("All Done")


def task_5():
    '''
    Convert the dataset to coco format using polygons and iscrowd = 0
    In order to use it in the polar_mask project
    '''
    print("Task 5, convert to coco format")
    img_id_start = 100000
    img_cnt = 0
    ann_id_start = 500000
    ann_cnt = 0
    p_folder = '/home/data/duke_liver'
    input_dataset_folder = os.path.join(p_folder, 'dataset')
    modalities = [os.path.basename(n) for n in glob.glob(input_dataset_folder + "/img/*")]

    output_folder = '/home/data/duke_liver/duke_liver_coco'
    annotation_output_folder = os.path.join(output_folder, 'annotations')
    img_output_folder = os.path.join(output_folder, 'imgs')

    # category_map = get_category_map()
    emp_cnt = 0
    category_list = [{"supercategory": "liver",
                      "name": "liver",
                      "id": 1}]

    for modality in modalities:
        json_dict = {"images": [],
                     "info": "Duke liver segmentation dataset",
                     "license": "Duke internal use only",
                     "annotations": [],
                     "categories": category_list}
        output_json_file = os.path.join(annotation_output_folder, f"{modality}.json")

        imgs_output_dir = os.path.join(img_output_folder, modality)
        if not os.path.exists(imgs_output_dir):
            os.makedirs(imgs_output_dir)

        modality_dir = os.path.join(input_dataset_folder, "img", modality)
        exam_dir_list = glob.glob(modality_dir + "/*")

        for exam_dir in exam_dir_list:
            exam_id = os.path.basename(exam_dir)
            mask_dir = os.path.join(input_dataset_folder, "mask", modality, exam_id)
            img_file_list = glob.glob(exam_dir + "/*")
            for img_file in img_file_list:
                img_name = os.path.basename(img_file)
                new_img_name = f"{modality}_{exam_id}_{img_name}"
                new_img_id = img_id_start + img_cnt

                mask_file = os.path.join(mask_dir, img_name)
                bimask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                height, width = bimask.shape[0], bimask.shape[1]

                bimask = bimask / 255
                bimask = np.expand_dims(bimask, axis=2)
                bimask = bimask.astype('uint8')
                bimask = np.asfortranarray(bimask)
                Rs = mask.encode(bimask)
                assert len(Rs) == 1
                Rs = Rs[0]
                area = int(mask.area(Rs))
                if area == 0:
                    emp_cnt += 1
                    continue  # skip the "empty" image
                bbox = mask.toBbox(Rs).tolist()

                segmentation = binary_mask_to_polygon(binary_mask=bimask[:,:,0], tolerance=0)
                ann_id = ann_id_start + ann_cnt
                ann_cnt += 1
                annotation_dict = {}
                annotation_dict['image_id'] = new_img_id
                annotation_dict['category_id'] = 1  # category_map[modality]
                annotation_dict['segmentation'] = segmentation
                annotation_dict['iscrowd'] = 0
                annotation_dict['id'] = ann_id
                annotation_dict['area'] = area
                annotation_dict['bbox'] = bbox

                json_dict['annotations'].append(annotation_dict)

                img_cnt += 1
                new_img_file = os.path.join(imgs_output_dir, new_img_name)
                shutil.copy(img_file, new_img_file)
                image_dict = {
                    "file_name": new_img_name,
                    "height": height,
                    "width": width,
                    "id": new_img_id,
                }
                json_dict["images"].append(image_dict)
        with open(output_json_file, 'w') as writer:
            json.dump(json_dict, writer)
        print(f"{modality} Done")
    print(f"{emp_cnt} images skipped")
    print("All Done")


def task_6():
    '''
    Draw & save gt segmentation
    '''
    print("Task 6, draw & save gt")
    img_dir = '/home/data/duke_liver/duke_liver_coco/train_val/t1nfs_val'
    mask_dir = '/home/data/duke_liver/dataset/mask/t1nfs'
    output_dir = '/home/data/duke_liver/dataset/gt_draw/t1nfs_val'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    draw_save_gt(img_dir,mask_dir,output_dir)
    print("Done")


def task_7():
    '''
    Draw & save gt for all the series
    '''
    print("Task 7, draw & save gt for all the series")
    series_list = [os.path.basename(name) for name in glob.glob('/home/data/duke_liver/dataset/img/*')]
    img_dir_p = '/home/data/duke_liver/duke_liver_coco/train_val'
    mask_dir_p = '/home/data/duke_liver/dataset/mask'
    output_dir_p = '/home/data/duke_liver/dataset/gt_draw'

    for series_type in series_list:
        img_dir = os.path.join(img_dir_p,series_type+"_val")
        mask_dir = os.path.join(mask_dir_p,series_type)
        output_dir = os.path.join(output_dir_p,series_type+"_val")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        draw_save_gt(img_dir, mask_dir, output_dir)
        print(f"{series_type} Done")


def task_8():
    '''
    Draw & save the average face for each series type
    '''
    img_dir = '/home/data/duke_liver/dataset/img'
    output_dir = '/home/data/duke_liver/dataset/average_img'

    series_list = [os.path.basename(d) for d in glob.glob(img_dir+"/*")]

    for series_name in series_list:
        series_dir = os.path.join(img_dir,series_name)
        vol_list = glob.glob(series_dir+"/*")
        img_sum = np.zeros((512,512),dtype=float)
        img_cnt = 0
        for vol_dir in vol_list:
            img_file_list = glob.glob(vol_dir+"/*")
            for img_file in img_file_list:
                img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img,(512,512))
                img_sum += img
                img_cnt += 1
        print(f"{series_name} has {img_cnt} slices")
        output_file = os.path.join(output_dir,f"{series_name}.png")
        img_avg = img_sum/img_cnt
        img_avg.astype(np.uint8)
        cv2.imwrite(output_file,img_avg)


if __name__ == "__main__":
    #task_1()
    #task_2()
    #task_3()
    #task_4()
    #task_5()
    #task_6()
    #task_7()
    task_8()