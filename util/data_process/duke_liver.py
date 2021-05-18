import nibabel as nib
import os
import glob
import cv2
import numpy as np
import json
import shutil

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
def get_category_dict():
    category_dict = {"liver": 1,
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
    return category_dict


def task_2():
    '''
    Convert the dataset processed in step 1 into coco format
    Zhe Zhu, 20210517
    '''
    print("Task 2, convert to coco format")
    img_id_start = 100000
    img_cnt = 0
    p_folder = '/home/data/duke_liver'
    input_dataset_folder = os.path.join(p_folder,'dataset')
    modalities = [os.path.basename(n) for n in glob.glob(input_dataset_folder+"/imgs/*")]

    output_folder = '/home/data/duke_liver/duke_liver_coco'
    annotation_output_folder = os.path.join(output_folder,'annotations')
    img_output_folder = os.path.join(output_folder,'imgs')

    category_dict = get_category_dict()
    json_dict = {"images": [],
                 "info": "Duke liver segmentation dataset",
                 "license": "Duke internal use only",
                 "annotations": [],
                 "categories": category_dict}

    for modality in modalities:
        output_json_file = os.path.join(annotation_output_folder,f"{modality}.json")


        imgs_output_dir = os.path.join(img_output_folder,modality)
        if not os.path.exists(imgs_output_dir):
            os.makedirs(imgs_output_dir)

        modality_dir = os.path.join(input_dataset_folder,"imgs",modality)
        exam_dir_list = glob.glob(modality_dir+"/*")

        for exam_dir in exam_dir_list:
            exam_id = os.path.basename(exam_dir)
            mask_dir = os.path.join(input_dataset_folder, "mask", modality, exam_id)
            img_file_list = glob.glob(exam_dir+"/*")
            for img_file in img_file_list:
                img_name = os.path.basename(img_file)
                new_img_name = f"{modality}_{exam_id}_{img_name}"
                new_img_id = img_id_start + img_cnt
                img_cnt += 1
                new_img_file = os.path.join(imgs_output_dir,new_img_name)
                shutil.copy(img_file,new_img_file)

                mask_file = os.path.join(mask_dir,img_name)
                mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
                height, width = mask.shape[0], mask.shape[1]

                image_dict = {
                    "file_name": new_img_name,
                    "height": height,
                    "width": width,
                    "id": new_img_id,
                }
                json_dict["images"].append(image_dict)

                mask /= 255

                area = np.sum(mask)











if __name__ == "__main__":
    #task_1()
    task_2()