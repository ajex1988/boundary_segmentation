import nibabel as nib
import os
import glob

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
        modality = get_modality(name)
        exam_name = get_exam_name(name)
        print(modality)



if __name__ == "__main__":
    task_1()