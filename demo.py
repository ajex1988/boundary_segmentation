import torch, torchvision
from util.dataset import DukeLiverDataset


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


def main():
    '''
    Main function for demo
    '''
    demo_2()

if __name__ == "__main__":
    main()