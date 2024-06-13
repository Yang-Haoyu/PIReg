import torch
import os
# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision.transforms import v2



def transform_mnist(data_dir, trans_method, train=True):

    mnist_dat = MNIST(data_dir, train=train, download=True,
                      transform=trans_method)

    all_images = []
    all_labels = []

    for images, labels in mnist_dat:
        # Append the current batch of images and labels to the lists
        all_images.append(images)
        all_labels.append(labels)

    images = torch.stack(all_images)
    labels = torch.Tensor(all_labels)
    return images, labels


def generate_corrupted_data(data_dir):

    # regular feature transformation
    trans_blur = v2.Compose(
        [
            v2.GaussianBlur(kernel_size=(13, 13), sigma=1),
            v2.RandomPerspective(distortion_scale=0.5, p = 1.0),
            
            v2.Resize((25, 25)),
            v2.ToTensor(),
            v2.Lambda(lambda x: x.view(x.size(1) * x.size(2)))
        ]
    )

    # privileged feature transformation
    trans_elastic = v2.Compose(
        [
            v2.ElasticTransform(alpha=100.0),
            v2.Resize((25, 25)),
            
            v2.ToTensor(),
            v2.Lambda(lambda x: x.view(x.size(1) * x.size(2)))
        ]
    )

    trans_baseline = v2.Compose(
        [

            v2.Resize((25, 25)),
            v2.ToTensor(),
            v2.Lambda(lambda x: x.view(x.size(1) * x.size(2)))
        ]
    )

    trans_affine = v2.Compose(
        [
            v2.RandomAffine(degrees=30, translate=(0.1, 0.3), scale=(0.5, 0.75)),
            v2.Resize((25, 25)),
            v2.ToTensor(),
            v2.Lambda(lambda x: x.view(x.size(1) * x.size(2)))
        ]
    )



    trans_dict = {
        "Blur": trans_blur,
        "Elastic": trans_elastic,
        "Resize": trans_baseline,
        "Affine": trans_affine

    }




    if not os.path.exists(data_dir + '/processed'):
        os.mkdir(data_dir + '/processed')
        


    for trans_name, trans_method in trans_dict.items():

        images_reg_tr, labels_reg_tr = transform_mnist(
            data_dir, trans_method, train=True)
        
        images_reg_tst, labels_reg_tst = transform_mnist(
            data_dir, trans_method, train=False)
        
        if not os.path.exists(data_dir + '/processed/{}'.format(trans_name)):
            os.mkdir(data_dir + '/processed/{}'.format(trans_name))
        torch.save(images_reg_tr, data_dir + '/processed/{}/train.dat'.format(trans_name))
        torch.save(images_reg_tst, data_dir + '/processed/{}/test.dat'.format(trans_name))

        torch.save(labels_reg_tr, data_dir + '/processed/tr_label.dat')
        torch.save(labels_reg_tst, data_dir + '/processed/tst_label.dat')

if __name__ == "__main__":

    data_dir = "PATH TO MNIST"
    generate_corrupted_data(data_dir)