import os
import torchvision
from skimage import io
import torchvision.datasets.mnist as mnist
import gzip
import shutil

def convert_to_img(train=True):
    if (train):
        f = open(root + 'train.txt', 'w')
        data_path = root + '/train/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            img_path = data_path + str(i) + '.png'
            io.imsave(img_path, img.numpy())
            int_label = str(label).replace('tensor(', '')
            int_label = int_label.replace(')', '')
            f.write(img_path + ' ' + str(int_label) + '\n')
        f.close()
    else:
        f = open(root + 'test.txt', 'w')
        data_path = root + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = data_path + str(i) + '.png'
            io.imsave(img_path, img.numpy())
            int_label = str(label).replace('tensor(', '')
            int_label = int_label.replace(')', '')
            f.write(img_path + ' ' + str(int_label) + '\n')
        f.close()

def un_gz(file_name):
    f_name = file_name.replace(".gz","")
    g_file = gzip.GzipFile(file_name)
    open(f_name, 'wb+').write(g_file.read())
    g_file.close()

if __name__=="__main__":
    DOWNLOAD_MNIST = False
    # Mnist digits dataset
    if not (os.path.exists('./MNIST')) or not os.listdir('./MNIST'):
        # not mnist dir or mnist is empyt dir
        DOWNLOAD_MNIST = True
        train_data = torchvision.datasets.MNIST(
            root='./',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=DOWNLOAD_MNIST,
        )
        gz_paths = os.listdir('./MNIST/raw/')
        for gz in gz_paths:
            un_gz('./MNIST/raw/'+gz)#### 不知道为啥，解压出来的有两个文件是空的，手动解压吧～

    root = "./MNIST/raw/"
    train_set = (
        mnist.read_image_file(os.path.join(root, 'train-images.idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 'train-labels.idx1-ubyte'))
    )

    test_set = (
        mnist.read_image_file(os.path.join(root, 't10k-images.idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 't10k-labels.idx1-ubyte'))
    )


    print("train set:", train_set[0].size())
    print("test set:", test_set[0].size())

    # convert_to_img(True)
    # convert_to_img(False)

    pic_save = "number_data/"
    if not os.path.exists(pic_save):
        os.mkdir(pic_save)
    txt_save = "number_label.txt"
    with open(txt_save,'w') as f:
        for i in range(len(train_set[1])):
            img = train_set[0][i]
            label = train_set[1][i]
            io.imsave(pic_save + str(i) + '.png', img.numpy())
            int_label = str(label).replace('tensor(', '')
            int_label = int_label.replace(')', '')
            f.write(str(i)+'.png' + '\t' + str(int_label) + '\n')
        count = len(train_set[1])
        for i in range(len(test_set[1])):
            img = test_set[0][i]
            label = test_set[1][i]
            io.imsave(pic_save + str(i+count) + '.png', img.numpy())
            int_label = str(label).replace('tensor(', '')
            int_label = int_label.replace(')', '')
            f.write(str(i+count)+'.png' + '\t' + str(int_label) + '\n')
    shutil.rmtree("MNIST")