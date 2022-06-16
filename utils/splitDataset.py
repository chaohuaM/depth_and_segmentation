import os
import random
random.seed(0)

# ----------------------------------------------------------------------#
#   想要增加测试集修改trainval_percent
#   修改train_percent用于改变验证集的比例 9:1
#
#   当前该库将测试集当作验证集使用，不单独划分测试集
# ----------------------------------------------------------------------#
trainval_percent = 1
train_percent = 0.83
# -------------------------------------------------------#
#   指向数据集所在的文件夹
#   默认指向根目录下的数据集
# -------------------------------------------------------#
dataset_path = '../dataset/MSL/'

if __name__ == "__main__":

    print("Generate txt in ImageSets.")
    segfilepath = os.path.join(dataset_path, 'semantic_01_label')
    saveBasePath = os.path.join(dataset_path, 'ImageSets')
    if not os.path.exists(saveBasePath):
        os.mkdir(saveBasePath)

    temp_seg = os.listdir(segfilepath)
    # total_seg = []
    # for seg in temp_seg:
    #     if 'Left' in seg:
    #         if random.randint(0, 1):
    #             seg = seg.replace('Left', 'Right')
    #         total_seg.append(seg)
    total_seg = temp_seg

    num = len(total_seg)
    data_list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(data_list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("traub suze", tr)
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in data_list:
        name = total_seg[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")
