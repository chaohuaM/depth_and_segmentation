import os

import cv2
from tqdm import tqdm

from predict import PredictModel, blend_image, show_depth
from utils.utils_metrics import compute_mIoU, show_results

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照JPG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
3、仅有按照VOC格式数据训练的模型可以利用这个文件进行miou的计算。
'''
if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 1
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 2
    # --------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    # --------------------------------------------#
    name_classes = ["background", "rock"]
    # -------------------------------------------------------#
    #   指向数据集所在的文件夹
    #   数据集路径
    # -------------------------------------------------------#
    dataset_path = '/home/ch5225/chaohua/MSL_Mastcam_R_DLRC_png/'

    # 有val.txt的时候
    # image_ids = open(os.path.join(dataset_path, "ImageSets/val.txt"), 'r').read().splitlines()
    # 直接读取文件夹里的文件名
    image_ids = os.listdir(os.path.join(dataset_path, ''))
    image_ids = [image_id[:-4] for image_id in image_ids]
    gt_dir = os.path.join(dataset_path, "")
    miou_out_path = "/home/ch5225/chaohua/MSL_Mastcam_R_DLRC_png_MIOU"
    pred_dir = os.path.join(miou_out_path, 'detection-results')
    pred_col_dir = os.path.join(miou_out_path, 'detection-col-results')
    pred_depth_dir = os.path.join(miou_out_path, 'detection-depth-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        if not os.path.exists(pred_col_dir):
            os.makedirs(pred_col_dir)

        if not os.path.exists(pred_depth_dir):
            os.makedirs(pred_depth_dir)

        print("Load model.")
        config_path = 'logs/2022_03_11_17_49_48/2022_03_11_17_49_48_config.yaml'
        model_weights_path = 'logs/2022_03_11_17_49_48/ep100.pth'
        pr_net = PredictModel(config_path=config_path, model_weights_path=model_weights_path)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(dataset_path, image_id + '.png')
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pr_seg, pr_depth = pr_net.detect_image(img)

            cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pr_seg)

            col_seg = blend_image(img, pr_seg, 0.3)
            col_depth = show_depth(pr_depth)

            cv2.imwrite(os.path.join(pred_col_dir, image_id + ".png"), cv2.cvtColor(col_seg, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(pred_depth_dir, image_id + ".png"), col_depth)

        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
