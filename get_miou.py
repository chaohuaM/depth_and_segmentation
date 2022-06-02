import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import yaml
import glob

import cv2
from tqdm import tqdm

from predict_model import create_predict_model_pl, blend_image, show_depth
from utils.utils_metrics import compute_mIoU, show_results
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照JPG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集。
3、仅有按照VOC格式数据训练的模型可以利用这个文件进行miou的计算。
'''
if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    # ------------------------------#
    #   输入预测的图片类型：gray和rgb   #
    # ------------------------------#
    input_type = 'rgb'
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 2
    # --------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    # --------------------------------------------#
    name_classes = ["Non-rock", "Rock"]
    # -------------------------------------------------------#
    #   指向数据集所在的文件夹
    #   数据集路径
    # -------------------------------------------------------#
    model_paths = glob.glob("525-logs/*sa/*06_01*/checkpoints/*")
    for model_path in model_paths:
        print("model path:", model_path)
        log_path = '/'.join(model_path.split('/')[:-2]) + '/' + model_path.split("/")[-1][:-5]
        config_path = '/'.join(model_path.split('/')[:-2]) + '/' + "hparams.yaml"
        ckpt_path = model_path

        with open(config_path, 'r') as f:
            config_params = yaml.safe_load(f)

        model_name = config_params['model_name']
        backbone = config_params['backbone']
        in_channels = config_params['in_channels']

        dataset_dir = 'dataset/oaisys-new/'
        image_dir = os.path.join(dataset_dir, 'rgb')
        gt_dir = os.path.join(dataset_dir, "semantic_01_label")

        # 输出路径设置
        miou_out_path = dataset_dir + log_path + '/val'
        pred_dir = os.path.join(miou_out_path, 'detection-results')

        # 有val.txt的时候
        image_ids = open(os.path.join(dataset_dir, "ImageSets/val.txt"), 'r').read().splitlines()
        # 直接读取文件夹里的文件名
        # image_ids = os.listdir(image_dir)
        # image_ids = [image_id[:-4] for image_id in image_ids]

        print("images number：", len(image_ids))

        # 获得预测结果
        if miou_mode == 0 or miou_mode == 1:
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)

            pred_col_dir = os.path.join(miou_out_path, 'detection-col-results')

            if not os.path.exists(pred_col_dir):
                os.makedirs(pred_col_dir)

            if 'dual_decoder' in model_name:
                pred_depth_dir = os.path.join(miou_out_path, 'detection-depth-results')
                if not os.path.exists(pred_depth_dir):
                    os.makedirs(pred_depth_dir)

            pr_net = create_predict_model_pl(checkpoint_path=ckpt_path, config_path=config_path)
            print("Load model done.")

            print("Get predict result.")
            for image_id in tqdm(image_ids):
                image_path = os.path.join(image_dir, image_id + '.png')
                if input_type == 'rgb':
                    img = cv2.imread(image_path, 1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = cv2.imread(image_path, 0)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                pr_outputs = pr_net.detect_image(img)

                # blend混合显示
                pr_seg = pr_outputs[0]

                cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pr_seg)

                col_seg = blend_image(img, pr_seg, 0.3)
                cv2.imwrite(os.path.join(pred_col_dir, image_id + ".png"), cv2.cvtColor(col_seg, cv2.COLOR_RGB2BGR))

                if len(pr_outputs) > 1:
                    pr_depth = pr_outputs[1]
                    col_depth = show_depth(pr_depth)
                    cv2.imwrite(os.path.join(pred_depth_dir, image_id + ".png"), col_depth)

            print("Get predict result done.")

        # 计算精度指标
        if miou_mode == 0 or miou_mode == 2:
            print("Get miou.")
            hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                            name_classes)  # 执行计算mIoU的函数
            print("Get miou done.")
            show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

        if 'sa' in model_name and miou_mode == 3:
            layers = ['sa_blocks.0', 'sa_blocks.1', 'sa_blocks.2', 'sa_blocks.3', 'sa_blocks.4']
            pred_dsa_mask_dir = os.path.join(miou_out_path, 'detection-dsa-mask-results')

            if not os.path.exists(pred_dsa_mask_dir):
                os.makedirs(pred_dsa_mask_dir)

            pr_net = create_predict_model_pl(checkpoint_path=ckpt_path, config_path=config_path)
            print("Load model done.")

            print("Get predict dsa-mask.")
            for image_id in tqdm(image_ids):
                image_path = os.path.join(image_dir, image_id + '.png')
                if input_type == 'rgb':
                    img = cv2.imread(image_path, 1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = cv2.imread(image_path, 0)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                dsa_masks = pr_net.get_dsa_mask(input_image=img)
                for idx in range(len(layers)):
                    layer = layers[idx]
                    fig = plt.figure(dpi=400)
                    # plt.imshow(feature_maps[0], cmap='jet')
                    # plt.axis('off')
                    # plt.colorbar()
                    im = plt.imshow(dsa_masks[idx], cmap='jet')
                    plt.axis('off')

                    fig.tight_layout()  # 调整整体空白
                    plt.subplots_adjust(right=0.9, wspace=-0.5, hspace=0.1)  # 调整子图间距
                    position = fig.add_axes([0.9, 0.1, 0.02, 0.78])  # 位置[左,下,右,上]
                    fig.colorbar(im, cax=position)

                    # plt.show()
                    fig_path = os.path.join(pred_dsa_mask_dir, image_id + '-' + layer + '.png')
                    plt.savefig(fig_path)

                    plt.clf()
                    plt.close()




