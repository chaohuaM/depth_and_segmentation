import numpy as np
from open3d import read_point_cloud, draw_geometries
import time
from utils.utils import load_exr
import cv2


class point_cloud_generator:
    """
    根据RGB图像和深度图像生成深度图
    """
    def __init__(self, focal_length, scalingfactor, rgb=None, depth=None):
        """
        参数包括焦距、缩放因子、rgb图像矩阵和depth图像矩阵
        :param focal_length:  焦距，可以从相机内参K矩阵获取
        :param scalingfactor: 缩放比例，通常是焦距mm和m之间的转换
        :param rgb:  ndarray rgb image
        :param depth: ndarray depth image
        """
        self.focal_length = focal_length
        self.scalingfactor = scalingfactor
        self.rgb = rgb      # rgb image,ndarray
        self.depth = depth    # depth image, depth.shape == rgb.shape
        self.X = None        # 所有点X坐标
        self.Y = None        # 所有点Y坐标
        self.Z = None        # 所有点Z坐标
        self.df = None       # 点云表格 按字段存储
        self.pc_file = None  # 点云存储文件名  *.ply

    # 设置为property,可以随着其他变量的改变动态更新
    @property
    def height(self):
        return self.rgb.shape[0]

    @property
    def width(self):
        return self.rgb.shape[1]

    def calculate(self):
        t1 = time.time()
        depth = self.depth.T
        self.Z = depth / self.scalingfactor
        X = np.zeros((self.width, self.height))
        Y = np.zeros((self.width, self.height))
        for i in range(self.width):
            X[i, :] = np.full(X.shape[1], i)

        self.X = ((X - self.width / 2) * self.Z) / self.focal_length
        for i in range(self.height):
            Y[:, i] = np.full(Y.shape[0], i)
        self.Y = ((Y - self.height / 2) * self.Z) / self.focal_length

        df = np.zeros((6, self.width * self.height))
        df[0] = self.X.T.reshape(-1)
        df[1] = -self.Y.T.reshape(-1)
        df[2] = -self.Z.T.reshape(-1)
        img = np.array(self.rgb)
        df[3] = img[:, :, 0].reshape(-1)
        df[4] = img[:, :, 1].reshape(-1)
        df[5] = img[:, :, 2].reshape(-1)
        self.df = df
        t2 = time.time()
        print('calcualte 3d point cloud Done.', t2 - t1)

    def write_ply(self, pc_file):
        self.pc_file = pc_file
        t1 = time.time()
        float_formatter = lambda x: "%.4f" % x
        points = []
        for i in self.df.T:
            points.append("{} {} {} {} {} {} 0\n".format
                          (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                           int(i[3]), int(i[4]), int(i[5])))

        file = open(pc_file, "w")
        file.write('''ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property uchar alpha
        end_header
        %s
        ''' % (len(points), "".join(points)))
        file.close()

        t2 = time.time()
        print("Write into .ply file Done.", t2 - t1)

    def show_point_cloud(self):
        pcd = read_point_cloud(self.pc_file)
        draw_geometries([pcd])


if __name__ == '__main__':
    rgb_path = '*.png'
    depth_path = '*.exr'

    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth = load_exr(depth_path)

    a = point_cloud_generator(focal_length=595.90, scalingfactor=1.0, rgb=rgb, depth=depth)

    a.calculate()
    a.write_ply('pc1.ply')
    a.show_point_cloud()
    df = a.df
    np.save('pc.npy', df)
