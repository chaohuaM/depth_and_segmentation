from PIL import Image
import numpy as np
from open3d import read_point_cloud, draw_geometries
import time
from utils.utils import load_exr
import cv2


class point_cloud_generator:

    def __init__(self, focal_length, scalingfactor):
        self.focal_length = focal_length
        self.scalingfactor = scalingfactor
        self.rgb = None
        self.depth = None
        self.pc_file = None

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
        df[3] = img[:, :, 0:1].reshape(-1)
        df[4] = img[:, :, 1:2].reshape(-1)
        df[5] = img[:, :, 2:3].reshape(-1)
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

    rgb_path = '/home/ch5225/chaohua/oaisys/oaisys_tmp/2022-03-03-15-15-02/batch_0002/sensorLeft/0008sensorLeft_semantic_label_01.png'
    depth_path = '/home/ch5225/chaohua/oaisys/oaisys_tmp/2022-03-03-15-15-02/batch_0002/sensorLeft/0008sensorLeft_pinhole_depth_00.exr'
    # a = point_cloud_generator(rgb_path, depth_path, 'pc1.ply',
    #                           focal_length=13.11, scalingfactor=1)

    a = point_cloud_generator(focal_length=595.90, scalingfactor=1.0)
    rgb = cv2.imread(rgb_path)
    # rgb = cv2.resize(rgb, (2048, 2048))

    a.rgb = rgb

    depth = load_exr(depth_path)
    # depth = cv2.resize(depth, (2048, 2048))
    a.depth = depth

    a.calculate()
    a.write_ply('pc1.ply')
    a.show_point_cloud()
    df = a.df
    np.save('pc.npy', df)
