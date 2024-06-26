import os.path
import numpy as np
import torchvision.transforms as transforms
from yaml import load, BaseLoader
from data.prepare import BaseDataset, pgm, augmentor, uniform_data, FREE_PIXEL, OCC_PIXEL, UNKNOWN_PIXEL

# from multiprocessing import Pool
# from tqdm import tqdm

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        dir_A_n = dir_A_val = dir_B_n = dir_B_val = ""
        
        ### input A (label maps)
        dir_A_ = '_mmwave'
        dir_A = os.path.join(opt.dataroot, opt.phase + dir_A_)
        if opt.saveImage: dir_A_n = os.path.join(opt.saveroot, opt.phase + dir_A_)
        if opt.splitVal: dir_A_val = os.path.join(opt.valroot, "test" + dir_A_)
        A = pgm(dir_A)

        ### input B (real images)
        if opt.isTrain:
            dir_B_ = '_lidar'
            dir_B = os.path.join(opt.dataroot, opt.phase + dir_B_)  
            if opt.saveImage: dir_B_n = os.path.join(opt.saveroot, opt.phase + dir_B_)
            if opt.splitVal: dir_B_val = os.path.join(opt.valroot, "test" + dir_B_)
            B = pgm(dir_B)

        A.load_data(split=opt.splitVal, ratio=0.1, dst=dir_A_val)
        if opt.isTrain:
            B.load_data(split=opt.splitVal, ratio=0.1, dst=dir_B_val, idx=A.idx_.tolist())
            ### Align a pair of maps 
            uniform_data(B.data, A.data, tolerance=opt.tolerance)
        
        ### Split train_validation and Augment data
        map_name = []
        obj_name = []
        val_name = []
        for i, data in enumerate(A.data):
            if "val" in data["name"]: val_name.append(i)
            else:
                if "obj" in data["name"]: obj_name.append(i)
                else : map_name.append(i)
        mmwave_map = augmentor([A.data[i] for i in map_name], opt.crop_and_resize, opt.flip, opt.rotate, opt.randomCrop, False, size=opt.loadSize, times=opt.times, write=opt.saveImage)
        mmwave_obj = augmentor([A.data[i] for i in obj_name], False, True, True, False, False, size=opt.loadSize, times=opt.times, write=opt.saveImage)
        val_mmwave = augmentor([A.data[i] for i in val_name], False, False, False, False, False, size=opt.loadSize, times=opt.times, write=opt.splitVal)
        if opt.isTrain:
            lidar_map = augmentor([B.data[i] for i in map_name], opt.crop_and_resize, opt.flip, opt.rotate, opt.randomCrop, True, size=opt.loadSize, times=opt.times, write=opt.saveImage)
            lidar_obj = augmentor([B.data[i] for i in obj_name], False, True, True, False, False, size=opt.loadSize, times=opt.times, write=opt.saveImage)
            val_lidar = augmentor([B.data[i] for i in val_name], False, False, False, False, False, size=opt.loadSize, times=opt.times, write=opt.splitVal)
        try:
            mmwave_map.save_images(dir_A_n)
            mmwave_obj.save_images(dir_A_n)
            val_mmwave.save_images(dir_A_val)
            if opt.isTrain:
                lidar_map.save_images(dir_B_n, mmwave_map.pos)
                lidar_obj.save_images(dir_B_n)
                val_lidar.save_images(dir_B_val)
        except Exception:
            print("data augmentation failed", Exception.args())
            return
        self.dataset_size = mmwave_map.total + mmwave_obj.total if opt.isTrain else len(val_mmwave.val_session)
        
        ### Update origin coodinates of pgm images for validation in yaml files
        if opt.isTrain and opt.splitVal:
            try:
                for val_id in val_name:
                    with open("{fname}.yaml".format(fname=os.path.join(dir_B_val, B.data[val_id]["name"]))) as fp: old_lidar = load(fp, Loader=BaseLoader)
                    with open("{fname}.yaml".format(fname=os.path.join(dir_A_val, A.data[val_id]["name"]))) as fp: old_mmwave = load(fp, Loader=BaseLoader)
                    updated_lidar = "image: {image}\nresolution: {resolution}\norigin: {origin}\nnegate: {negate}\noccupied_thresh: {occ}\nfree_thresh: {free}\n".format(
                        image = old_lidar["image"], resolution = str(old_lidar["resolution"]), 
                        origin = str([float(B.data[val_id]["origin"][0]), float(B.data[val_id]["origin"][1]), float(old_lidar["origin"][-1])]),
                        negate = str(old_lidar["negate"]), occ = str(old_lidar["occupied_thresh"]), free = str(old_lidar["free_thresh"]))
                    updated_mmwave = "image: {image}\nresolution: {resolution}\norigin: {origin}\nnegate: {negate}\noccupied_thresh: {occ}\nfree_thresh: {free}\n".format(
                        image = old_mmwave["image"], resolution = str(old_mmwave["resolution"]), 
                        origin = str([float(B.data[val_id]["origin"][0]), float(B.data[val_id]["origin"][1]), float(old_mmwave["origin"][-1])]),
                        negate = str(old_mmwave["negate"]), occ = str(old_mmwave["occupied_thresh"]), free = str(old_mmwave["free_thresh"]))
                    with open("{yname}.yaml".format(yname=os.path.join(dir_B_val, B.data[val_id]["name"])), "w") as fp: fp.write(updated_lidar)
                    with open("{yname}.yaml".format(yname=os.path.join(dir_A_val, A.data[val_id]["name"])), "w") as fp: fp.write(updated_mmwave)
            except Exception:
                print("yaml files of pgm for test were not successfully updated:", Exception.args())
        if opt.isTrain:
            self.A_input = sorted([(e[0], e[1]) for e in mmwave_map.session + mmwave_obj.session], key = lambda map: map[0])
            self.B_input = sorted([(e[0], e[1]) for e in lidar_map.session + lidar_obj.session], key = lambda map: map[0])
        else:
            self.A_input = [(e[0], e[1]) for e in val_mmwave.val_session]

        ### Mask Values
        if opt.mask_output:
            assert opt.output_nc==3, "There are only three unique mask values of 2d OccupancyGrid map: Occ, Free, Unknown"
            self.mask_values = [OCC_PIXEL, UNKNOWN_PIXEL, FREE_PIXEL]

    def im2masktensor(self, img_arr):
        if isinstance(img_arr, np.ndarray):
            mask = np.zeros((img_arr.shape[0], img_arr.shape[1], self.opt.output_nc))
            if len(img_arr.shape) > 2: img_arr = img_arr[:,:,0]
            for i, row in enumerate(img_arr):
                for j, col in enumerate(row):
                    if int(col) == OCC_PIXEL: mask[i][j][0] = 1
                    if int(col) == UNKNOWN_PIXEL: mask[i][j][1] = 1
                    if int(col) == FREE_PIXEL: mask[i][j][2] = 1
            totensor = transforms.ToTensor()
            return totensor(mask.astype(np.float32))

    def __getitem__(self, index):        
        A_path = self.A_input[index][0]
        if self.opt.isTrain:
            B_path = self.B_input[index][0]
            assert(A_path.split("/")[-1] == B_path.split("/")[-1])
        transform_list = [transforms.ToTensor()]
        if not self.opt.no_norm_input:
            transform_list += [transforms.Normalize((0.5), (0.5))] if not self.opt.define_norm else [transforms.Normalize((0.80), (0.24))]
        transform_ = transforms.Compose(transform_list)
        A_tensor = transform_(self.A_input[index][1] / 255.0) if not self.opt.mask_output else self.im2masktensor(self.A_input[index][1])
        B_tensor = 0
        if self.opt.isTrain: 
            B_tensor = transform_(self.B_input[index][1] / 255.0) if not self.opt.mask_output else self.im2masktensor(self.B_input[index][1])

        input_dict = {'label': A_tensor, 'image': B_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'