import os
import cv2 as cv
import numpy as np
import torch.utils.data as data 
from yaml import load, BaseLoader
from shutil import copy

UNKNOWN_PIXEL = 205

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

## A class of 2d pgm dataset loader with preprocessing
class pgm:
    def __init__(self, folder):
        """
        folder: path of source (raw pgm images)
        """
        self.path = folder
        self.data = []

    def load_data(self, split=False, ratio=0.2, dst="", idx=[]):
        """
        split: if True, choose a ratio of all data for validation
        ratio: 0~1
        dst: path which the splitted data will be saved to
        idx: idx list of specified data for validation
        """
        file_no = []
        yaml_template = '{name}.yaml' 
        pgm_template = '{name}.pgm'
        for root, _, fnames in sorted(os.walk(self.path)):
            for fname in fnames:
                file_no.append(os.path.splitext(fname)[0])
        file_no = np.unique(file_no)
        if not idx:
            seed = np.random.default_rng()
            self.idx_ = idx = seed.choice(len(file_no), int(len(file_no)*ratio), replace=False)
        count = 0
        for f in file_no:
            try:
                if split and count in idx: 
                    copy(yaml_template.format(name=os.path.join(self.path, f)), 
                                                             yaml_template.format(name=os.path.join(dst, "val_"+f)))
                x0, x1, res = self.read_config(yaml_template.format(name=os.path.join(self.path, f)))
            except:
                print(f + ".yaml not found.")
                return
            try:
                img = cv.imread(pgm_template.format(name=os.path.join(self.path, f)))
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            except:
                print(f + ".pgm not found.")
                return
            if split and count in idx: temp = {"name": "val_"+f, "origin": [x0, x1], "resolution": res, "data": img}
            else: temp = {"name": f, "origin": [x0, x1], "resolution": res, "data": img}
            self.data.append(temp)
            count += 1

    def read_config(self, yaml_file):
        with open(yaml_file) as cfg:
            map_config = [line.rstrip("\r\n") for line in cfg]
            resolution = float(map_config[1].split(":")[-1].replace(" ",""))
            origin_x = float(map_config[2].split(":")[-1].split(",")[0].replace("[",""))
            origin_y = float(map_config[2].split(":")[-1].split(",")[1].replace(" ",""))
            return origin_x, origin_y, resolution

    @staticmethod 
    # compute the origin deviation of mmwave map based on a fact that the origins of maps built by lidar and mmwave should be the same
    def align_(aim, fixed):
        ## todo : modify align_() by accepting an extrinsic matrix as prior knowledge from lidar to mmwave (or verse)
        div_x = int(aim["origin"][0]/aim["resolution"] - fixed["origin"][0]/fixed["resolution"])
        div_y = int(aim["origin"][1]/aim["resolution"] - fixed["origin"][1]/fixed["resolution"])

        if(div_x == 0 and div_y == 0):
            aim_ = aim["data"]
            fixed_ = fixed["data"]
        else:
            # Prepare for translation by filling the unknown areas
            ## to right upper
            if(div_x >= 0 and div_y >= 0): 
                aim_ = cv.copyMakeBorder(aim["data"], abs(div_y), 0, 0, abs(div_x), borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
                fixed_ = fixed["data"]
            ## to right bottom
            if(div_x >= 0 and div_y <= 0): 
                aim_ = cv.copyMakeBorder(aim["data"], 0, abs(div_y), 0, abs(div_x), borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
                fixed_ = cv.copyMakeBorder(fixed["data"], 0, abs(div_y), 0, 0, borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
                fixed["origin"][1] -= abs(div_y) * fixed["resolution"]
            ## to left upper
            if(div_x <= 0 and div_y >= 0): 
                aim_ = cv.copyMakeBorder(aim["data"], abs(div_y), 0, abs(div_x), 0, borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
                fixed_ = cv.copyMakeBorder(fixed["data"], 0, 0, abs(div_x), 0, borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
                fixed["origin"][0] -= abs(div_x) * fixed["resolution"]
            ## to left bottom
            if(div_x <= 0 and div_y <= 0): 
                aim_ = cv.copyMakeBorder(aim["data"], 0, abs(div_y), abs(div_x), 0, borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
                fixed_ = cv.copyMakeBorder(fixed["data"], 0, abs(div_y), abs(div_x), 0, borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
                fixed["origin"][0] -= abs(div_x) * fixed["resolution"]
                fixed["origin"][1] -= abs(div_y) * fixed["resolution"]
        translate_mat = np.float32([
            [1, 0, div_x],
            [0, 1, -div_y]
        ])
        adjst = cv.warpAffine(aim_, translate_mat, (aim_.shape[1], aim_.shape[0]),borderMode=cv.BORDER_CONSTANT,borderValue=UNKNOWN_PIXEL)

        # Padding the translated mmwave image and origial lidar map
        div_h = fixed_.shape[0]-adjst.shape[0]
        div_w = fixed_.shape[1]-adjst.shape[1]
        if(fixed_.shape == adjst.shape):             
            adjst_ = adjst
            fixed_ = fixed_
        else:
            if(div_h >= 0 and div_w >= 0):
                adjst_ = cv.copyMakeBorder(adjst, abs(div_h), 0, 0, abs(div_w), borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
                fixed_ = fixed_
            if(div_h >= 0 and div_w <= 0):
                adjst_ = cv.copyMakeBorder(adjst, abs(div_h), 0, 0, 0, borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
                fixed_ = cv.copyMakeBorder(fixed_, 0, 0, 0, abs(div_w), borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
            if(div_h <= 0 and div_w >= 0):
                adjst_ = cv.copyMakeBorder(adjst, 0, 0, 0, abs(div_w), borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
                fixed_ = cv.copyMakeBorder(fixed_, abs(div_h), 0, 0, 0, borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
            if(div_h <= 0 and div_w <= 0):
                adjst_ = adjst
                fixed_ = cv.copyMakeBorder(fixed_, abs(div_h), 0, 0, abs(div_w), borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)

        return adjst_, fixed_

    @staticmethod
    # Get the corners positions of the Known Region Square
    def corner_extractor(img):
        rows = []
        cols = []
        if isinstance(img, np.ndarray):
            for row_idx, row in enumerate(img):
                for col_idx, col in enumerate(row):
                    if col != UNKNOWN_PIXEL: 
                        rows.append(row_idx)
                        cols.append(col_idx)
        return [min(rows), min(cols), max(rows), max(cols)]

    @staticmethod
    def swap(dst, src, idx):
        if isinstance(src, list) and isinstance(dst, list):
            tmp = src[idx]
            src[idx] = dst[idx]
            dst[idx] = tmp
        return dst

class augmentor:
    def __init__(self, pgm_dic, if_crop, if_flip, if_rotate, if_random=False, keep_same=False, size=128, times=6, write=True):
        """
        pgm_dic: a list of pgm_dictionaries
        if_crop: if crop the raw image into pieces or not
        if_flip: if flip the raw image vertically and horizontally or not
        if_rotate: if rotate the raw image in 90, 180 and 270 degrees or not
        if_random: only used when if_crop is true. if true, then use random cropping, otherwise just crop the raw image into 2x2 pieces with the same size
        keep_same: only used when if_crop is true. if true, then crop the same region as specified
        size: the final size of augmented images
        times: int n: if random cropping, then random crop n times, else crop the raw image into nxn pieces
        write: bool, if True save the generated input data to folders, else save to session
        """
        self.raw_image = pgm_dic
        self.temp = []
        self.crop_ = if_crop
        self.flip_ = if_flip
        self.rotate_ = if_rotate
        self.random_crop_ = if_random
        self.size_ = size
        self.keep_same_pos = keep_same
        self.pos = []
        self.times = times
        self.total = 0
        self.write_ = write
        self.session = []
        self.val_session = []

    def __crop(self, img, times, folder, name, pos_x=[], pos_y=[]):
        """
        times: int n: if random cropping, then random crop n times, else crop the raw image into nxn pieces
        folder: dst folder path
        name: dst image name
        pos_x: A list of starting row indexes of the specified regions, only used when self.keep_same_pos is true.
        pos_y: A list of starting col indexes of the specified regions, only used when self.keep_same_pos is true.
        """
        h = img.shape[0]
        w = img.shape[1]
        if not self.random_crop_:
            for i in range(times):
                for j in range(times):
                    box = img[int(i*h/times):int((i+1)*h/times), int(j*w/times):int((j+1)*w/times)]
                    name_ = "{name}-{i}-{j}".format(name=name, i=str(i), j=str(j))
                    self.temp.append({"name":name_, "data":box})
                    self.save("{folder_}/{fname}.pgm".format(folder_=folder, fname=name_), box)
        else:
            if h < self.size_+times or w <self.size_+times:
                img_ = self.__resize(img, self.size_)
                self.temp.append({"name":name, "data":img_})
                if not self.keep_same_pos: self.pos.append(['',''])  # Keep the length of pos same
                self.save("{folder_}/{fname}.pgm".format(folder_=folder, fname=name), img_)
                return
            else: img_ = img
            if not self.keep_same_pos:
                rng_x = np.random.default_rng()
                rng_y = np.random.default_rng()
                x = rng_x.choice(np.arange(img_.shape[1]-self.size_),times,replace=False)
                y = rng_y.choice(np.arange(img_.shape[0]-self.size_),times,replace=False)
                self.pos.append([x, y])
            else: 
                if len(pos_x) == times and len(pos_y) == times:
                    x, y = pos_x, pos_y
                    # print(y, x)
                else: 
                    print("The random cropping times for lidar and mmwave are different... Stoped!")
                    return
            for i in range(times):
                box = img_[y[i]:y[i]+self.size_, x[i]:x[i]+self.size_]
                name_ = "{name}-{i}".format(name=name, i=str(i))
                self.temp.append({"name":name_, "data":box})
                self.save("{folder_}/{fname}.pgm".format(folder_=folder, fname=name_), box)
        
    def __flip(self, img, folder, name):
        img_vertical = cv.flip(img, 0)
        img_horizontal = cv.flip(img, 1)
        self.save("{folder_}/{name_}-f0.pgm".format(folder_=folder, name_=name), img_vertical)
        self.save("{folder_}/{name_}-f1.pgm".format(folder_=folder, name_=name), img_horizontal)
    
    def __rotate(self, img, folder, name):
        img_90 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        img_180 = cv.rotate(img, cv.ROTATE_180)
        img_270 = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
        self.save("{folder_}/{name_}-r90.pgm".format(folder_=folder, name_=name), img_90)
        self.save("{folder_}/{name_}-r180.pgm".format(folder_=folder, name_=name), img_180)
        self.save("{folder_}/{name_}-r270.pgm".format(folder_=folder, name_=name), img_270)

    def __resize(self, img, target_size, direct_resize=False):
        """
        direct_resize: only used when self.crop_ is true. 
        If true, then directly use cv.resize with the nearest interpolation,
        else resize by padding
        """
        oh = img.shape[0]
        ow = img.shape[1]
        if self.crop_:
            if not direct_resize:
                img_ = img
                if ow < target_size: img_ = cv.copyMakeBorder(img_, 0, 0, int((target_size-ow)/2), target_size-ow-int((target_size-ow)/2), borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
                if oh < target_size: img_ = cv.copyMakeBorder(img_, int((target_size-oh)/2), target_size-oh-int((target_size-oh)/2), 0, 0, borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
            else: img_ = cv.resize(img, (target_size, target_size), interpolation=cv.INTER_NEAREST)
        else:
            if oh == target_size and ow == target_size: return img
            else:
                if not direct_resize:
                    if oh > target_size and ow > target_size: img_ = img[:target_size , :target_size]
                    if oh > target_size and ow < target_size: img_ = cv.copyMakeBorder(img[:target_size , :], 0, 0, int((target_size-ow)/2), target_size-ow-int((target_size-ow)/2), borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
                    if oh < target_size and ow > target_size: img_ = cv.copyMakeBorder(img[: , :target_size], int((target_size-oh)/2), target_size-oh-int((target_size-oh)/2), 0, 0, borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
                    if oh < target_size and ow < target_size: img_ = cv.copyMakeBorder(img, int((target_size-oh)/2), target_size-oh-int((target_size-oh)/2), int((target_size-ow)/2), target_size-ow-int((target_size-ow)/2), borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
                else: img_ = cv.resize(img, (target_size, target_size), interpolation=cv.INTER_NEAREST)
        return img_
        
    def save_images(self, path, box=[]):
        if not self.raw_image: return
        if self.crop_: 
            if self.random_crop_:
                if self.keep_same_pos: 
                    if path: print("{dataset} used the same partial image regions of the paired lidar/mmwave maps".format(dataset=path.split("/")[-1]))
                    for idx, raw in enumerate(self.raw_image): self.__crop(raw["data"], times=self.times, folder=path, name=raw["name"], pos_x=box[idx][0], pos_y=box[idx][1])
                else: 
                    if path: print("{dataset} generated random partial image regions.".format(dataset=path.split("/")[-1]))
                    for idx, raw in enumerate(self.raw_image): self.__crop(raw["data"], times=self.times, folder=path, name=raw["name"])
            else: 
                for idx, raw in enumerate(self.raw_image): self.__crop(raw["data"], times=2, folder=path, name=raw["name"])
            for idx, output in enumerate(self.temp):
                if(self.flip_): self.__flip(output["data"], folder=path, name=output["name"])
                if(self.rotate_): self.__rotate(output["data"], folder=path, name=output["name"])
        else:
            for idx, raw in enumerate(self.raw_image): 
                if "val" not in raw["name"]: raw["data"] = self.__resize(raw["data"], self.size_, direct_resize=True)
                self.save("{path_}/{name_}.pgm".format(path_=path, name_=raw["name"]), raw["data"])
                if(self.flip_): self.__flip(raw["data"], folder=path, name=raw["name"])
                if(self.rotate_): self.__rotate(raw["data"], folder=path, name=raw["name"])

    def save(self, path, img):
        if "val" in path:
            if self.write_: cv.imwrite(path, img)
            self.val_session.append((path, img.astype(np.float32)))
            return
        if self.quick_check(img):
            self.total += 1
            self.session.append((path, img.astype(np.float32)))
            if self.write_: cv.imwrite(path, img)
        else:
            print("Writing images failed due to incorrect shape: {fpath}".format(fpath=path))
    
    def quick_check(self, img):
        if img.shape == (self.size_, self.size_): return True
        else: False

def uniform_data(lidar_lst, mm_lst, tolerance=0.2):
    """
    tolerance: 0~1, expand image borders by shape*tolerance after ROI extraction
    """
    if len(lidar_lst) != len(mm_lst): 
        print("The sizes of two dataset are not same... shut down.")
        return
    for i in range(len(lidar_lst)):
        if lidar_lst[i]["name"] != mm_lst[i]["name"]:
            print("the names of two pgms in lidar and mmwave folders are not same... shut down.")
            return
        try:
            # Align images with the same physical origin and img size
            mm_img, lidar_img = pgm.align_(mm_lst[i], lidar_lst[i])
            if lidar_img.shape != mm_img.shape:
                print("the shapes of translated images are different!")
                return
            
            ## For debug
            # cv.imshow(lidar_lst[i]["name"]+"_lidar", lidar_img)
            # cv.imshow(lidar_lst[i]["name"]+"_mmwave", mm_img)
            # cv.waitKey(0)

            # Extract ROI (occupancy/free information) from unknown areas
            lidar_edge = pgm.corner_extractor(lidar_img)
            mm_edge = pgm.corner_extractor(mm_img)
            for j in range(4):
                if lidar_edge[j] == mm_edge[j]: continue
                if j < 2:
                    # left-upper corner, get min
                    if lidar_edge[j] > mm_edge[j]: lidar_edge = pgm.swap(lidar_edge, mm_edge, j)
                    if j == 1: lidar_lst[i]["origin"][0] -= (mm_edge[j] - lidar_edge[j]) * lidar_lst[i]["resolution"]
                else:
                    # right-bottom corner, get max
                    if lidar_edge[j] < mm_edge[j]: lidar_edge = pgm.swap(lidar_edge, mm_edge, j)
                    if j == 2: lidar_lst[i]["origin"][1] -= (lidar_edge[j] - mm_edge[j]) * lidar_lst[i]["resolution"]
            lidar_img = lidar_img[lidar_edge[0]:lidar_edge[2], lidar_edge[1]:lidar_edge[3]]
            mm_img = mm_img[lidar_edge[0]:lidar_edge[2], lidar_edge[1]:lidar_edge[3]]

            # Expand edge
            roi_h = lidar_img.shape[0]
            roi_w = lidar_img.shape[1]
            lidar_img = cv.copyMakeBorder(lidar_img, int(roi_h*tolerance), int(roi_h*tolerance), int(roi_w*tolerance), int(roi_w*tolerance), borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
            mm_img = cv.copyMakeBorder(mm_img, int(roi_h*tolerance), int(roi_h*tolerance), int(roi_w*tolerance), int(roi_w*tolerance), borderType=cv.BORDER_CONSTANT, value=UNKNOWN_PIXEL)
            lidar_lst[i]["origin"][0] -= int(roi_w*tolerance) * lidar_lst[i]["resolution"]
            lidar_lst[i]["origin"][1] -= int(roi_h*tolerance) * lidar_lst[i]["resolution"]
            lidar_lst[i]["data"] = lidar_img
            mm_lst[i]["data"] = mm_img

            ## For debug
            # cv.imshow(lidar_lst[i]["name"]+"_lidar", lidar_lst[i]["data"])
            # cv.imshow(lidar_lst[i]["name"]+"_mmwave", mm_lst[i]["data"])
            # cv.waitKey(0)

        except Exception:
            print("image resizing went wrong:", Exception.args())
            return
    return 

def main():
    # Parse paths def function()
    lidar = pgm("./map/lidar")
    mmwave = pgm("./map/mmwave")
    path_lidar = "./w/train_B"
    path_mmwave = "./w/train_A"
    val_path_lidar = "./val/test_B"
    val_path_mmwave = "./val/test_A"
    lidar.load_data(dst=val_path_lidar)
    mmwave.load_data(dst=val_path_mmwave)

    # Align a pair of maps 
    uniform_data(lidar.data, mmwave.data, tolerance=0.2)

    # Ready before final
    ready = True if os.path.exists(path_lidar) and os.path.exists(path_mmwave) and os.path.exists(val_path_mmwave) and os.path.exists(val_path_lidar) else False
    
    if(ready):
        map_name = []
        obj_name = []
        val_name = []
        for i, data in enumerate(lidar.data):
            if "val" in data["name"]: val_name.append(i)
            elif "obj" in data["name"]: obj_name.append(i)
            else : map_name.append(i)
        # print(map_name, obj_name, val_name)
        lidar_map = augmentor([lidar.data[i] for i in map_name], True, True, True, True, False, size=128)
        mmwave_map = augmentor([mmwave.data[i] for i in map_name], True, True, True, True, True, size=128)
        lidar_obj = augmentor([lidar.data[i] for i in obj_name], False, True, True, False, False, size=128)
        mmwave_obj = augmentor([mmwave.data[i] for i in obj_name], False, True, True, False, False, size=128)
        val_lidar = augmentor([lidar.data[i] for i in val_name], False, False, False)
        val_mmwave = augmentor([mmwave.data[i] for i in val_name], False, False, False)
        try:
            lidar_map.save_images(path_lidar)
            mmwave_map.save_images(path_mmwave, lidar_map.pos)
            lidar_obj.save_images(path_lidar)
            mmwave_obj.save_images(path_mmwave)
            val_lidar.save_images(val_path_lidar)
            val_mmwave.save_images(val_path_mmwave)
        except Exception:
            print("data augmentation failed", Exception.args())
            return
        lidar_total = lidar_map.total + lidar_obj.total
        mm_total = mmwave_map.total + mmwave_obj.total
        if lidar_total != mm_total: 
            print("The lidar and mmwave images are not paired, something went wrong!")
            return
        else: print("Dataset prepared already, there are {num} images for training in total".format(num=lidar_total))

        # Update origin coodinates of validation pgm images in yaml files
        try:
            for val_id in val_name:
                with open("{fname}.yaml".format(fname=os.path.join(val_path_lidar, lidar.data[val_id]["name"]))) as fp: old_lidar = load(fp, Loader=BaseLoader)
                with open("{fname}.yaml".format(fname=os.path.join(val_path_mmwave, mmwave.data[val_id]["name"]))) as fp: old_mmwave = load(fp, Loader=BaseLoader)
                updated_lidar = "image: {image}\nresolution: {resolution}\norigin: {origin}\nnegate: {negate}\noccupied_thresh: {occ}\nfree_thresh: {free}\n".format(
                    image = old_lidar["image"], resolution = str(old_lidar["resolution"]), 
                    origin = str([float(lidar.data[val_id]["origin"][0]), float(lidar.data[val_id]["origin"][1]), float(old_lidar["origin"][-1])]),
                    negate = str(old_lidar["negate"]), occ = str(old_lidar["occupied_thresh"]), free = str(old_lidar["free_thresh"]))
                updated_mmwave = "image: {image}\nresolution: {resolution}\norigin: {origin}\nnegate: {negate}\noccupied_thresh: {occ}\nfree_thresh: {free}\n".format(
                    image = old_mmwave["image"], resolution = str(old_mmwave["resolution"]), 
                    origin = str([float(lidar.data[val_id]["origin"][0]), float(lidar.data[val_id]["origin"][1]), float(old_mmwave["origin"][-1])]),
                    negate = str(old_mmwave["negate"]), occ = str(old_mmwave["occupied_thresh"]), free = str(old_mmwave["free_thresh"]))
                with open("{yname}.yaml".format(yname=os.path.join(val_path_lidar, lidar.data[val_id]["name"])), "w") as fp: fp.write(updated_lidar)
                with open("{yname}.yaml".format(yname=os.path.join(val_path_mmwave, mmwave.data[val_id]["name"])), "w") as fp: fp.write(updated_mmwave)
        except Exception:
            print("yaml files of pgm for test were not successfully updated:", Exception.args())
    else:
        print("the target folders don't exist")
        return 

if __name__ == "__main__":
    main()