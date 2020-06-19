from torchvision import transforms
from torch.utils.data import dataset,dataloader
from torchvision.datasets.folder import default_loader
from utils import RandomErasing,RandomSampler
from opt import opt
import os,re
import os.path as osp
from scipy.io import loadmat
import numpy as np
import torch
from PIL import Image

class Data():
    def __init__(self):

        train_transform = transforms.Compose([
            transforms.Resize((384, 192), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((384, 192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # self.trainset = MSMT17(train_transform, 'train', opt.data_path)
        # self.testset = MSMT17(test_transform, 'test', opt.data_path)
        # self.queryset = MSMT17(test_transform, 'query', opt.data_path)
        self.trainset = Market1501(train_transform, 'train', opt.data_path)
        # self.trainset = Market1501(train_transform, 'cam_transfer', opt.data_path)
        # self.trainset = Market1501(train_transform, 'motion_transfer', opt.data_path)
        print('Source Train Set:{}'.format(len(self.trainset)))
        self.testset = Market1501(test_transform, 'test', opt.data_path)
        print('Source Test Set:{}'.format(len(self.testset)))
        self.queryset = Market1501(test_transform, 'query', opt.data_path)
        print('Source Query Set:{}'.format(len(self.queryset)))

        # self.mqueryset = Market1501(test_transform, 'multi-query', opt.data_path)

        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=opt.batchid, batch_image=opt.batchimage),
                                                  drop_last=True,
                                                  batch_size=opt.batchid*opt.batchimage, num_workers=8,pin_memory=True)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=8,pin_memory=True)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, num_workers=8,pin_memory=True)
        # self.mquery_loader = dataloader.DataLoader(self.mqueryset, batch_size=opt.batchtest, num_workers=8,
        #                                           pin_memory=True)



        ###########################MARS######################################
        # mars = Mars()
        #
        # # decompose tracklets into images
        # new_train = []
        # for img_paths, pid, camid in mars.train:
        #     for img_path in img_paths:
        #         new_train.append((img_path, pid, camid))
        # # new_test = []
        # # for img_paths, pid, camid in mars.gallery:
        # #     for img_path in img_paths:
        # #         new_test.append((img_path, pid, camid))
        # # new_query = []
        # # for img_paths, pid, camid in mars.query:
        # #     for img_path in img_paths:
        # #         new_query.append((img_path, pid, camid))
        #
        # self.trainset = ImageDataset(new_train, transform=train_transform)
        # self.testset = VideoDataset(mars.gallery, seq_len=15, sample='evenly', transform=test_transform)
        # self.queryset = VideoDataset(mars.query, seq_len=15, sample='evenly', transform=test_transform)
        #
        # self.train_loader = dataloader.DataLoader(
        #     self.trainset,
        #     sampler=RandomSampler(self.trainset, batch_id=opt.batchid, batch_image=opt.batchimage),
        #     batch_size=opt.batchid*opt.batchimage, num_workers=8,
        #     pin_memory=True, drop_last=True,
        # )
        #
        # self.query_loader = dataloader.DataLoader(
        #     self.queryset,
        #     batch_size=opt.batchtest, shuffle=False, num_workers=8,
        #     pin_memory=True, drop_last=False,
        # )
        #
        # self.test_loader = dataloader.DataLoader(
        #     self.testset,
        #     batch_size=opt.batchtest, shuffle=False, num_workers=0,
        #     pin_memory=True, drop_last=False,
        # )


class Market1501(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):

        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        if dtype == 'train':
            self.data_path += '/bounding_box_train'
        elif dtype == 'test':
            self.data_path += '/bounding_box_test'
        elif dtype == 'multi-query':
            self.data_path += '/gt_bbox'
        else:
            self.data_path += '/query'

        self.imgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]
        # cam = self.camera(path)-1

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    def list_pictures(self,directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])


class MSMT17(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):

        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        if dtype == 'train':
            self.data_path += '/bounding_box_train'
        elif dtype == 'test':
            self.data_path += '/bounding_box_test'
        else:
            self.data_path += '/query'

        self.imgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]
        # cam = self.camera(path)-1

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[2])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    def list_pictures(self,directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])


class Mars(object):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.

    URL: http://www.liangzheng.com.cn/Project/project_mars.html

    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
    # cameras: 6
    """
    dataset_dir = 'MARS'

    def __init__(self, root='data', min_seq_len=0, verbose=True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_name_path = osp.join(self.dataset_dir, 'info/train_name.txt')
        self.test_name_path = osp.join(self.dataset_dir, 'info/test_name.txt')
        self.track_train_info_path = osp.join(self.dataset_dir, 'info/tracks_train_info.mat')
        self.track_test_info_path = osp.join(self.dataset_dir, 'info/tracks_test_info.mat')
        self.query_IDX_path = osp.join(self.dataset_dir, 'info/query_IDX.mat')

        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        # print(train_names[0])
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        query_IDX -= 1 # index from 0
        # print(track_train[0:5,:])
        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
          self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
          self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
          self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        if verbose:
            print("=> MARS loaded")
            print("Dataset statistics:")
            print("  ------------------------------------------")
            print("  subset   | # ids | # tracklets")
            print("  ------------------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(np.min(num_train_imgs), np.max(num_train_imgs), np.mean(num_train_imgs)))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(np.min(num_query_imgs), np.max(num_query_imgs), np.mean(num_query_imgs)))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(np.min(num_gallery_imgs), np.max(num_gallery_imgs), np.mean(num_gallery_imgs)))
            print("  ------------------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
            print("  ------------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))  # unique pids
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            # camid -= 1 # index starts from 0
            img_names = names[start_index-1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.dataset_dir, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(dataset.Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        # print(dataset)
        self.imgs = []
        self.ids = []
        self.cameras = []
        for i in range(len(dataset)):
            self.imgs.append(dataset[i][0])
            self.ids.append(self.id(dataset[i][0]))
            self.cameras.append(self.camera(dataset[i][0]))
        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}
        # print(self.imgs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1][0:4])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1][5])

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))


class VideoDataset(dataset.Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample == 'random':
            """
            Randomly sample seq_len items from num items,
            if num is smaller than seq_len, then replicate items
            """
            indices = np.arange(num)
            replace = False if num >= self.seq_len else True
            indices = np.random.choice(indices, size=self.seq_len, replace=replace)
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)
        elif self.sample == 'evenly':
            """
            Evenly sample seq_len items from num items.
            """
            if num >= self.seq_len:
                num -= num % self.seq_len
                indices = np.arange(0, num, num / self.seq_len)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32) * (num - 1)])
            assert len(indices) == self.seq_len
        elif self.sample == 'all':
            """
            Sample all items, seq_len is useless now and batch_size needs
            to be set to 1.
            """
            indices = np.arange(num)
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

        imgs = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, camid


# train_transform = transforms.Compose([
#             transforms.Resize((384, 192), interpolation=3),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

# mars = Mars()
# # decompose tracklets into images
# new_train = []
# for img_paths, pid, camid in mars.train:
#     for img_path in img_paths:
#         new_train.append((img_path, pid, camid))
#
# trainset = ImageDataset(new_train, transform=train_transform)
# print(trainset[0])

# data = Data()
# trainset = data.trainset
# queryset = data.queryset
# testset = data.testset
# print(len(queryset.imgs))
# print(len(testset.imgs))
# for item in testset.unique_ids:
#     if item not in queryset.unique_ids: print(item)
# # print(trainset.find_id('000001.jpg'))
# samples = next(iter(data.train_loader))
# print(samples[2])