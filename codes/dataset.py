from codes.utils import *
import glob
from torchvision import transforms
# Spike Dataset
class SpikeData(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_type = 'GOPRO', stage = 'train',
                 use_resize = False,use_roi = False,roi_size = [256,256],
                 use_small = False,spike_full = False):
        """ Spike Dataset

        Args:
            root_dir (str): base folder of the dataset
            data_type (str, optional): data type. Defaults to 'GOPRO'.
            stage (str, optional): train / test. Defaults to 'train'.
            use_resize (bool, optional): . Defaults to False.
            use_roi (bool, optional): ROI operation. Defaults to False.
            roi_size (list, optional): ROI size for the image, [ROI/4,ROI/4] for the spike. Defaults to [256,256].
            use_small (bool, optional): small dataset for debugging. Defaults to False.
            spike_full (bool, optional): full spike size (1280 * 720) instead of (320 * 180). Defaults to False.
        """
        self.root_dir = root_dir
        self.data_type = data_type
        self.use_resize = use_resize
        self.use_roi = use_roi
        self.roi_size = roi_size
        self.spike_full = spike_full
        # Real DataSet
        if data_type == 'real':
            pattern = os.path.join(self.root_dir,'spike_data', '*','*')
            self.spike_list = sorted(glob.glob(pattern))
            self.width = 400 * 4
            self.height = 250 * 4
        # GOPRO Synthetic DataSet
        elif data_type == 'GOPRO':
            if self.spike_full:
                pattern = os.path.join(self.root_dir, stage,'spike_full', '*','*')
            else:
                pattern = os.path.join(self.root_dir, stage,'spike_data', '*','*')
            self.spike_list = sorted(glob.glob(pattern))
            if use_small == True:
                self.spike_list = self.spike_list[::10]
            self.width = 1280
            self.height = 720
        self.resize = transforms.Resize((self.height // 2,self.width // 2),interpolation=transforms.InterpolationMode.NEAREST)
        self.img_size = (self.height,self.width) if use_resize == False else (self.height // 2,self.width // 2)
        self.length = len(self.spike_list)
        
    def __getitem__(self, index: int):
        # blur and spike load
        if  self.data_type in ['real']:
            spike_name = self.spike_list[index]
            spike = load_vidar_dat(spike_name,width=self.width //4 ,height=self.height // 4)
            blur_name = spike_name.replace('.dat','.jpg').replace('spike_data','blur_data')
            blur = cv2.imread(blur_name)
        elif self.data_type in ['GOPRO']:
            spike_name = self.spike_list[index]
            if self.spike_full:
                spike = load_vidar_dat(spike_name,width=self.width ,height=self.height )
                blur_name = spike_name.replace('.dat','.png').replace('spike_full','blur_data')
            else:
                spike = load_vidar_dat(spike_name,width=self.width // 4,height=self.height // 4)
                blur_name = spike_name.replace('.dat','.png').replace('spike_data','blur_data')
            blur = cv2.imread(blur_name)
        
        # sharp load
        if self.data_type in ['real']:
            sharp = np.zeros_like(blur)
        elif self.data_type in ['GOPRO']:
            if self.spike_full:
                sharp_name = spike_name.replace('.dat','.png').replace('spike_full','sharp_data')
            else:
                sharp_name = spike_name.replace('.dat','.png').replace('spike_data','sharp_data')
            sharp = cv2.imread(sharp_name)
        
        # channel & property exchange
        blur = torch.from_numpy(blur).permute((2,0,1)).float() / 255
        sharp = torch.from_numpy(sharp).permute((2,0,1)).float() / 255
        spike = torch.from_numpy(spike)
        # resize method (set true for synthetic NeRF dataset and false for real dataset)
        if self.use_resize == True:
            blur,spike,sharp = self.resize(blur),self.resize(spike),self.resize(sharp)
        # randomly crop
        if self.use_roi == True:
            if self.data_type not in ['GOPRO','real']:
                roiTL = (np.random.randint(0, self.img_size[0] -self.roi_size[0]+1), np.random.randint(0, self.img_size[1] -self.roi_size[1]+1))
                roiBR = (roiTL[0]+self.roi_size[0],roiTL[1]+self.roi_size[1])
                blur = blur[:,roiTL[0]:roiBR[0], roiTL[1]:roiBR[1]]
                spike = spike[:,roiTL[0]:roiBR[0], roiTL[1]:roiBR[1]]
                sharp = sharp[:,roiTL[0]:roiBR[0], roiTL[1]:roiBR[1]]
            else:
                roiTL = (np.random.randint(0, self.height // 4 - self.roi_size[0] // 4 +1), np.random.randint(0, self.width // 4 - self.roi_size[1] // 4+1))
                roiBR = (roiTL[0]+self.roi_size[0]//4,roiTL[1]+self.roi_size[1]//4)
                blur = blur[:,4 * roiTL[0]:4 * roiBR[0], 4 * roiTL[1]:4 * roiBR[1]]
                if self.spike_full:
                    spike = spike[:,4 * roiTL[0]:4 * roiBR[0], 4 * roiTL[1]:4 * roiBR[1]]
                else:
                    spike = spike[:,roiTL[0]:roiBR[0], roiTL[1]:roiBR[1]]
                sharp = sharp[:,4 * roiTL[0]:4 *roiBR[0], 4 *roiTL[1]:4 * roiBR[1]]
        return blur,spike,sharp

    def __len__(self):
        return self.length