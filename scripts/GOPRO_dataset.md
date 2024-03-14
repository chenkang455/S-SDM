## Step 1: Obtain the Original GOPRO Dataset
Step into the  `scripts` subfolder:
```
cd scripts
``` 

Download the [GOPRO_Large_all](https://drive.google.com/file/d/1rJTmM9_mLCNzBUUhYIGldBYgup279E_f/view) from the [GOPRO website](https://seungjunnah.github.io/Datasets/gopro) to get the sharp sequence for simulating spikes and synthesizing blurry frames. After downloading the data, rename the data file to `GOPRO` and place it in the `scripts` directory. Add a subfolder `raw_folder` in both `train` and `test` folders. The file structure is as follows:
```
scripts
├── XVFI-main
├── run.sh
├── ...
└── GOPRO
    ├── train
    │   └── raw_data
    │       ├── GOPR0372_07_00
    │       ├── ...
    │       └── GOPR0884_11_00
    └── test
        └── raw_data
            ├── GOPR0384_11_00
            ├── ...
            └── GOPR0881_11_01
```

## Step 2: Frame Interpolation
We use the XVFI frame interpolation algorithm to insert 7 additional imgs between two adjacent imgs, increasing the frame rate of the GOPRO image sequence. Run

```
cd XVFI-main/
python main.py --custom_path ../GOPRO/test/raw_data --gpu 0 --phase test_custom --exp_num 1 --dataset X4K1000FPS --module_scale_factor 4 --S_tst 5 --multiple 8 
python main.py --custom_path ../GOPRO/train/raw_data --gpu 0 --phase test_custom --exp_num 1 --dataset X4K1000FPS --module_scale_factor 4 --S_tst 5 --multiple 8 
cd ..
```

## Step 3: Blur Synthesis
We synthesize a blurred frame using 97 images in the dataset after frame interpolation (corresponding to 13 images before interpolation). Run

```
python blur_syn.py --raw_folder GOPRO/test/raw_data --overlap_len 7 --blur_num 13
python blur_syn.py --raw_folder GOPRO/train/raw_data --overlap_len 7 --blur_num 13
```

## Step 4: Spike Simulation
We resize the image size from `720×1280` to `180×320` and apply a spike generation physical model to simulate low-resolution spikes, obtaining the spike stream corresponding to the virtual exposure time in `Step 3: Blur Synthesis`. Run

```
python spike_simulate.py --raw_folder GOPRO/test/raw_data --overlap_len 7 --use_resize --blur_num 13
python spike_simulate.py --raw_folder GOPRO/train/raw_data --overlap_len 7 --use_resize --blur_num 13
```

## Step 5: Sharp Extract
For obtaining the single sharp frame corresponding to the blurry frame:

```
python sharp_extract.py --raw_folder GOPRO/test/raw_data --overlap_len 7 --blur_num 13
python sharp_extract.py --raw_folder GOPRO/train/raw_data --overlap_len 7 --blur_num 13
```

For obtaining the sharp sequence (13 images in this example) corresponding to the blurry frame:
```
python sharp_extract.py --raw_folder GOPRO/test/raw_data --overlap_len 7 --blur_num 13 --multi
python sharp_extract.py --raw_folder GOPRO/train/raw_data --overlap_len 7 --blur_num 13 --multi
```

## Step 6: Final
Omit the raw_folder, the structure of the `Spike-GOPRO` dataset is as follows:
```
GOPRO
├── test
│   ├── blur_data
│   ├── sharp_data
│   └── spike_data
└── train
    ├── blur_data
    ├── sharp_data
    └── spike_data
```