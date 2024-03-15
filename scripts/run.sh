# todo GOPRO Data 
# 1. Download Data

#? Structure:
#? GOPRO
#? ├── train
#? │   ├── raw_data
#? │   │   ├── GOPR0372_07_00
#? │   │   ├──      ...
#? │   │   └── GOPR0884_11_00
#? │
#? ├── test
#? │   ├── raw_data
#? │   │   ├── GOPR0384_11_00
#? │   │   ├──      ...
#? │   │   └── GOPR0881_11_01

# 2. Interpolate frames
cd XVFI-main/
python main.py --custom_path ../GOPRO/test/raw_data --gpu 0 --phase test_custom --exp_num 1 --dataset X4K1000FPS --module_scale_factor 4 --S_tst 5 --multiple 8 
python main.py --custom_path ../GOPRO/train/raw_data --gpu 0 --phase test_custom --exp_num 1 --dataset X4K1000FPS --module_scale_factor 4 --S_tst 5 --multiple 8 
cd ..

# 3. Synthesize the blurry image
python blur_syn.py --raw_folder GOPRO/test/raw_data --overlap_len 7 --blur_num 13
python blur_syn.py --raw_folder GOPRO/train/raw_data --overlap_len 7 --blur_num 13

# 4. Simulate the spike
python spike_simulate.py --raw_folder GOPRO/test/raw_data --overlap_len 7 --use_resize --blur_num 13
python spike_simulate.py --raw_folder GOPRO/train/raw_data --overlap_len 7 --use_resize --blur_num 13

# 5. extract GT from raw_folder 
## single frame
python sharp_extract.py --raw_folder GOPRO/test/raw_data --overlap_len 7 --blur_num 13
python sharp_extract.py --raw_folder GOPRO/train/raw_data --overlap_len 7 --blur_num 13

## sequence
# python sharp_extract.py --raw_folder GOPRO/test/raw_data --overlap_len 7 --blur_num 13 --multi
# python sharp_extract.py --raw_folder GOPRO/train/raw_data --overlap_len 7 --blur_num 13 --multi