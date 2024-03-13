# todo GOPRO Data 
#? Structure:
#? GOPRO
#? ├── train
#? │   ├── raw_data
#? │   │   ├── 000
#? │   │   └── 001
#? │
#? ├── test
#? │   ├── raw_data
#? │   │   ├── 000
#? │   │   └── 001
# 1. Download Data

# 2. Interpolate frames
cd XVFI-main/
python main.py --gpu 0 --phase test_custom --exp_num 1 --dataset X4K1000FPS --module_scale_factor 4 --S_tst 5 --multiple 8 --custom_path E:\chenk\GOPRO\test\raw_data
python main.py --gpu 0 --phase test_custom --exp_num 1 --dataset X4K1000FPS --module_scale_factor 4 --S_tst 5 --multiple 8 --custom_path E:\chenk\GOPRO\train\raw_data
cd ..
# 3. Synthesize the blurry image
python blur_syn.py --sharp_folder E:\chenk\Data\GOPRO\test\raw_data --overlap_len 7 --blur_num 13
python blur_syn.py --sharp_folder E:\chenk\Data\GOPRO\train\raw_data --overlap_len 7 --blur_num 13

# 4. Simulate the spike
python spike_simulate.py --sharp_folder E:\chenk\Data\GOPRO\test\raw_data --overlap_len 7 --use_resize --blur_num 13
python spike_simulate.py --sharp_folder E:\chenk\Data\GOPRO\train\raw_data --overlap_len 7 --use_resize --blur_num 13

# todo no resize 
python spike_simulate.py --sharp_folder E:\chenk\Data\GOPRO\test\raw_data --overlap_len 7 --blur_num 13


# 5. extract GT from raw_folder
python sharp_extract.py --raw_folder E:\chenk\Data\GOPRO\test\raw_data --overlap_len 7 --blur_num 13
python sharp_extract.py --raw_folder E:\chenk\Data\GOPRO\train\raw_data --overlap_len 7 --blur_num 13

python sharp_extract.py --raw_folder E:\chenk\Data\GOPRO\test\raw_data --overlap_len 7 --blur_num 13 --multi
python sharp_extract.py --raw_folder E:\chenk\Data\GOPRO\train\raw_data --overlap_len 7 --blur_num 13 --multi

# test
python blur_syn.py --sharp_folder E:\chenk\GOPRO_test\raw_data --overlap_len 7  

python spike_simulate.py --sharp_folder E:\chenk\GOPRO_test\raw_data --overlap_len 7 

python sharp_extract.py --raw_folder E:\chenk\GOPRO_test\raw_data --overlap_len 7 

#todo GOPRO test
# 3. Synthesize the blurry image
# overlap 1 img [0,1,2,3,4,5,6] -> blur0 ; [7,8,9,10,11,12,13] -> blur2
python blur_syn.py --sharp_folder E:/chenk/Data/GOPRO_test/test/raw_data --overlap_len 7  --blur_num 13

# 4. Simulate the spike
python spike_simulate.py --sharp_folder E:/chenk/Data/GOPRO_test/test/raw_data --overlap_len 7  --blur_num 13 --use_resize

# 5. Extract the sharp images from raw_folder and construct the corresponding sharp imgs according to the blurry image
python sharp_extract.py --raw_folder E:/chenk/Data/GOPRO_test/test/raw_data --overlap_len 7 --blur_num 13