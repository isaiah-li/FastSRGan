#python infer.py --image_dir 'path/to/your/image/directory' --output_dir 'path/to/save/super/resolution/images'
#python infer.py --image_dir '/mnt/SuperResolution/data/test_Internet' --output_dir 'result/test_Internet_result'
#python infer.py --image_dir '/mnt/SuperResolution/data/test_Internet' --output_dir 'result/test_Internet_result-1' --model 'models/generator_test.h5'
python infer.py --image_dir '/mnt/SuperResolution/data/test_Internet' --output_dir 'result/test_Internet_result-2k' --model 'models/generator_test_2k.h5'

