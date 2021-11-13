echo "" > sliced_Celeb1M/train_img.csv
echo "" > sliced_Celeb1M/train_id.txt
echo "" > sliced_Celeb1M/test_img.csv
echo "" > sliced_Celeb1M/test_id.txt
python3 slice_celeb1m.py --root2files '.' --output_root 'sliced_Celeb1M' --train_portion 0.7 > output.txt 2>&1 &

