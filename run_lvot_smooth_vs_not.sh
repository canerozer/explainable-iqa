# Used to collect the results for the LVOT merged-column of Table 2 in the paper
# (Uses the best model from the LVOT detection experiments)


mkdir dump_smooth_vs_not_lvot

echo "GradCAM LVOT Detection Test"
python custom_gradcam.py --yaml=configs/LVOT/R50_LVOT_test.yaml --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/gc_lvot_test.txt

echo "GradCAM LVOT Detection Test Smoothed"
python custom_gradcam.py --yaml=configs/LVOT/R50_LVOT_test_smoothed.yaml --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/gc_lvot_test_s.txt

echo "Guided Grad-CAM LVOT Detection Test"
python custom_guided_gradcam.py --yaml=configs/LVOT/R50_LVOT_test.yaml --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/ggc_lvot_test.txt

echo "Guided Grad-CAM LVOT Detection Test Smoothed"
python custom_guided_gradcam.py --yaml=configs/LVOT/R50_LVOT_test_smoothed.yaml --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/ggc_lvot_test_s.txt

echo "Input x Grad LVOT Detection Test"
python custom_grad_times_image.py --yaml=configs/LVOT/R50_LVOT_test.yaml --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/ixg_lvot_test.txt

echo "Input x Grad LVOT Detection Test Smoothed"
python custom_grad_times_image.py --yaml=configs/LVOT/R50_LVOT_test_smoothed.yaml --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/ixg_lvot_test_s.txt

echo "Guided Backpropagation LVOT Detection Test"
python custom_guided_backprop.py --yaml=configs/LVOT/R50_LVOT_test.yaml --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/gbp_lvot_test.txt

echo "Guided Backpropagation LVOT Detection Test Smoothed"
python custom_guided_backprop.py --yaml=configs/LVOT/R50_LVOT_test_smoothed.yaml --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/gbp_lvot_test_s.txt

echo "NormGrad LVOT Detection Test Scaling"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_scaling.yaml --device cuda --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --device cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/ng_lvot_test_scaling.txt

echo "NormGrad LVOT Detection Test Scaling Smoothed"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_scaling_smoothed.yaml --device cuda --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --device cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/ng_lvot_test_scaling_s.txt

echo "NormGrad LVOT Detection Test Combined Layers Scaling"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_combined_scaling.yaml --device cuda --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --device cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/ng_lvot_test_scaling_combined.txt

echo "NormGrad LVOT Detection Test Combined Layers Scaling Smoothed"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_combined_scaling_smoothed.yaml --device cuda --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --device cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/ng_lvot_test_scaling_combined_s.txt

echo "NormGrad LVOT Detection Test Conv1x1"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_conv1x1.yaml --device cuda --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --device cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/ng_lvot_test_conv1x1.txt

echo "NormGrad LVOT Detection Test Conv1x1 Smoothed"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_conv1x1_smoothed.yaml --device cuda --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --device cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/ng_lvot_test_conv1x1_s.txt

echo "NormGrad LVOT Detection Test Combined Layers Conv1x1"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_combined_conv1x1.yaml --device cuda --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --device cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/ng_lvot_test_conv1x1_combined.txt

echo "NormGrad LVOT Detection Test Combined Layers Conv1x1 Smoothed"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_combined_conv1x1_smoothed.yaml --device cuda --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --device cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/ng_lvot_test_conv1x1_combined_s.txt

echo "NormGrad LVOT Detection Test Conv3x3"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_conv3x3.yaml --device cuda --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --device cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/ng_lvot_test_conv3x3.txt

echo "NormGrad LVOT Detection Test Conv3x3 Smoothed"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_conv3x3_smoothed.yaml --device cuda --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --device cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/ng_lvot_test_conv3x3_s.txt

echo "NormGrad LVOT Detection Test Combined Layers Conv3x3"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_combined_conv3x3.yaml --device cuda --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --device cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/ng_lvot_test_conv3x3_combined.txt

echo "NormGrad LVOT Detection Test Combined Layers Conv3x3 Smoothed"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_combined_conv3x3_smoothed.yaml --device cuda --model-path models/LVOT/R50_3x_DA_224_0.9559.pt --reprod-id=2 --device cuda 2>&1 | tee -a dump_smooth_vs_not_lvot/ng_lvot_test_conv3x3_combined_s.txt

