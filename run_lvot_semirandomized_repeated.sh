# Used to collect the results for the SR column of Table 3 in the paper
# (Semi-Randomized experiments, repeated for 3 times with only different random parameters at the classifier layer)


mkdir dump_semirandomized_repeat_lvot

echo "GradCAM LVOT Detection Test Smoothed 1"
python custom_gradcam.py --yaml=configs/LVOT/R50_LVOT_test_smoothed.yaml --reprod-id=1 --randomized --cuda --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/gc_lvot_test_s_1.txt

echo "Guided Grad-CAM LVOT Detection Test Smoothed 1"
python custom_guided_gradcam.py --yaml=configs/LVOT/R50_LVOT_test_smoothed.yaml --reprod-id=1 --randomized --cuda --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ggc_lvot_test_s_1.txt

echo "Input x Grad LVOT Detection Test Smoothed 1"
python custom_grad_times_image.py --yaml=configs/LVOT/R50_LVOT_test_smoothed.yaml --reprod-id=1 --randomized --cuda --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ixg_lvot_test_s_1.txt

echo "Guided Backpropagation LVOT Detection Test Smoothed 1"
python custom_guided_backprop.py --yaml=configs/LVOT/R50_LVOT_test_smoothed.yaml --reprod-id=1 --randomized --cuda --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/gbp_lvot_test_s_1.txt

echo "NormGrad LVOT Detection Test Scaling Smoothed 1"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_scaling_smoothed.yaml --device cuda --reprod-id=1 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_scaling_s_1.txt

echo "NormGrad LVOT Detection Test Combined Layers Scaling Smoothed 1"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_combined_scaling_smoothed.yaml --device cuda --reprod-id=1 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_scaling_combined_s_1.txt

echo "NormGrad LVOT Detection Test Conv1x1 Smoothed 1"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_conv1x1_smoothed.yaml --device cuda --reprod-id=1 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_conv1x1_s_1.txt

echo "NormGrad LVOT Detection Test Combined Layers Conv1x1 Smoothed 1"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_combined_conv1x1_smoothed.yaml --device cuda --reprod-id=1 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_conv1x1_combined_s_1.txt

echo "NormGrad LVOT Detection Test Conv3x3 Smoothed 1"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_conv3x3_smoothed.yaml --device cuda --reprod-id=1 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_conv3x3_s_1.txt

echo "NormGrad LVOT Detection Test Combined Layers Conv3x3 Smoothed 1"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_combined_conv3x3_smoothed.yaml --device cuda --reprod-id=1 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_conv3x3_combined_s_1.txt

echo "GradCAM LVOT Detection Test Smoothed 2"
python custom_gradcam.py --yaml=configs/LVOT/R50_LVOT_test_smoothed.yaml --reprod-id=2 --randomized --cuda --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/gc_lvot_test_s_2.txt

echo "Guided Grad-CAM LVOT Detection Test Smoothed 2"
python custom_guided_gradcam.py --yaml=configs/LVOT/R50_LVOT_test_smoothed.yaml --reprod-id=2 --randomized --cuda --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ggc_lvot_test_s_2.txt

echo "Input x Grad LVOT Detection Test Smoothed 2"
python custom_grad_times_image.py --yaml=configs/LVOT/R50_LVOT_test_smoothed.yaml --reprod-id=2 --randomized --cuda --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ixg_lvot_test_s_2.txt

echo "Guided Backpropagation LVOT Detection Test Smoothed 2"
python custom_guided_backprop.py --yaml=configs/LVOT/R50_LVOT_test_smoothed.yaml --reprod-id=2 --randomized --cuda --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/gbp_lvot_test_s_2.txt

echo "NormGrad LVOT Detection Test Scaling Smoothed 2"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_scaling_smoothed.yaml --device cuda --reprod-id=2 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_scaling_s_2.txt

echo "NormGrad LVOT Detection Test Combined Layers Scaling Smoothed 2"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_combined_scaling_smoothed.yaml --device cuda --reprod-id=2 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_scaling_combined_s_2.txt

echo "NormGrad LVOT Detection Test Conv1x1 Smoothed 2"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_conv1x1_smoothed.yaml --device cuda --reprod-id=2 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_conv1x1_s_2.txt

echo "NormGrad LVOT Detection Test Combined Layers Conv1x1 Smoothed 2"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_combined_conv1x1_smoothed.yaml --device cuda --reprod-id=2 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_conv1x1_combined_s_2.txt

echo "NormGrad LVOT Detection Test Conv3x3 Smoothed 2"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_conv3x3_smoothed.yaml --device cuda --reprod-id=2 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_conv3x3_s_2.txt

echo "NormGrad LVOT Detection Test Combined Layers Conv3x3 Smoothed 2"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_combined_conv3x3_smoothed.yaml --device cuda --reprod-id=2 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_conv3x3_combined_s_2.txt

echo "GradCAM LVOT Detection Test Smoothed 3"
python custom_gradcam.py --yaml=configs/LVOT/R50_LVOT_test_smoothed.yaml --reprod-id=3 --randomized --cuda --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/gc_lvot_test_s_3.txt

echo "Guided Grad-CAM LVOT Detection Test Smoothed 3"
python custom_guided_gradcam.py --yaml=configs/LVOT/R50_LVOT_test_smoothed.yaml --reprod-id=3 --randomized --cuda --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ggc_lvot_test_s_3.txt

echo "Input x Grad LVOT Detection Test Smoothed 3"
python custom_grad_times_image.py --yaml=configs/LVOT/R50_LVOT_test_smoothed.yaml --reprod-id=3 --randomized --cuda --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ixg_lvot_test_s_3.txt

echo "Guided Backpropagation LVOT Detection Test Smoothed 3"
python custom_guided_backprop.py --yaml=configs/LVOT/R50_LVOT_test_smoothed.yaml --reprod-id=3 --randomized --cuda --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/gbp_lvot_test_s_3.txt

echo "NormGrad LVOT Detection Test Scaling Smoothed 3"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_scaling_smoothed.yaml --device cuda --reprod-id=3 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_scaling_s_3.txt

echo "NormGrad LVOT Detection Test Combined Layers Scaling Smoothed 3"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_combined_scaling_smoothed.yaml --device cuda --reprod-id=3 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_scaling_combined_s_3.txt

echo "NormGrad LVOT Detection Test Conv1x1 Smoothed 3"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_conv1x1_smoothed.yaml --device cuda --reprod-id=3 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_conv1x1_s_3.txt

echo "NormGrad LVOT Detection Test Combined Layers Conv1x1 Smoothed 3"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_combined_conv1x1_smoothed.yaml --device cuda --reprod-id=3 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_conv1x1_combined_s_3.txt

echo "NormGrad LVOT Detection Test Conv3x3 Smoothed 3"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_conv3x3_smoothed.yaml --device cuda --reprod-id=3 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_conv3x3_s_3.txt

echo "NormGrad LVOT Detection Test Combined Layers Conv3x3 Smoothed 3"
python custom_normgrad.py --yaml=configs/LVOT/R50_LVOT_test_combined_conv3x3_smoothed.yaml --device cuda --reprod-id=3 --randomized --tau=15 2>&1 | tee -a dump_semirandomized_repeat_lvot/ng_lvot_test_conv3x3_combined_s_3.txt

