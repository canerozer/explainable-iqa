# Used to collect the results for the Object-CXR merged-column of Table 2 in the paper
# (Uses the best model from the LVOT detection experiments)


mkdir dump_smooth_vs_not_ocxr

echo "GradCAM Object-CXR Test"
python custom_gradcam.py --yaml=configs/object_CXR/R34_object-CXR_cls_test.yaml --cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/gc_ocxr_test.txt

echo "GradCAM Object-CXR Test Smoothed"
python custom_gradcam.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_smoothed.yaml --cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/gc_ocxr_test_s.txt

echo "Guided Grad-CAM Object-CXR Test"
python custom_guided_gradcam.py --yaml=configs/object_CXR/R34_object-CXR_cls_test.yaml --cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/ggc_ocxr_test.txt

echo "Guided Grad-CAM Object-CXR Test Smoothed"
python custom_guided_gradcam.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_smoothed.yaml --cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/ggc_ocxr_test_s.txt

echo "Input x Grad Object-CXR Test"
python custom_grad_times_image.py --yaml=configs/object_CXR/R34_object-CXR_cls_test.yaml --cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/ixg_ocxr_test.txt

echo "Input x Grad Object-CXR Test Smoothed"
python custom_grad_times_image.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_smoothed.yaml --cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/ixg_ocxr_test_s.txt

echo "Guided Backpropagation Object-CXR Test"
python custom_guided_backprop.py --yaml=configs/object_CXR/R34_object-CXR_cls_test.yaml --cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/gbp_ocxr_test.txt

echo "Guided Backpropagation Object-CXR Test Smoothed"
python custom_guided_backprop.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_smoothed.yaml --cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/gbp_ocxr_test_s.txt

echo "NormGrad Object-CXR Test Scaling"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_scaling.yaml --device cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/ng_ocxr_test_scaling.txt

echo "NormGrad Object-CXR Test Scaling Smoothed"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_scaling_smoothed.yaml --device cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/ng_ocxr_test_scaling_s.txt

echo "NormGrad Object-CXR Test Combined Layers Scaling"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_combined_scaling.yaml --device cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/ng_ocxr_test_scaling_combined.txt

echo "NormGrad Object-CXR Test Combined Layers Scaling Smoothed"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_combined_scaling_smoothed.yaml --device cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/ng_ocxr_test_scaling_combined_s.txt

echo "NormGrad Object-CXR Test Conv1x1"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_conv1x1.yaml --device cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/ng_ocxr_test_conv1x1.txt

echo "NormGrad Object-CXR Test Conv1x1 Smoothed"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_conv1x1_smoothed.yaml --device cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/ng_ocxr_test_conv1x1_s.txt

echo "NormGrad Object-CXR Test Combined Layers Conv1x1"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_combined_conv1x1.yaml --device cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/ng_ocxr_test_conv1x1_combined.txt

echo "NormGrad Object-CXR Test Combined Layers Conv1x1 Smoothed"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_combined_conv1x1_smoothed.yaml --device cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/ng_ocxr_test_conv1x1_combined_s.txt

echo "NormGrad Object-CXR Test Conv3x3"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_conv3x3.yaml --device cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/ng_ocxr_test_conv3x3.txt

echo "NormGrad Object-CXR Test Conv3x3 Smoothed"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_conv3x3_smoothed.yaml --device cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/ng_ocxr_test_conv3x3_s.txt

echo "NormGrad Object-CXR Test Combined Layers Conv3x3"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_combined_conv3x3.yaml --device cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/ng_ocxr_test_conv3x3_combined.txt

echo "NormGrad Object-CXR Test Combined Layers Conv3x3 Smoothed"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_combined_conv3x3_smoothed.yaml --device cuda --tau=15 2>&1 | tee -a dump_smooth_vs_not_ocxr/ng_ocxr_test_conv3x3_combined_s.txt

