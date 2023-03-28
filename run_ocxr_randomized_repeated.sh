# Used to collect the results for the FR column of Table 4 in the paper
# (Randomized experiments, repeated for 3 times with different random parameters)


mkdir dump_randomized_repeat_ocxr

echo "GradCAM Object-CXR Foreign Object Detection Test Smoothed 1"
python custom_gradcam.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_smoothed.yaml --reprod-id=1 --randomized --no-pretrained --cuda --tau=15 2>&1 | tee -a dump_randomized_repeat_ocxr/gc_ocxr_test_s_1.txt

echo "Guided Grad-CAM Object-CXR Foreign Object Detection Test Smoothed 1"
python custom_guided_gradcam.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_smoothed.yaml --reprod-id=1 --randomized --no-pretrained --cuda --tau=15 2>&1 | tee -a dump_randomized_repeat_ocxr/ggc_ocxr_test_s_1.txt

echo "Input x Grad Object-CXR Foreign Object Detection Test Smoothed 1"
python custom_grad_times_image.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_smoothed.yaml --reprod-id=1 --randomized --no-pretrained --cuda --tau=15 2>&1 | tee -a dump_randomized_repeat_ocxr/ixg_ocxr_test_s_1.txt

echo "Guided Backpropagation Object-CXR Foreign Object Detection Test Smoothed 1"
python custom_guided_backprop.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_smoothed.yaml --reprod-id=1 --randomized --no-pretrained --cuda --tau=15 2>&1 | tee -a dump_randomized_repeat_ocxr/gbp_ocxr_test_s_1.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Scaling Smoothed 1"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_scaling_smoothed.yaml --device cuda --tau=15 --reprod-id=1 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_scaling_s_1.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Combined Layers Scaling Smoothed 1"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_combined_scaling_smoothed.yaml --device cuda --tau=15 --reprod-id=1 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_scaling_combined_s_1.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Conv1x1 Smoothed 1"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_conv1x1_smoothed.yaml --device cuda --tau=15 --reprod-id=1 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_conv1x1_s_1.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Combined Layers Conv1x1 Smoothed 1"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_combined_conv1x1_smoothed.yaml --device cuda --tau=15 --reprod-id=1 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_conv1x1_combined_s_1.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Conv3x3 Smoothed 1"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_conv3x3_smoothed.yaml --device cuda --tau=15 --reprod-id=1 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_conv3x3_s_1.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Combined Layers Conv3x3 Smoothed 1"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_combined_conv3x3_smoothed.yaml --device cuda --tau=15 --reprod-id=1 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_conv3x3_combined_s_1.txt

echo "GradCAM Object-CXR Foreign Object Detection Test Smoothed 2"
python custom_gradcam.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_smoothed.yaml --reprod-id=2 --randomized --no-pretrained --cuda --tau=15 2>&1 | tee -a dump_randomized_repeat_ocxr/gc_ocxr_test_s_2.txt

echo "Guided Grad-CAM Object-CXR Foreign Object Detection Test Smoothed 2"
python custom_guided_gradcam.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_smoothed.yaml --reprod-id=2 --randomized --no-pretrained --cuda --tau=15 2>&1 | tee -a dump_randomized_repeat_ocxr/ggc_ocxr_test_s_2.txt

echo "Input x Grad Object-CXR Foreign Object Detection Test Smoothed 2"
python custom_grad_times_image.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_smoothed.yaml --reprod-id=2 --randomized --no-pretrained --cuda --tau=15 2>&1 | tee -a dump_randomized_repeat_ocxr/ixg_ocxr_test_s_2.txt

echo "Guided Backpropagation Object-CXR Foreign Object Detection Test Smoothed 2"
python custom_guided_backprop.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_smoothed.yaml --reprod-id=2 --randomized --no-pretrained --cuda --tau=15 2>&1 | tee -a dump_randomized_repeat_ocxr/gbp_ocxr_test_s_2.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Scaling Smoothed 2"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_scaling_smoothed.yaml --device cuda --tau=15 --reprod-id=2 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_scaling_s_2.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Combined Layers Scaling Smoothed 2"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_combined_scaling_smoothed.yaml --device cuda --tau=15 --reprod-id=2 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_scaling_combined_s_2.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Conv1x1 Smoothed 2"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_conv1x1_smoothed.yaml --device cuda --tau=15 --reprod-id=2 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_conv1x1_s_2.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Combined Layers Conv1x1 Smoothed 2"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_combined_conv1x1_smoothed.yaml --device cuda --tau=15 --reprod-id=2 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_conv1x1_combined_s_2.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Conv3x3 Smoothed 2"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_conv3x3_smoothed.yaml --device cuda --tau=15 --reprod-id=2 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_conv3x3_s_2.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Combined Layers Conv3x3 Smoothed 2"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_combined_conv3x3_smoothed.yaml --device cuda --tau=15 --reprod-id=2 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_conv3x3_combined_s_2.txt

echo "GradCAM Object-CXR Foreign Object Detection Test Smoothed 3"
python custom_gradcam.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_smoothed.yaml --reprod-id=3 --randomized --no-pretrained --cuda --tau=15 2>&1 | tee -a dump_randomized_repeat_ocxr/gc_ocxr_test_s_3.txt

echo "Guided Grad-CAM Object-CXR Foreign Object Detection Test Smoothed 3"
python custom_guided_gradcam.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_smoothed.yaml --reprod-id=3 --randomized --no-pretrained --cuda --tau=15 2>&1 | tee -a dump_randomized_repeat_ocxr/ggc_ocxr_test_s_3.txt

echo "Input x Grad Object-CXR Foreign Object Detection Test Smoothed 3"
python custom_grad_times_image.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_smoothed.yaml --reprod-id=3 --randomized --no-pretrained --cuda --tau=15 2>&1 | tee -a dump_randomized_repeat_ocxr/ixg_ocxr_test_s_3.txt

echo "Guided Backpropagation Object-CXR Foreign Object Detection Test Smoothed 3"
python custom_guided_backprop.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_smoothed.yaml --reprod-id=3 --randomized --no-pretrained --cuda --tau=15 2>&1 | tee -a dump_randomized_repeat_ocxr/gbp_ocxr_test_s_3.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Scaling Smoothed 3"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_scaling_smoothed.yaml --device cuda --tau=15 --reprod-id=3 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_scaling_s_3.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Combined Layers Scaling Smoothed 3"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_combined_scaling_smoothed.yaml --device cuda --tau=15 --reprod-id=3 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_scaling_combined_s_3.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Conv1x1 Smoothed 3"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_conv1x1_smoothed.yaml --device cuda --tau=15 --reprod-id=3 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_conv1x1_s_3.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Combined Layers Conv1x1 Smoothed 3"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_combined_conv1x1_smoothed.yaml --device cuda --tau=15 --reprod-id=3 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_conv1x1_combined_s_3.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Conv3x3 Smoothed 3"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_conv3x3_smoothed.yaml --device cuda --tau=15 --reprod-id=3 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_conv3x3_s_3.txt

echo "NormGrad Object-CXR Foreign Object Detection Test Combined Layers Conv3x3 Smoothed 3"
python custom_normgrad.py --yaml=configs/object_CXR/R34_object-CXR_cls_test_combined_conv3x3_smoothed.yaml --device cuda --tau=15 --reprod-id=3 --randomized --no-pretrained 2>&1 | tee -a dump_randomized_repeat_ocxr/ng_ocxr_test_conv3x3_combined_s_3.txt

