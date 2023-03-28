# Used to collect the results for the LVOT EB0 column of Table 5 in the paper
# (Pre-trained experiments using the EfficientNet-B0 model, repeated for 3 times by using the models trained with different seeds)


mkdir dump_repeat_ocxr_eb0

echo "GradCAM Object-CXR Test Smoothed (1)"
python custom_gradcam.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA.pt --reprod-id=1 --tau=15 --cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/gc_ocxr_test_s_1.txt

echo "Guided Grad-CAM Object-CXR Test Smoothed (1)"
python custom_guided_gradcam.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA.pt --reprod-id=1 --tau=15 --cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ggc_ocxr_test_s_1.txt

echo "Input x Grad Object-CXR Test Smoothed (1)"
python custom_grad_times_image.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA.pt --reprod-id=1 --tau=15 --cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ixg_ocxr_test_s_1.txt

echo "Guided Backpropagation Object-CXR Test Smoothed (1)"
python custom_guided_backprop.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA.pt --reprod-id=1 --tau=15 --cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/gbp_ocxr_test_s_1.txt

echo "NormGrad Object-CXR Test Scaling Smoothed (1)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_scaling_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA.pt --reprod-id=1 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_scaling_s_1.txt

echo "NormGrad Object-CXR Test Combined Layers Scaling Smoothed (1)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_combined_scaling_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA.pt --reprod-id=1 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_scaling_combined_s_1.txt

echo "NormGrad Object-CXR Test Conv1x1 Smoothed (1)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_conv1x1_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA.pt --reprod-id=1 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_conv1x1_s_1.txt

echo "NormGrad Object-CXR Test Combined Layers Conv1x1 Smoothed (1)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_combined_conv1x1_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA.pt --reprod-id=1 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_conv1x1_combined_s_1.txt

echo "NormGrad Object-CXR Test Conv3x3 Smoothed (1)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_conv3x3_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA.pt --reprod-id=1 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_conv3x3_s_1.txt

echo "NormGrad Object-CXR Test Combined Layers Conv3x3 Smoothed (1)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_combined_conv3x3_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA.pt --reprod-id=1 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_conv3x3_combined_s_1.txt


echo "GradCAM Object-CXR Test Smoothed (2)"
python custom_gradcam.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_2.pt --reprod-id=2 --tau=15 --cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/gc_ocxr_test_s_2.txt

echo "Guided Grad-CAM Object-CXR Test Smoothed (2)"
python custom_guided_gradcam.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_2.pt --reprod-id=2 --tau=15 --cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ggc_ocxr_test_s_2.txt

echo "Input x Grad Object-CXR Test Smoothed (2)"
python custom_grad_times_image.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_2.pt --reprod-id=2 --tau=15 --cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ixg_ocxr_test_s_2.txt

echo "Guided Backpropagation Object-CXR Test Smoothed (2)"
python custom_guided_backprop.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_2.pt --reprod-id=2 --tau=15 --cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/gbp_ocxr_test_s_2.txt

echo "NormGrad Object-CXR Test Scaling Smoothed (2)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_scaling_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_2.pt --reprod-id=2 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_scaling_s_2.txt

echo "NormGrad Object-CXR Test Combined Layers Scaling Smoothed (2)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_combined_scaling_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_2.pt --reprod-id=2 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_scaling_combined_s_2.txt

echo "NormGrad Object-CXR Test Conv1x1 Smoothed (2)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_conv1x1_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_2.pt --reprod-id=2 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_conv1x1_s_2.txt

echo "NormGrad Object-CXR Test Combined Layers Conv1x1 Smoothed (2)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_combined_conv1x1_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_2.pt --reprod-id=2 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_conv1x1_combined_s_2.txt

echo "NormGrad Object-CXR Test Conv3x3 Smoothed (2)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_conv3x3_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_2.pt --reprod-id=2 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_conv3x3_s_2.txt

echo "NormGrad Object-CXR Test Combined Layers Conv3x3 Smoothed (2)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_combined_conv3x3_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_2.pt --reprod-id=2 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_conv3x3_combined_s_2.txt


echo "GradCAM Object-CXR Test Smoothed (3)"
python custom_gradcam.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_3.pt --reprod-id=3 --tau=15 --cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/gc_ocxr_test_s_3.txt

echo "Guided Grad-CAM Object-CXR Test Smoothed (3)"
python custom_guided_gradcam.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_3.pt --reprod-id=3 --tau=15 --cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ggc_ocxr_test_s_3.txt

echo "Input x Grad Object-CXR Test Smoothed (3)"
python custom_grad_times_image.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_3.pt --reprod-id=3 --tau=15 --cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ixg_ocxr_test_s_3.txt

echo "Guided Backpropagation Object-CXR Test Smoothed (3)"
python custom_guided_backprop.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_3.pt --reprod-id=3 --tau=15 --cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/gbp_ocxr_test_s_3.txt

echo "NormGrad Object-CXR Test Scaling Smoothed (3)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_scaling_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_3.pt --reprod-id=3 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_scaling_s_3.txt

echo "NormGrad Object-CXR Test Combined Layers Scaling Smoothed (3)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_combined_scaling_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_3.pt --reprod-id=3 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_scaling_combined_s_3.txt

echo "NormGrad Object-CXR Test Conv1x1 Smoothed (3)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_conv1x1_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_3.pt --reprod-id=3 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_conv1x1_s_3.txt

echo "NormGrad Object-CXR Test Combined Layers Conv1x1 Smoothed (3)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_combined_conv1x1_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_3.pt --reprod-id=3 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_conv1x1_combined_s_3.txt

echo "NormGrad Object-CXR Test Conv3x3 Smoothed (3)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_conv3x3_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_3.pt --reprod-id=3 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_conv3x3_s_3.txt

echo "NormGrad Object-CXR Test Combined Layers Conv3x3 Smoothed (3)"
python custom_normgrad.py --yaml=configs/object_CXR/EB0_object-CXR_cls_test_combined_conv3x3_smoothed.yaml --model-path=models/object_CXR/EB0_1x_DA_3.pt --reprod-id=3 --tau=15 --device cuda 2>&1 | tee -a dump_repeat_ocxr_eb0/ng_ocxr_test_conv3x3_combined_s_3.txt

