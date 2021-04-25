echo "NormGrad Object-CXR Val Scaling"
python run_normgrad.py --yaml=configs/R34_object-CXR_cls_val_ng_scaling.yaml

echo "NormGrad Object-CXR Test Scaling"
python run_normgrad.py --yaml=configs/R34_object-CXR_cls_test_ng_scaling.yaml

echo "NormGrad Object-CXR Val Combined Layers Scaling"
python run_normgrad.py --yaml=configs/R34_object-CXR_cls_val_combined_ng_scaling.yaml

echo "NormGrad Object-CXR Test Combined Layers Scaling"
python run_normgrad.py --yaml=configs/R34_object-CXR_cls_test_combined_ng_scaling.yaml

echo "NormGrad Object-CXR Val Conv1x1"
python run_normgrad.py --yaml=configs/R34_object-CXR_cls_val_ng_conv1x1.yaml

echo "NormGrad Object-CXR Test Conv1x1"
python run_normgrad.py --yaml=configs/R34_object-CXR_cls_test_ng_conv1x1.yaml

echo "NormGrad Object-CXR Val Combined Layers Conv1x1"
python run_normgrad.py --yaml=configs/R34_object-CXR_cls_val_combined_ng_conv1x1.yaml

echo "NormGrad Object-CXR Test Combined Layers Conv1x1"
python run_normgrad.py --yaml=configs/R34_object-CXR_cls_test_combined_ng_conv1x1.yaml

echo "NormGrad Object-CXR Val Conv3x3"
python run_normgrad.py --yaml=configs/R34_object-CXR_cls_val_ng_conv3x3.yaml

echo "NormGrad Object-CXR Test Conv3x3"
python run_normgrad.py --yaml=configs/R34_object-CXR_cls_test_ng_conv3x3.yaml

echo "NormGrad Object-CXR Val Combined Layers Conv3x3"
python run_normgrad.py --yaml=configs/R34_object-CXR_cls_val_combined_ng_conv3x3.yaml

echo "NormGrad Object-CXR Test Combined Layers Conv3x3"
python run_normgrad.py --yaml=configs/R34_object-CXR_cls_test_combined_ng_conv3x3.yaml

echo "NormGrad Object-CXR Val Bias"
python run_normgrad.py --yaml=configs/R34_object-CXR_cls_val_ng_bias.yaml

echo "NormGrad Object-CXR Test Bias"
python run_normgrad.py --yaml=configs/R34_object-CXR_cls_test_ng_bias.yaml

echo "NormGrad Object-CXR Val Combined Layers Bias"
python run_normgrad.py --yaml=configs/R34_object-CXR_cls_val_combined_ng_bias.yaml

echo "NormGrad Object-CXR Test Combined Layers Bias"
python run_normgrad.py --yaml=configs/R34_object-CXR_cls_test_combined_ng_bias.yaml

echo "Input x Grad Object-CXR Val"
python run_grad_times_image.py --yaml=configs/R34_object-CXR_cls_val_other.yaml

echo "Input x Grad Object-CXR Test"
python run_grad_times_image.py --yaml=configs/R34_object-CXR_cls_test_other.yaml

echo "Guided Backpropagation Object-CXR Val"
python run_guided_backprop.py --yaml=configs/R34_object-CXR_cls_val_other.yaml

echo "Guided Backpropagation Object-CXR Test"
python run_guided_backprop.py --yaml=configs/R34_object-CXR_cls_test_other.yaml

echo "Guided Grad-CAM Object-CXR Val"
python run_guided_gradcam.py --yaml=configs/R34_object-CXR_cls_val_other.yaml

echo "Guided Grad-CAM Object-CXR Test"
python run_guided_gradcam.py --yaml=configs/R34_object-CXR_cls_test_other.yaml

echo "GradCAM Object-CXR Val"
python run_gradcam.py --yaml=configs/R34_object-CXR_cls_val_other.yaml

echo "GradCAM Object-CXR Test"
python run_gradcam.py --yaml=configs/R34_object-CXR_cls_test_other.yaml
