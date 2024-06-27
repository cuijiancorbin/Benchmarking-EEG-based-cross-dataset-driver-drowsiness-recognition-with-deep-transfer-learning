# Benchmarking-EEG-based-cross-dataset-driver-drowsiness-recognition-with-deep-transfer-learning
It usually takes a long time to collect data for calibration when using electroencephalography (EEG) for driver drowsiness monitoring. Cross-dataset recognition is desirable since it can significantly save the calibration time when an existing dataset is used. However, the recognition accuracy is affected by the distribution drift problem caused by different experimental environments when building different datasets. In order to solve the problem, we propose a deep transfer learning model named Entropy-Driven Joint Adaptation Network (EDJAN), which can learn useful information from source and target domains simultaneously. An entropy-driven loss function is used to promote clustering of target-domain representations and an individual-level domain adaptation technique is proposed to alleviate the distribution discrepancy problem of test subjects. 

The code implements the Entropy-Driven Joint Adaptation Network (EDJAN) for cross-dataset driver drowsiness recognition:
     
Cui, Jian, et al. "Benchmarking EEG-based cross-dataset driver drowsiness recognition with deep transfer learning." 2023 45th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). IEEE, 2023. DOI: 10.1109/EMBC40787.2023.10340982    
 
The processed SADT dataset can be downloaded here:
https://figshare.com/articles/dataset/EEG_driver_drowsiness_dataset/14273687
  
The processed SEED-VIG dataset can be downloaded here:
https://figshare.com/articles/dataset/Extracted_SEED-VIG_dataset_for_cross-dataset_driver_drowsiness_recognition/26104987
  
   
Description on the backbone ICNN model can be found from:
     
Cui J, Lan Z, Sourina O, et al. EEG-based cross-subject driver drowsiness recognition with an interpretable convolutional neural network[J]. IEEE Transactions on Neural Networks and Learning Systems, 2022, 34(10): 7921-7933. DOI: 10.1109/TNNLS.2022.3147208   
  
If you have met any problems, you can contact Dr. Cui Jian at cuijian@zhejianglab.com

