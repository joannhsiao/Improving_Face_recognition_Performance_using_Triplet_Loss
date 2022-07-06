# Improving_Face_recognition_Performance_using_Triplet_Loss
## Goal
- To improve face recognition performance on the Face Recognition System:
	
	<img src="https://user-images.githubusercontent.com/59599987/177505876-4192e688-9c21-4dd5-8959-431fa8baa21e.jpg" width="250" height="400">

## Method
apply Triplet Loss on EFM model, DeepFace model and BEGAN-CS model

### Architecture
1. EFM model

	![efm](https://user-images.githubusercontent.com/59599987/177505602-92ff03c8-686b-454a-894c-af24f7dd1dde.png)

2. Deepface

	![deepface](https://user-images.githubusercontent.com/59599987/177505640-bb750da7-50d8-4424-8f3e-608fdfff5da3.png)

3. BEGAN-CS with Triplet Loss

	![began-cs_w_tl](https://user-images.githubusercontent.com/59599987/177505679-8271a5b6-53e9-4469-ba6a-fcab462710c5.png)

## Dataset collection
1. Training set: Celeb1M dataset (4,621,640 Images; 78,579 Identities)
	- 0.7 for training, 0.3 for validation
2. Testing set: LFW dataset

## Scripts
1. extract_feacture_v2.py -> feature extraction
2. pre-trained_efm_v3.py -> using pre-trained edm model to optimize the similarity with Triplet loss
3. draw_cos_dis_real.py -> draw the figure of cosine similarity

## Figures (results)
- experiment result

	![result](https://user-images.githubusercontent.com/59599987/177505842-11d9abc1-3d6b-43ec-b4e8-8128168559da.png)

- reconstructed image by BEGAN-CS with Triplet Loss
![began_result](https://user-images.githubusercontent.com/59599987/177505790-deeddef0-29cd-472b-b2a7-f83514bd3ae0.png)


## DEMO (Deepface)

