# Improving_Face_recognition_Performance_using_Triplet_Loss
## Goal
To improve face recognition performance on:

## Method
apply Triplet Loss on EFM model, DeepFace model and BEGAN-CS model

### Architecture
1. EFM model
2. Deepface
3. BEGAN-CS

## Dataset collection
1. Training set: Celeb1M dataset (4,621,640 Images; 78,579 Identities)
	- 0.7 for training, 0.3 for validation
2. Testing set: LFW dataset

## Scripts
1. extract_feacture_v2.py -> feature extraction
2. pre-trained_efm_v3.py -> using pre-trained edm model to optimize the similarity with Triplet loss
3. draw_cos_dis_real.py -> draw the figure of cosine similarity

## Figures (results)


## DEMO (Deepface)

