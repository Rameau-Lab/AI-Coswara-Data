Update:
- tried MFCC and LogmelSpec, mfcc is better
- working with wav2vec, need still some bugs, cant work yet
- want to work with other clinical data how to access?
- can i do updates via emials/slack instead over the summer

Todo:
- Change to classification problem age groups 

Data
- Train
  - 1152 Samples 
- Validation
  - 307 Samples
- Test
  - 626 Samples

Results
Transformer
Validation MAE: 9.870915018386095
Test MAE: 10.68286142562525

Validation Average Loss: 156.91608123779298
Validation RMSE: 12.575531005859375
Test Average Loss: 180.3241928100586
Test RMSE: 13.402342796325684


Linear Regression
Validation MAE: 10.498983358591843
Test MAE: 10.895241956568306

Validation RMSE: 13.160218082177959
Test RMSE: 13.56746724233586

Lasso Regression
Validation MAE: 10.477250478857025
Test MAE: 10.871145597521059

Validation RMSE: 13.13619862797565
Test RMSE: 13.538048354362797


Baseline:
Sadjadi et al., iVectors and SVM, NIST SRE 2010 telephony test set, 4.7 years MAE
Hechmi et al., CNN using spectrograms, part of VoxCeleb2, 9.44 years MAE
Zazo et al., LSTM, NIST test set, 6.58 years MAE
Gupta et al., wav2vec 2.0, Timit, 5.54 years MAE (male), 6.49 years MAE (female)
https://arxiv.org/abs/2203.11774


Current pipeline
- extract MFCC signal from raw audio file
- pass through to model (transformer)
- extract RMSE

Feature Extraction
- CNN
- Wav2Vec

Model 
- Transformer based architecture 
- https://www.sciencedirect.com/science/article/pii/S1746809424002167
- marginally better

papers:
- https://www.isca-archive.org/interspeech_2021/schuller21_interspeech.pdf
- https://arxiv.org/abs/2306.16962