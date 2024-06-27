import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.speaker import SpeakerRecognition

# annotations
# breathing-deep.csv
# breathing-shallow.csv
# cough-heavy.csv
# cough-shallow.csv
# counting-fast.csv
# counting-normal.csv
# vowel-a.csv
# vowel-e.csv
# vowel-o.csv
annotation_dir = 'annotations/LABELS'
meta_data_path = 'metadata_files/combined_data.csv'
path_file_path = 'path_files/wav.scp'

base = '../../Extracted_data/20220224/zq953YOTkoNRH1nmQo5LogJsHH32'
diff = '../../Extracted_data/20220224/zmfoK4aATPVlxBYRzKNxNzIDVGm1'

verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
print("performing verification 1")
score, prediction = verification.verify_files(f'{base}/counting-fast.wav', f'{base}/cough-heavy.wav') # Different Speakers
print(score, prediction)
print("performing verification 1")
score, prediction = verification.verify_files(f'{base}/cough-shallow.wav', f'{diff}/cough-shallow.wav') # Different Speakers
print(score, prediction)
print("performing verification 2")
score1, prediction1 = verification.verify_files(f'{base}/counting-fast.wav', f'{diff}/counting-fast.wav') # same
score1, prediction1 = verification.verify_files(f'{base}/counting-fast.wav', f'{diff}/counting-normal.wav') # same
print(score1, prediction1)
