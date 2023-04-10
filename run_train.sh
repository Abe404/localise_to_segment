echo Starting training 
python3 train.py --imdir data/spleen/Task09_Spleen/imagesTr --annotdir data/spleen/Task09_Spleen/labelsTr --organname spleen
python3 train.py --imdir data/pancreas/Task07_Pancreas/imagesTr --annotdir data/pancreas/Task07_Pancreas/labelsTr --organname pancreas
python3 train.py --imdir data/prostate/Task05_Prostate/imagesTr --annotdir data/prostate/Task05_Prostate/labelsTr --organname prostate
python3 train.py --imdir data/liver/Task03_Liver/imagesTr --annotdir data/liver/Task03_Liver/labelsTr --organname liver
python3 train.py --imdir data/heart/Task02_Heart/imagesTr --annotdir data/heart/Task02_Heart/labelsTr --organname heart
