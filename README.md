# java-bert-predict
turn bert pretrain checkpoint into saved model for a feature extracting demo in java

# usage

1. download google bert pretrain from [here](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip) and unzip it into a proper path.

2. because java cannot work with tensorflow checkpoint directly, we need to transform checkpoint into saved_model with following command 
	> python script/checkpoint_to_saved_model.py path/to/		unzipped/pretrain/checkpoint path/to/save/model

1. run java program with path/to/save/model as params


