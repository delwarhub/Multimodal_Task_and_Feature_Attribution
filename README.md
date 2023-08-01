# XAI Project 3
## Project 3 - Multimodal Task and Feature Attribution

## Task Data: e-SNLI-VE

The e-SNLI-VE dataset is derived from the well-known SNLI (Stanford Natural Language Inference) dataset and extends it with visual information. It combines textual premises, hypotheses, and corresponding images to enable the investigation of visual entailment, where the goal is to determine if a textual hypothesis can be inferred from a given image and a textual premise. 

The dataset consists of train, dev, and test splits, which can be found in the data folder of this repository. The data is stored in .csv files, and each file contains Flickr30k Image IDs. To access the actual image data, you can download the Flickr30k dataset separately from [Flickr30K](https://www.kaggle.com/hsankesara/flickr-image-dataset).

### Data Format: ###
The dataset is available in a structured format, consisting of CSV files. The e-SNLI-VE dataset has the following data format:

- **pairID**: A unique identifier for each pair of image and textual data.
- **Flickr30kID**: A string or object representing the Flickr30k Image ID associated with the data entry.
- **hypothesis**: The textual hypothesis or prediction associated with the image.
- **gold_label**: The gold or ground truth label for the image-text pair.
- **explanation**: An object or string containing the natural language explanation for the image-text pair.

The below link provide access to the e-SNLI-VE dataset and corresponding Flick30K images.

- [e-VIL](https://github.com/maximek3/e-ViL/tree/main)
- [Flickr30K](https://www.kaggle.com/hsankesara/flickr-image-dataset)

The below link provides access to already downloaded version of the above mention dataset via Google drive.

- [e-SNLI-VE](https://drive.google.com/file/d/105KNRwMRseTtGrYQ0PUVTpvCXSTO5zB0/view?usp=drive_link)
- [Flick30K](https://drive.google.com/file/d/1-2xugLRKZMsbV-8bQz5IJsE46ex0mt03 /view?usp=sharing)

```
# Download Fickr30K dataset from the either of above provided links.
unzip $PATH_TO_DOWNLOADED_FLICKR30K_DATA.zip -d "./data/"
```

## Training & Testing Set-Up  (Linux SetUp)

1. Clone the repository

```
git clone [git clone https URL]
```

2. Create a Python virtual environment

```
# Update and upgrade
sudo apt update
sudo apt -y upgrade

# check for python version
python3 -V

# install python3-pip
sudo apt install -y python3-pip

# install-venv
sudo apt install -y python3-venv

# Create virtual environment
python3 -m venv my_env


# Activate virtual environment
source my_env/bin/activate
```

3. Install project dependent files

```
pip install requirements.txt
```

4. Run main.py

```
python3 main.py
```

## Topic Modeling

The below link provides access to already trained topic-model w/ 20 topic clusters alongwith other files required to train the model.

- [Topic-Model](https://drive.google.com/file/d/1n4kl5uxml96lJZ5Kb0XrUIhkVmU7mKDI/view?usp=sharing)
- [Cached-Embeddings](https://drive.google.com/file/d/1--EcUuMvmwsV1L3jEgNhkbelml9ef5R0/view?usp=sharing)

```
Note: Download the above mentioned files as follows:
1. Topic-Model => "./trained_models/topic-model"
2. Cached-Embeddings => "./trained_models/sentence_transformers_all-MiniLM-L6-v2_embeddings.npy"
```

### Training Topic-Model

```
# edit ./config.yaml file
vim ./config.yaml

# change following attributes
APPLY_TOPIC_MODELING: True
TRAIN_TOPIC_MODEL: True

# train-topic model
python3 main.py
```


## Multimodal Features Preparation.

To extract mutimodal text and image features via zero-shot-classification of already defined features

```
# edit ./config.yaml file
vim ./config.yaml

# change following attributes 
APPLY_TEXT_FEATURE_EXTRACTION: True
APPLY_IMAGE_FEATURE_EXTRACTION: True

Appropriate feature_type (`topic-oriented` or `emotion-oriented`) can implemented by changing the attributes;
TEXTUAL_FEATURES_TYPE: feature-type
IMAGE_FEATURES_TYPE: feature-type

Note: When modeling XGBoost we add additional feature-type `skip` to perform single modality training.

# train-topic model
python3 main.py
```

## Training XGBoost Model

```
# edit ./config.yaml file
vim ./config.yaml

# change following attributes 
TRAIN_XGBOOST: True

Appropriate feature_type (`topic-oriented`, `emotion-oriented` or `skip`) can implemented by changing the attributes;
TEXTUAL_FEATURES_TYPE: feature-type
IMAGE_FEATURES_TYPE: feature-type

# train-topic model
python3 main.py
```



# Project Directory Tree

```
└── Project2/
    ├── data/
    │   ├── sample_esnlive_train.csv
    │   ├── image_features/
    |   |   ├── sample_esnlive_emotion_oriented.csv
    |   |   └── sample_esnlive_topic_oriented.csv
    │   ├── text_features/
    |   |   ├── sample_esnlive_emotion_oriented.csv
    |   |   └── sample_esnlive_topic_oriented.csv
    │   ├── sample_esnlive_common_columns.pkl
    |   ├── topic_oriented_features_1.pkl
    |   ├── topic_oriented_features_2.pkl
    |   └── emotion_oriented_features.pkl
    ├── trained_models/
    │   ├── sentence_transformers_all-MiniLM-L6-v2_embeddings.npy
    │   └── topic_model
    ├── plots/
    ├── zero_shot_text_features.py
    ├── zero_shot_image_features.py
    ├── topic_model.py
    ├── config.yaml
    ├── main.py
    ├── utils.py
    ├── utils.py
    ├── README.md
    └── requirements.txt
```

# NOTE

```
If there are any dependency issues related to SHAP or LIME not compatibile on local environment try using the .ipynb notebook instead. 
```
## Tasks
1.	Pick one multimodal dataset that includes both image & text content. For ex: Multimodal Sentiment - MVSA Single, (pass: mvsa-2023-uni-p), or choose another dataset where the task uses both image and text (both modalities are inputs) to predict. It doesn't apply to tasks where one modality is input and the other is output (e.g. text2image).
2.	Extract various visual features (not embeddings) using pre-trained visual models (classifiers, captioning models, etc.). Example: tuples of feature names with probability (or some other score value) => "image includes a dog": 0.8, "kitchen is the scene": 0.6. In this way, each feature has a name and can be back-traced for the explanation.
You can use any pre-trained model(s) of your choice. Some notable examples are CLIP, Recognize Anything Model, OWL-ViT
3.	Extract textual features (not embeddings) using pre-trained textual models. Example: tuples of feature names with probability (or some other score value) => "text includes positive word: amazing": 0.8, "text mentions entity: Berlin": 0.6, "text has informal writing style": 0.9, ...
You can use any pre-trained model(s) of your choice.
4.	Implement a model architecture that uses XGBoost and integrates the extracted features to train a model.
5.	Perform feature importance analysis and present the findings (with visualization of explanations), counterfactual examples etc. (as it was done in the other two projects)

## Reference

1. Virginie Do, Oana-Maria Camburu, Zeynep Akata, Thomas Lukasiewicz. e-SNLI-VE: Corrected Visual-Textual Entailment with Natural Language Explanations. arXiv preprint arXiv:2004.03744.
2. Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. 2014. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:67–78.

