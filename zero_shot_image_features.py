import os
import gc
import torch
import flair
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from collections import Counter, defaultdict
from tqdm import tqdm
import nltk
from PIL import Image
from nltk.corpus import wordnet
from transformers import pipeline
from utils import load_from_pickle, load_from_json

nltk.download("wordnet")

flair.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Zero_Shot_Image_Features:
    def __init__(self, config: dict):
        super(Zero_Shot_Image_Features, self).__init__()

        self.config = config
        # retrieve image-features
        print("retrieve image-features")
        self.load_image_features()
        # load data
        print("load data")
        self.load_data()
        # initialize zero-shot image classification pipeline
        print("initalize zero-shot image classification pipeline")
        self.initialize_pipeline()
        # initiaize features-dict
        self.features_dict = defaultdict(list, \
         {f"{feature_name}_img": [] for feature_name in self.image_features})
        # generate image-features
        print("generate text-features")
        self.generate_image_features_full()

    def load_data(self):
        self.data = pd.read_csv(self.config["PATH_TO_SAMPLE_TRAIN_DATA"])
        if self.config["SAMPLE_DATA_IMAGE_FEATURES"]:
            sample_data = self.data.sample(self.config["SAMPLE_SIZE_IMAGE_FEATURES"])
            sample_data.reset_index(drop=True, inplace=True)
            self.data = sample_data
            del sample_data
            gc.collect()

    def load_image_features(self):
        self.feature_type = self.config["IMAGE_FEATURES_TYPE"]
        if self.feature_type == "topic_oriented":
            self.image_features = load_from_pickle(self.config["PATH_TO_TOPIC_ORIENTED_FEATURES"])
        elif self.feature_type == "emotion_oriented":
            self.image_features = load_from_pickle(self.config["PATH_TO_EMOTION_ORIENTED_FEATURES"])

    def initialize_pipeline(self):
        # Need to seek a way to have multi-label classification
        self.classifier = pipeline("zero-shot-image-classification",
                                   model="openai/clip-vit-large-patch14-336",
                                   framework="pt",
                                   device=0)

    def get_PIL_image(self, filename: str):
        image = None
        path_to_file = os.path.join(self.config['PATH_TO_IMAGE_DIR'], filename)
        if os.path.exists(path_to_file):
            image = Image.open(path_to_file)
        return image

    def __getitem__(self, index):
        return self.get_PIL_image(self.data.iloc[index][self.config["SOURCE_IMAGE_COLUMN"]])

    def dataset(self):
        column_name = self.config["SOURCE_IMAGE_COLUMN"]
        return (self.get_PIL_image(row[column_name]) for _, row in self.data.iterrows())

    def generate_image_features_full(self):
        # iterate over the entire data efficiently w/ gpu access!
        outputs = []
        for image in tqdm(self.dataset(), total=self.data.shape[0]):
            if image == None:
                outputs.append(None)
            else:
                output = self.classifier(image, candidate_labels=self.image_features)
                outputs.append(output)
        for output in tqdm(outputs, total=len(outputs)):
            if output == None:
                for feature_name in self.image_features:
                    self.features_dict[f"{feature_name}_img"].append(None)
            else:
                for item in output:
                    self.features_dict[item["label"] + "_img"].append(item["score"])
        self.concatenate_features()
        del outputs
        gc.collect()

    def concatenate_features(self):
        features = pd.DataFrame(self.features_dict)
        self.data = pd.concat([self.data, features], axis=1)
        filename = f"sample_esnlive_{self.feature_type}.csv"
        path_to_save_loc = os.path.join(config["PATH_TO_IMAGE_FEATURES_DIR"], filename)
        self.data.to_csv(path_to_save_loc, index=False)
        print(f"saved {self.feature_type} text-features @ {path_to_save_loc}")
        del features
        del filename
        del path_to_save_loc
        gc.collect()

    # ---------------------------- Legacy Approach ---------------------------- #

    def generate_entities(self):
        # legacy approach
        path_to_entities = self.config["PATH_TO_SAMPLE_ENTITIES"]
        if os.path.exists(path_to_entities):
            entity_counter = Counter(load_from_json(path_to_entities))
        else:
            tagger = SequenceTagger.load("flair/pos-english")
            entities = []
            hypothesis_list = self.data[self.config["SOURCE_TEXT_COLUMN"]].tolist()
            for hypothesis in tqdm(hypothesis_list, total=len(hypothesis_list)):
                sentence = Sentence(hypothesis)
                tagger.predict(sentence)
                for token in sentence:
                    if token.tag in ["NN"]:
                        entities.append(token.text)
            entity_counter = Counter(entities)
        return entity_counter

    def get_word_root(self, word: str):
        # legacy approach
        synsets = wordnet.synsets(word)
        if synsets:
            most_common_synset = synsets[0]
            root = most_common_synset.lemmas()[0].name()
            return root
        return None

    def get_feature_names(self, k: int=50):
        # legacy approach
        entities, _ = zip(*self.entity_counter.most_common(k))
        self.image_features = list(filter(lambda x: x is not None, [self.get_word_root(entity) for entity in entities]))
        return self.image_features