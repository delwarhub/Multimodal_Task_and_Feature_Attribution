import os
import gc
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from transformers import pipeline
from utils import load_from_pickle

class Zero_Shot_Text_Features:
    def __init__(self, config):
        super(Zero_Shot_Text_Features, self).__init__()

        self.config = config
        # retrieve text-features
        print("retrieve text-features")
        self.load_textual_features()
        # load data
        print("load data")
        self.load_data()
        # initialize zero-shot classification pipeline
        print("initalize zero-shot classification pipeline")
        self.initialize_pipeline()
        # initiaize features-dict
        self.features_dict = defaultdict(list, \
         {f"{feature_name}_txt": [] for feature_name in self.text_features})
        # generate text-features
        print("generate text-features")
        if self.config["FULL_PROCESS"]:
            self.generate_text_features_full()
        else:
            self.generate_text_features()

    def load_data(self):
        self.data = pd.read_csv(self.config["PATH_TO_SAMPLE_TRAIN_DATA"])
        if self.config["SAMPLE_DATA_TEXT_FEATURES"]:
            sample_data = self.data.sample(self.config["SAMPLE_SIZE_TEXT_FEATURES"])
            sample_data.reset_index(drop=True, inplace=True)
            self.data = sample_data
            del sample_data 
        gc.collect()

    def load_textual_features(self):
        self.feature_type = self.config["TEXTUAL_FEATURES_TYPE"]
        if self.feature_type == "topic_oriented":
            self.text_features = load_from_pickle(self.config["PATH_TO_TOPIC_ORIENTED_FEATURES"])
        elif self.feature_type == "emotion_oriented":
            self.text_features = load_from_pickle(self.config["PATH_TO_EMOTION_ORIENTED_FEATURES"])

    def initialize_pipeline(self):
        self.classifier = pipeline("zero-shot-classification",
                                   model="facebook/bart-large-mnli",
                                   framework="pt",
                                   device=0)

    def __getitem__(self, index):
        return self.data.iloc[index][self.config["SOURCE_TEXT_COLUMN"]]

    def dataset(self):
        for index, row in self.data.iterrows():
            yield row[self.config["SOURCE_TEXT_COLUMN"]]

    def get_scores(self, index):
        output = self.classifier(self[index],
                                 self.text_features,
                                 multi_label=True)
        for label, score in zip(output["labels"], output["scores"]):
            self.features_dict[label].append(score)
        del output
        del label
        del score
        gc.collect()

    def generate_text_features(self):
        # Iterate over entire data
        for index in tqdm(range(self.data.shape[0])):
            self.get_scores(index)
        # Concatenate data w/ features
        self.concatenate_features()
        del index
        gc.collect()

    def generate_text_features_full(self):
        # iterate over the entire data efficiently w/ gpu access!
        outputs = [output for output in tqdm(self.classifier(self.dataset(), self.text_features, multi_label=True, batch_size=24), total=self.data.shape[0])]
        _ = [self.features_dict[f"{label}_txt"].append(score) for output in outputs for label, score in zip(output["labels"], output["scores"])]
        # Concatenate data w/ features
        self.concatenate_features()
        del outputs
        gc.collect()

    def concatenate_features(self):
        features = pd.DataFrame(self.features_dict)
        self.data = pd.concat([self.data, features], axis=1)
        filename = f"sample_esnlive_{self.feature_type}.csv"
        path_to_save_loc = os.path.join(self.config["PATH_TO_TEXT_FEATURES_DIR"], filename)
        self.data.to_csv(path_to_save_loc, index=False)
        print(f"saved {self.feature_type} text-features @ {path_to_save_loc}")
        del features
        del filename
        del path_to_save_loc
        gc.collect()