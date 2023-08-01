import os
import numpy as np
import pandas as pd
from typing import Literal
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost
from xgboost import plot_importance, XGBClassifier
from utils import load_from_yaml, load_from_pickle, feature_types_, modality_types_, save_to_pickle

feature_types_ = Literal["topic_oriented", "emotion_oriented", "skip"]
modality_types_ = Literal["txt", "img"]

class Modeling_XGBoost:
    def __init__(self, config):
        super(Modeling_XGBoost, self)

        self.config = config
        # load text & image features-type
        self.text_feature_type = self.config["TEXTUAL_FEATURES_TYPE"]
        self.image_feature_type = self.config["IMAGE_FEATURES_TYPE"]
        assert self.text_feature_type in feature_types_.__args__; "provide appropriate text-feature-type"
        assert self.image_feature_type in feature_types_.__args__; "provide appropriate image-feature-type"
        # construct filename
        self.construct_filename()
        # load features
        self.load_features()
        # load data
        self.load_data()
        # train/load xgboost classifier
        if self.config["TRAIN_XGBOOST"]:
            # train XGBoost model
            self.train_XGBoost()
            # evaluate XGBoost model
            self.evaluate_XGBoost()
        else:
            # load XGBoost model
            self.load_XGBoost()
            # evaluate XGBoost model
            self.evaluate_XGBoost()

    def load_features(self):
        feature_type_2_path = {
            "topic_oriented": self.config["PATH_TO_TOPIC_ORIENTED_FEATURES"],
            "emotion_oriented": self.config["PATH_TO_EMOTION_ORIENTED_FEATURES"],
            "skip": None
        }

        def load_cached_features(modality_type: modality_types_):
            features = []
            modality_type_2_feature_type = {"txt": self.text_feature_type, "img": self.image_feature_type}
            path_to_cached_features = feature_type_2_path[modality_type_2_feature_type[modality_type]]
            if path_to_cached_features is not None:
                features = [f"{feature}_{modality_type}" for feature in load_from_pickle(path_to_cached_features)]
            return features

        self.text_features = load_cached_features(modality_type="txt")
        self.image_features = load_cached_features(modality_type="img")
        self.total_features = self.text_features + self.image_features

    def load_data(self):
        # :xD trying a bit of hack!
        current_text_feature_type = self.text_feature_type if self.text_feature_type != "skip" else "topic_oriented"
        current_image_feature_type = self.image_feature_type if self.image_feature_type != "skip" else "topic_oriented"

        path_to_text_features = os.path.join(self.config["PATH_TO_TEXT_FEATURES_DIR"], f"./sample_esnlive_{current_text_feature_type}.csv")
        path_to_image_features = os.path.join(self.config["PATH_TO_IMAGE_FEATURES_DIR"], f"./sample_esnlive_{current_image_feature_type}.csv")
        assert os.path.exists(path_to_text_features); f"couldn't find file/folder, {path_to_text_features}"
        assert os.path.exists(path_to_image_features); f"couldn't find file/folder, {path_to_image_features}"
        text_df = pd.read_csv(path_to_text_features)
        image_df = pd.read_csv(path_to_image_features)
        common_columns = load_from_pickle(self.config["PATH_TO_COMMON_COLUMNS"])
        self.merge_df = pd.merge(text_df, image_df, on=common_columns)

        self.X = self.merge_df[self.total_features].to_numpy()
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(self.merge_df[[self.config["TARGET_COLUMN"]]].values.ravel())
        self.class_labels = label_encoder.classes_
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42) # modulate test-size here!
        
        # prepare DMatrix
        self.d_train = xgboost.DMatrix(self.X_train, label=self.y_train)   
        self.d_test = xgboost.DMatrix(self.X_test, label=self.y_test)

    def construct_filename(self):
        self.filename = f"xgboost_clf_txt_{self.text_feature_type}_img_{self.image_feature_type}.pkl"

    def load_XGBoost(self):
        path_to_file = os.path.join(self.config["PATH_TO_TRAINED_MODELS_DIR"], self.filename)
        if os.path.exists(path_to_file) and not self.config["TRAIN_XGBOOST"]:
            self.model = self.load_model()
        else:
            raise ValueError(f"No model saved @ loc: {path_to_file}")

    def train_XGBoost(self):
        params = {
            "eta": 0.01,
            "num_class": 3,
            "objective": "multi:softmax",
            "subsample": 0.5,
            "base_score": np.mean(self.y_train),
            "eval_metric": "mlogloss"
        }

        self.model = xgboost.train(params, self.d_train, 1500, evals=[(self.d_test, "test")], verbose_eval=100, early_stopping_rounds=20)
        self.save_model()

    def predict(self, input: np.ndarray):
        return self.model.predict(input)

    def evaluate_XGBoost(self):
        y_pred = self.predict(self.d_test)
        print(classification_report(self.y_test, y_pred, target_names=self.class_labels))

    def save_model(self):
        path_to_file = os.path.join(self.config["PATH_TO_TRAINED_MODELS_DIR"], self.filename)
        save_to_pickle(self.model, path_to_file)
    
    def load_model(self):
        path_to_file = os.path.join(self.config["PATH_TO_TRAINED_MODELS_DIR"], self.filename)
        return load_from_pickle(path_to_file)