import os
import yaml
import json
import pickle
import pandas as pd
from typing import Literal
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

def load_from_yaml(path_to_file: str):
    print(f"loading data from .yaml file @ {path_to_file}")
    with open(path_to_file) as file:
        _dict = yaml.safe_load(file)
    return _dict

def load_from_txt(path_to_file: str):
    print(f"loading data from .txt file @ {path_to_file}")
    with open (path_to_file, "r") as myfile:
        data = myfile.read().splitlines()
    return data

def save_to_json(path_to_file: str, data: list):
    with open(path_to_file, 'w') as outfile:
        json.dump(data, outfile)
    print(f"file saved @ loc: {path_to_file}")

def load_from_json(path_to_file: str):
    print(f"loading data from .json file @ {path_to_file}")
    with open(path_to_file, "r") as json_file:
        _dict = json.load(json_file)
    return _dict

def save_to_pickle(data_list, path_to_file):
    with open(path_to_file, 'wb') as file:
        pickle.dump(data_list, file)
    print(f"file saved @ loc: {path_to_file}")

def load_from_pickle(path_to_file):
    print(f"loading data from .pkl file @ {path_to_file}")
    with open(path_to_file, 'rb') as file:
        data_list = pickle.load(file)
    return data_list
    
data_types_ = Literal["train", "test", "dev"]
def load_data(path_to_dir: str, data_type: data_types_):
    path_to_file = os.path.join(path_to_dir, f"esnlive_{data_type}.csv")
    df = pd.read_csv(path_to_file)
    df.drop(["Unnamed: 0"], axis=1, inplace=True)
    return df

def get_PIL_image(filename: str):
    path_to_file = os.path.join(config['IMAGE_DIR'], filename)
    if os.path.exists(path_to_file):
        image = Image.open(path_to_file)
    return image    

def plot_correlation_matrix(data: pd.DataFrame, features: list):
    corr_matrix = data[features].corr(method="pearson")
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot the correlation matrix using a heatmap
    sns.heatmap(corr_matrix, ax=ax, cmap='coolwarm', annot=False, fmt=".2f")
    # Set the x-axis and y-axis tick labels
    ax.set_xticklabels(features, rotation=90)
    ax.set_yticklabels(data[features][::-1], rotation=0)
    # Set the axis labels
    ax.set_xlabel('Features')
    ax.set_ylabel('Features')
    # Set the title
    ax.set_title('Correlation Plot')
    # Show the plot
    plt.tight_layout()
    plt.show()

config = load_from_yaml("./config.yaml")

feature_types_ = Literal["topic_oriented", "emotion_oriented", "skip"]
modality_types_ = Literal["txt", "img"]