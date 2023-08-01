import os
from PIL import Image
import pandas as pd
import shap
from modeling_xgboost import Modeling_XGBoost

import os
from PIL import Image
import pandas as pd
import shap

import os
from PIL import Image
import pandas as pd
import shap

class shap_interpretability:
  def __init__(self, config):

    self.config = config
    # train/retrieve xgboost model
    self.modeling_xgboost = Modeling_XGBoost(self.config)
    # related attributes
    self.features = self.modeling_xgboost.total_features
    self.class_labels = self.modeling_xgboost.class_labels
    self.model = self.modeling_xgboost.model
    self.X = self.modeling_xgboost.X
    self.X_display = pd.DataFrame(self.X, columns=[self.features])
    # initalize shap-explainer
    self.explainer = shap.TreeExplainer(self.model)
    self.shap_values = self.explainer.shap_values(self.X_display)

  def get_PIL_image(self, filename: str):
      image = None
      path_to_file = os.path.join(self.config['PATH_TO_IMAGE_DIR'], filename)
      if os.path.exists(path_to_file):
          image = Image.open(path_to_file)
      return image

  def retrieve_text(self, row_index):
    return self.X_display.iloc[row_index][self.config["SOURCE_TEXT_COLUMN"]]

  def retrieve_image(self, row_index):
    filename = self.X_display.iloc[row_index][self.config["SOURCE_IMAGE_COLUMN"]]
    return self.get_PIL_image(filename=filename)

  def show_global_force_plot(self, class_index: int=0, sample_size: int=1000):

    shap.initjs()
    shap.force_plot(self.explainer.expected_value[class_index],
                    self.shap_values[class_index][:sample_size, :],
                    self.X_display.iloc[:sample_size, :])

  def show_local_force_plot(self, class_index: int=0, row_index: int=0):

    shap.initjs()
    shap.force_plot(self.explainer.expected_value[class_index],
                    self.shap_values[class_index][row_index, :],
                    self.X_display.iloc[:row_index, :])
    
  def summary_plot(self, class_index):
    
    shap.initjs()
    shap.summary_plot(self.shap_values[class_index], self.X, feature_names=self.features)

  def show_dependency_plot(self, class_index: int=0, feature_index: int=0,
                           sample_size: int=1000):
    shap.initjs()
    shap.dependence_plot(feature_index,
                         self.shap_values[class_index][:sample_size, :],
                         self.X[:sample_size, :],
                         self.X_display.iloc[:sample_size, :])