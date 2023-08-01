from utils import config, plot_correlation_matrix
from topic_model import Topic_Model
from zero_shot_text_features import Zero_Shot_Text_Features
from zero_shot_image_features import Zero_Shot_Image_Features
from modeling_xgboost import Modeling_XGBoost
from shap_interpreter import shap_interpretability

if __name__ == "__main__":
    
    # for generating topic-representation for training data /
    # and studying topics to formulate features
    if config["APPLY_TOPIC_MODELING"]:
        topic_model = Topic_Model(config=config)    

        # perform visualization
        topic_model.visualize_documents() # visualize documents `aggregate docs`
        topic_model.visualize_heatmap() # visualize heatmap `confusion plot (similarity b/w topics)`

    # for generating textul features based on predefined /
    # topic-oriented and emotion-oriented features
    if config["APPLY_TEXT_FEATURE_EXTRACTION"]:
        zero_shot_text_feautres = Zero_Shot_Text_Features(config)

    # for generating visual feature based on predefind /
    # entities extracted from the text-data.
    if config["APPLY_IMAGE_FEATURE_EXTRACTION"]:
        zero_shot_image_features = Zero_Shot_Image_Features(config)
        print(zero_shot_image_features.data.shape)
        print(zero_shot_image_features.data.columns.tolist())

    # train and test xgboost
    if config["TRAIN_XGBOOST"]:
        modeling_xgboost = Modeling_XGBoost(config)
        # plot correlation for text-features
        plot_correlation_matrix(
            data=modeling_xgboost.merge_df, 
            features=modeling_xgboost.text_features)
        # plot correlation for image-features
        plot_correlation_matrix(
            data=modeling_xgboost.merge_df,
            features=modeling_xgboost.image_features)

    # shap-interpretability
    if config["INTERPRET_W_SHAP"]:
        shap_interpreter = shap_interpretability(config)
        # global force-plot
        shap_interpreter.show_global_force_plot(class_index=0)
        shap_interpreter.show_global_force_plot(class_index=1)
        shap_interpreter.show_global_force_plot(class_index=2)
        # local force-plot
        shap_interpreter.show_local_force_plot(class_index=2, row_index=899)
        shap_interpreter.show_local_force_plot(class_index=0, row_index=985)
        shap_interpreter.show_local_force_plot(class_index=1, row_index=1503)
        shap_interpreter.show_local_force_plot(class_index=1, row_index=80)
        # summary plots
        shap_interpreter.summary_plot(class_index=0)
        shap_interpreter.summary_plot(class_index=1)