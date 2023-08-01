import os
import gc
import numpy as np
import collections
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from utils import load_data, config, load_from_pickle, save_to_pickle

class Topic_Model:
    def __init__(self, config: dict):
        super(Topic_Model, self).__init__()

        self.config = config
        # sample/load docs
        if not os.path.exists(self.config["PATH_TO_SAVED_SAMPLED_DOCS"]) or self.config["TOPIC_MODEL_SAMPLE_DOCS"]:
            print("sample docs")
            self._sample_docs()
        else:
            print("load docs")
            self._load_docs()
        # generate/load embeddings
        if not os.path.exists(self.config["PATH_TO_SAVED_EMBEDDINGS"]) or self.config["GENERATE_EMBEDDINGS"]:
            print("generate embeddings")
            self._generate_embeddings()
        else:
            print("load saved embeddings")
            self._load_embeddings()
        # generate vocabulary
        print("generate vocabulary")
        self._generate_vocab()
        # train/load topic-model
        if not os.path.exists(self.config["PATH_TO_SAVED_TOPIC_MODEL"]) or self.config["TRAIN_TOPIC_MODEL"]:
            print("train topic-model")
            self._apply_to_topic_model()
        else:
            print("load topic-model")
            self._load_topic_model()

    def _sample_docs(self):
        data = load_data(path_to_dir=self.config["PATH_TO_DATA_DIR"], data_type="train")
        if self.config["TOPIC_MODEL_SAMPLE_DOCS"]:
            print("generate sample-data")
            sample_count = int(data.shape[0]*self.config["TOPIC_MODEL_SAMPLE_SIZE"])
            sample_data = data.sample(n=sample_count)
            sample_data.reset_index(inplace=True)
            sample_data.to_csv(self.config["PATH_TO_SAMPLE_TRAIN_DATA"], index=False)
            self.docs = sample_data[self.config["SOURCE_TEXT_COLUMN"]].tolist()
        else:
            print("using original non-sample data")
            self.docs = data[self.config["SOURCE_TEXT_COLUMN"]].tolist()
        save_to_pickle(data_list=self.docs,
                       path_to_file=self.config["PATH_TO_SAVED_SAMPLED_DOCS"])
        del data
        gc.collect()

    def _load_docs(self):
        print("using cached sample/non-sample docs")
        self.docs = load_from_pickle(path_to_file=self.config["PATH_TO_SAVED_SAMPLED_DOCS"])

    def _generate_embeddings(self):
        model_checkpoint = self.config["TOPIC_MODEL_CHECKPOINT"]
        model = SentenceTransformer(model_checkpoint)
        self.embeddings = model.encode(self.docs, show_progress_bar=True)
        path_to_saved_embeddings = config["PATH_TO_SAVED_EMBEDDINGS"]
        with open(path_to_saved_embeddings, "wb") as f:
            np.save(f, self.embeddings)
        del model_checkpoint
        del model
        del path_to_saved_embeddings
        gc.collect()

    def _load_embeddings(self):
        path_to_saved_embeddings = self.config["PATH_TO_SAVED_EMBEDDINGS"]
        self.embeddings = np.load(path_to_saved_embeddings)
        del path_to_saved_embeddings

    def _generate_vocab(self):
        vocab = collections.Counter()
        tokenizer = CountVectorizer().build_tokenizer()
        for doc in tqdm(self.docs):
            vocab.update(tokenizer(doc))
        self.vocab = [word for word, frequency in vocab.items() if frequency >= 15]
        del tokenizer
        gc.collect()

    def _apply_to_topic_model(self):
        model_checkpoint = self.config["TOPIC_MODEL_CHECKPOINT"]
        embedding_model = SentenceTransformer(model_checkpoint)
        vectorizer_model = CountVectorizer(vocabulary=self.vocab, stop_words="english")
        self.topic_model = BERTopic(
            nr_topics=self.config["NUM_TOPICS"],
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            verbose=True
        ).fit(self.docs, embeddings=self.embeddings)
        self.topic_model.save(self.config["PATH_TO_SAVED_TOPIC_MODEL"])
        del model_checkpoint
        del embedding_model
        del vectorizer_model
        gc.collect()

    def _load_topic_model(self):
        path_to_saved_topic_model = self.config["PATH_TO_SAVED_TOPIC_MODEL"]
        self.topic_model = BERTopic.load(path_to_saved_topic_model)
        self.topic_model.transform(self.docs, self.embeddings)
        del path_to_saved_topic_model
        gc.collect()

    def get_topic_representations(self):
        representations = self.topic_model.get_topic_info()["Representation"].tolist()
        return representations

    def visualize_heatmap(self):
        fig = self.topic_model.visualize_heatmap()
        fig.write_html("./plots/topic_model_visualize_heatmap.html")

    def visualize_topics(self):
        fig = self.topic_model.visualize_topics()
        fig.write_html("./plots/topic_model_visualize_topics.html")
    
    def visualize_documents(self):
        fig = self.topic_model.visualize_documents(self.docs)
        fig.write_html("./plots/topic_model_visualize_documents.html")