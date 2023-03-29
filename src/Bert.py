import os
import tensorflow_hub as hub
import tensorflow_text
from src.metadata.tf_hub_mapping import model_names, model_preprocess_pipelines
import requests
import tarfile


class BertModel:
    
    model_name = 'experts_wiki_books_mnli'
    preprocess_model_name = 'bert_en_uncased_preprocess_3'
    preprocess_model = None
    model = None
    preprocessed_documents = []

    def __init__(self):

        if not os.path.exists('src/models/' + self.model_name):
            self.fetch_model_file(model_names[self.model_name], self.model_name)

        if not os.path.exists('src/models/preprocess/' + self.preprocess_model_name):
            self.fetch_model_file(model_preprocess_pipelines[self.preprocess_model_name], 'preprocess/' + self.preprocess_model_name)

        self.preprocess_model = hub.load('src/models/preprocess/' + self.preprocess_model_name)
        self.model = hub.load('src/models/' + self.model_name)

    def fetch_model_file(self, model_url, model_name):

        print('fetching ' + model_url + ' ...' )
        url = model_url + '?tf-hub-format=compressed'
        filename = 'src/models/' + model_name
        filetype = '.tar.gz'

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename + filetype, 'wb') as f:
                f.write(response.raw.read())

            file = tarfile.open(filename + filetype)
            file.extractall(filename)
            os.remove(filename + filetype)

            file.close()

    def preprocess_documents(self, documents):

        self.preprocessed_documents = self.preprocess_model(documents)
        # print(self.preprocessed_documents["input_word_ids"].shape)

    def run(self):

        return self.model(self.preprocessed_documents)