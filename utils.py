import nltk
import numpy as np
from nltk.corpus import stopwords
import re
import string
from sklearn.metrics.pairwise import cosine_similarity
import pandas
from gensim.utils import simple_preprocess

custom_words = [
                   'none', 'ie', 'etc', 'eg', '',
               ] + list(string.ascii_lowercase) + list(string.digits)

punctuation = string.punctuation.replace('-', '')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english') + list(punctuation) + custom_words)
lemmatizer = nltk.stem.WordNetLemmatizer()


def pre_process(document):

    words = simple_preprocess(document)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return words


def write_output(row, file, soft_matrix, catalog_length, sp_length, csf_length, index_list):
    top_sp = soft_matrix[row.name][catalog_length:(catalog_length + sp_length)].argsort()[-8:][::-1]
    top_csf = soft_matrix[row.name][(catalog_length + sp_length):].argsort()[-5:][::-1]

    file.write(row['NON-TECHNICAL'])
    file.write("\n")
    file.write(row['Sub-Capability'])
    file.write("\n")
    file.write(row['Element/Action'])
    file.write("\n")
    file.write(row['Sub-Element/Sub-Action'])
    file.write("\n"),
    file.write(str([index_list[catalog_length + ind] for ind in top_sp]))
    file.write('\n')
    file.write(str([soft_matrix[row.name][catalog_length + ind] for ind in top_sp]))
    file.write('\n')
    file.write(str([index_list[catalog_length + sp_length + ind] for ind in top_csf]))
    file.write('\n')
    file.write(str([soft_matrix[row.name][catalog_length + sp_length + ind] for ind in top_csf]))
    file.write("\n\n\n")


def write_catalog_row(row, file):
    file.write('-----------------------------------------------------------------\n')
    file.write("[NON-TECHNICAL] ")
    file.write("NaN" if pandas.isna(row['NON-TECHNICAL']) else row['NON-TECHNICAL'])
    file.write("\n")
    file.write("[Sub-Capability] ")
    file.write("NaN" if pandas.isna(row['Sub-Capability']) else row['Sub-Capability'])
    file.write("\n")
    file.write("[Element/Action] ")
    file.write("NaN" if pandas.isna(row['Element/Action']) else row['Element/Action'])
    file.write("\n")
    file.write("[Sub-Element/Sub-Action] ")
    file.write("NaN" if pandas.isna(row['Sub-Element/Sub-Action']) else row['Sub-Element/Sub-Action'])
    file.write("\n")
    file.write('-----------------------------------------------------------------\n\n')

