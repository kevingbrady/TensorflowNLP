import tensorflow as tf
import pandas as pd
from src import utils


class SimilarityMatrix:

    catalog_dataframe = None
    matrix = None

    def __init__(self, DocumentManager, embeddings_vector):

        self.document_manager = DocumentManager
        self.catalog_dataframe = pd.read_pickle("src/metadata/catalog_dataframe.pkl")
        self.matrix = self.compute_similarity_matrix(embeddings_vector["pooled_output"])

    @tf.function
    def compute_similarity_matrix(self, embeddings_vector):

        A = tf.nn.l2_normalize(embeddings_vector, 0)
        return 1 - tf.matmul(A, A, transpose_b=True)

    def write_output_to_file(self, filename):

        output_file = open(filename, 'w+')
        self.catalog_dataframe.apply(utils.write_output,
                                     args=(output_file,
                                      self.matrix,
                                      self.document_manager.document_metadata['800-213A']['shape'][0],
                                      self.document_manager.document_metadata['sp800_53']['shape'][0],
                                      self.document_manager.document_metadata['csf']['shape'][0],
                                      self.document_manager.indexes), axis=1)
        output_file.close()
