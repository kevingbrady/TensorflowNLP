import time
from src.ArgumentManager import ArgumentManager
from src.DocumentManager import DocumentManager
from src.Bert import BertModel
from src.SimilarityMatrix import SimilarityMatrix
import src.utils as utils

if __name__ == '__main__':

    argument_manager = ArgumentManager()
    document_manager = DocumentManager(argument_manager)

    start = time.time()
    bert_model = BertModel()
    bert_model.preprocess_documents(document_manager.documents)
    bert_results = bert_model.run()

    similarity_matrix = SimilarityMatrix(document_manager, bert_results)
    similarity_matrix.write_output_to_file("output.txt")
    end = time.time()

    print('Total Run Time: ' + utils.print_run_time(end - start))


    '''
    print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
    print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
    print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
    print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')
    '''




    


