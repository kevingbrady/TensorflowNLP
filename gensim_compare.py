import sys
import pandas
from src import utils
import numpy as np
import gensim
from gensim import corpora
import os
import hashlib

import fasttext.util as fasttext_util
from gensim.models import fasttext

if __name__ == '__main__':
    _sheet_one_ = sys.argv[1]
    _sheet_two_ = sys.argv[2]
    _sheet_three_ = sys.argv[3]

    catalog_check = open("catalog.txt", "w+")

    sp800_53 = pandas.read_excel(_sheet_one_)
    catalog = pandas.read_excel(_sheet_two_, sheet_name="(new)Non-Tech Manufactures")
    csf = pandas.read_excel(_sheet_three_)
    sp800_53.drop(0, inplace=True)
    catalog.drop(0, inplace=True)

    catalog.columns = ["NON-TECHNICAL", "Sub-Capability", "Element/Action", "Sub-Element/Sub-Action", "800-53", "CSF",
                       "Comment", "Filter by X for comments only", "NIST Reviewer", 'Unnamed: 10', 'Unnamed: 11',
                       'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17',
                       'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23']

    sp800_53.fillna("", inplace=True)
    catalog.fillna(method='ffill', inplace=True)
    csf.fillna(method='ffill', inplace=True)

    csf.drop(columns=['Informative References'], inplace=True)
    csf.drop_duplicates(subset=['Category', 'Subcategory'], inplace=True)


    sp800_53_ids = sp800_53['Control Identifier'].values.tolist()
    csf['Control Identifier'] = csf['Subcategory'].apply(lambda r: r[0:7])

    csf_ids = csf['Control Identifier'].values.tolist()

    # Filter out sections with no controls
    catalog = catalog[catalog['Element/Action'].str.contains("^[a-z]\.") == False]

    catalog.reset_index(inplace=True)
    sp800_53.reset_index(inplace=True)
    csf.reset_index(inplace=True)

    catalog.apply(utils.write_catalog_row, args=(catalog_check,), axis=1)

    indexes = [i for i in range(1, catalog.shape[0] + 1)] + sp800_53_ids + csf_ids

    documents = catalog[['NON-TECHNICAL', 'Sub-Capability', 'Element/Action', 'Sub-Element/Sub-Action']].values.tolist() \
                + sp800_53[['Control (or Control Enhancement) Name', 'Control Text', 'Discussion']].values.tolist() \
                + csf[['Function', 'Category', 'Subcategory']].values.tolist()

    documents = [' '.join(x) for x in documents]

    catalog_hash = hashlib.sha256(repr(catalog[['NON-TECHNICAL', 'Sub-Capability', 'Element/Action',
                                                'Sub-Element/Sub-Action']].values.tolist()).encode('utf-8'))
    sp800_53_hash = hashlib.sha256(
        repr(sp800_53[['Control (or Control Enhancement) Name', 'Control Text', 'Discussion']].values.tolist()).encode(
            'utf-8'))

    csf_hash = hashlib.sha256(repr(csf[['Function', 'Category', 'Subcategory']].values.tolist()).encode('utf-8'))

    reset = False

    print('[FAST_TEXT] Checking for hash files ...')
    if os.path.exists('catalog_hash.txt') and os.path.exists('sp800_53_hash.txt') and os.path.exists('csf_hash.txt'):
        print('[FAST_TEXT] Catalog hash file found')
        with open('catalog_hash.txt', 'r+') as file:
            hash = file.readline()
            if catalog_hash.hexdigest() != hash:
                print('[FAST_TEXT] Catalog hash file out of date ...')
                file.truncate()
                file.write(catalog_hash.hexdigest())
                reset = True

        print('[FAST_TEXT] 800-53 hash file found')
        with open('sp800_53_hash.txt', 'r+') as file:
            hash = file.readline()
            if sp800_53_hash.hexdigest() != hash:
                print('[FAST_TEXT] 800-53 hash file out of date ...')
                file.truncate()
                file.write(sp800_53_hash.hexdigest())
                reset = True

        print('[FAST_TEXT] CSF hash file found')
        with open('csf_hash.txt', 'r+') as file:
            hash = file.readline()
            if csf_hash.hexdigest() != hash:
                print('[FAST_TEXT] CSF hash file out of date ...')
                file.truncate()
                file.write(csf_hash.hexdigest())
                reset = True

    else:
        reset = True
        print('[FAST_TEXT] Hash files not found reset enabled')

        with open('catalog_hash.txt', 'w+') as file:
            file.write(catalog_hash.hexdigest())
        with open('sp800_53_hash.txt', 'w+') as file:
            file.write(sp800_53_hash.hexdigest())
        with open('csf_hash.txt', 'w+') as file:
            file.write(csf_hash.hexdigest())

    # Download the FastText model
    print("[FAST_TEXT] LOADING FAST TEXT MODEL ...")

    # fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
    if os.path.exists('FAST_TEXT'):
        print("[FAST_TEXT] LOADING FAST TEXT MODEL FROM FILE 'FAST_TEXT'")
        fasttext_model300 = fasttext.FastText.load('FAST_TEXT')
        print("[FAST_TEXT] FAST TEXT MODEL LOADED FROM FILE 'FAST_TEXT'")

    else:
        fasttext_util.download_model('en', if_exists="ignore")
        print("[FAST_TEXT] LOADING FAST TEXT MODEL FROM FILE 'cc.en.300.bin'")
        fasttext_model300 = fasttext.load_facebook_model('cc.en.300.bin')
        fasttext_model300.save('FAST_TEXT')
        print("[FAST_TEXT] FAST TEXT MODEL LOADED FROM FILE 'cc.en.300.bin'")

    print("[FAST_TEXT] FAST TEST MODEL LOADED")

    print("[FAST_TEXT] PREPROCESSING DOCUMENTS ...")
    preprocessed_documents = [utils.pre_process(doc) for doc in documents]
    print("[FAST_TEXT] DOCUMENTS PREPROCESSED")

    if os.path.exists("FAST_TEXT_Dictionary") and reset is False:
        print("[FAST_TEXT] LOADING DICTIONARY FROM FILE 'FAST_TEXT_Dictionary'")
        dictionary = corpora.Dictionary.load("FAST_TEXT_Dictionary")
        print("[FAST_TEXT] DICTIONARY LOADED")
    else:

        print("[FAST_TEXT] BUILDING DICTIONARY ...")
        dictionary = corpora.Dictionary(preprocessed_documents)
        print("[FAST_TEXT] DICTIONARY BUILT")

        print("[FAST_TEXT] SAVING DICTIONARY TO FILE 'FAST_TEXT_Dictionary'")
        dictionary.save('FAST_TEXT_Dictionary')

    tfidf = gensim.models.TfidfModel(dictionary=dictionary)

    if os.path.exists("similarity_matrix_fasttext") and reset is False:
        print("[FAST_TEXT] LOADING SIMILARITY MATRIX FROM FILE 'similarity_matrix_fasttext'")
        similarity_matrix = gensim.similarities.SparseTermSimilarityMatrix.load("similarity_matrix_fasttext")
        print("[FAST_TEXT] LOADED SIMILARITY MATRIX ...")
    else:
        print("[FAST_TEXT] BUILDING WORD EMBEDDING SIMILARITY INDEX ...")
        similarity_index = gensim.models.WordEmbeddingSimilarityIndex(fasttext_model300.wv)
        print("[FAST_TEXT] WORD EMBEDDING SIMILARITY INDEX BUILT ...")

        print("[FAST_TEXT] BUILDING SIMILARITY MATRIX ...")
        similarity_matrix = gensim.similarities.SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf=tfidf,
                                                                          nonzero_limit=100)
        similarity_matrix.save("similarity_matrix_fasttext")
        print("[FAST_TEXT] SIMILARITY MATRIX BUILT")

    print("[FAST_TEXT] CONVERTING DOCUMENTS TO BAG OF WORDS VECTORS")
    bow_documents = [dictionary.doc2bow(x) for x in preprocessed_documents]
    print("[FAST_TEXT] CONVERSION COMPLETE")

    if os.path.exists("softcossim_matrix_fasttext") and reset is False:
        print("[FAST_TEXT] LOADING SOFT COSINE SIMILARITY MATRIX FROM FILE 'softcossim_matrix_fasttext'")
        soft_matrix_gensim = gensim.similarities.SoftCosineSimilarity.load("softcossim_matrix_fasttext")
        print("[FAST_TEXT] SOFT COSINE SIMILARITY MATRIX LOADED")
    else:
        print("[FAST_TEXT] BUILDING SOFT COSINE SIMILARITY MATRIX ...")
        soft_matrix_gensim = gensim.similarities.SoftCosineSimilarity(tfidf[bow_documents], similarity_matrix)
        soft_matrix_gensim.save("softcossim_matrix_fasttext")
        print("[FAST_TEXT] SOFT COSINE SIMILARITY MATRIX BUILT")

    soft_matrix = np.array(soft_matrix_gensim)
    output_file = open('output.txt', 'w+')
    catalog.apply(utils.write_output, args=(output_file, soft_matrix, catalog.shape[0], sp800_53.shape[0], csf.shape[0], indexes), axis=1)
    output_file.close()

    catalog_check.close()
