import os
import hashlib
from pathlib import Path
import json
import pickle
from src.DataFrameManager import DataFrameManager


class DocumentManager:

    document_metadata = {}
    documents = []
    indexes = []

    def __init__(self, ArgumentManager):

        self.document_metadata.update({
            'sp800_53': {'hash': self.compute_filehash(ArgumentManager.sp800_53, 'md5')},
            '800-213A': {'hash': self.compute_filehash(ArgumentManager.catalog, 'md5')},
            'csf': {'hash': self.compute_filehash(ArgumentManager.csf, 'md5')}
        })

        if os.path.exists('src/metadata/document_metadata.json') \
                and os.path.exists('src/metadata/documents.pkl')\
                and os.path.exists('src/metadata/indexes.pkl'):

            with open('src/metadata/document_metadata.json', 'r') as r:
                document_metadata_saved = json.loads(r.read())

            if self.check_documents_updated(self.document_metadata, document_metadata_saved):
                print('Document check passed, continuing ...')

                self.document_metadata = document_metadata_saved
                with open('src/metadata/documents.pkl', 'rb') as h:
                    self.documents = pickle.load(h)
                with open('src/metadata/indexes.pkl', 'rb') as h:
                    self.indexes = pickle.load(h)

        else:

            dataframes = DataFrameManager(ArgumentManager)
            
            sp800_53_ids = dataframes.sp800_53['Control Identifier'].values.tolist()
            csf_ids = dataframes.csf['Control Identifier'].values.tolist()

            self.document_metadata['sp800_53'].update({'shape': dataframes.sp800_53.shape})
            self.document_metadata['800-213A'].update({'shape': dataframes.catalog.shape})
            self.document_metadata['csf'].update({'shape': dataframes.csf.shape})

            # catalog.apply(utils.write_catalog_row, args=(catalog_check,), axis=1)

            indexes = [i for i in range(1, dataframes.catalog.shape[0] + 1)] + sp800_53_ids + csf_ids

            documents = dataframes.catalog[['NON-TECHNICAL', 'Sub-Capability', 'Element/Action',
                                 'Sub-Element/Sub-Action']].values.tolist() \
                        + dataframes.sp800_53[
                            ['Control (or Control Enhancement) Name', 'Control Text', 'Discussion']].values.tolist() \
                        + dataframes.csf[['Function', 'Category', 'Subcategory']].values.tolist()

            documents = [' '.join(x) for x in documents]

            with open('src/metadata/documents.pkl', 'wb') as pb:
                pickle.dump(documents, pb, protocol=pickle.HIGHEST_PROTOCOL)
            with open('src/metadata/document_metadata.json', 'w') as fp:
                json.dump(self.document_metadata, fp)
            with open('src/metadata/indexes.pkl', 'wb') as pb:
                pickle.dump(indexes, pb, protocol=pickle.HIGHEST_PROTOCOL)

            dataframes.catalog.to_pickle('src/metadata/catalog_dataframe.pkl')

    def check_documents_updated(self, document_metadata, document_metadata_saved):

        for key in document_metadata:

            if document_metadata[key]['hash'] != document_metadata_saved[key]['hash']:
                return False

        return True

    def compute_filehash(self, filepath: str, hashtype: str) -> str:
        """Computes the requested hash for the given file.

        Args:
            filepath: The path to the file to compute the hash for.
            hashtype: The hash type to compute.

              Available hash types:
                md5, sha1, sha224, sha256, sha384, sha512, sha3_224,
                sha3_256, sha3_384, sha3_512, shake_128, shake_256

        Returns:
            A string that represents the hash.

        Raises:
            ValueError: If the hash type is not supported.
        """
        if hashtype not in ['md5', 'sha1', 'sha224', 'sha256', 'sha384',
                            'sha512', 'sha3_224', 'sha3_256', 'sha3_384',
                            'sha3_512', 'shake_128', 'shake_256']:
            raise ValueError(f'Hash type {hashtype} is not supported.')

        return getattr(hashlib, hashtype)(
            Path(filepath).read_bytes()).hexdigest()

