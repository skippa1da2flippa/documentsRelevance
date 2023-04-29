from src.model.dataManager import DataManager
from nltk import word_tokenize
from rank_bm25 import BM25Okapi
from numpy import ndarray, array, inner
from sentence_transformers import SentenceTransformer
import torch
import re
from src.model.resultManager import resultStorer

"""
    This class computes the score for both sparse and dense vectors representation
"""


class MultipleRepresentation:
    def __init__(self, queryPath: str, documentsPath: str, solutionPath):
        self.dataManager: DataManager = DataManager(queryPath, documentsPath, solutionPath)
        self._tokenizedQueries: dict[str, list[str]] = {}
        self._tokenizedDocuments: dict[str, list[str]] = {}

    """
        Method which tokenizes all the queries and documents inside the dataset
    """
    def _sparseTokenization(self):
        for query in self.dataManager.getQueries():
            queryText: str = re.sub(r'[^\w\s]', '', query["text"])
            self._tokenizedQueries[query["_id"]] = word_tokenize(queryText)

        for doc in self.dataManager.getDocuments():
            docCorpus: str = re.sub(r'[^\w\s]', '', doc["title"] + " " + doc["text"])
            self._tokenizedDocuments[doc["_id"]] = word_tokenize(docCorpus)

    """
        getSparseScores compute the scores for the sparse representation by using bm25 index
    """
    def getSparseScores(self) -> dict[str, ndarray[tuple[str, float]]]:
        self._sparseTokenization()
        bm25 = BM25Okapi(list(self._tokenizedDocuments.values()))
        solution: dict[str, ndarray[tuple[str, float]]] = {}
        documentKeys = list(self._tokenizedDocuments.keys())
        for key in self._tokenizedQueries:
            scores: ndarray[float] = array(bm25.get_scores(self._tokenizedQueries[key]))
            solution["query{}".format(key)] = array([(documentKeys[idx], scores[idx])
                                                     for idx in range(0, len(scores))])

        return solution

    """
        this method return the score related to the dense representation using inner product
    """
    def getDenseScores(self) -> dict[str, ndarray[tuple[str, float]]]:
        model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        vectorizedDocs = model.encode(
            [f"{doc['title']} {doc['text']}" for doc in self.dataManager.getDocuments()],
            convert_to_numpy=True,
        )
        solution: dict[str, ndarray[tuple[str, float]]] = {}

        for query in self.dataManager.getQueries():
            print("Im at the query: ", query["_id"])
            vectorizedQuery = model.encode(
                query["text"],
                convert_to_numpy=True,
            )

            solution["query" + query["_id"]] = array(
                [(self.dataManager.getDocuments()[idx]["_id"], inner(vectorizedQuery, vectorizedDocs[idx]))
                 for idx in range(0, len(self.dataManager.getDocuments()))]
            )

        return solution


def saveDatas():
    base: str = "data/"
    manager = MultipleRepresentation(base + "queries.jsonl", base + "corpus.jsonl", base + "test.tsv")
    struct1 = manager.getSparseScores()
    struct2 = manager.getDenseScores()
    resultStorer([struct1, struct2])
