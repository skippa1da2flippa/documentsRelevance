import json
from nltk import word_tokenize
from numpy import ndarray, array, append
from pandas import DataFrame, read_csv
from rank_bm25 import BM25Okapi


class DataManager(object):
    def __init__(self, queryPath: str, documentsPath: str, solutionPath):
        self.queries: ndarray[dict[str, str]] = array([])
        self.documents: ndarray[dict[str, str]] = array([])
        self.tokenizedQueries: dict[str, list[str]] = {}
        self.tokenizedDocuments: dict[str, list[str]] = {}

        self.solutions: DataFrame = read_csv(solutionPath)
        self._readJson(queryPath)
        self._readJson(documentsPath, False)

    def _readJson(self, path: str, isQuery: bool = True):
        with open(path, "r") as file:
            data: str = file.read()
            objects: list[str] = data.strip().split('\n')

            if isQuery:
                def doOperation(jsonElem: dict[str, str]):
                    self.queries = append(self.queries, jsonElem)

            else:
                def doOperation(jsonElem: dict[str, str]):
                    self.documents = append(self.documents, jsonElem)

            for obj in objects:
                try:
                    json_data = json.loads(obj)
                    doOperation(json_data)
                except json.JSONDecodeError:
                    print("Bro we got some issue with the json encoding")

    def _tokenize(self):
        for query in self.queries:
            self.tokenizedQueries[query["_id"]] = word_tokenize(query["text"])

        for doc in self.documents:
            self.tokenizedDocuments[doc["_id"]] = word_tokenize(doc["title"] + " " + doc["text"])

    def _getSparseScores(self) -> dict[str, ndarray[tuple[str, float]]]:
        bm25 = BM25Okapi(list(self.tokenizedDocuments.values()))
        solution: dict[str, ndarray[tuple[str, float]]] = {}
        documentKeys = list(self.tokenizedDocuments.keys())
        for key in self.tokenizedQueries:
            scores: ndarray[float] = array(bm25.get_scores(self.tokenizedQueries[key]))
            solution["query{}".format(key)] = array([(documentKeys[idx], scores[idx]) for idx in range(0, len(scores))])

        return solution


base: str = "data/"

manager: DataManager = DataManager(base + "queries.jsonl", base + "corpus.jsonl", base + "qrels/test.tsv")

print(manager.queries)
