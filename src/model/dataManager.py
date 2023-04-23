import json

from nltk import word_tokenize
from numpy import ndarray, array, append
from pandas import DataFrame, read_csv
import nltk


class DataManager(object):
    def __init__(self, queryPath: str, documentsPath: str, solutionPath):
        self.queries: ndarray[dict[str, str]] = array([])
        self.documents: ndarray[dict[str, str]] = array([])
        self.tokenizedQueries: ndarray[list[str]] = array([])
        self.tokenizedDocuments: ndarray[list[str]] = array([])

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
            self.tokenizedQueries = append(self.tokenizedQueries, word_tokenize(query["text"]))

        for doc in self.documents:
            self.tokenizedDocuments = append(self.tokenizedDocuments, word_tokenize(doc["title"] + doc["text"]))


base: str = "C:\\Users\\biagi\\OneDrive\\Documents\\Desktop\\trec-covid\\"

manager: DataManager = DataManager(base + "queries.jsonl", base + "corpus.jsonl", base + "qrels\\test.tsv")

print(manager.queries)
