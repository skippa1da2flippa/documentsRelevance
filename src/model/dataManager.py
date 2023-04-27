import json
from nltk import word_tokenize
from numpy import ndarray, array, append, inner, argsort, flip
from pandas import DataFrame, read_csv


class DataManager(object):
    def __init__(self, queryPath: str, documentsPath: str, solutionPath):
        self._queries: ndarray[dict[str, str]] = array([])
        self._documents: ndarray[dict[str, str]] = array([])
        self._solutions: DataFrame = read_csv(solutionPath)

        self._readJson(queryPath)
        self._readJson(documentsPath, False)

    def _readJson(self, path: str, isQuery: bool = True):
        with open(path, "r") as file:
            data: str = file.read()
            objects: list[str] = data.strip().split('\n')

            if isQuery:
                def doOperation(jsonElem: dict[str, str]):
                    self._queries = append(self._queries, jsonElem)

            else:
                def doOperation(jsonElem: dict[str, str]):
                    self._documents = append(self._documents, jsonElem)

            for obj in objects:
                try:
                    json_data = json.loads(obj)
                    doOperation(json_data)
                except json.JSONDecodeError:
                    print("Bro we got some issue with the json encoding")

    def getQueries(self) -> ndarray[dict[str, str]]:
        return self._queries

    def getDocuments(self) -> ndarray[dict[str, str]]:
        return self._documents

    def getSolutions(self) -> DataFrame:
        return self._solutions





