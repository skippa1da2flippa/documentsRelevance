import json
import numpy
from numpy import ndarray, array, append


class DataManager(object):
    def __int__(self, queryPath: str, documentsPath: str):
        self.queries: ndarray[dict[str, str]] = array([])
        self.documents: ndarray[dict[str, str]] = array([])

        self._readJson(queryPath)
        self._readJson(documentsPath, False)

    def _readJson(self, path: str, isQuery: bool = True):
        with open(path) as file:
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

#%%
