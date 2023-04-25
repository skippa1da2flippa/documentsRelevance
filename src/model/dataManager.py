import json
from nltk import word_tokenize
from numpy import ndarray, array, append, inner, argsort, flip
from pandas import DataFrame, read_csv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class DataManager(object):
    def __init__(self, queryPath: str, documentsPath: str, solutionPath, autoTokenize: bool = True):
        self._queries: ndarray[dict[str, str]] = array([])
        self._documents: ndarray[dict[str, str]] = array([])
        self._tokenizedQueries: dict[str, list[str]] = {}
        self._tokenizedDocuments: dict[str, list[str]] = {}

        self._solutions: DataFrame = read_csv(solutionPath)
        self._readJson(queryPath)
        self._readJson(documentsPath, False)

        if autoTokenize:
            self._tokenize()

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

    def getTokenizedQueries(self) -> dict[str, list[str]]:
        return self._tokenizedQueries

    def getTokenizedDocuments(self) -> dict[str, list[str]]:
        return self._tokenizedDocuments

    def getSolutions(self) -> DataFrame:
        return self._solutions

    def _tokenize(self):
        for query in self._queries:
            self._tokenizedQueries[query["_id"]] = word_tokenize(query["text"])

        for doc in self._documents:
            self._tokenizedDocuments[doc["_id"]] = word_tokenize(doc["title"] + " " + doc["text"])

    def _minimizeTokenization(self):
        pass

    # this method should not be here
    def getSparseScores(self) -> dict[str, ndarray[tuple[str, float]]]:
        bm25 = BM25Okapi(list(self._tokenizedDocuments.values()))
        solution: dict[str, ndarray[tuple[str, float]]] = {}
        documentKeys = list(self._tokenizedDocuments.keys())
        for key in self._tokenizedQueries:
            scores: ndarray[float] = array(bm25.get_scores(self._tokenizedQueries[key]))
            solution["query{}".format(key)] = array([(documentKeys[idx], scores[idx])
                                                     for idx in range(0, len(scores))])

        return solution

    # this method should not be here
    def doDenseThing(self) -> dict[str, ndarray[tuple[str, float]]]:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        vectorizedDocs = model.encode(
            [f"{doc['title']} {doc['text']}" for doc in self._documents],
            convert_to_numpy=True,
        )
        solution: dict[str, ndarray[tuple[str, float]]] = {}

        for query in self._queries:
            vectorizedQuery = model.encode(
                query,
                convert_to_numpy=True,
            )

            solution[query["_id"]] = array(
                [(self._documents[idx]["_id"], inner(vectorizedQuery, vectorizedDocs[idx]))
                 for idx in range(0, len(self._documents))]
            )

        return solution

    # this method should not be here (make solution self.solution)
    def takeTopKPrime(self, solution: dict[str, ndarray[tuple[str, float]]], kPrime: int):
        for key in solution:
            solution[key] = (solution[key][argsort(solution[:1])])[-kPrime:]

    # assuming these solutions have already gone through takeTopKPrime function
    def computeTopK(self, denseSolution: dict[str, ndarray[tuple[str, float]]],
                    sparseSolution: dict[str, ndarray[tuple[str, float]]],
                    k: int) -> dict[str, ndarray[tuple[str, float]]]:
        topK: dict[str, ndarray[tuple[str, float]]] = {}

        for key in denseSolution:
            topK[key] = self.inOrder(denseSolution[key], sparseSolution[key], k)

        return topK

    # Assuming two already sorted data structure
    def inOrder(self, denseScore: ndarray[tuple[str, float]],
                sparseScore: ndarray[tuple[str, float]], k: int) -> ndarray[tuple[str, float]]:

        solution: set[tuple[str, float]] = set()
        sparseIdx: int = 0
        denseIdx: int = 0

        while (sparseIdx < len(sparseScore) or denseIdx < len(denseScore)) and k > 0:
            majorityCondition: bool = denseScore[denseIdx][1] < sparseScore[sparseIdx][1]
            sparseCondition: bool = sparseIdx < len(sparseScore)
            denseCondition: bool = denseIdx < len(denseScore)

            if sparseCondition and denseCondition:
                if majorityCondition:
                    solution.add((denseScore[denseIdx][0], denseScore[denseIdx][1]))
                    denseIdx += 1
                else:
                    solution.add((sparseScore[sparseIdx][0], sparseScore[sparseIdx][1]))
                    sparseIdx += 1

                k -= 1

            elif sparseCondition:
                while sparseIdx < len(sparseScore) and k > 0:
                    k -= 1
                    solution.add((sparseScore[sparseIdx][0], sparseScore[sparseIdx][1]))
                    sparseIdx += 1

            else:
                while denseIdx < len(denseScore) and k > 0:
                    k -= 1
                    solution.add((denseScore[denseIdx][0], sparseScore[denseIdx][1]))
                    denseIdx += 1

        return flip(array(solution))

    def computeScore(self, topK: dict[str, ndarray[tuple[str, float]]]):
        for index, row in self._solutions.iterrows():
            pass


base: str = "data/"

manager: DataManager = DataManager(base + "queries.jsonl", base + "corpus.jsonl", base + "test.tsv", False)

print(manager.doDenseThing())
