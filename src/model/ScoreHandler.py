from matplotlib import pyplot as plt
from numpy import ndarray, flip, argsort, union1d, intersect1d, append, where, mean

from src.model.resultManager import resultGetter


def sort(arr: ndarray[tuple[str, float]]):
    sortedIndices = flip(argsort(arr[:, 1]))
    arr = arr[sortedIndices]


class ScoreHandler:
    def __init__(self, sparseResult: dict[str, ndarray[tuple[str, float]]],
                 denseResult: dict[str, ndarray[tuple[str, float]]]):
        self._sparseResult: dict[str, ndarray[tuple[str, float]]] = sparseResult
        self._denseResult: dict[str, ndarray[tuple[str, float]]] = denseResult

        self._groundTruth: dict[str, ndarray[tuple[str, float]]] = {}
        self._modelOutput: dict[str, ndarray[tuple[str, float]]] = {}
        self._recall: dict[int, dict[int, ndarray[tuple[str, float]]]] = {}

        self._sortSparseDense()
        self._computeGroundTruth()

    def plotGraph(self, k: int):
        # to modify
        plt.plot(self._recall[k])
        plt.title("k: " + str(k))
        plt.xlabel("k prime")
        plt.ylabel("recall")
        plt.show()

    def _takeKSamples(self, k: int) -> dict[str, ndarray[str]]:
        kSamplesDict: dict[str, ndarray[str]] = {}
        for queryId in self._groundTruth:
            kSamplesDict[queryId] = self._groundTruth[queryId][:k, 0]

        return kSamplesDict

    def _computeGroundTruth(self):
        for queryId in self._sparseResult:
            for idx in range(0, len(self._sparseResult[queryId])):
                self._groundTruth[queryId][idx][1] = self._sparseResult[queryId][idx][1] + \
                                                     self._denseResult[queryId][idx][1]

            sort(self._groundTruth[queryId])

    def _sortSparseDense(self):
        for queryId in self._sparseResult:
            sort(self._sparseResult[queryId])
            sort(self._denseResult[queryId])

    def _computeSPrime(self, k: int, kPrime: int) -> dict[str, ndarray[str]]:
        sPrimeScore: dict[str, ndarray[tuple[str, float]]] = {}
        sPrimeSample: dict[str, ndarray[str]] = {}
        for queryId in self._sparseResult:
            sPrimeSample[queryId] = union1d(self._sparseResult[queryId][:kPrime, 0],
                                            self._denseResult[queryId][:kPrime, 0])

        for queryId in sPrimeSample:
            for docId in sPrimeSample[queryId]:
                rightIdx: int = where(self._groundTruth[queryId][:, 0] == docId)[0][0]
                score: float = self._groundTruth[queryId][rightIdx][1]
                sPrimeScore[queryId] = append(sPrimeScore[queryId], (docId, score))

            sort(sPrimeScore[queryId])
            sPrimeSample[queryId] = sPrimeScore[queryId][:k, 0]

        return sPrimeSample

    def _computeMean(self):
        experimentResult: dict[int, ndarray[tuple[int, float]]] = {}
        for k in self._recall:
            for kPrime in self._recall[k]:
                kKPrimeMean: float = self._recall[k][kPrime][:, 1].mean()
                experimentResult[k] = append(experimentResult[k], (kPrime, kKPrimeMean))

    def _computeRecalls(self):
        for k in range(0, 10000):
            for kPrime in range(k, len(self._sparseResult["query1"])):
                modelTruthDict: dict[str, ndarray[str]] = self._computeSPrime(k, kPrime)
                righteousTruthDict: dict[str, ndarray[str]] = self._takeKSamples(k)
                for queryId in modelTruthDict:
                    recall: float = len(intersect1d(modelTruthDict[queryId], righteousTruthDict[queryId])) / k
                    self._recall[k][kPrime] = append(self._recall[k][kPrime], (queryId, recall))


l = resultGetter()

scoreHandler = ScoreHandler(l[0], l[1])
