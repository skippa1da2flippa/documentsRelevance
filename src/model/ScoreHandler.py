from matplotlib import pyplot as plt
from numpy import ndarray, flip, argsort, union1d, intersect1d, append, where, mean, array

from src.model.resultManager import resultGetter


def sort(arr: ndarray[tuple[str, float]]):
    sortedIndices = flip(argsort(arr[:, 1]))
    arr = arr[sortedIndices]


"""
    Class representing the scoring operation given some sparse and dense results 
"""


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

    """
        This method takes the first k best samples from the ground truth and return just the docId's related to them
    """

    def _takeKSamples(self, k: int) -> dict[str, ndarray[str]]:
        kSamplesDict: dict[str, ndarray[str]] = {}
        for queryId in self._groundTruth:
            kSamplesDict[queryId] = self._groundTruth[queryId][:k, 0]

        return kSamplesDict

    """
        This method compute the ground truth by adding for each document sparse and dense representation. 
        This operation is performed for each query
    """

    def _computeGroundTruth(self):
        for queryId in self._sparseResult:
            for idx in range(0, len(self._sparseResult[queryId])):
                sparseDenseSum: float = float(self._sparseResult[queryId][idx][1]) + \
                                        float(self._denseResult[queryId][idx][1])
                sparseDenseDocId: str = self._denseResult[queryId][idx][0]

                if idx == 0:
                    self._groundTruth[queryId] = array([(sparseDenseDocId, sparseDenseSum)])
                else:
                    self._groundTruth[queryId] = append(self._groundTruth[queryId],
                                                        [(sparseDenseDocId, sparseDenseSum)], axis=0)

            sort(self._groundTruth[queryId])

    def _sortSparseDense(self):
        for queryId in self._sparseResult:
            sort(self._sparseResult[queryId])
            sort(self._denseResult[queryId])

    """
        _computeSPrime generates the model output by first taking the best k' from the dense and the sparse 
        representation, after that a union operation is applied to the best k's. Lastly the union is sorted and the best
        k samples are returned
    """

    def _computeSPrime(self, k: int, kPrime: int) -> dict[str, ndarray[str]]:
        sPrimeScore: dict[str, ndarray[tuple[str, float]]] = {}
        sPrimeSample: dict[str, ndarray[str]] = {}
        for queryId in self._sparseResult:
            sPrimeSample[queryId] = union1d(self._sparseResult[queryId][:kPrime, 0],
                                            self._denseResult[queryId][:kPrime, 0])

        for queryId in sPrimeSample:
            sPrimeScore[queryId] = array([])
            for docId in sPrimeSample[queryId]:
                rightIdx: int = where(self._groundTruth[queryId][:, 0] == docId)[0][0]
                score: float = self._groundTruth[queryId][rightIdx][1]
                if sPrimeScore[queryId].size() > 0:
                    sPrimeScore[queryId] = append(sPrimeScore[queryId], (docId, score))
                else:
                    sPrimeScore[queryId] = array([(docId, score)])

            sort(sPrimeScore[queryId])
            sPrimeSample[queryId] = sPrimeScore[queryId][:k, 0]

        return sPrimeSample

    """
        This methods compute the mean for each k, k' given all the queries results
    """
    def _computeMean(self):
        experimentResult: dict[int, ndarray[tuple[int, float]]] = {}
        for k in self._recall:
            experimentResult[k] = array([])
            for kPrime in self._recall[k]:
                kKPrimeMean: float = self._recall[k][kPrime][:, 1].mean()
                if experimentResult[k].size() > 0:
                    experimentResult[k] = append(experimentResult[k], (kPrime, kKPrimeMean))
                else:
                    experimentResult[k] = array([(kPrime, kKPrimeMean)])

        return experimentResult

    """
        This method computes the recalls of all the queries for all the possible k and kPrimes
    """

    def computeRecalls(self):
        for k in range(0, 10000):
            for kPrime in range(k, len(self._sparseResult["query1"])):
                modelTruthDict: dict[str, ndarray[str]] = self._computeSPrime(k, kPrime)
                righteousTruthDict: dict[str, ndarray[str]] = self._takeKSamples(k)
                for queryId in modelTruthDict:
                    recall: float = len(intersect1d(modelTruthDict[queryId], righteousTruthDict[queryId])) / k
                    self._recall[k][kPrime] = append(self._recall[k][kPrime], (queryId, recall))






