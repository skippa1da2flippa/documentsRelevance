import pickle
from numpy import ndarray


def resultStorer(structures: list[dict[str, ndarray[tuple[str, float]]]]):
    file = open("data/resultDatabase.pkl", 'wb')
    pickle.dump(structures, file)
    file.close()


def resultGetter() -> list[dict[str, ndarray[tuple[str, float]]]]:
    file = open("data/resultDatabase.pkl", 'rb')
    structures: list[dict[str, ndarray[tuple[str, float]]]] = pickle.load(file)
    file.close()

    return structures
