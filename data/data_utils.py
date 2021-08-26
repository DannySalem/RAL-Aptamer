# from data.dataset import Aptamer_Dataset
from data.dataset import Aptamer_Dataset
from utils.oracle import oracle, filterDuplicateSamples
import numpy as np

params = {}
params["dataset seed"] = 0
params["variable sample length"] = False
params["max sample length"] = 20
params["test mode"] = True
params["pipeline iterations"] = 2
params["init dataset length"] = 400
params["queries per iter"] = 100
params["sampling time"] = int(1e3)
params["num samplers"] = 2
params["ensemble size"] = 2
params["max training epochs"] = 5
params["dataset"] = "potts"
params["model filters"] = 12
params["model layers"] = 1  # for cluster batching
params["embed dim"] = 12  # embedding dimension
params["batch size"] = 10  # model training batch size
params["min sample length"], params["max sample length"] = [
    10,
    20,
]  # minimum input sequence length and # maximum input sequence length (inclusive) - or fixed sample size if 'variable sample length' is false
params[
    "dict size"
] = 4  # number of possible choices per-state, e.g., [0,1] would be two, [1,2,3,4] (representing ATGC) would be 4

toy_oracle = oracle(params)
toy_oracle.initializeDataset()


def get_exp_data():
    """Create all datasets required for Training.

    :return: labelled_dataset, model_state_dataset, unlabelled_dataset
    """

    labelled_data = np.load("./datasets/potts.npy", allow_pickle=True).item()
    labelled_dataset = {}
    val_dataset = {}
    labelled_dataset["samples"] = labelled_data["samples"][
        int((len(labelled_data["samples"])) / 2) :
    ]
    labelled_dataset["scores"] = labelled_data["scores"][int((len(labelled_data["scores"])) / 2) :]
    val_dataset["samples"] = labelled_data["samples"][int((len(labelled_data["samples"])) / 2) :]
    val_dataset["scores"] = labelled_data["scores"][int((len(labelled_data["scores"])) / 2) :]
    model_state_dataset = generateRandomSamples(30, [10, 20], params["dict size"])

    labelled_dataset = Aptamer_Dataset(labelled_dataset, islabelled=True)
    val_dataset = Aptamer_Dataset(val_dataset, islabelled=True)
    model_state_dataset = Aptamer_Dataset(model_state_dataset)

    return model_state_dataset, labelled_dataset, val_dataset


def get_iter_data():

    unlabelled_dataset = generateRandomSamples(100, [10, 20], 4)
    unlabelled_dataset = Aptamer_Dataset(unlabelled_dataset)
    return unlabelled_dataset


def generateRandomSamples(
    nSamples, sampleLengthRange, dictSize, oldDatasetPath=None, variableLength=True
):
    """
    randomly generate a non-repeating set of samples of the appropriate size and composition
    :param nSamples:
    :param sampleLengthRange:
    :param dictSize:
    :param variableLength:
    :return:
    """

    if variableLength:
        samples = []
        while len(samples) < nSamples:
            for i in range(sampleLengthRange[0], sampleLengthRange[1] + 1):
                samples.extend(np.random.randint(1, dictSize + 1, size=(int(10 * dictSize * i), i)))

            samples = numpy_fillna(np.asarray(samples)).astype(
                int
            )  # pad sequences up to maximum length
            samples = filterDuplicateSamples(
                samples, oldDatasetPath
            )  # this will naturally proportionally punish shorter sequences
            if len(samples) < nSamples:
                samples = samples.tolist()

    else:  # fixed sample size
        samples = []
        while len(samples) < nSamples:
            samples.extend(
                np.random.randint(1, dictSize + 1, size=(2 * nSamples, sampleLengthRange[1]))
            )
            samples = numpy_fillna(np.asarray(samples)).astype(
                int
            )  # pad sequences up to maximum length
            samples = filterDuplicateSamples(
                samples, oldDatasetPath
            )  # this will naturally proportionally punish shorter sequences
            if len(samples) < nSamples:
                samples = samples.tolist()

    np.random.shuffle(
        samples
    )  # shuffle so that sequences with different lengths are randomly distributed
    samples = samples[
        :nSamples
    ]  # after shuffle, reduce dataset to desired size, with properly weighted samples

    return {"samples": samples}


def numpy_fillna(data):
    """
    function to pad uneven-length vectors up to the max with zeros
    :param data:
    :return:
    """
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out


def add_labeled_datapoint(labelled_dataset, action_datapoint):
    """This function...

    :param : (torch.utils.data.Dataset) Training set.
    :param action: Selected index to be labeled.
    :return: List of existing images, updated with the new image.
    """

    score = toy_oracle.getScore(action_datapoint.cpu().detach().numpy()[np.newaxis, :])
    labelled_dataset.add_datapoint([action_datapoint, score])
    return labelled_dataset
