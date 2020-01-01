# needed for RaggedTensorFeeder
import itertools
import random
from collections import namedtuple

import numpy as np


def pow_2_reduce(n: int, batch_size: int):
    """ returns the largest integer of the form
    2^m that is less than or equal to n.

    If this integer is greater than the allowed batch_size,
    then we simply return the batch_size
    """
    max_power = int(2 ** np.floor(np.log(n) / np.log(2)))
    return min(max_power, batch_size)


class RaggedTensorFeeder(object):
    """Given a ragged_data list of data points with
    varying first dimension and constant inner dimensions and a
    list of constant shape labels such that labels[i] corresponds
     to ragged_data[i] -- RaggedTensorFeeder is able to generate batches
    that of maximal size batch_size such that an tensorflow RNN
    that only operates on non-ragged tensors can still learn.

        [description]

        Arguments:
            ragged_data {list} -- ragged array of input data
            labels {list} -- constant shape labels

        Keyword Arguments:
            batch_size {int} -- Maximal batch size (default: {32})

        Raises:
            ValueError -- [description]
        """

    def __init__(self, ragged_data: list, labels: list, batch_size: int = 32):
        super(RaggedTensorFeeder, self).__init__()
        self.ragged_data = ragged_data
        self.labels = labels
        self.batch_size = batch_size

        # label[i] must correspond to ragged_data[i]
        if len(labels) != len(ragged_data):
            raise ValueError("Must have same amount of data and labels.")

        # zip data with their corresponding labels
        self.data_set = list(zip(ragged_data, labels))

        # Sorting-function that returns the number of timesteps of data_set[i]
        self.sort_func = lambda e: e[0].shape[0]

    def _get_distribution(self):
        """creates a dict with keys being time lengths and
        values are how many signals of such length are in the ragged_data.

        [description]
        """
        distr = {}
        self.groups = {}

        for key, group in itertools.groupby(self.data_set, self.sort_func):

            group = list(group)
            distr[key] = len(group)

            self.groups[key] = group

        return distr

    @property
    def time_distr(self):
        return self._get_distribution()

    def _get_sample_info(self):
        """returns a dict of
            time_seq_length : namedtuple[batch_size, poportion, group]
        where batch_size is the largest permissible power of 2 greater
        or equal to the number of sequences that have the same length.

        Proportion tells us how many % of the total data have this length.

        Group is the subset of data_set that has said length

        [description]
        """

        distr = {}
        lengths = []
        proportions = []

        total = len(self.data_set)
        Entry = namedtuple("Entry", ["batch_size", "proportion", "group"])

        for length in self.time_distr:

            num_entries = self.time_distr[length]
            size = pow_2_reduce(num_entries, self.batch_size)
            prop = num_entries / total
            group = self.groups[length]

            distr[length] = Entry(batch_size=size, proportion=prop, group=group)

            proportions.append(prop)
            lengths.append(length)

        self.proportions = proportions
        self.lengths = lengths

        return distr

    def sample(self):
        """samples one batch by choosing a length
        with distr. given by the proportions in sample_info
        and batch_size also given by sample_info.

        [description]
        """

        # Sort data_set according to the above function
        self.data_set = sorted(self.data_set, key=self.sort_func)

        self.sample_info = self._get_sample_info()

        # sample random length
        length = np.random.choice(a=self.lengths, p=self.proportions)

        info = self.sample_info[length]

        valid_data = info.group
        batch_size = info.batch_size

        sampled_data = random.sample(valid_data, batch_size)

        batch_data = [elem[0] for elem in sampled_data]
        batch_labels = [elem[1] for elem in sampled_data]

        # remove sampled data from data set
        data_set = []
        for elem in self.data_set:
            data_point, label = elem
            for selected in batch_data:
                match_found = False
                if np.all(data_point == selected):
                    match_found = True
                    break
            if not match_found:
                data_set.append(elem)

        self.data_set = data_set

        return np.stack(batch_data), np.stack(batch_labels)

    def gen_samples(self):

        while len(self.data_set) > 0:
            yield self.sample()

        self.data_set = list(zip(self.ragged_data, self.labels))
