import glob
import math
import awkward
import numpy

import tensorflow as tf
from tensorflow import keras

import logging
logger = logging.getLogger(__name__)

class EvergraphGenerator(keras.utils.Sequence):
    """

    """
    def __init__(self, input_dir, batch_size, targets, n_tuple, graph, DUMMY_VALUE, mode = "train", greedy = True, short = False):
        self.input_dir = input_dir
        self.batch_size = batch_size
        self.targets = targets
        self.n_tuple = n_tuple
        self.graph = graph
        self.DUMMY_VALUE = DUMMY_VALUE
        self.mode = mode
        self.greedy = greedy
        self.short = short
        self.debug = False

        self.files = glob.glob(self.input_dir + "/*.parquet")
        sorted(self.files)
        if self.mode == "train":
            self.files = self.files[0::2]
        elif self.mode == "test":
            self.files = self.files[1::2]

        self.n_files = len(self.files)

        self.n_events = {}
        self.n_events_total = 0
        self.file_map = {}
        for idx, f in enumerate(self.files):
            events = awkward.from_parquet(f)
            n_events = len(events)

            self.n_events[f] = len(events)
            self.n_events_total += n_events
            if self.greedy:
                self.file_map[idx] = events
            else:
                self.file_map[idx] = f

        logger.debug("[EvergraphGenerator : __init__] From input dir '%s', found %d files with %d total events." % (self.input_dir, self.n_files, self.n_events_total))

        events = self.load_events(0) 

        if self.graph:
            #self.n_objects = math.comb(self.max_objects, self.n_tuple)
            self.n_objects = math.comb(len(events.objects[0]), self.n_tuple)
            self.n_object_features = self.n_tuple * len(events.objects[0][0])
        else:
            self.n_objects = len(events.objects[0]) 
            self.n_object_features = len(events.objects[0][0])

        if self.debug:
            for i in range(len(events.objects[0])):
                for j in range(len(events.objects[0][0])):
                    mean = awkward.mean(events.objects[:,i,j])
                    std = awkward.std(events.objects[:,i,j])
                    logger.debug("[EvergraphGenerator : __init__] Object %d, feature %d : mean +/- std dev = %.4f +/- %.4f." % (i, j, mean, std))

        self.labels = [x for x in events.fields if "target" in x and x in self.targets]
        self.n_labels = len(self.labels)

        logger.debug("[EvergraphGenerator : __init__] Found %d objects per event with %d features per object." % (self.n_objects, self.n_object_features))
        for x in self.labels:
            logger.debug("\t %s : mean (std dev) of %.4f (%.4f)" % (x, awkward.mean(events[x][events[x] != self.DUMMY_VALUE]), awkward.std(events[x][events[x] != self.DUMMY_VALUE])))


    def __len__(self):
        if self.short:
            if self.mode == "train":
                return 200
            else:
                return 10
        return math.ceil(self.n_events_total / self.batch_size)


    def __getitem__(self, idx):
        evt_idx = idx * self.batch_size

        if evt_idx + self.batch_size >= self.n_events_total:
            events = self.load_events(self.n_files-1) 
            events_batch = events[-(self.n_events_total - evt_idx):] # assumes batch_size < n_file

        else:
            running_count = 0
            batch_file = None
            batch_file_evt_idx = None
            batch_file_idx = None
            for f_idx, f in enumerate(self.files):
                if evt_idx >= running_count and evt_idx < running_count + self.n_events[f]:
                    batch_file = f
                    batch_file_evt_idx = evt_idx - running_count 
                    batch_file_idx = f_idx
                    break

                running_count += self.n_events[f]

            events = self.load_events(batch_file_idx) 
            
            # Are batch events all contained within this file?
            if batch_file_evt_idx + self.batch_size <= self.n_events[batch_file]:
                events_batch = events[batch_file_evt_idx:batch_file_evt_idx + self.batch_size]
            # Otherwise load partial batch from this file and partial batch from next file
            else:
                events_batch_1 = events[batch_file_evt_idx:]
                n_remaining = self.batch_size - len(events_batch_1)
                batch_file_idx += 1
                if batch_file_idx >= self.n_files:
                    events_batch = events_batch_1

                else:
                    events_2 = self.load_events(batch_file_idx)
                    events_batch_2 = events_2[:n_remaining]

                    events_batch = awkward.concatenate([events_batch_1, events_batch_2])

        if len(events_batch) > self.batch_size:
            logger.warning("[EvergraphGenerator : __getitem__] Length of events (%d) does not equal batch_size (%d) for idx %d of generator." % (len(events_batch), self.batch_size, idx))

        return self.process_batch(events_batch)
            

    def load_events(self, idx):
        if self.greedy:
            return self.file_map[idx]
        else:
            return awkward.from_parquet(self.file_map[idx])


    def process_batch(self, events):
        ntuples = awkward.combinations(
                events["objects"],
                self.n_tuple,
                axis=1,
                fields=["obj_%d" % x for x in range(self.n_tuple)]
        )
        ntuples = awkward.concatenate(
                [ntuples["obj_%d" % x] for x in range(self.n_tuple)],
                axis=2
        )

        nearest_factor_of_two = 1
        while nearest_factor_of_two < self.n_objects:
            nearest_factor_of_two *= 2
        ntuples = awkward.concatenate(
                [ntuples, numpy.zeros((len(ntuples), nearest_factor_of_two - self.n_objects, self.n_object_features))], 
                axis=1
        )

        X = tf.convert_to_tensor(ntuples)
        X = tf.where(
                (tf.math.is_nan(X)) | (tf.math.is_inf(X)),
                tf.zeros_like(X),
                X
        )

        y = {}
        for x in self.labels:
            y["output_%s" % x] = tf.convert_to_tensor(events[x])

        return (X,y)

