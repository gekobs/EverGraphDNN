import awkward
import numpy
import os
import math

import tensorflow as tf
from tensorflow import keras

import logging
logger = logging.getLogger(__name__)

from evergraph.utils.misc_utils import update_dict
from evergraph.algorithms.losses import selective_mse

DUMMY_VALUE = -999.
DEFAULT_OPTIONS = {
        "targets" : [
            "target_has_HggHiggs", "target_HggHiggs_pt", "target_HggHiggs_eta", "target_HggHiggs_phi", "target_HggHiggs_mass",
            "target_has_HbbHiggs", "target_HbbHiggs_pt", "target_HbbHiggs_eta", "target_HbbHiggs_phi", "target_HbbHiggs_mass",
            "target_has_HttHiggs", "target_HttHiggs_pt", "target_HttHiggs_eta", "target_HttHiggs_phi", "target_HttHiggs_mass",
            "target_has_HwwHiggs", "target_HwwHiggs_pt", "target_HwwHiggs_eta", "target_HwwHiggs_phi", "target_HwwHiggs_mass",
            "target_has_Top_1", "target_Top_1_pt", "target_Top_1_eta", "target_Top_1_phi", "target_Top_1_mass",
            "target_has_Top_2", "target_Top_2_pt", "target_Top_2_eta", "target_Top_2_phi", "target_Top_2_mass",
        ],
        "learning_rate" : 0.0005,
        "model" : {
            "type" : "graph_cnn",
            "n_tuple" : 2
        }
}

class DNNHelper():
    """
    Class to wrap typical DNN tasks:
        - loading data
        - setting inputs, creating & compiling keras model
        - training & callbacks
        - saving models
    """
    def __init__(self, input_dir, output_dir, config = {}, **kwargs):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = update_dict(original = DEFAULT_OPTIONS, new = config)


    def run(self):
        """

        """
        self.load_data()
        self.create_model()
        self.train()
        self.evaluate()
        self.save()

    
    def load_data(self):
        """

        """
        self.f_input = self.input_dir + "/data.parquet"
        self.events = awkward.from_parquet(self.f_input)

        self.n_events = len(self.events)
        if self.config["model"]["type"] == "graph_cnn":
            self.n_objects = math.comb(len(self.events.objects[0]), self.config["model"]["n_tuple"])
            self.n_object_features = self.config["model"]["n_tuple"] * len(self.events.objects[0][0])
        else:
            self.n_objects = len(self.events.objects[0])
            self.n_object_features = len(self.events.objects[0][0])

        self.labels = [x for x in self.events.fields if "target" in x and x in self.config["targets"]]
        self.n_labels = len(self.labels)

        logger.debug("[DNNHelper : load_data] Loaded %d self.events from file '%s'." % (self.n_events, self.f_input))
        logger.debug("[DNNHelper : load_data] Found %d objects per event with %d features per object." % (self.n_objects, self.n_object_features))
        logger.debug("[DNNHelper : load_data] Found %d targets:" % (self.n_labels))
        for x in self.labels:
            logger.debug("\t %s : mean (std dev) of %.4f (%.4f)" % (x, awkward.mean(self.events[x][self.events[x] != DUMMY_VALUE]), awkward.std(self.events[x][self.events[x] != DUMMY_VALUE])))


        if self.config["model"]["type"] == "graph_cnn":
            ntuples = awkward.combinations(
                    self.events["objects"],
                    self.config["model"]["n_tuple"],
                    axis=1,
                    fields=["obj_%d" % x for x in range(self.config["model"]["n_tuple"])]
            )
            ntuples = awkward.concatenate(
                    [ntuples["obj_%d" % x] for x in range(self.config["model"]["n_tuple"])],
                    axis=2
            )
            self.X = tf.convert_to_tensor(ntuples)

        else:    
            self.X = tf.convert_to_tensor(self.events["objects"])

        self.X = tf.where(
                (tf.math.is_nan(self.X)) | (tf.math.is_inf(self.X)),
                DUMMY_VALUE * tf.ones_like(self.X),
                self.X
        )

        self.y = {}
        for x in self.labels:
            self.y["output_%s" % x] = tf.convert_to_tensor(self.events[x])

    def create_model(self):
        """

        """
        input_layer = keras.layers.Input(shape = (self.n_objects, self.n_object_features,), name = "input")
        if self.config["model"]["type"] == "1d_cnn" or self.config["model"]["type"] == "graph_cnn":
            layer = input_layer
            for i in range(5):
                layer = keras.layers.Conv1D(64, min(i+1,2), activation="elu", name = "layer_%d" % i)(layer)
                layer = keras.layers.BatchNormalization(name = "batch_norm_cnn_%d" % i)(layer)
            layer = keras.layers.Flatten()(layer)
            layer = keras.layers.Dense(32, activation="elu", name = "intermediate")(layer)
            for i in range(3):
                layer = keras.layers.BatchNormalization(name = "batch_norm_dense_%d" % i)(layer)
                layer = keras.layers.Dense(100, activation="elu", name = "dense_%d" % i)(layer)
 
        #elif self.config["model"] = "graph_cnn":
        #    layer = input_layer
        #    for i in range(2):
        #        layer = 

        outputs = {}
        losses = {}
        for x in self.labels:
            outputs["output_%s" % x] = keras.layers.Dense(
                    1,
                    activation = "sigmoid" if "has" in x else None,
                    name = "output_%s" % x
            )(layer)
            if "has" in x:
                losses["output_%s" % x] = keras.losses.BinaryCrossentropy()
            else:
                losses["output_%s" % x] = selective_mse(DUMMY_VALUE) 

        self.model = keras.models.Model(inputs = input_layer, outputs = outputs)
        self.model.summary()
        

        self.model.compile(
                optimizer = keras.optimizers.Adam(learning_rate = self.config["learning_rate"]),
                loss = losses
        )

    def train(self):
        """

        """
        self.model.fit(
                self.X,
                self.y,
                batch_size = 512,
                epochs = 100
        )

    def evaluate(self):
        """

        """
        predictions = self.model.predict(self.X, batch_size = 10000)
        for pred, values in predictions.items():
            array = awkward.from_numpy(numpy.array(values).flatten())
            self.events[pred] = array


        

    def save(self):
        """

        """

        os.system("mkdir -p %s" % self.output_dir)
        awkward.to_parquet(self.events, self.output_dir + "/data_and_preds.parquet")
