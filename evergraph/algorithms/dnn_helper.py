import awkward
import numpy
import os
import math

import tensorflow as tf
from tensorflow import keras

from sklearn import metrics

import logging
logger = logging.getLogger(__name__)

from evergraph.utils.misc_utils import update_dict
from evergraph.algorithms.losses import selective_mse
from evergraph.algorithms.generators import EvergraphGenerator

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

DUMMY_VALUE = -999.
DEFAULT_OPTIONS = {
        "targets" : [
            #"target_has_HggHiggs",
            "target_HggHiggs_pt", "target_HggHiggs_eta",
            "target_has_HbbHiggs", "target_HbbHiggs_pt", "target_HbbHiggs_eta",
            #"target_has_HggHiggs", "target_HggHiggs_pt", "target_HggHiggs_eta", "target_HggHiggs_phi", "target_HggHiggs_mass",
            #"target_has_HbbHiggs", "target_HbbHiggs_pt", "target_HbbHiggs_eta", "target_HbbHiggs_phi", "target_HbbHiggs_mass",
            #"target_has_HttHiggs", "target_HttHiggs_pt", "target_HttHiggs_eta", "target_HttHiggs_phi", "target_HttHiggs_mass",
            #"target_has_HwwHiggs", "target_HwwHiggs_pt", "target_HwwHiggs_eta", "target_HwwHiggs_phi", "target_HwwHiggs_mass",
            #"target_has_Top_1", "target_Top_1_pt", "target_Top_1_eta", "target_Top_1_phi", "target_Top_1_mass",
            #"target_has_Top_2", "target_Top_2_pt", "target_Top_2_eta", "target_Top_2_phi", "target_Top_2_mass",
        ],
        "learning_rate" : 0.001,
        "batch_size" : 256,
        "val_batch_size" : 1000000,
        "n_epochs" : 30,
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
        self.data_input_dir = self.input_dir + "/data/"
        self.generator = EvergraphGenerator(
                input_dir = self.data_input_dir,
                batch_size = self.config["batch_size"],
                targets = self.config["targets"],
                n_tuple = self.config["model"]["n_tuple"],
                graph = self.config["model"]["type"] == "graph_cnn",
                DUMMY_VALUE = DUMMY_VALUE,
                mode = "train",
                short = False
        )

        self.generator_val = EvergraphGenerator(
                input_dir = self.data_input_dir,
                batch_size = self.config["val_batch_size"],
                targets = self.config["targets"],
                n_tuple = self.config["model"]["n_tuple"],
                graph = self.config["model"]["type"] == "graph_cnn",
                DUMMY_VALUE = DUMMY_VALUE,
                mode = "test",
                short = False
        )  
        
    
    def create_model(self):
        """

        """
        n_objects = 1
        while n_objects < self.generator.n_objects:
            n_objects *= 2
        input_layer = keras.layers.Input(shape = (n_objects, self.generator.n_object_features,), name = "input")
        if self.config["model"]["type"] == "1d_cnn" or self.config["model"]["type"] == "graph_cnn":
            layer = input_layer
            for i in range(4):
                layer = keras.layers.Conv1D(128, 1, activation="elu", name = "layer_%d" % i)(layer)
                #layer = keras.layers.BatchNormalization(name = "batch_norm_cnn_%d" % i)(layer)
                #layer = keras.layers.Dropout(0.1)(layer)
            for i in range(6):
                layer = keras.layers.Conv1D(32*((i+1)*2), 2, strides = 2, activation="elu", name = "layer_2_%d" % i)(layer)
                #layer = keras.layers.BatchNormalization(name = "batch_norm_cnn_2_%d" % i)(layer)
                #layer = keras.layers.Dropout(0.1)(layer)
            layer = keras.layers.Flatten()(layer)
            layer = keras.layers.Dense(256, activation="elu", name = "intermediate")(layer)
            for i in range(3):
                #layer = keras.layers.BatchNormalization(name = "batch_norm_dense_%d" % i)(layer)
                #layer = keras.layers.Dropout(0.1)(layer)
                layer = keras.layers.Dense(256, activation="elu", name = "dense_%d" % i)(layer)
 
        #elif self.config["model"] = "graph_cnn":
        #    layer = input_layer
        #    for i in range(2):
        #        layer = 

        outputs = {}
        losses = {}
        for x in self.generator.labels:
            outputs["output_%s" % x] = keras.layers.Dense(
                    1,
                    activation = "sigmoid" if "has" in x else None,
                    name = "output_%s" % x
            )(layer)
            if "has" in x:
                losses["output_%s" % x] = keras.losses.BinaryCrossentropy()
            else:
                if "phi" in x:
                    periodic = True
                else:
                    periodic = False
                losses["output_%s" % x] = keras.losses.Huber(delta = 0.25)
                #losses["output_%s" % x] = selective_mse(DUMMY_VALUE, periodic)
                #losses["output_%s" % x] = keras.losses.MeanSquaredError()
                #losses["output_%s" % x] = selective_mse(DUMMY_VALUE) 

        self.model = keras.models.Model(inputs = {"input" : input_layer}, outputs = outputs)
        self.model.summary()

        self.model.compile(
                optimizer = keras.optimizers.Adam(learning_rate = self.config["learning_rate"]),
                loss = losses
        )

    def train(self):
        """

        """
        for i in range(self.config["n_epochs"]):
            self.model.fit(
                    self.generator,
                    validation_data = self.generator_val,
                    epochs = 1,
                    use_multiprocessing = False
            )

            for j in range(1):
                X, y = self.generator_val.__getitem__(j)
                predictions = self.model.predict(X, batch_size = len(X))

                for label, array in y.items():
                    pred = awkward.flatten(predictions[label])
                    target = awkward.Array(array.numpy())
                    if "_has_" in label:
                        continue
                        auc = metrics.roc_auc_score(target, pred)
                        logger.debug("[DNNHelper : train] For target '%s', AUC is %.3f." % (label, auc))
                        continue

                    if "_mass" in label or "_pt" in label:
                        pred = numpy.exp(pred)
                        target = numpy.exp(target)

                    if "_phi" in label:
                        mae = numpy.mean(
                                numpy.abs(
                                    (pred - target) % (2 * math.pi)
                                )
                        )
                    else:
                        mae = numpy.mean(
                                numpy.abs(pred - target)
                        )
                    logger.debug("[DNNHelper : train] For target '%s', MAE is %.3f." % (label, mae))



    def evaluate(self):
        """

        """
        os.system("mkdir -p %s" % (self.output_dir))
        os.system("mkdir -p %s/data" % (self.output_dir))

        self.generator.batch_size = self.config["val_batch_size"]
        for train_idx, gen in enumerate([self.generator, self.generator_val]):
            for idx, f in enumerate(gen.files):
                print(idx, f)
                events = gen.load_events(idx)
                features, labels = gen.process_batch(events)
                pred = self.model.predict(features)

                for p, v in pred.items():
                    array = awkward.from_numpy(numpy.array(v).flatten())
                    events[p] = array

                if train_idx == 0:
                    events["train_label"] = awkward.zeros_like(array)
                elif train_idx == 1:
                    events["train_label"] = awkward.ones_like(array)

                awkward.to_parquet(events, self.output_dir + "/data/file%d.parquet" % idx)

        awkward.to_parquet.dataset(self.output_dir + "/data")
        

    def evaluate_old(self):
        """

        """
        predictions = self.model.predict(self.X, batch_size = 10000)
        for pred, values in predictions.items():
            array = awkward.from_numpy(numpy.array(values).flatten())
            self.events[pred] = array


        

    def save(self):
        """

        """

        #os.system("mkdir -p %s" % self.output_dir)
        #awkward.to_parquet(self.events, self.output_dir + "/data_and_preds.parquet")
