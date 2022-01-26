import awkward

import tensorflow as tf
from tensorflow import keras

import logging
logger = logging.getLogger(__name__)

from evergraph.utils.misc_utils import update_dict

DUMMY_VALUE = -999.
DEFAULT_OPTIONS = {
        "targets" : ["target_has_HggHiggs", "target_has_HttHiggs"],
        "learning_rate" : 0.001,
        "model" : "1d_cnn"
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
        events = awkward.from_parquet(self.f_input)

        self.n_events = len(events)
        self.n_objects = len(events.objects[0])
        self.n_object_features = len(events.objects[0][0])
        self.labels = [x for x in events.fields if "target" in x and x in self.config["targets"]]
        self.n_labels = len(self.labels)

        logger.debug("[DNNHelper : load_data] Loaded %d events from file '%s'." % (self.n_events, self.f_input))
        logger.debug("[DNNHelper : load_data] Found %d objects per event with %d features per object." % (self.n_objects, self.n_object_features))
        logger.debug("[DNNHelper : load_data] Found %d targets:" % (self.n_labels))
        for x in self.labels:
            logger.debug("\t %s : mean (std dev) of %.4f (%.4f)" % (x, awkward.mean(events[x][events[x] != DUMMY_VALUE]), awkward.std(events[x][events[x] != DUMMY_VALUE])))


       
        self.X = tf.convert_to_tensor(events["objects"])

        self.X = tf.where(
                (tf.math.is_nan(self.X)) | (tf.math.is_inf(self.X)),
                DUMMY_VALUE * tf.ones_like(self.X),
                self.X
        )

        self.y = {}
        for x in self.labels:
            self.y["output_%s" % x] = tf.convert_to_tensor(events[x])

    def create_model(self):
        """

        """
        input_layer = keras.layers.Input(shape = (self.n_objects, self.n_object_features,), name = "input")
        if self.config["model"] == "1d_cnn":
            layer = input_layer
            for i in range(2):
                layer = keras.layers.Conv1D(32, 2, activation="relu", name = "layer_%d" % i)(layer)
            layer = keras.layers.Flatten()(layer)
            for i in range(2):
                layer = keras.layers.Dense(50, activation="relu", name = "dense_%d" % i)(layer)
            

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
                losses["output_%s" % x] = keras.losses.MeanSquaredError()

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
                batch_size = 1024,
                epochs = 10
        )

    def evaluate(self):
        """

        """


    def save(self):
        """

        """


