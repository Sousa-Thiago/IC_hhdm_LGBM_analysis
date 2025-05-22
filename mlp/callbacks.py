import json

import tensorflow as tf


class EvaluateOnDataset(tf.keras.callbacks.Callback):
    def __init__(self, filepath, train_data=None, test_data=None, validation_step=10):
        if train_data is None and test_data is None:
            raise ValueError("At least one set must be specificy, data or train")

        self.filepath = filepath
        self.train_data = train_data
        self.test_data = test_data
        self.validation_step = validation_step

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.validation_step != 0:
            return

        line = {"epoch": epoch}
        if self.train_data:
            X_train, Y_train, W_train = self.train_data
            eval_train_loss, eval_train_acc = self.model.evaluate(
                X_train, Y_train, sample_weight=W_train, verbose=False
            )
            line["eval_train_loss"] = eval_train_loss
            line["eval_train_acc"] = eval_train_acc
            logs["eval_train_loss"] = eval_train_loss
            logs["eval_train_acc"] = eval_train_acc

        if self.test_data:
            X_test, Y_test, W_test = self.test_data
            eval_test_loss, eval_test_acc = self.model.evaluate(
                X_test, Y_test, sample_weight=W_test, verbose=False
            )
            line["eval_test_loss"] = eval_test_loss
            line["eval_test_acc"] = eval_test_acc
            logs["eval_test_loss"] = eval_test_loss
            logs["eval_test_acc"] = eval_test_acc

        with open(self.filepath, "a+") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(line)
