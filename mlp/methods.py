import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from .utils import batch_generator


class MLP:
    def __init__(
        self,
        neurons_per_layer,
        activation_functions,
        optimizer,
        loss,
        metrics,
        num_classes,
    ):
        self.neurons_per_layer = neurons_per_layer
        self.activation_functions = activation_functions
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.num_classes = num_classes
        self.model = None

    def build(self, features_size):
        """
        Generate MLP model
        """
        # Input Layer
        input_layer = Input(shape=(features_size,))

        # First hidden layer
        hidden_layer = Dense(
            self.neurons_per_layer[0], activation=self.activation_functions[0]
        )(input_layer)

        # Other hidden layers
        for layer_number, neurons_in_layer in enumerate(
            self.neurons_per_layer[1:], start=1
        ):
            activation_function = self.activation_functions[layer_number]
            hidden_layer = Dense(neurons_in_layer, activation=activation_function)(
                hidden_layer
            )

        # Output layer
        output_layer = Dense(
            self.num_classes, activation=self.activation_functions[-1]
        )(hidden_layer)

        # Model
        model = Model(inputs=input_layer, outputs=[output_layer])

        # Compile model
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        self.model = model

    def train(
        self,
        X_train,
        Y_train,
        W_train,
        X_test,
        Y_test,
        W_test,
        n_epocs=5000,
        batch_size=100,
        verbose=1,
    ):
        """
        Train MLP model
        """
        if self.model is None:
            raise ValueError("Model is undefined")

        # Store model status
        epochs = []
        train_acc = []
        test_acc = []
        train_loss = []
        test_loss = []
        best_weights = []
        epoch_min_loss = 0
        min_loss = 99999

        # Create batch samples
        train_batches = batch_generator(
            [X_train, to_categorical(Y_train, num_classes=self.num_classes), W_train],
            batch_size,
        )

        # Train each epoch
        for i in range(n_epocs):

            # Generate training batches
            train_x_b, train_y_b, train_w_b = next(train_batches)

            # Train model to learn class
            _ = self.model.train_on_batch(train_x_b, train_y_b, sample_weight=train_w_b)

            # Each 10 epochs evaluate model on train batches and test batches
            test_acc_i = []
            if (i + 1) % 10 == 0:
                train_acc_i = self.model.evaluate(
                    X_train,
                    to_categorical(Y_train, num_classes=self.num_classes),
                    sample_weight=W_train,
                    verbose=verbose,
                )
                test_acc_i = self.model.evaluate(
                    X_test,
                    to_categorical(Y_test, num_classes=self.num_classes),
                    sample_weight=W_test,
                    verbose=verbose,
                )

                epochs.append(i + 1)
                train_acc.append(train_acc_i[1])
                test_acc.append(test_acc_i[1])
                train_loss.append(train_acc_i[0])
                test_loss.append(test_acc_i[0])

                if test_acc_i[0] < min_loss:
                    min_loss = test_acc_i[0]
                    epoch_min_loss = i + 1
                    # Save best weights
                    best_weights[:] = []
                    for layer in self.model.layers:
                        best_weights.append(layer.get_weights())

                if verbose > 0:
                    print(
                        "Epoch %d, class loss =  %.10f, class accuracy =  %.3f"
                        % (i, test_acc_i[0], test_acc_i[1])
                    )

        if epoch_min_loss > 0:
            # Set weights of the best classification model
            k = 0
            for layer in self.model.layers:
                layer.set_weights(best_weights[k])
                k += 1

        return {
            "model": self.model,
            "epoch_min_loss": epoch_min_loss,
            "epochs": np.array(epochs),
            "train_acc": np.array(train_acc),
            "test_acc": np.array(test_acc),
            "train_loss": np.array(train_loss),
            "test_loss": np.array(test_loss),
        }
