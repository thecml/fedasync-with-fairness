"""
A federated learning server using FedAsync.

Reference:

Xie, C., Koyejo, S., Gupta, I. "Asynchronous federated optimization,"
in Proc. 12th Annual Workshop on Optimization for Machine Learning (OPT 2020).

https://opt-ml.org/papers/2020/paper_28.pdf
"""
import logging
from collections import OrderedDict
import numpy as np

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedAsync algorithm."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(
            model=model, datasource=None, algorithm=algorithm, trainer=trainer
        )

        # The hyperparameter of FedAsync with a range of (0, 1)
        self.mixing_hyperparam = 1
        # Whether adjust mixing hyperparameter after each round
        self.adaptive_mixing = False


    def configure(self):
        """Configure the mixing hyperparameter for the server, as well as
        other parameters from the configuration file.
        """
        super().configure()

        for i in range(Config().clients.total_clients):
            self.client_aggregations[i+1] = 0

        # Configuring the mixing hyperparameter for FedAsync
        self.adaptive_mixing = (
            hasattr(Config().server, "adaptive_mixing")
            and Config().server.adaptive_mixing
        )

        if not hasattr(Config().server, "mixing_hyperparameter"):
            logging.warning(
                "FedAsync: Variable mixing hyperparameter is required for the FedAsync server."
            )
        else:
            self.mixing_hyperparam = Config().server.mixing_hyperparameter

            if 0 < self.mixing_hyperparam < 1:
                logging.info(
                    "FedAsync: Mixing hyperparameter is set to %s.",
                    self.mixing_hyperparam,
                )
            else:
                logging.warning(
                    "FedAsync: Invalid mixing hyperparameter. "
                    "The hyperparameter needs to be between 0 and 1 (exclusive)."
                )

        # Calculating size function divisor dependant on size function
        self.size_divisor = 1
        if hasattr(Config().server, "size_weighting_function"):
            size_func_param = Config().server.size_weighting_function
            func_type = size_func_param.type.lower()
            trainsetDF = self.datasource.trainset.data
            if func_type == "total":
                self.size_divisor = trainsetDF.shape[0]
            elif func_type == "largest":
                self.size_divisor = max(self.datasource.client_sizes.values())
            elif func_type == "positives":
                raise NotImplementedError("positive ratios not implemented")
            else:
                logging.warning(
                    "FedAsync: Unknown size weighting function type. "
                    "Type needs to be total, largest, or positives."
                )


    def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Process the client reports by aggregating their weights."""
        # Calculate the new mixing hyperparameter with client's staleness
        updated_model = None
        print(f"aggregate_weights with {len(updates)} updates")
        for index, update in enumerate(updates):
            client_size = 0
            client_positive_ratio = 0
            if hasattr(Config().clients, "random_staleness"):
                client_staleness = update.report.staleness
            else:
                client_staleness = update.staleness
                client_size = update.report.num_samples
                client_positive_ratio = update.report.positive_rate

            mixing_hyperparam = self.mixing_hyperparam
            if self.adaptive_mixing:
                mixing_hyperparam *= self._size_function(client_size, client_positive_ratio, self.size_divisor)
                mixing_hyperparam *= self._staleness_function(client_staleness)

            print(f"Round({self.current_round}): aggregating client {update.client_id} with staleness {client_staleness} resulting staleness function returning {mixing_hyperparam}")
            if updated_model is None:
                updated_model = self.algorithm.aggregate_weights(
                    baseline_weights, weights_received[index], mixing=mixing_hyperparam
                )
            else:
                updated_model = self.algorithm.aggregate_weights(
                    updated_model, weights_received[index], mixing=mixing_hyperparam
                )

            self.client_aggregations[update.client_id] +=1

        return updated_model

    @staticmethod
    def _staleness_function(staleness) -> float:
        """Staleness function used to adjust the mixing hyperparameter"""
        if hasattr(Config().server, "staleness_weighting_function"):
            staleness_func_param = Config().server.staleness_weighting_function
            func_type = staleness_func_param.type.lower()
            if func_type == "constant":
                return Server._constant_function()
            elif func_type == "polynomial":
                a = staleness_func_param.a
                return Server._polynomial_function(staleness, a)
            elif func_type == "hinge":
                a = staleness_func_param.a
                b = staleness_func_param.b
                return Server._hinge_function(staleness, a, b)
            else:
                logging.warning(
                    "FedAsync: Unknown staleness weighting function type. "
                    "Type needs to be constant, polynomial, or hinge."
                )
        else:
            return Server.constant_function()

    @staticmethod
    def _constant_function() -> float:
        """Constant staleness function as proposed in Sec. 5.2, Evaluation Setup."""
        return 1

    @staticmethod
    def _polynomial_function(staleness, a) -> float:
        """Polynomial staleness function as proposed in Sec. 5.2, Evaluation Setup."""
        return float(staleness + 1) ** -a

    @staticmethod
    def _hinge_function(staleness, a, b) -> float:
        """Hinge staleness function as proposed in Sec. 5.2, Evaluation Setup."""
        if staleness <= b:
            return 1
        else:
            return 1 / (a * (staleness - b) + 1)

    # Functionality for adjusting the mixing parameter alpha according to the amount and relevance of samples in the client aggregated.
    @staticmethod
    def _size_function(size, ratio, divisor) -> float:
        """Size function used to adjust the mixing hyperparameter"""
        if hasattr(Config().server, "size_weighting_function"):
            size_func_param = Config().server.size_weighting_function
            func_type = size_func_param.type.lower()
            if func_type == "total":
                return Server._total_function(size, divisor)
            elif func_type == "largest":
                return Server._largest_function(size, divisor)
            elif func_type == "positives":
                return Server._positives_function(ratio, divisor)
            else:
                logging.warning(
                    "FedAsync: Unknown size weighting function type. "
                    "Type needs to be total, largest, or positives."
                )
        else:
            return Server._none_function()

    @staticmethod
    def _none_function() -> float:
        """None - size weight is just constant if nothing else is set"""
        return 1

    @staticmethod
    def _total_function(client_size, divisor) -> float:
        """Proposal 1: Weighted according to total samples in run"""
        return client_size/divisor

    @staticmethod
    def _largest_function(client_size, divisor) -> float:
        """Proposal 2: Weighted according to largest client in run"""
        return client_size/divisor

    @staticmethod
    def _positives_function(client_rate, divisor) -> float:
        """Proposal 3: Weighted according to rate of positive samples"""
        return client_rate/divisor