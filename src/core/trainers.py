# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import uuid
import pathlib
from typing import List, Optional
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from .util import get_opts, move_to
from .callbacks import Callback, ConsoleLogger, Checkpoint, CheckpointSaver


def _add_dicts(a, b):
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) + v
    return result


def _div_dict(d, n):
    result = dict(d)
    for k in result:
        result[k] /= n
    return result


class Trainer:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """
    def __init__(
            self,
            game: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_data: DataLoader,
            validation_data: Optional[DataLoader] = None,
            device: torch.device = None,
            callbacks: Optional[List[Callback]] = None
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer = optimizer
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device
        self.game.to(self.device)
        # NB: some optimizers pre-allocate buffers before actually doing any steps
        # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
        # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
        self.optimizer.state = move_to(self.optimizer.state, self.device)
        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks

        if common_opts.load_from_checkpoint is not None:
            print(f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}")
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        if common_opts.preemptable:
            assert common_opts.checkpoint_dir, 'checkpointing directory has to be specified'
            d = self._get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
            self.checkpoint_path = d
            self.load_from_latest(d)
            checkpointer = CheckpointSaver(self.checkpoint_path)
            self.callbacks.append(checkpointer)
        else:
            self.checkpoint_path = None if common_opts.checkpoint_dir is None \
                else pathlib.Path(common_opts.checkpoint_dir)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

    def _get_preemptive_checkpoint_dir(self, checkpoint_root):
        if 'SLURM_JOB_ID' not in os.environ:
            print('Preemption flag set, but I am not running under SLURM?')

        job_id = os.environ.get('SLURM_JOB_ID', uuid.uuid4())
        task_id = os.environ.get('SLURM_PROCID', 0)

        d = pathlib.Path(checkpoint_root) / f'{job_id}_{task_id}'
        d.mkdir(exist_ok=True)

        return d

    def eval(self):
        mean_loss = 0.0
        mean_rest = {}

        n_batches = 0
        self.game.eval()
        with torch.no_grad():
            for batch in self.validation_data:
                batch = move_to(batch, self.device)
                optimized_loss, rest = self.game(*batch)
                mean_loss += optimized_loss
                mean_rest = _add_dicts(mean_rest, rest)
                n_batches += 1
        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)

        return mean_loss.item(), mean_rest

    def train_epoch(self):
        mean_loss = 0
        mean_rest = {}
        n_batches = 0
        self.game.train()
        for batch in self.train_data:
            self.optimizer.zero_grad()
            batch = move_to(batch, self.device)
            optimized_loss, rest = self.game(*batch)
            mean_rest = _add_dicts(mean_rest, rest)
            optimized_loss.backward()
            self.optimizer.step()

            n_batches += 1
            mean_loss += optimized_loss

        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)
        return mean_loss.item(), mean_rest

    def train(self, n_epochs):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin()

            train_loss, train_rest = self.train_epoch()

            for callback in self.callbacks:
                callback.on_epoch_end(train_loss, train_rest)

            if self.validation_data is not None and self.validation_freq > 0 and epoch % self.validation_freq == 0:
                for callback in self.callbacks:
                    callback.on_test_begin()
                validation_loss, rest = self.eval()
                for callback in self.callbacks:
                    callback.on_test_end(validation_loss, rest)

            if self.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.starting_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f'# loading trainer state from {path}')
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob('*.tar'):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)


class CompoTrainer:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """
    def __init__(
            self,
            n_attributes:int,
            n_values:int,
            game: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_data: DataLoader,
            validation_data: Optional[DataLoader] = None,
            device: torch.device = None,
            callbacks: Optional[List[Callback]] = None,
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer = optimizer
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device
        self.game.to(self.device)
        # NB: some optimizers pre-allocate buffers before actually doing any steps
        # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
        # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
        self.optimizer.state = move_to(self.optimizer.state, self.device)
        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks
        self.n_attributes=n_attributes
        self.n_values=n_values

        if common_opts.load_from_checkpoint is not None:
            print(f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}")
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        if common_opts.preemptable:
            assert common_opts.checkpoint_dir, 'checkpointing directory has to be specified'
            d = self._get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
            self.checkpoint_path = d
            self.load_from_latest(d)
            checkpointer = CheckpointSaver(self.checkpoint_path)
            self.callbacks.append(checkpointer)
        else:
            self.checkpoint_path = None if common_opts.checkpoint_dir is None \
                else pathlib.Path(common_opts.checkpoint_dir)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

    def _get_preemptive_checkpoint_dir(self, checkpoint_root):
        if 'SLURM_JOB_ID' not in os.environ:
            print('Preemption flag set, but I am not running under SLURM?')

        job_id = os.environ.get('SLURM_JOB_ID', uuid.uuid4())
        task_id = os.environ.get('SLURM_PROCID', 0)

        d = pathlib.Path(checkpoint_root) / f'{job_id}_{task_id}'
        d.mkdir(exist_ok=True)

        return d

    def eval(self):
        mean_loss = 0.0
        mean_rest = {}

        n_batches = 0
        self.game.eval()
        with torch.no_grad():
            for batch in self.validation_data:
                batch = move_to(batch, self.device)
                optimized_loss, rest = self.game(*batch)
                mean_loss += optimized_loss
                mean_rest = _add_dicts(mean_rest, rest)
                n_batches += 1
        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)

        return mean_loss.item(), mean_rest

    def train_epoch(self):
        mean_loss = 0
        mean_rest = {}
        n_batches = 0
        self.game.train()
        for batch in self.train_data:
            self.optimizer.zero_grad()
            batch = move_to(batch, self.device)
            optimized_loss, rest = self.game(*batch)
            mean_rest = _add_dicts(mean_rest, rest)
            optimized_loss.backward()
            self.optimizer.step()

            n_batches += 1
            mean_loss += optimized_loss

        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)
        return mean_loss.item(), mean_rest

    def train(self, n_epochs):

        #scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5, threshold=10**(-2))
        scheduler = StepLR(self.optimizer, step_size=25, gamma=0.1)

        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):

            print(epoch,flush=True)

            for callback in self.callbacks:
                callback.on_epoch_begin()

            train_loss, train_rest = self.train_epoch()
            scheduler.step()

            print("Train loss: "+str(train_loss),flush=True)

            for callback in self.callbacks:
                callback.on_epoch_end(train_loss, train_rest)

            if self.validation_data is not None and self.validation_freq > 0 and epoch % self.validation_freq == 0:
                for callback in self.callbacks:
                    callback.on_test_begin()
                validation_loss, rest = self.eval()
                print("Eval loss: "+str(validation_loss),flush=True)
                print(rest,flush=True)
                for callback in self.callbacks:
                    callback.on_test_end(validation_loss, rest)

            if self.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.starting_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f'# loading trainer state from {path}')
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob('*.tar'):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)

class TrainerDialog:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """
    def __init__(
            self,
            game: torch.nn.Module,
            optimizer_1: torch.optim.Optimizer,
            optimizer_2: torch.optim.Optimizer,
            train_data: DataLoader,
            validation_data: Optional[DataLoader] = None,
            device: torch.device = None,
            callbacks: Optional[List[Callback]] = None
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer_1 = optimizer_1
        self.optimizer_2 = optimizer_2
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device
        self.game.to(self.device)
        # NB: some optimizers pre-allocate buffers before actually doing any steps
        # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
        # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
        self.optimizer_1.state = move_to(self.optimizer_1.state, self.device)
        self.optimizer_2.state = move_to(self.optimizer_2.state, self.device)
        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks

        if common_opts.load_from_checkpoint is not None:
            print(f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}")
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        if common_opts.preemptable:
            assert common_opts.checkpoint_dir, 'checkpointing directory has to be specified'
            d = self._get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
            self.checkpoint_path = d
            self.load_from_latest(d)
            checkpointer = CheckpointSaver(self.checkpoint_path)
            self.callbacks.append(checkpointer)
        else:
            self.checkpoint_path = None if common_opts.checkpoint_dir is None \
                else pathlib.Path(common_opts.checkpoint_dir)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

    def _get_preemptive_checkpoint_dir(self, checkpoint_root):
        if 'SLURM_JOB_ID' not in os.environ:
            print('Preemption flag set, but I am not running under SLURM?')

        job_id = os.environ.get('SLURM_JOB_ID', uuid.uuid4())
        task_id = os.environ.get('SLURM_PROCID', 0)

        d = pathlib.Path(checkpoint_root) / f'{job_id}_{task_id}'
        d.mkdir(exist_ok=True)

        return d

    def eval(self):
        mean_loss = 0.0
        mean_rest = {}

        n_batches = 0
        self.game.eval()
        with torch.no_grad():
            for batch in self.validation_data:
                batch = move_to(batch, self.device)
                optimized_loss_1,optimized_loss_2, rest = self.game(*batch)
                mean_loss += 0.5*(optimized_loss_1 + optimized_loss_2)
                mean_rest = _add_dicts(mean_rest, rest)
                n_batches += 1
        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)

        return mean_loss.item(), mean_rest

    def train_epoch(self):
        mean_loss = 0
        mean_rest = {}
        n_batches = 0
        self.game.train()
        for batch in self.train_data:
            optimized_loss_1,optimized_loss_2, rest = self.game(*batch)
            self.optimizer_1.zero_grad()
            batch = move_to(batch, self.device)
            mean_rest = _add_dicts(mean_rest, rest)
            optimized_loss_1.backward()
            self.optimizer_1.step()

            self.optimizer_2.zero_grad()
            optimized_loss_2.backward()
            self.optimizer_2.step()

            n_batches += 1
            mean_loss += 0.5*(optimized_loss_1+optimized_loss_2)


        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)
        return mean_loss.item(), mean_rest

    def train(self, n_epochs):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin()

            train_loss, train_rest = self.train_epoch()

            for callback in self.callbacks:
                callback.on_epoch_end(train_loss, train_rest)

            if self.validation_data is not None and self.validation_freq > 0 and epoch % self.validation_freq == 0:
                for callback in self.callbacks:
                    callback.on_test_begin()
                validation_loss, rest = self.eval()
                for callback in self.callbacks:
                    callback.on_test_end(validation_loss, rest)

            if self.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.starting_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f'# loading trainer state from {path}')
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob('*.tar'):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)

class TrainerDialogModel1:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """
    def __init__(
            self,
            game: torch.nn.Module,
            optimizer_sender_1: torch.optim.Optimizer,
            optimizer_sender_2: torch.optim.Optimizer,
            optimizer_receiver_1: torch.optim.Optimizer,
            optimizer_receiver_2: torch.optim.Optimizer,
            train_data: DataLoader,
            validation_data: Optional[DataLoader] = None,
            device: torch.device = None,
            callbacks: Optional[List[Callback]] = None
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer_sender_1 = optimizer_sender_1
        self.optimizer_sender_2 = optimizer_sender_2
        self.optimizer_receiver_1 = optimizer_receiver_1
        self.optimizer_receiver_2 = optimizer_receiver_2
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device
        self.game.to(self.device)
        # NB: some optimizers pre-allocate buffers before actually doing any steps
        # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
        # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
        self.optimizer_sender_1.state = move_to(self.optimizer_sender_1.state, self.device)
        self.optimizer_sender_2.state = move_to(self.optimizer_sender_2.state, self.device)
        self.optimizer_receiver_1.state = move_to(self.optimizer_receiver_1.state, self.device)
        self.optimizer_receiver_2.state = move_to(self.optimizer_receiver_2.state, self.device)
        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks

        if common_opts.load_from_checkpoint is not None:
            print(f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}")
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        if common_opts.preemptable:
            assert common_opts.checkpoint_dir, 'checkpointing directory has to be specified'
            d = self._get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
            self.checkpoint_path = d
            self.load_from_latest(d)
            checkpointer = CheckpointSaver(self.checkpoint_path)
            self.callbacks.append(checkpointer)
        else:
            self.checkpoint_path = None if common_opts.checkpoint_dir is None \
                else pathlib.Path(common_opts.checkpoint_dir)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

    def _get_preemptive_checkpoint_dir(self, checkpoint_root):
        if 'SLURM_JOB_ID' not in os.environ:
            print('Preemption flag set, but I am not running under SLURM?')

        job_id = os.environ.get('SLURM_JOB_ID', uuid.uuid4())
        task_id = os.environ.get('SLURM_PROCID', 0)

        d = pathlib.Path(checkpoint_root) / f'{job_id}_{task_id}'
        d.mkdir(exist_ok=True)

        return d

    def eval(self):
        mean_loss = 0.0
        mean_rest = {}

        n_batches = 0
        self.game.eval()
        with torch.no_grad():
            for batch in self.validation_data:
                batch = move_to(batch, self.device)
                optimized_loss_11, optimized_loss_12, optimized_loss_21, optimized_loss_22, rest = self.game(*batch)
                mean_loss += 0.25*(optimized_loss_11 + optimized_loss_12 + optimized_loss_21 + optimized_loss_22)
                mean_rest = _add_dicts(mean_rest, rest)
                n_batches += 1
        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)

        return mean_loss.item(), mean_rest

    def train_epoch(self):
        mean_loss = 0
        mean_rest = {}
        n_batches = 0
        self.game.train()
        for batch in self.train_data:
            optimized_loss_11, optimized_loss_12, optimized_loss_21, optimized_loss_22, rest = self.game(*batch)
            batch = move_to(batch, self.device)
            mean_rest = _add_dicts(mean_rest, rest)

            optimized_loss_sender_1=optimized_loss_11+optimized_loss_12
            optimized_loss_sender_2=optimized_loss_21+optimized_loss_22

            if np.random.rand()>0.5:
              self.optimizer_sender_1.zero_grad()
              self.optimizer_receiver_1.zero_grad()
              self.optimizer_receiver_2.zero_grad()
              optimized_loss_sender_1.backward()
              self.optimizer_sender_1.step()
              self.optimizer_receiver_1.step()
              self.optimizer_receiver_2.step()
            else:
              self.optimizer_sender_2.zero_grad()
              self.optimizer_receiver_1.zero_grad()
              self.optimizer_receiver_2.zero_grad()
              optimized_loss_sender_2.backward()
              self.optimizer_sender_2.step()
              self.optimizer_receiver_1.step()
              self.optimizer_receiver_2.step()


            n_batches += 1
            mean_loss += 0.25*(optimized_loss_11+optimized_loss_12+optimized_loss_21+optimized_loss_22)


        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)
        return mean_loss.item(), mean_rest

    def train(self, n_epochs):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin()

            train_loss, train_rest = self.train_epoch()

            for callback in self.callbacks:
                callback.on_epoch_end(train_loss, train_rest)

            if self.validation_data is not None and self.validation_freq > 0 and epoch % self.validation_freq == 0:
                for callback in self.callbacks:
                    callback.on_test_begin()
                validation_loss, rest = self.eval()
                for callback in self.callbacks:
                    callback.on_test_end(validation_loss, rest)

            if self.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.starting_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f'# loading trainer state from {path}')
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob('*.tar'):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)


class TrainerDialogModel2:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """
    def __init__(
            self,
            game: torch.nn.Module,
            optimizer_1: torch.optim.Optimizer,
            optimizer_2: torch.optim.Optimizer,
            train_data: DataLoader,
            validation_data: Optional[DataLoader] = None,
            device: torch.device = None,
            callbacks: Optional[List[Callback]] = None
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer_1 = optimizer_1
        self.optimizer_2 = optimizer_2
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device
        self.game.to(self.device)
        # NB: some optimizers pre-allocate buffers before actually doing any steps
        # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
        # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
        self.optimizer_1.state = move_to(self.optimizer_1.state, self.device)
        self.optimizer_2.state = move_to(self.optimizer_2.state, self.device)
        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks

        if common_opts.load_from_checkpoint is not None:
            print(f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}")
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        if common_opts.preemptable:
            assert common_opts.checkpoint_dir, 'checkpointing directory has to be specified'
            d = self._get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
            self.checkpoint_path = d
            self.load_from_latest(d)
            checkpointer = CheckpointSaver(self.checkpoint_path)
            self.callbacks.append(checkpointer)
        else:
            self.checkpoint_path = None if common_opts.checkpoint_dir is None \
                else pathlib.Path(common_opts.checkpoint_dir)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

    def _get_preemptive_checkpoint_dir(self, checkpoint_root):
        if 'SLURM_JOB_ID' not in os.environ:
            print('Preemption flag set, but I am not running under SLURM?')

        job_id = os.environ.get('SLURM_JOB_ID', uuid.uuid4())
        task_id = os.environ.get('SLURM_PROCID', 0)

        d = pathlib.Path(checkpoint_root) / f'{job_id}_{task_id}'
        d.mkdir(exist_ok=True)

        return d

    def eval(self):
        mean_loss = 0.0
        mean_rest = {}

        n_batches = 0
        self.game.eval()
        with torch.no_grad():
            for batch in self.validation_data:
                batch = move_to(batch, self.device)
                optimized_loss_1, optimized_loss_2, rest = self.game(*batch)
                mean_loss += 0.5*(optimized_loss_1 + optimized_loss_2)
                mean_rest = _add_dicts(mean_rest, rest)
                n_batches += 1
        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)

        return mean_loss.item(), mean_rest

    def train_epoch(self):
        mean_loss = 0
        mean_rest = {}
        n_batches = 0
        self.game.train()
        for batch in self.train_data:
            optimized_loss_1, optimized_loss_2, rest = self.game(*batch)
            batch = move_to(batch, self.device)
            mean_rest = _add_dicts(mean_rest, rest)

            if np.random.rand()>0.5:
              self.optimizer_1.zero_grad()
              optimized_loss_1.backward()
              self.optimizer_1.step()
            else:
              self.optimizer_2.zero_grad()
              optimized_loss_2.backward()
              self.optimizer_2.step()

            n_batches += 1
            mean_loss += 0.5*(optimized_loss_1+optimized_loss_2)


        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)
        return mean_loss.item(), mean_rest

    def train(self, n_epochs):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin()

            train_loss, train_rest = self.train_epoch()

            for callback in self.callbacks:
                callback.on_epoch_end(train_loss, train_rest)

            if self.validation_data is not None and self.validation_freq > 0 and epoch % self.validation_freq == 0:
                for callback in self.callbacks:
                    callback.on_test_begin()
                validation_loss, rest = self.eval()
                for callback in self.callbacks:
                    callback.on_test_end(validation_loss, rest)

            if self.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.starting_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f'# loading trainer state from {path}')
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob('*.tar'):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)

class TrainerDialogModel3:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """
    def __init__(
            self,
            game: torch.nn.Module,
            optimizer_1_comm: torch.optim.Optimizer,
            optimizer_1_imitation: torch.optim.Optimizer,
            optimizer_2_comm: torch.optim.Optimizer,
            optimizer_2_imitation: torch.optim.Optimizer,
            train_data: DataLoader,
            validation_data: Optional[DataLoader] = None,
            device: torch.device = None,
            callbacks: Optional[List[Callback]] = None
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer_1_comm = optimizer_1_comm
        self.optimizer_1_imitation = optimizer_1_imitation
        self.optimizer_2_comm = optimizer_2_comm
        self.optimizer_2_imitation = optimizer_2_imitation
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device
        self.game.to(self.device)
        # NB: some optimizers pre-allocate buffers before actually doing any steps
        # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
        # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
        self.optimizer_1_comm.state = move_to(self.optimizer_1_comm.state, self.device)
        self.optimizer_1_imitation.state = move_to(self.optimizer_1_imitation.state, self.device)
        self.optimizer_2_comm.state = move_to(self.optimizer_2_comm.state, self.device)
        self.optimizer_2_imitation.state = move_to(self.optimizer_2_imitation.state, self.device)
        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks

        if common_opts.load_from_checkpoint is not None:
            print(f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}")
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        if common_opts.preemptable:
            assert common_opts.checkpoint_dir, 'checkpointing directory has to be specified'
            d = self._get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
            self.checkpoint_path = d
            self.load_from_latest(d)
            checkpointer = CheckpointSaver(self.checkpoint_path)
            self.callbacks.append(checkpointer)
        else:
            self.checkpoint_path = None if common_opts.checkpoint_dir is None \
                else pathlib.Path(common_opts.checkpoint_dir)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

    def _get_preemptive_checkpoint_dir(self, checkpoint_root):
        if 'SLURM_JOB_ID' not in os.environ:
            print('Preemption flag set, but I am not running under SLURM?')

        job_id = os.environ.get('SLURM_JOB_ID', uuid.uuid4())
        task_id = os.environ.get('SLURM_PROCID', 0)

        d = pathlib.Path(checkpoint_root) / f'{job_id}_{task_id}'
        d.mkdir(exist_ok=True)

        return d

    def eval(self):
        mean_loss = 0.0
        mean_rest = {}

        n_batches = 0
        self.game.eval()
        with torch.no_grad():
            for batch in self.validation_data:
                batch = move_to(batch, self.device)
                optimized_loss_1,loss_1_imitation, optimized_loss_2, loss_2_imitation, rest = self.game(*batch)
                mean_loss += 0.25*(optimized_loss_1+loss_1_imitation+optimized_loss_2+loss_2_imitation)
                mean_rest = _add_dicts(mean_rest, rest)
                n_batches += 1
        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)

        return mean_loss.item(), mean_rest

    def train_epoch(self):
        mean_loss = 0
        mean_rest = {}
        n_batches = 0
        self.game.train()
        for batch in self.train_data:
            optimized_loss_1,loss_1_imitation, optimized_loss_2, loss_2_imitation, rest = self.game(*batch)
            batch = move_to(batch, self.device)
            mean_rest = _add_dicts(mean_rest, rest)

            #if np.random.rand()>0.5:
            #  self.optimizer_1_comm.zero_grad()
            #  optimized_loss_1.backward()
            #  self.optimizer_1_comm.step()
            #  self.optimizer_1_imitation.zero_grad()
            #  loss_1_imitation.backward()
            #  self.optimizer_1_imitation.step()
            #else:
            #  self.optimizer_2_comm.zero_grad()
            #  optimized_loss_2.backward()
            #  self.optimizer_2_comm.step()
            #  self.optimizer_2_imitation.zero_grad()
            #  loss_2_imitation.backward()
            #  self.optimizer_2_imitation.step()

            if np.random.rand()>0.5:
              loss_1=0.5*(optimized_loss_1+loss_1_imitation)
              self.optimizer_1_comm.zero_grad()
              self.optimizer_1_imitation.zero_grad()
              optimized_loss_1.backward()
              self.optimizer_1_comm.step()
              self.optimizer_1_imitation.step()
            else:
              loss_2=0.5*(optimized_loss_2+loss_2_imitation)
              self.optimizer_2_comm.zero_grad()
              self.optimizer_2_imitation.zero_grad()
              loss_2.backward()
              self.optimizer_2_comm.step()
              self.optimizer_2_imitation.step()

            n_batches += 1
            mean_loss += 0.25*(optimized_loss_1+loss_1_imitation+optimized_loss_2+loss_2_imitation)

        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)
        return mean_loss.item(), mean_rest

    def train(self, n_epochs):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin()

            train_loss, train_rest = self.train_epoch()

            for callback in self.callbacks:
                callback.on_epoch_end(train_loss, train_rest)

            if self.validation_data is not None and self.validation_freq > 0 and epoch % self.validation_freq == 0:
                for callback in self.callbacks:
                    callback.on_test_begin()
                validation_loss, rest = self.eval()
                for callback in self.callbacks:
                    callback.on_test_end(validation_loss, rest)

            if self.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.starting_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f'# loading trainer state from {path}')
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob('*.tar'):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)




class TrainerDialogModel4:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """
    def __init__(
            self,
            game: torch.nn.Module,
            optimizer_sender_1: torch.optim.Optimizer,
            optimizer_sender_2: torch.optim.Optimizer,
            optimizer_receiver_1: torch.optim.Optimizer,
            optimizer_receiver_2: torch.optim.Optimizer,
            train_data: DataLoader,
            validation_data: Optional[DataLoader] = None,
            device: torch.device = None,
            callbacks: Optional[List[Callback]] = None
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer_sender_1 = optimizer_sender_1
        self.optimizer_sender_2 = optimizer_sender_2
        self.optimizer_receiver_1 = optimizer_receiver_1
        self.optimizer_receiver_2 = optimizer_receiver_2
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device
        self.game.to(self.device)
        # NB: some optimizers pre-allocate buffers before actually doing any steps
        # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
        # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
        self.optimizer_sender_1.state = move_to(self.optimizer_sender_1.state, self.device)
        self.optimizer_sender_2.state = move_to(self.optimizer_sender_2.state, self.device)
        self.optimizer_receiver_1.state = move_to(self.optimizer_receiver_1.state, self.device)
        self.optimizer_receiver_2.state = move_to(self.optimizer_receiver_2.state, self.device)
        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks

        if common_opts.load_from_checkpoint is not None:
            print(f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}")
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        if common_opts.preemptable:
            assert common_opts.checkpoint_dir, 'checkpointing directory has to be specified'
            d = self._get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
            self.checkpoint_path = d
            self.load_from_latest(d)
            checkpointer = CheckpointSaver(self.checkpoint_path)
            self.callbacks.append(checkpointer)
        else:
            self.checkpoint_path = None if common_opts.checkpoint_dir is None \
                else pathlib.Path(common_opts.checkpoint_dir)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

    def _get_preemptive_checkpoint_dir(self, checkpoint_root):
        if 'SLURM_JOB_ID' not in os.environ:
            print('Preemption flag set, but I am not running under SLURM?')

        job_id = os.environ.get('SLURM_JOB_ID', uuid.uuid4())
        task_id = os.environ.get('SLURM_PROCID', 0)

        d = pathlib.Path(checkpoint_root) / f'{job_id}_{task_id}'
        d.mkdir(exist_ok=True)

        return d

    def eval(self):
        mean_loss = 0.0
        mean_rest = {}

        n_batches = 0
        self.game.eval()
        with torch.no_grad():
            for batch in self.validation_data:
                batch = move_to(batch, self.device)
                optimized_loss_11,loss_11_imitation, optimized_loss_12,loss_12_imitation, optimized_loss_21,loss_21_imitation, optimized_loss_22,loss_22_imitation, rest = self.game(*batch)
                mean_loss += 0.25*(optimized_loss_11 + optimized_loss_12 + optimized_loss_21 + optimized_loss_22)
                mean_rest = _add_dicts(mean_rest, rest)
                n_batches += 1
        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)

        return mean_loss.item(), mean_rest

    def train_epoch(self):
        mean_loss = 0
        mean_rest = {}
        n_batches = 0
        self.game.train()
        for batch in self.train_data:
            optimized_loss_11,loss_11_imitation, optimized_loss_12,loss_12_imitation, optimized_loss_21,loss_21_imitation, optimized_loss_22,loss_22_imitation, rest= self.game(*batch)
            batch = move_to(batch, self.device)
            mean_rest = _add_dicts(mean_rest, rest)

            alpha=2*rest["acc_21"]
            beta=2*rest["acc_12"]

            optimized_loss_sender_1=optimized_loss_11+optimized_loss_12+alpha*loss_21_imitation
            optimized_loss_sender_2=optimized_loss_21+optimized_loss_22+beta*loss_12_imitation

            if np.random.rand()>0.5:
              self.optimizer_sender_1.zero_grad()
              self.optimizer_receiver_1.zero_grad()
              self.optimizer_receiver_2.zero_grad()
              #self.optimizer_sender_2.zero_grad()
              optimized_loss_sender_1.backward()
              self.optimizer_sender_1.step()
              self.optimizer_receiver_1.step()
              self.optimizer_receiver_2.step()
              #self.optimizer_sender_2.step()
            else:
              self.optimizer_sender_2.zero_grad()
              self.optimizer_receiver_1.zero_grad()
              self.optimizer_receiver_2.zero_grad()
              #self.optimizer_sender_1.zero_grad()
              optimized_loss_sender_2.backward()
              self.optimizer_sender_2.step()
              self.optimizer_receiver_1.step()
              self.optimizer_receiver_2.step()
              #self.optimizer_sender_1.step()


            n_batches += 1
            mean_loss += 0.25*(optimized_loss_11+optimized_loss_12+optimized_loss_21+optimized_loss_22)


        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)
        return mean_loss.item(), mean_rest

    def train(self, n_epochs):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin()

            train_loss, train_rest = self.train_epoch()

            for callback in self.callbacks:
                callback.on_epoch_end(train_loss, train_rest)

            if self.validation_data is not None and self.validation_freq > 0 and epoch % self.validation_freq == 0:
                for callback in self.callbacks:
                    callback.on_test_begin()
                validation_loss, rest = self.eval()
                for callback in self.callbacks:
                    callback.on_test_end(validation_loss, rest)

            if self.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.starting_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f'# loading trainer state from {path}')
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob('*.tar'):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)

class TrainerDialogModel5:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """
    def __init__(
            self,
            game: torch.nn.Module,
            optimizer_sender_1: torch.optim.Optimizer,
            optimizer_sender_2: torch.optim.Optimizer,
            optimizer_receiver_1: torch.optim.Optimizer,
            optimizer_receiver_2: torch.optim.Optimizer,
            optimizer_embedding_1: torch.optim.Optimizer,
            optimizer_embedding_2: torch.optim.Optimizer,
            train_data: DataLoader,
            validation_data: Optional[DataLoader] = None,
            device: torch.device = None,
            callbacks: Optional[List[Callback]] = None
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer_sender_1 = optimizer_sender_1
        self.optimizer_sender_2 = optimizer_sender_2
        self.optimizer_receiver_1 = optimizer_receiver_1
        self.optimizer_receiver_2 = optimizer_receiver_2
        self.optimizer_embedding_1 = optimizer_embedding_1
        self.optimizer_embedding_2 = optimizer_embedding_2
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device
        self.game.to(self.device)
        # NB: some optimizers pre-allocate buffers before actually doing any steps
        # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
        # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
        self.optimizer_sender_1.state = move_to(self.optimizer_sender_1.state, self.device)
        self.optimizer_sender_2.state = move_to(self.optimizer_sender_2.state, self.device)
        self.optimizer_receiver_1.state = move_to(self.optimizer_receiver_1.state, self.device)
        self.optimizer_receiver_2.state = move_to(self.optimizer_receiver_2.state, self.device)
        self.optimizer_embedding_1.state = move_to(self.optimizer_embedding_1.state, self.device)
        self.optimizer_embedding_2.state = move_to(self.optimizer_embedding_2.state, self.device)
        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks

        if common_opts.load_from_checkpoint is not None:
            print(f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}")
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        if common_opts.preemptable:
            assert common_opts.checkpoint_dir, 'checkpointing directory has to be specified'
            d = self._get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
            self.checkpoint_path = d
            self.load_from_latest(d)
            checkpointer = CheckpointSaver(self.checkpoint_path)
            self.callbacks.append(checkpointer)
        else:
            self.checkpoint_path = None if common_opts.checkpoint_dir is None \
                else pathlib.Path(common_opts.checkpoint_dir)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

    def _get_preemptive_checkpoint_dir(self, checkpoint_root):
        if 'SLURM_JOB_ID' not in os.environ:
            print('Preemption flag set, but I am not running under SLURM?')

        job_id = os.environ.get('SLURM_JOB_ID', uuid.uuid4())
        task_id = os.environ.get('SLURM_PROCID', 0)

        d = pathlib.Path(checkpoint_root) / f'{job_id}_{task_id}'
        d.mkdir(exist_ok=True)

        return d

    def eval(self):
        mean_loss = 0.0
        mean_rest = {}

        n_batches = 0
        self.game.eval()
        with torch.no_grad():
            for batch in self.validation_data:
                batch = move_to(batch, self.device)
                optimized_loss_11,loss_11_imitation, optimized_loss_12,loss_12_imitation, optimized_loss_21,loss_21_imitation, optimized_loss_22,loss_22_imitation, rest = self.game(*batch)
                mean_loss += 0.25*(optimized_loss_11 + optimized_loss_12 + optimized_loss_21 + optimized_loss_22)
                mean_rest = _add_dicts(mean_rest, rest)
                n_batches += 1
        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)

        return mean_loss.item(), mean_rest

    def train_epoch(self):
        mean_loss = 0
        mean_rest = {}
        n_batches = 0
        self.game.train()
        for batch in self.train_data:
            optimized_loss_11,loss_11_imitation, optimized_loss_12,loss_12_imitation, optimized_loss_21,loss_21_imitation, optimized_loss_22,loss_22_imitation, rest= self.game(*batch)
            batch = move_to(batch, self.device)
            mean_rest = _add_dicts(mean_rest, rest)

            optimized_loss_sender_1=optimized_loss_11+optimized_loss_12#+loss_12_imitation
            optimized_loss_sender_2=optimized_loss_22+optimized_loss_21#+loss_21_imitation

            if np.random.rand()>0.5:
              self.optimizer_sender_1.zero_grad()
              self.optimizer_receiver_1.zero_grad()
              self.optimizer_embedding_1.zero_grad()
              self.optimizer_receiver_2.zero_grad()
              optimized_loss_sender_1.backward()
              self.optimizer_sender_1.step()
              self.optimizer_receiver_1.step()
              self.optimizer_receiver_2.step()
              self.optimizer_embedding_1.step()
            else:
              self.optimizer_sender_2.zero_grad()
              self.optimizer_receiver_1.zero_grad()
              self.optimizer_receiver_2.zero_grad()
              self.optimizer_embedding_2.zero_grad()
              optimized_loss_sender_2.backward()
              self.optimizer_sender_2.step()
              self.optimizer_receiver_1.step()
              self.optimizer_receiver_2.step()
              self.optimizer_embedding_2.step()


            n_batches += 1
            mean_loss += 0.25*(optimized_loss_11+optimized_loss_12+optimized_loss_21+optimized_loss_22)


        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)
        return mean_loss.item(), mean_rest

    def train(self, n_epochs):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin()

            train_loss, train_rest = self.train_epoch()

            for callback in self.callbacks:
                callback.on_epoch_end(train_loss, train_rest)

            if self.validation_data is not None and self.validation_freq > 0 and epoch % self.validation_freq == 0:
                for callback in self.callbacks:
                    callback.on_test_begin()
                validation_loss, rest = self.eval()
                for callback in self.callbacks:
                    callback.on_test_end(validation_loss, rest)

            if self.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.starting_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f'# loading trainer state from {path}')
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob('*.tar'):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)

class TrainerPretraining:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """
    def __init__(
            self,
            game: torch.nn.Module,
            optimizer_sender_1: torch.optim.Optimizer,
            optimizer_receiver_1: torch.optim.Optimizer,
            train_data: DataLoader,
            validation_data: Optional[DataLoader] = None,
            device: torch.device = None,
            callbacks: Optional[List[Callback]] = None
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer_sender_1 = optimizer_sender_1
        self.optimizer_receiver_1 = optimizer_receiver_1
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device
        self.game.to(self.device)
        # NB: some optimizers pre-allocate buffers before actually doing any steps
        # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
        # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
        self.optimizer_sender_1.state = move_to(self.optimizer_sender_1.state, self.device)
        self.optimizer_receiver_1.state = move_to(self.optimizer_receiver_1.state, self.device)
        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks

        if common_opts.load_from_checkpoint is not None:
            print(f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}")
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        if common_opts.preemptable:
            assert common_opts.checkpoint_dir, 'checkpointing directory has to be specified'
            d = self._get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
            self.checkpoint_path = d
            self.load_from_latest(d)
            checkpointer = CheckpointSaver(self.checkpoint_path)
            self.callbacks.append(checkpointer)
        else:
            self.checkpoint_path = None if common_opts.checkpoint_dir is None \
                else pathlib.Path(common_opts.checkpoint_dir)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

    def _get_preemptive_checkpoint_dir(self, checkpoint_root):
        if 'SLURM_JOB_ID' not in os.environ:
            print('Preemption flag set, but I am not running under SLURM?')

        job_id = os.environ.get('SLURM_JOB_ID', uuid.uuid4())
        task_id = os.environ.get('SLURM_PROCID', 0)

        d = pathlib.Path(checkpoint_root) / f'{job_id}_{task_id}'
        d.mkdir(exist_ok=True)

        return d

    def eval(self):
        mean_loss = 0.0
        mean_rest = {}

        n_batches = 0
        self.game.eval()
        with torch.no_grad():
            for batch in self.validation_data:
                batch = move_to(batch, self.device)
                optimized_loss_11,loss_11_imitation, rest = self.game(*batch)
                mean_loss += optimized_loss_11+loss_11_imitation
                mean_rest = _add_dicts(mean_rest, rest)
                n_batches += 1
        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)

        return mean_loss.item(), mean_rest

    def train_epoch(self):
        mean_loss = 0
        mean_rest = {}
        n_batches = 0
        self.game.train()
        for batch in self.train_data:
            optimized_loss_11,loss_11_imitation, rest= self.game(*batch)
            batch = move_to(batch, self.device)
            mean_rest = _add_dicts(mean_rest, rest)

            optimized_loss_sender_1=optimized_loss_11+loss_11_imitation

            self.optimizer_sender_1.zero_grad()
            self.optimizer_receiver_1.zero_grad()
            optimized_loss_sender_1.backward()
            self.optimizer_sender_1.step()
            self.optimizer_receiver_1.step()



            n_batches += 1
            mean_loss += optimized_loss_11+loss_11_imitation


        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)
        return mean_loss.item(), mean_rest

    def train(self, n_epochs):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin()

            train_loss, train_rest = self.train_epoch()

            for callback in self.callbacks:
                callback.on_epoch_end(train_loss, train_rest)

            if self.validation_data is not None and self.validation_freq > 0 and epoch % self.validation_freq == 0:
                for callback in self.callbacks:
                    callback.on_test_begin()
                validation_loss, rest = self.eval()
                for callback in self.callbacks:
                    callback.on_test_end(validation_loss, rest)

            if self.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.starting_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f'# loading trainer state from {path}')
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob('*.tar'):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)



class TrainerDialogModel6:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """
    def __init__(
            self,
            game: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_data: DataLoader,
            validation_data: Optional[DataLoader] = None,
            device: torch.device = None,
            callbacks: Optional[List[Callback]] = None
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer = optimizer
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device
        self.game.to(self.device)
        # NB: some optimizers pre-allocate buffers before actually doing any steps
        # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
        # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
        self.optimizer.state = move_to(self.optimizer.state, self.device)
        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks

        if common_opts.load_from_checkpoint is not None:
            print(f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}")
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        if common_opts.preemptable:
            assert common_opts.checkpoint_dir, 'checkpointing directory has to be specified'
            d = self._get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
            self.checkpoint_path = d
            self.load_from_latest(d)
            checkpointer = CheckpointSaver(self.checkpoint_path)
            self.callbacks.append(checkpointer)
        else:
            self.checkpoint_path = None if common_opts.checkpoint_dir is None \
                else pathlib.Path(common_opts.checkpoint_dir)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

    def _get_preemptive_checkpoint_dir(self, checkpoint_root):
        if 'SLURM_JOB_ID' not in os.environ:
            print('Preemption flag set, but I am not running under SLURM?')

        job_id = os.environ.get('SLURM_JOB_ID', uuid.uuid4())
        task_id = os.environ.get('SLURM_PROCID', 0)

        d = pathlib.Path(checkpoint_root) / f'{job_id}_{task_id}'
        d.mkdir(exist_ok=True)

        return d

    def eval(self):
        mean_loss = 0.0
        mean_rest = {}

        n_batches = 0
        self.game.eval()
        with torch.no_grad():
            for batch in self.validation_data:
                batch = move_to(batch, self.device)
                optimized_loss_11, optimized_loss_12, optimized_loss_21, optimized_loss_22, rest = self.game(*batch)
                mean_loss += 0.25*(optimized_loss_11 + optimized_loss_12 + optimized_loss_21 + optimized_loss_22)
                mean_rest = _add_dicts(mean_rest, rest)
                n_batches += 1
        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)

        return mean_loss.item(), mean_rest

    def train_epoch(self):
        mean_loss = 0
        mean_rest = {}
        n_batches = 0
        self.game.train()
        for batch in self.train_data:
            optimized_loss_11, optimized_loss_12, optimized_loss_21, optimized_loss_22, rest = self.game(*batch)
            batch = move_to(batch, self.device)
            mean_rest = _add_dicts(mean_rest, rest)

            optimized_loss_sender_1=optimized_loss_11+optimized_loss_12
            optimized_loss_sender_2=optimized_loss_21+optimized_loss_22

            self.optimizer.zero_grad()
            if np.random.rand()>0.5:
                optimized_loss_sender_1.backward()
            else:
                optimized_loss_sender_2.backward()
            self.optimizer.step()

            n_batches += 1
            mean_loss += 0.25*(optimized_loss_11+optimized_loss_12+optimized_loss_21+optimized_loss_22)


        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)
        return mean_loss.item(), mean_rest

    def train(self, n_epochs):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin()

            train_loss, train_rest = self.train_epoch()

            for callback in self.callbacks:
                callback.on_epoch_end(train_loss, train_rest)

            if self.validation_data is not None and self.validation_freq > 0 and epoch % self.validation_freq == 0:
                for callback in self.callbacks:
                    callback.on_test_begin()
                validation_loss, rest = self.eval()
                for callback in self.callbacks:
                    callback.on_test_end(validation_loss, rest)

            if self.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.starting_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f'# loading trainer state from {path}')
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob('*.tar'):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)
