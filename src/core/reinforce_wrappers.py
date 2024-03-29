# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict
import numpy as np


from .transformer import TransformerEncoder, TransformerDecoder
from .rnn import RnnEncoder, RnnEncoderImpatient,RnnEncoderExternalEmbedding
from .util import find_lengths


class ReinforceWrapper(nn.Module):
    """
    Reinforce Wrapper for an agent. Assumes that the during the forward,
    the wrapped agent returns log-probabilities over the potential outputs. During training, the wrapper
    transforms them into a tuple of (sample from the multinomial, log-prob of the sample, entropy for the multinomial).
    Eval-time the sample is replaced with argmax.

    >>> agent = nn.Sequential(nn.Linear(10, 3), nn.LogSoftmax(dim=1))
    >>> agent = ReinforceWrapper(agent)
    >>> sample, log_prob, entropy = agent(torch.ones(4, 10))
    >>> sample.size()
    torch.Size([4])
    >>> (log_prob < 0).all().item()
    1
    >>> (entropy > 0).all().item()
    1
    """
    def __init__(self, agent):
        super(ReinforceWrapper, self).__init__()
        self.agent = agent

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)

        distr = Categorical(logits=logits)
        entropy = distr.entropy()

        if self.training:
            sample = distr.sample()
        else:
            sample = logits.argmax(dim=1)
        log_prob = distr.log_prob(sample)

        return sample, log_prob, entropy


class ReinforceDeterministicWrapper(nn.Module):
    """
    Simple wrapper that makes a deterministic agent (without sampling) compatible with Reinforce-based game, by
    adding zero log-probability and entropy values to the output. No sampling is run on top of the wrapped agent,
    it is passed as is.
    >>> agent = nn.Sequential(nn.Linear(10, 3), nn.LogSoftmax(dim=1))
    >>> agent = ReinforceDeterministicWrapper(agent)
    >>> sample, log_prob, entropy = agent(torch.ones(4, 10))
    >>> sample.size()
    torch.Size([4, 3])
    >>> (log_prob == 0).all().item()
    1
    >>> (entropy == 0).all().item()
    1
    """
    def __init__(self, agent):
        super(ReinforceDeterministicWrapper, self).__init__()
        self.agent = agent

    def forward(self, *args, **kwargs):
        out = self.agent(*args, **kwargs)

        return out, torch.zeros(1).to(out.device), torch.zeros(1).to(out.device)


class SymbolGameReinforce(nn.Module):
    """
    A single-symbol Sender/Receiver game implemented with Reinforce.
    """
    def __init__(self, sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0):
        """
        :param sender: Sender agent. On forward, returns a tuple of (message, log-prob of the message, entropy).
        :param receiver: Receiver agent. On forward, accepts a message and the dedicated receiver input. Returns
            a tuple of (output, log-probs, entropy).
        :param loss: The loss function that accepts:
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs the end-to-end loss. Can be non-differentiable; if it is differentiable, this will be leveraged
        :param sender_entropy_coeff: The entropy regularization coefficient for Sender
        :param receiver_entropy_coeff: The entropy regularizatino coefficient for Receiver
        """
        super(SymbolGameReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.sender_entropy_coeff = sender_entropy_coeff

        self.mean_baseline = 0.0
        self.n_points = 0.0

    def forward(self, sender_input, labels, receiver_input=None):
        message, sender_log_prob, sender_entropy = self.sender(sender_input)
        receiver_output, receiver_log_prob, receiver_entropy = self.receiver(message, receiver_input)

        loss, rest_info = self.loss(sender_input, message, receiver_input, receiver_output, labels)
        policy_loss = ((loss.detach() - self.mean_baseline) * (sender_log_prob + receiver_log_prob)).mean()
        entropy_loss = -(sender_entropy.mean() * self.sender_entropy_coeff + receiver_entropy.mean() * self.receiver_entropy_coeff)

        if self.training:
            self.n_points += 1.0
            self.mean_baseline += (loss.detach().mean().item() -
                                   self.mean_baseline) / self.n_points

        full_loss = policy_loss + entropy_loss + loss.mean()

        for k, v in rest_info.items():
            if hasattr(v, 'mean'):
                rest_info[k] = v.mean().item()

        rest_info['baseline'] = self.mean_baseline
        rest_info['loss'] = loss.mean().item()
        rest_info['sender_entropy'] = sender_entropy.mean()
        rest_info['receiver_entropy'] = receiver_entropy.mean()

        return full_loss, rest_info


class RnnSenderReinforce(nn.Module):
    """
    Reinforce Wrapper for Sender in variable-length message game. Assumes that during the forward,
    the wrapped agent returns the initial hidden state for a RNN cell. This cell is the unrolled by the wrapper.
    During training, the wrapper samples from the cell, getting the output message. Evaluation-time, the sampling
    is replaced by argmax.

    >>> agent = nn.Linear(10, 3)
    >>> agent = RnnSenderReinforce(agent, vocab_size=5, embed_dim=5, hidden_size=3, max_len=10, cell='lstm', force_eos=False)
    >>> input = torch.FloatTensor(16, 10).uniform_(-0.1, 0.1)
    >>> message, logprob, entropy = agent(input)
    >>> message.size()
    torch.Size([16, 10])
    >>> (entropy > 0).all().item()
    1
    >>> message.size()  # batch size x max_len
    torch.Size([16, 10])
    """
    def __init__(self, agent, vocab_size, embed_dim, hidden_size, max_len, num_layers=1, cell='rnn', force_eos=True):
        """
        :param agent: the agent to be wrapped
        :param vocab_size: the communication vocabulary size
        :param embed_dim: the size of the embedding used to embed the output symbols
        :param hidden_size: the RNN cell's hidden state size
        :param max_len: maximal length of the output messages
        :param cell: type of the cell used (rnn, gru, lstm)
        :param force_eos: if set to True, each message is extended by an EOS symbol. To ensure that no message goes
        beyond `max_len`, Sender only generates `max_len - 1` symbols from an RNN cell and appends EOS.
        """
        super(RnnSenderReinforce, self).__init__()
        self.agent = agent

        self.force_eos = force_eos

        self.max_len = max_len
        if force_eos:
            self.max_len -= 1

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.norm_h = nn.LayerNorm(hidden_size)
        self.norm_c = nn.LayerNorm(hidden_size)
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cells = None

        cell = cell.lower()
        cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'lstm': nn.LSTMCell}

        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        cell_type = cell_types[cell]
        self.cells = nn.ModuleList([
            cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
            cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x):
        prev_hidden = [self.agent(x)]
        prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)])

        prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    h_t = self.norm_h(h_t)
                    c_t = self.norm_c(c_t)
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                    h_t = self.norm_h(h_t)
                prev_hidden[i] = h_t
                input = h_t


            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)

            logits.append(distr.log_prob(x))

            input = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy

class RnnSenderReinforceModel3(nn.Module):
    """
    Reinforce Wrapper for Sender in variable-length message game. Assumes that during the forward,
    the wrapped agent returns the initial hidden state for a RNN cell. This cell is the unrolled by the wrapper.
    During training, the wrapper samples from the cell, getting the output message. Evaluation-time, the sampling
    is replaced by argmax.

    >>> agent = nn.Linear(10, 3)
    >>> agent = RnnSenderReinforce(agent, vocab_size=5, embed_dim=5, hidden_size=3, max_len=10, cell='lstm', force_eos=False)
    >>> input = torch.FloatTensor(16, 10).uniform_(-0.1, 0.1)
    >>> message, logprob, entropy = agent(input)
    >>> message.size()
    torch.Size([16, 10])
    >>> (entropy > 0).all().item()
    1
    >>> message.size()  # batch size x max_len
    torch.Size([16, 10])
    """
    def __init__(self, agent, vocab_size, embed_dim, hidden_size, max_len, num_layers=1, cell='rnn', force_eos=True):
        """
        :param agent: the agent to be wrapped
        :param vocab_size: the communication vocabulary size
        :param embed_dim: the size of the embedding used to embed the output symbols
        :param hidden_size: the RNN cell's hidden state size
        :param max_len: maximal length of the output messages
        :param cell: type of the cell used (rnn, gru, lstm)
        :param force_eos: if set to True, each message is extended by an EOS symbol. To ensure that no message goes
        beyond `max_len`, Sender only generates `max_len - 1` symbols from an RNN cell and appends EOS.
        """
        super(RnnSenderReinforceModel3, self).__init__()
        self.agent = agent

        self.force_eos = force_eos

        self.max_len = max_len
        if force_eos:
            self.max_len -= 1

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.norm_h = torch.nn.LayerNorm(hidden_size)
        self.norm_c = torch.nn.LayerNorm(hidden_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cells = None

        cell = cell.lower()
        cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'lstm': nn.LSTMCell}

        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        cell_type = cell_types[cell]
        self.cells = nn.ModuleList([
            cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
            cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x, imitate=False):
        prev_hidden = [self.agent(x)]
        prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)])

        prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    h_t = self.norm_h(h_t)
                    c_t = self.norm_h(c_t)
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                    h_t = self.norm_h(h_t)
                prev_hidden[i] = h_t
                input = h_t


            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)

            if imitate:
                logits.append(distr.probs)
            else:
                logits.append(distr.log_prob(x))

            input = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        if imitate:
          logits = torch.stack(logits).permute(1,2, 0)
        else:
          logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy

class RnnSenderReinforceExternalEmbedding(nn.Module):
    """
    Reinforce Wrapper for Sender in variable-length message game. Assumes that during the forward,
    the wrapped agent returns the initial hidden state for a RNN cell. This cell is the unrolled by the wrapper.
    During training, the wrapper samples from the cell, getting the output message. Evaluation-time, the sampling
    is replaced by argmax.

    >>> agent = nn.Linear(10, 3)
    >>> agent = RnnSenderReinforce(agent, vocab_size=5, embed_dim=5, hidden_size=3, max_len=10, cell='lstm', force_eos=False)
    >>> input = torch.FloatTensor(16, 10).uniform_(-0.1, 0.1)
    >>> message, logprob, entropy = agent(input)
    >>> message.size()
    torch.Size([16, 10])
    >>> (entropy > 0).all().item()
    1
    >>> message.size()  # batch size x max_len
    torch.Size([16, 10])
    """
    def __init__(self, agent,embedding_layer, vocab_size, embed_dim, hidden_size, max_len, num_layers=1, cell='rnn', force_eos=True):
        """
        :param agent: the agent to be wrapped
        :param vocab_size: the communication vocabulary size
        :param embed_dim: the size of the embedding used to embed the output symbols
        :param hidden_size: the RNN cell's hidden state size
        :param max_len: maximal length of the output messages
        :param cell: type of the cell used (rnn, gru, lstm)
        :param force_eos: if set to True, each message is extended by an EOS symbol. To ensure that no message goes
        beyond `max_len`, Sender only generates `max_len - 1` symbols from an RNN cell and appends EOS.
        """
        super(RnnSenderReinforceExternalEmbedding, self).__init__()
        self.agent = agent

        self.force_eos = force_eos

        self.max_len = max_len
        if force_eos:
            self.max_len -= 1

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.norm_h = torch.nn.LayerNorm(hidden_size)
        self.norm_c = torch.nn.LayerNorm(hidden_size)
        self.embedding = embedding_layer
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cells = None

        cell = cell.lower()
        cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'lstm': nn.LSTMCell}

        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        cell_type = cell_types[cell]
        self.cells = nn.ModuleList([
            cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
            cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x, imitate=False):
        prev_hidden = [self.agent(x)]
        prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)])

        prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    h_t = self.norm_h(h_t)
                    c_t = self.norm_h(c_t)
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                    h_t = self.norm_h(h_t)
                prev_hidden[i] = h_t
                input = h_t


            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)

            if imitate:
                logits.append(distr.probs)
            else:
                logits.append(distr.log_prob(x))

            input = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        if imitate:
          logits = torch.stack(logits).permute(1,2, 0)
        else:
          logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy


class RnnReceiverReinforce(nn.Module):
    """
    Reinforce Wrapper for Receiver in variable-length message game. The wrapper logic feeds the message into the cell
    and calls the wrapped agent on the hidden state vector for the step that either corresponds to the EOS input to the
    input that reaches the maximal length of the sequence.
    This output is assumed to be the tuple of (output, logprob, entropy).
    """
    def __init__(self, agent, vocab_size, embed_dim, hidden_size, cell='rnn', num_layers=1):
        super(RnnReceiverReinforce, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoder(vocab_size, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message, input=None, lengths=None):
        encoded = self.encoder(message)
        sample, logits, entropy = self.agent(encoded, input)

        return sample, logits, entropy

class RnnReceiverCompositionality(nn.Module):
    """
    Reinforce Wrapper for Receiver in variable-length message game with several attributes (for compositionality experiments).
    RnnReceiverCompositionality is equivalent to RnnReceiverReinforce but treated each attribute independently.
    This output is assumed to be the tuple of (output, logprob, entropy).
    """
    def __init__(self, agent, vocab_size, embed_dim, hidden_size,max_len,n_attributes, n_values, cell='rnn', num_layers=1):
        super(RnnReceiverCompositionality, self).__init__()
        self.agent = agent
        self.n_attributes=n_attributes
        self.n_values=n_values
        self.encoder = RnnEncoder(vocab_size, embed_dim, hidden_size, cell, num_layers)
        self.hidden_to_output = nn.Linear(hidden_size, n_attributes*n_values)

    def forward(self, message, input=None, lengths=None):
        encoded = self.encoder(message)
        logits = F.log_softmax(self.hidden_to_output(encoded).reshape(encoded.size(0),self.n_attributes,self.n_values), dim=2)
        #entropy=-torch.exp(logits)*logits
        entropy=[]
        slogits= []
        for i in range(logits.size(1)):
          distr = Categorical(logits=logits[:,i,:])
          entropy.append(distr.entropy())
          x = distr.sample()
          slogits.append(distr.log_prob(x))

        entropy = torch.stack(entropy).permute(1, 0)
        slogits = torch.stack(slogits).permute(1, 0)

        return logits, slogits, entropy

class RnnReceiverDeterministic(nn.Module):
    """
    Reinforce Wrapper for a deterministic Receiver in variable-length message game. The wrapper logic feeds the message
    into the cell and calls the wrapped agent with the hidden state that either corresponds to the end-of-sequence
    term or to the end of the sequence. The wrapper extends it with zero-valued log-prob and entropy tensors so that
    the agent becomes compatible with the SenderReceiverRnnReinforce game.

    As the wrapped agent does not sample, it has to be trained via regular back-propagation. This requires that both the
    the agent's output and  loss function and the wrapped agent are differentiable.

    >>> class Agent(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> agent = RnnReceiverDeterministic(Agent(), vocab_size=10, embed_dim=10, hidden_size=5)
    >>> message = torch.zeros((16, 10)).long().random_(0, 10)  # batch of 16, 10 symbol length
    >>> output, logits, entropy = agent(message)
    >>> (logits == 0).all().item()
    1
    >>> (entropy == 0).all().item()
    1
    >>> output.size()
    torch.Size([16, 3])
    """

    def __init__(self, agent, vocab_size, embed_dim, hidden_size, cell='rnn', num_layers=1):
        super(RnnReceiverDeterministic, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoder(vocab_size, embed_dim, hidden_size, cell, num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, message, input=None, lengths=None,imitate=False):
        encoded = self.encoder(message)
        encoded=self.norm(encoded)
        agent_output = self.agent(encoded, input)

        if imitate:
            step_logits = F.log_softmax(agent_output, dim=1)
            distr = Categorical(logits=step_logits)
            entropy=distr.entropy()
            logits=distr.probs
            entropy=entropy.to(agent_output.device)
            logits=logits.to(agent_output.device)

            det_logits=torch.zeros(agent_output.size(0)).to(agent_output.device)
            det_entropy=det_logits

            return agent_output, logits, entropy, det_logits, det_entropy

        else:
            logits = torch.zeros(agent_output.size(0)).to(agent_output.device)
            entropy = logits

            return agent_output, logits, entropy


class RnnReceiverDeterministicExternalEmbedding(nn.Module):
    """
    Reinforce Wrapper for a deterministic Receiver in variable-length message game. The wrapper logic feeds the message
    into the cell and calls the wrapped agent with the hidden state that either corresponds to the end-of-sequence
    term or to the end of the sequence. The wrapper extends it with zero-valued log-prob and entropy tensors so that
    the agent becomes compatible with the SenderReceiverRnnReinforce game.

    As the wrapped agent does not sample, it has to be trained via regular back-propagation. This requires that both the
    the agent's output and  loss function and the wrapped agent are differentiable.

    >>> class Agent(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> agent = RnnReceiverDeterministic(Agent(), vocab_size=10, embed_dim=10, hidden_size=5)
    >>> message = torch.zeros((16, 10)).long().random_(0, 10)  # batch of 16, 10 symbol length
    >>> output, logits, entropy = agent(message)
    >>> (logits == 0).all().item()
    1
    >>> (entropy == 0).all().item()
    1
    >>> output.size()
    torch.Size([16, 3])
    """

    def __init__(self, agent, embedding_layer, vocab_size, embed_dim, hidden_size, cell='rnn', num_layers=1):
        super(RnnReceiverDeterministicExternalEmbedding, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoderExternalEmbedding(embedding_layer, vocab_size, embed_dim, hidden_size, cell, num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, message, input=None, lengths=None,imitate=False):
        encoded = self.encoder(message)
        encoded=self.norm(encoded)
        agent_output = self.agent(encoded, input)

        if imitate:
            step_logits = F.log_softmax(agent_output, dim=1)
            distr = Categorical(logits=step_logits)
            entropy=distr.entropy()
            logits=distr.probs
            entropy=entropy.to(agent_output.device)
            logits=logits.to(agent_output.device)

            det_logits=torch.zeros(agent_output.size(0)).to(agent_output.device)
            det_entropy=det_logits

            return agent_output, logits, entropy, det_logits, det_entropy

        else:
            logits = torch.zeros(agent_output.size(0)).to(agent_output.device)
            entropy = logits

            return agent_output, logits, entropy

class RnnReceiverImpatient(nn.Module):
    """
    Impatient Listener.
    The wrapper logic feeds the message into the cell and calls the wrapped agent.
    The wrapped agent has to returns the intermediate hidden states for every position.
    All the hidden states are mapped to a categorical distribution with a single
    Linear layer (hidden_to_ouput) followed by a softmax.
    Thess categorical probabilities (step_logits) will then be used to compute the Impatient loss function.
    """

    def __init__(self, agent, vocab_size, embed_dim, hidden_size,max_len,n_features, cell='rnn', num_layers=1):
        super(RnnReceiverImpatient, self).__init__()

        self.max_len = max_len
        self.hidden_to_output = nn.Linear(hidden_size, n_features)
        self.encoder = RnnEncoderImpatient(vocab_size, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message, input=None, lengths=None):
        encoded = self.encoder(message)

        sequence = []
        logits = []
        entropy = []

        for step in range(encoded.size(0)):
            h_t=encoded[step,:,:]
            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample() # Sampling useless ?
            else:
                x = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(x))
            sequence.append(step_logits)

        sequence = torch.stack(sequence).permute(1, 0, 2)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        return sequence, logits, entropy

class RnnReceiverImpatientCompositionality(nn.Module):

    """
    RnnReceiverImpatientCompositionality is an adaptation of RnnReceiverImpatientCompositionality
    for inputs with several attributes (compositionality experiments).
    Each attribute is treated independently.
    """

    def __init__(self, agent, vocab_size, embed_dim, hidden_size,max_len,n_attributes, n_values, cell='rnn', num_layers=1):
        super(RnnReceiverImpatientCompositionality, self).__init__()

        self.max_len = max_len
        self.n_attributes=n_attributes
        self.n_values=n_values
        self.hidden_to_output = nn.Linear(hidden_size, n_attributes*n_values)
        self.encoder = RnnEncoderImpatient(vocab_size, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message, input=None, lengths=None):

        encoded = self.encoder(message)

        sequence = []
        slogits = []
        entropy = []

        for step in range(encoded.size(0)):

            h_t=encoded[step,:,:]
            step_logits = F.log_softmax(self.hidden_to_output(h_t).reshape(h_t.size(0),self.n_attributes,self.n_values), dim=2)
            distr = Categorical(logits=step_logits)

            sequence.append(step_logits)

            entropy_step=[]
            slogits_step=[]

            for i in range(step_logits.size(1)):
              distr = Categorical(logits=step_logits[:,i,:])
              entropy_step.append(distr.entropy())
              x = distr.sample()
              slogits_step.append(distr.log_prob(x))

            entropy_step = torch.stack(entropy_step).permute(1, 0)
            slogits_step = torch.stack(slogits_step).permute(1, 0)

            entropy.append(entropy_step)
            slogits.append(slogits_step)

        sequence = torch.stack(sequence).permute(1,0,2,3)
        entropy = torch.stack(entropy).permute(1,0,2)
        slogits= torch.stack(slogits).permute(1,0,2)
        #logits = torch.stack(logits).permute(1, 0)
        #entropy = torch.stack(entropy).permute(1, 0)

        return sequence, slogits, entropy

class RnnReceiverWithHiddenStates(nn.Module):
    """
    Impatient Listener.
    The wrapper logic feeds the message into the cell and calls the wrapped agent.
    The wrapped agent has to returns the intermediate hidden states for every position.
    All the hidden states are mapped to a categorical distribution with a single
    Linear layer (hidden_to_ouput) followed by a softmax.
    Thess categorical probabilities (step_logits) will then be used to compute the Impatient loss function.
    """

    def __init__(self, agent, vocab_size, embed_dim, hidden_size,max_len,n_features, cell='rnn', num_layers=1):
        super(RnnReceiverWithHiddenStates, self).__init__()

        self.max_len = max_len
        self.hidden_to_output = nn.Linear(hidden_size, n_features)
        self.norm_h=nn.LayerNorm(hidden_size)
        self.norm_c=nn.LayerNorm(hidden_size)
        self.encoder = RnnEncoderImpatient(vocab_size, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message, input=None, lengths=None):
        encoded = self.encoder(message)

        sequence = []
        logits = []
        entropy = []
        hidden_states = []

        for step in range(encoded.size(0)):
            h_t=encoded[step,:,:]
            h_t=norm_h(h_t)
            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample() # Sampling useless ?
            else:
                x = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(x))
            sequence.append(step_logits)
            hidden_states.append(h_t)

        sequence = torch.stack(sequence).permute(1, 0, 2)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)
        hidden_states = torch.stack(hidden_states).permute(1, 0, 2)

        return sequence, logits, entropy, hidden_states


class AgentBaseline(nn.Module):

    """
    AgentBaseline is composed of a couple of modalities:
        - sender
        - receiver
    In AgentBaseline, Sender and Receiver parts are independent
    """

    def __init__(self, receiver, sender):
        super(AgentBaseline, self).__init__()

        self.receiver=receiver
        self.sender=sender

    def forward(self, message, input=None, lengths=None):

        raise NotImplementedError

class AgentModel2(nn.Module):

    """
    AgentBaseline is composed of a couple of modalities:
        - sender
        - receiver
    In AgentBaseline, Sender and Receiver parts are independent
    """

    def __init__(self, receiver, sender):
        super(AgentModel2, self).__init__()

        self.receiver=receiver
        self.sender=sender

    def send(self, sender_input):

      return self.sender(sender_input)

    def receive(self,message, receiver_input, message_lengths):

      receiver_output, log_prob_r, entropy_r,hidden_states = self.receiver(message, receiver_input, message_lengths)

      sequence_lm=[]
      logits_lm=[]

      for step in range(hidden_states.size(1)):
        h_t=hidden_states[:,step,:]
        step_logits_lm = F.log_softmax(self.sender.hidden_to_output(h_t), dim=1)
        distr_lm = Categorical(logits=step_logits_lm)
        #entropy_lm.append(distr_lm.entropy())

        x = step_logits_lm.argmax(dim=1)
        logits_lm.append(distr_lm.log_prob(x))
        sequence_lm.append(step_logits_lm)

      sequence_lm = torch.stack(sequence_lm).permute(1, 0, 2)
      logits_lm = torch.stack(logits_lm).permute(1, 0)

      return receiver_output, log_prob_r, entropy_r, sequence_lm, logits_lm


class AgentModel3(nn.Module):

    """
    AgentBaseline is composed of a couple of modalities:
        - sender
        - receiver
    In AgentBaseline, Sender and Receiver parts are independent
    """

    def __init__(self, receiver, sender):
        super(AgentModel3, self).__init__()

        self.receiver=receiver
        self.sender=sender

    def send(self, sender_input):

      return self.sender(sender_input)

    def receive(self,message, receiver_input, message_lengths,imitate=True):

      return self.receiver(message, receiver_input, message_lengths,imitate)

    def imitate(self,sender_input,imitate=True):

      return self.sender(sender_input,imitate)

# New class agent

class AgentBaseline2(nn.Module):

    """
    AgentBaseline is composed of a couple of modalities:
        - sender
        - receiver
    In AgentBaseline, Sender and Receiver parts are independent
    """

    def __init__(self,
                n_features,
                vocab_size,
                max_len,
                embed_dim,
                sender_hidden_size,
                receiver_hidden_size,
                sender_cell,
                receiver_cell,
                sender_num_layers,
                receiver_num_layers,
                force_eos):
        super(AgentBaseline2, self).__init__()

        # Common to sender and receiver
        self.force_eos = force_eos

        self.max_len = max_len
        if force_eos:
            self.max_len -= 1

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.sender_hidden_size=sender_hidden_size
        self.receiver_hidden_size=receiver_hidden_size
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))

        cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'lstm': nn.LSTMCell}

        # Sender
        self.agent_sender = nn.Linear(n_features, sender_hidden_size) #nn.Linear(n_features, n_hidden)
        self.sender_cells = None
        self.sender_num_layers = sender_num_layers
        self.sender_norm_h = nn.LayerNorm(sender_hidden_size)
        self.sender_norm_c = nn.LayerNorm(sender_hidden_size)
        self.hidden_to_output = nn.Linear(sender_hidden_size, vocab_size)
        self.sender_embedding = nn.Embedding(vocab_size, embed_dim)

        sender_cell = sender_cell.lower()

        if sender_cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {sender_cell}")

        cell_type = cell_types[sender_cell]
        self.sender_cells = nn.ModuleList([
            cell_type(input_size=embed_dim, hidden_size=sender_hidden_size) if i == 0 else \
            cell_type(input_size=sender_hidden_size, hidden_size=sender_hidden_size) for i in range(self.sender_num_layers)])

        self.reset_parameters()

        # Receiver
        self.agent_receiver = nn.Linear(receiver_hidden_size, n_features) #nn.Linear(n_hidden, n_features)
        self.receiver_cells = None
        self.receiver_num_layers = receiver_num_layers
        self.receiver_norm_h = nn.LayerNorm(receiver_hidden_size)
        self.receiver_norm_c = nn.LayerNorm(receiver_hidden_size)
        #self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.receiver_embedding = nn.Embedding(vocab_size, embed_dim)

        receiver_cell = receiver_cell.lower()

        if receiver_cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {receiver_cell}")

        cell_types_r = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

        cell_type = cell_types_r[receiver_cell]
        self.receiver_cell = cell_types_r[receiver_cell](input_size=embed_dim, batch_first=True,
                               hidden_size=receiver_hidden_size, num_layers=receiver_num_layers)
        #self.receiver_cells = nn.ModuleList([
        #    cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
        #    cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.receiver_num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def send(self, x, eval=False,return_policies=False):
        prev_hidden = [self.agent_sender(x)]
        prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers - 1)])

        prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers)]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        logits = []
        entropy = []
        whole_logits = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.sender_cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    h_t = self.sender_norm_h(h_t)
                    c_t = self.sender_norm_c(c_t)
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                    h_t = self.sender_norm_h(h_t)
                prev_hidden[i] = h_t
                input = h_t


            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training and not eval:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)

            logits.append(distr.log_prob(x))
            whole_logits.append(distr.probs)

            input = self.sender_embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        whole_logits = torch.stack(whole_logits).permute(1,0, 2)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)
            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        if return_policies:
            return sequence,logits,whole_logits, entropy
        else:
            return sequence,logits, entropy

    def receive(self,message, receiver_input, message_lengths):

      if message_lengths is None:
        message_lengths=find_lengths(message)

      prev_hidden = [torch.zeros((message.size(0),self.hidden_size)).to("cuda")]
      prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.receiver_num_layers - 1)])

      prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.receiver_num_layers)]  # only used for LSTM

      inputs = self.receiver_embedding(message)

      sequence = []
      logits = []
      entropy = []

      for step in range(self.max_len):

          input=inputs[:,step,:]

          for i, layer in enumerate(self.receiver_cells):
              if isinstance(layer, nn.LSTMCell):
                  h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                  h_t = self.receiver_norm_h(h_t)
                  c_t = self.receiver_norm_c(c_t)
                  prev_c[i] = c_t
              else:
                  h_t = layer(input, prev_hidden[i])
                  h_t = self.norm_h(h_t)
              prev_hidden[i] = h_t


          #step_logits = F.log_softmax(self.agent_receiver(h_t,None), dim=1)
          agent_output = self.agent_receiver(h_t)
          log = torch.zeros(agent_output.size(0)).to(agent_output.device)
          ent = log

          logits.append(log)
          entropy.append(ent)
          sequence.append(agent_output)

      sequence = torch.stack(sequence).permute(1, 0, 2)
      logits = torch.stack(logits).permute(1, 0)
      entropy = torch.stack(entropy).permute(1, 0)

      # Here choose EOS
      #sequence=sequence[:,-1,:]
      #logits=logits[:,-1]
      #entropy=entropy[:,-1]

      output=[]
      for j in range(sequence.size(0)):
        output.append(sequence[j,message_lengths[j]-1,:])

      output=torch.stack(output)
      logits=logits[:,-1]
      entropy=entropy[:,-1]

      return output, logits, entropy

    def receive_2(self,message, receiver_input, message_lengths):

      emb = self.receiver_embedding(message)

      if message_lengths is None:
        message_lengths = find_lengths(message)

      packed = nn.utils.rnn.pack_padded_sequence(
          emb, message_lengths.cpu(), batch_first=True, enforce_sorted=False)
      _, rnn_hidden = self.receiver_cell(packed)

      if isinstance(self.receiver_cell, nn.LSTM):
          rnn_hidden, _ = rnn_hidden

      encoded = rnn_hidden[-1]
      #encoded=self.receiver_norm_h(encoded)
      agent_output = self.agent_receiver(encoded)

      logits = torch.zeros(agent_output.size(0)).to(agent_output.device)
      entropy = logits

      return agent_output, logits, entropy

    def imitate(self,x):

      prev_hidden = [self.agent_sender(x)]
      prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers - 1)])

      prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers)]  # only used for LSTM

      input = torch.stack([self.sos_embedding] * x.size(0))

      sequence = []
      logits = []
      entropy = []

      for step in range(self.max_len):
          for i, layer in enumerate(self.sender_cells):
              if isinstance(layer, nn.LSTMCell):
                  h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                  h_t = self.sender_norm_h(h_t)
                  c_t = self.sender_norm_c(c_t)
                  prev_c[i] = c_t
              else:
                  h_t = layer(input, prev_hidden[i])
                  h_t = self.sender_norm_h(h_t)
              prev_hidden[i] = h_t
              input = h_t


          step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)

          distr = Categorical(logits=step_logits)
          entropy.append(distr.entropy())

          if self.training:
              x = distr.sample()
          else:
              x = step_logits.argmax(dim=1)

          logits.append(distr.probs)


          input = self.sender_embedding(x)
          sequence.append(x)

      sequence = torch.stack(sequence).permute(1, 0)
      logits = torch.stack(logits).permute(1,2, 0)
      entropy = torch.stack(entropy).permute(1, 0)

      if self.force_eos:
          zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

          sequence = torch.cat([sequence, zeros.long()], dim=1)
          logits = torch.cat([logits, zeros], dim=1)
          entropy = torch.cat([entropy, zeros], dim=1)

      return sequence, logits, entropy


class AgentBaselineCompositionality(nn.Module):

    """
    AgentBaseline is composed of a couple of modalities:
        - sender
        - receiver
    In AgentBaseline, Sender and Receiver parts are independent
    """

    def __init__(self,
                n_values,
                n_attributes,
                vocab_size,
                max_len,
                embed_dim,
                sender_hidden_size,
                receiver_hidden_size,
                sender_cell,
                receiver_cell,
                sender_num_layers,
                receiver_num_layers,
                force_eos):
        super(AgentBaselineCompositionality, self).__init__()

        # Common to sender and receiver
        self.force_eos = force_eos

        self.max_len = max_len
        if force_eos:
            self.max_len -= 1

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.n_attributes=n_attributes
        self.n_values=n_values
        self.sender_hidden_size=sender_hidden_size
        self.receiver_hidden_size=receiver_hidden_size
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))

        cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'lstm': nn.LSTMCell}

        # Sender
        self.agent_sender = nn.Linear(n_values*n_attributes, sender_hidden_size) #nn.Linear(n_features, n_hidden)
        self.sender_cells = None
        self.sender_num_layers = sender_num_layers
        self.sender_norm_h = nn.LayerNorm(sender_hidden_size)
        self.sender_norm_c = nn.LayerNorm(sender_hidden_size)
        self.hidden_to_output = nn.Linear(sender_hidden_size, vocab_size)
        self.sender_embedding = nn.Embedding(vocab_size, embed_dim)

        sender_cell = sender_cell.lower()

        if sender_cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {sender_cell}")

        cell_type = cell_types[sender_cell]
        self.sender_cells = nn.ModuleList([
            cell_type(input_size=embed_dim, hidden_size=sender_hidden_size) if i == 0 else \
            cell_type(input_size=sender_hidden_size, hidden_size=sender_hidden_size) for i in range(self.sender_num_layers)])

        self.reset_parameters()

        # Receiver
        self.agent_receiver = nn.Linear(receiver_hidden_size, n_values*n_attributes) #nn.Linear(n_hidden, n_features)
        self.receiver_cells = None
        self.receiver_num_layers = receiver_num_layers
        self.receiver_norm_h = nn.LayerNorm(receiver_hidden_size)
        self.receiver_norm_c = nn.LayerNorm(receiver_hidden_size)
        #self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.receiver_embedding = nn.Embedding(vocab_size, embed_dim)

        receiver_cell = receiver_cell.lower()

        if receiver_cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {receiver_cell}")

        cell_types_r = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

        cell_type = cell_types_r[receiver_cell]
        self.receiver_cell = cell_types_r[receiver_cell](input_size=embed_dim, batch_first=True,
                               hidden_size=receiver_hidden_size, num_layers=receiver_num_layers)
        #self.receiver_cells = nn.ModuleList([
        #    cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
        #    cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.receiver_num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def send(self, x, eval=False,return_policies=False):
        prev_hidden = [self.agent_sender(x)]
        prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers - 1)])

        prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers)]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        logits = []
        entropy = []
        whole_logits = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.sender_cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    h_t = self.sender_norm_h(h_t)
                    c_t = self.sender_norm_c(c_t)
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                    h_t = self.sender_norm_h(h_t)
                prev_hidden[i] = h_t
                input = h_t


            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training and not eval:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)

            logits.append(distr.log_prob(x))
            whole_logits.append(distr.probs)

            input = self.sender_embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        whole_logits = torch.stack(whole_logits).permute(1,0, 2)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)
            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        if return_policies:
            return sequence,logits,whole_logits, entropy
        else:
            return sequence,logits, entropy

    def receive(self,message, receiver_input, message_lengths,return_policies=False,return_sample=False):

      emb = self.receiver_embedding(message)

      if message_lengths is None:
        message_lengths = find_lengths(message)

      packed = nn.utils.rnn.pack_padded_sequence(
          emb, message_lengths.cpu(), batch_first=True, enforce_sorted=False)
      _, rnn_hidden = self.receiver_cell(packed)

      if isinstance(self.receiver_cell, nn.LSTM):
          rnn_hidden, _ = rnn_hidden

      encoded = rnn_hidden[-1]

      agent_output = self.agent_receiver(encoded).reshape(encoded.size(0),self.n_attributes,self.n_values)
      logits = F.log_softmax(agent_output,dim=2)

      entropy=[]
      slogits= []
      sample = []
      for i in range(logits.size(1)):
        distr = Categorical(logits=logits[:,i,:])
        entropy.append(distr.entropy())
        if self.training:
            #x = distr.sample()
            x = logits[:,i,:].argmax(dim=1)
            sample.append(x)
        else:
            x = logits[:,i,:].argmax(dim=1)
        slogits.append(distr.log_prob(x))

      entropy = torch.stack(entropy).permute(1, 0)
      slogits = torch.stack(slogits).permute(1, 0)
      sample = torch.stack(sample).permute(1, 0)

      if return_sample:
          return sample, agent_output, slogits,logits, entropy
      elif return_policies:
          return agent_output, slogits,logits, entropy
      else:
          return agent_output, slogits, entropy

    def imitate(self,x):

      raise NotImplementedError


class AgentBaselineKL(nn.Module):

    """
    AgentBaseline is composed of a couple of modalities:
        - sender
        - receiver
    In AgentBaseline, Sender and Receiver parts are independent
    """

    def __init__(self,
                n_features,
                vocab_size,
                max_len,
                embed_dim,
                hidden_size,
                sender_cell,
                receiver_cell,
                sender_num_layers,
                receiver_num_layers,
                force_eos):
        super(AgentBaselineKL, self).__init__()

        # Common to sender and receiver
        self.force_eos = force_eos

        self.max_len = max_len
        if force_eos:
            self.max_len -= 1

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.hidden_size=hidden_size
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))

        cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'lstm': nn.LSTMCell}

        # Sender
        self.agent_sender = nn.Linear(n_features, hidden_size) #nn.Linear(n_features, n_hidden)
        self.sender_cells = None
        self.sender_num_layers = sender_num_layers
        self.sender_norm_h = nn.LayerNorm(hidden_size)
        self.sender_norm_c = nn.LayerNorm(hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.sender_embedding = nn.Embedding(vocab_size, embed_dim)

        sender_cell = sender_cell.lower()

        if sender_cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {sender_cell}")

        cell_type = cell_types[sender_cell]
        self.sender_cells = nn.ModuleList([
            cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
            cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.sender_num_layers)])

        self.reset_parameters()

        # Receiver
        self.agent_receiver = nn.Linear(hidden_size, n_features) #nn.Linear(n_hidden, n_features)
        self.receiver_cells = None
        self.receiver_num_layers = receiver_num_layers
        self.receiver_norm_h = nn.LayerNorm(hidden_size)
        self.receiver_norm_c = nn.LayerNorm(hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.receiver_embedding = nn.Embedding(vocab_size, embed_dim)

        receiver_cell = receiver_cell.lower()

        if receiver_cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {receiver_cell}")

        cell_type = cell_types[receiver_cell]
        self.receiver_cells = nn.ModuleList([
            cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
            cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.receiver_num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def send(self, x, eval=False):
        prev_hidden = [self.agent_sender(x)]
        prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers - 1)])

        prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers)]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        logits = []
        whole_logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.sender_cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    h_t = self.sender_norm_h(h_t)
                    c_t = self.sender_norm_c(c_t)
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                    h_t = self.sender_norm_h(h_t)
                prev_hidden[i] = h_t
                input = h_t


            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training and not eval:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)

            logits.append(distr.log_prob(x))
            whole_logits.append(step_logits)

            input = self.sender_embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        whole_logits = torch.stack(whole_logits).permute(1,0, 2)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)
            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits,whole_logits, entropy

    def receive(self,message, receiver_input, message_lengths):

      if message_lengths is None:
        message_lengths=find_lengths(message)

      prev_hidden = [torch.zeros((message.size(0),self.hidden_size)).to("cuda")]
      prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.receiver_num_layers - 1)])

      prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.receiver_num_layers)]  # only used for LSTM

      inputs = self.receiver_embedding(message)

      sequence = []
      logits = []
      entropy = []

      for step in range(self.max_len):

          input=inputs[:,step,:]

          for i, layer in enumerate(self.receiver_cells):
              if isinstance(layer, nn.LSTMCell):
                  h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                  h_t = self.receiver_norm_h(h_t)
                  c_t = self.receiver_norm_c(c_t)
                  prev_c[i] = c_t
              else:
                  h_t = layer(input, prev_hidden[i])
                  h_t = self.norm_h(h_t)
              prev_hidden[i] = h_t


          #step_logits = F.log_softmax(self.agent_receiver(h_t,None), dim=1)
          agent_output = self.agent_receiver(h_t)
          log = torch.zeros(agent_output.size(0)).to(agent_output.device)
          ent = log

          logits.append(log)
          entropy.append(ent)
          sequence.append(agent_output)

      sequence = torch.stack(sequence).permute(1, 0, 2)
      logits = torch.stack(logits).permute(1, 0)
      entropy = torch.stack(entropy).permute(1, 0)

      # Here choose EOS
      #sequence=sequence[:,-1,:]
      #logits=logits[:,-1]
      #entropy=entropy[:,-1]

      output=[]
      for j in range(sequence.size(0)):
        output.append(sequence[j,message_lengths[j]-1,:])

      output=torch.stack(output)
      logits=logits[:,-1]
      entropy=entropy[:,-1]

      return output, logits, entropy

    def imitate(self,x):

      prev_hidden = [self.agent_sender(x)]
      prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers - 1)])

      prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers)]  # only used for LSTM

      input = torch.stack([self.sos_embedding] * x.size(0))

      sequence = []
      logits = []
      entropy = []

      for step in range(self.max_len):
          for i, layer in enumerate(self.sender_cells):
              if isinstance(layer, nn.LSTMCell):
                  h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                  h_t = self.sender_norm_h(h_t)
                  c_t = self.sender_norm_c(c_t)
                  prev_c[i] = c_t
              else:
                  h_t = layer(input, prev_hidden[i])
                  h_t = self.sender_norm_h(h_t)
              prev_hidden[i] = h_t
              input = h_t


          step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)

          distr = Categorical(logits=step_logits)
          entropy.append(distr.entropy())

          if self.training:
              x = distr.sample()
          else:
              x = step_logits.argmax(dim=1)

          logits.append(distr.probs)


          input = self.sender_embedding(x)
          sequence.append(x)

      sequence = torch.stack(sequence).permute(1, 0)
      logits = torch.stack(logits).permute(1,2, 0)
      entropy = torch.stack(entropy).permute(1, 0)

      if self.force_eos:
          zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

          sequence = torch.cat([sequence, zeros.long()], dim=1)
          logits = torch.cat([logits, zeros], dim=1)
          entropy = torch.cat([entropy, zeros], dim=1)

      return sequence, logits, entropy


class AgentPol(nn.Module):

    """
    AgentBaseline is composed of a couple of modalities:
        - sender
        - receiver
    In AgentBaseline, Sender and Receiver parts are independent
    """

    def __init__(self,
                n_features,
                vocab_size,
                max_len,
                embed_dim,
                hidden_size,
                sender_cell,
                receiver_cell,
                sender_num_layers,
                receiver_num_layers,
                force_eos):
        super(AgentPol, self).__init__()

        # Common to sender and receiver
        self.force_eos = force_eos

        self.max_len = max_len
        if force_eos:
            self.max_len -= 1

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.hidden_size=hidden_size
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))

        cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'lstm': nn.LSTMCell}

        # Memory
        self.mem={}
        self.w_mem={}
        self.est_policy={}

        for k in range(n_features):
            self.mem[k]=[]
            self.w_mem[k]=[]
            self.est_policy[k]=torch.zeros([self.max_len,self.vocab_size]).to("cuda")

        # Sender
        self.agent_sender = nn.Linear(n_features, hidden_size) #nn.Linear(n_features, n_hidden)
        self.sender_cells = None
        self.sender_num_layers = sender_num_layers
        self.sender_norm_h = nn.LayerNorm(hidden_size)
        self.sender_norm_c = nn.LayerNorm(hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.sender_embedding = nn.Embedding(vocab_size, embed_dim)

        sender_cell = sender_cell.lower()

        if sender_cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {sender_cell}")

        cell_type = cell_types[sender_cell]
        self.sender_cells = nn.ModuleList([
            cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
            cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.sender_num_layers)])

        self.reset_parameters()

        # Receiver
        self.agent_receiver = nn.Linear(hidden_size, n_features) #nn.Linear(n_hidden, n_features)
        self.receiver_cells = None
        self.receiver_num_layers = receiver_num_layers
        self.receiver_norm_h = nn.LayerNorm(hidden_size)
        self.receiver_norm_c = nn.LayerNorm(hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.receiver_embedding = nn.Embedding(vocab_size, embed_dim)

        receiver_cell = receiver_cell.lower()

        if receiver_cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {receiver_cell}")

        cell_type = cell_types[receiver_cell]
        self.receiver_cells = nn.ModuleList([
            cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
            cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.receiver_num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def send(self, x, eval=False):
        prev_hidden = [self.agent_sender(x)]
        prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers - 1)])

        prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers)]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.sender_cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    h_t = self.sender_norm_h(h_t)
                    c_t = self.sender_norm_c(c_t)
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                    h_t = self.sender_norm_h(h_t)
                prev_hidden[i] = h_t
                input = h_t


            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training and not eval:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)

            logits.append(distr.log_prob(x))

            input = self.sender_embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)
            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy

    def receive(self,message, receiver_input, message_lengths):

      if message_lengths is None:
        message_lengths=find_lengths(message)

      prev_hidden = [torch.zeros((message.size(0),self.hidden_size)).to("cuda")]
      prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.receiver_num_layers - 1)])

      prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.receiver_num_layers)]  # only used for LSTM

      inputs = self.receiver_embedding(message)

      sequence = []
      logits = []
      entropy = []

      for step in range(self.max_len):

          input=inputs[:,step,:]

          for i, layer in enumerate(self.receiver_cells):
              if isinstance(layer, nn.LSTMCell):
                  h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                  h_t = self.receiver_norm_h(h_t)
                  c_t = self.receiver_norm_c(c_t)
                  prev_c[i] = c_t
              else:
                  h_t = layer(input, prev_hidden[i])
                  h_t = self.norm_h(h_t)
              prev_hidden[i] = h_t


          #step_logits = F.log_softmax(self.agent_receiver(h_t,None), dim=1)
          agent_output = self.agent_receiver(h_t)
          log = torch.zeros(agent_output.size(0)).to(agent_output.device)
          ent = log

          logits.append(log)
          entropy.append(ent)
          sequence.append(agent_output)

      sequence = torch.stack(sequence).permute(1, 0, 2)
      logits = torch.stack(logits).permute(1, 0)
      entropy = torch.stack(entropy).permute(1, 0)

      # Here choose EOS
      #sequence=sequence[:,-1,:]
      #logits=logits[:,-1]
      #entropy=entropy[:,-1]

      output=[]
      for j in range(sequence.size(0)):
        output.append(sequence[j,message_lengths[j]-1,:])

      output=torch.stack(output)
      logits=logits[:,-1]
      entropy=entropy[:,-1]

      return output, logits, entropy

    def imitate(self,x):

      prev_hidden = [self.agent_sender(x)]
      prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers - 1)])

      prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers)]  # only used for LSTM

      input = torch.stack([self.sos_embedding] * x.size(0))

      sequence = []
      logits = []
      entropy = []

      for step in range(self.max_len):
          for i, layer in enumerate(self.sender_cells):
              if isinstance(layer, nn.LSTMCell):
                  h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                  h_t = self.sender_norm_h(h_t)
                  c_t = self.sender_norm_c(c_t)
                  prev_c[i] = c_t
              else:
                  h_t = layer(input, prev_hidden[i])
                  h_t = self.sender_norm_h(h_t)
              prev_hidden[i] = h_t
              input = h_t


          step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)

          distr = Categorical(logits=step_logits)
          entropy.append(distr.entropy())

          if self.training:
              x = distr.sample()
          else:
              x = step_logits.argmax(dim=1)

          logits.append(distr.probs)


          input = self.sender_embedding(x)
          sequence.append(x)

      sequence = torch.stack(sequence).permute(1, 0)
      logits = torch.stack(logits).permute(1,2, 0)
      entropy = torch.stack(entropy).permute(1, 0)

      if self.force_eos:
          zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

          sequence = torch.cat([sequence, zeros.long()], dim=1)
          logits = torch.cat([logits, zeros], dim=1)
          entropy = torch.cat([entropy, zeros], dim=1)

      return sequence, logits, entropy


class AgentSharedRNN(nn.Module):

    """
    AgentBaseline is composed of a couple of modalities:
        - sender
        - receiver
    In AgentBaseline, Sender and Receiver parts are independent
    """

    def __init__(self,
                vocab_size,
                max_len,
                embed_dim,
                hidden_size,
                cell,
                num_layers,
                force_eos):
        super(AgentSharedRNN, self).__init__()

        self.agent_receiver = nn.Linear(n_hidden, n_features) #nn.Linear(n_hidden, n_features)
        self.agent_sender = nn.Linear(n_features, n_hidden) #nn.Linear(n_features, n_hidden)

        self.force_eos = force_eos

        self.max_len = max_len
        if force_eos:
            self.max_len -= 1

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.norm_h = nn.LayerNorm(hidden_size)
        self.norm_c = nn.LayerNorm(hidden_size)
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cells = None
        self.hidden_size=hidden_size

        cell = cell.lower()
        cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'lstm': nn.LSTMCell}

        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        cell_type = cell_types[cell]
        self.cells = nn.ModuleList([
            cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
            cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def send(self, x):
        prev_hidden = [self.agent_sender(x)]
        prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)])

        prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    h_t = self.norm_h(h_t)
                    c_t = self.norm_c(c_t)
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                    h_t = self.norm_h(h_t)
                prev_hidden[i] = h_t
                input = h_t


            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)

            logits.append(distr.log_prob(x))

            input = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy

    def receive(self,message, receiver_input, message_lengths):

      if message_lengths is None:
        message_lengths=find_lengths(message)

      prev_hidden = [torch.zeros((message.size(0),self.hidden_size)).to("cuda")]
      prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)])

      prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)]  # only used for LSTM

      inputs = self.embedding(message)

      sequence = []
      logits = []
      entropy = []

      for step in range(self.max_len):

          input=inputs[:,step,:]

          for i, layer in enumerate(self.cells):
              if isinstance(layer, nn.LSTMCell):
                  h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                  h_t = self.norm_h(h_t)
                  c_t = self.norm_c(c_t)
                  prev_c[i] = c_t
              else:
                  h_t = layer(input, prev_hidden[i])
                  h_t = self.norm_h(h_t)
              prev_hidden[i] = h_t


          #step_logits = F.log_softmax(self.agent_receiver(h_t,None), dim=1)
          agent_output = self.agent_receiver(h_t, None)
          log = torch.zeros(agent_output.size(0)).to(agent_output.device)
          ent = log

          #distr = Categorical(logits=step_logits)
          #entropy.append(distr.entropy())
          #x=step_logits.argmax(dim=1)

          logits.append(log)
          entropy.append(ent)
          sequence.append(agent_output)

      sequence = torch.stack(sequence).permute(1, 0, 2)
      logits = torch.stack(logits).permute(1, 0)
      entropy = torch.stack(entropy).permute(1, 0)

      # Here choose EOS
      #sequence=sequence[:,-1,:]
      #logits=logits[:,-1]
      #entropy=entropy[:,-1]

      output=[]
      for j in range(sequence.size(0)):
        output.append(sequence[j,message_lengths[j]-1,:])

      output=torch.stack(output)
      logits=logits[:,-1]
      entropy=entropy[:,-1]

      return output, logits, entropy

    def imitate(self,x):

      prev_hidden = [self.agent_sender(x)]
      prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)])

      prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)]  # only used for LSTM

      input = torch.stack([self.sos_embedding] * x.size(0))

      sequence = []
      logits = []
      entropy = []

      for step in range(self.max_len):
          for i, layer in enumerate(self.cells):
              if isinstance(layer, nn.LSTMCell):
                  h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                  h_t = self.norm_h(h_t)
                  c_t = self.norm_c(c_t)
                  prev_c[i] = c_t
              else:
                  h_t = layer(input, prev_hidden[i])
                  h_t = self.norm_h(h_t)
              prev_hidden[i] = h_t
              input = h_t


          step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)

          distr = Categorical(logits=step_logits)
          entropy.append(distr.entropy())

          if self.training:
              x = distr.sample()
          else:
              x = step_logits.argmax(dim=1)

          logits.append(distr.probs)


          input = self.embedding(x)
          sequence.append(x)

      sequence = torch.stack(sequence).permute(1, 0)

      logits = torch.stack(logits).permute(1,2, 0)

      entropy = torch.stack(entropy).permute(1, 0)

      if self.force_eos:
          zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

          sequence = torch.cat([sequence, zeros.long()], dim=1)
          logits = torch.cat([logits, zeros], dim=1)
          entropy = torch.cat([entropy, zeros], dim=1)

      return sequence, logits, entropy

class AgentSharedEmbedding(nn.Module):

    """
    AgentBaseline is composed of a couple of modalities:
        - sender
        - receiver
    In AgentBaseline, Sender and Receiver parts are independent
    """

    def __init__(self,
                n_features,
                vocab_size,
                max_len,
                embed_dim,
                hidden_size,
                cell_sender,
                cell_receiver,
                num_layers_sender,
                num_layers_receiver,
                force_eos):
        super(AgentSharedEmbedding, self).__init__()

        assert embed_dim==hidden_size, "embed_dim has to be equal to hidden_size"

        self.FC_features = nn.Linear(n_features,hidden_size,bias=False) #nn.Linear(n_hidden, n_features)
        self.FC_vocabulary = nn.Linear(hidden_size,vocab_size,bias=False)

        self.force_eos = force_eos

        self.max_len = max_len
        if force_eos:
            self.max_len -= 1

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding_speaker = nn.Embedding(vocab_size, embed_dim)
        self.embedding_listener = nn.Embedding(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.norm_h = nn.LayerNorm(hidden_size)
        self.norm_c = nn.LayerNorm(hidden_size)
        self.vocab_size = vocab_size
        self.num_layers_sender = num_layers_sender
        self.num_layers_receiver = num_layers_receiver
        self.cells = None
        self.hidden_size=hidden_size

        cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'lstm': nn.LSTMCell}

        cell_sender = cell_sender.lower()

        if cell_sender not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        cell_type = cell_types[cell_sender]
        self.cells_sender = nn.ModuleList([
            cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
            cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.num_layers_sender)])

        cell_receiver = cell_receiver.lower()

        if cell_receiver not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        cell_type = cell_types[cell_receiver]
        self.cells_receiver = nn.ModuleList([
            cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
            cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.num_layers_receiver)])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def send(self, x, eval=False):
        prev_hidden = [self.FC_features(x)]
        prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers_sender - 1)])

        prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers_sender)]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells_sender):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    h_t = self.norm_h(h_t)
                    c_t = self.norm_c(c_t)
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                    h_t = self.norm_h(h_t)
                prev_hidden[i] = h_t
                input = h_t


            step_logits = F.log_softmax(self.FC_vocabulary(h_t), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training and not eval:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)

            logits.append(distr.log_prob(x))

            #input = F.embedding(x,weight=self.FC_vocabulary.weight)
            input = self.embedding_speaker(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy

    def receive(self,message, receiver_input, message_lengths):

      if message_lengths is None:
        message_lengths=find_lengths(message)

      prev_hidden = [torch.zeros((message.size(0),self.hidden_size)).to("cuda")]
      prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers_receiver - 1)])

      prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers_receiver)]  # only used for LSTM

      #inputs = self.embedding(message)
      inputs = F.embedding(message,weight=self.FC_vocabulary.weight)

      sequence = []
      logits = []
      entropy = []

      for step in range(self.max_len):

          input=inputs[:,step,:]

          for i, layer in enumerate(self.cells_receiver):
              if isinstance(layer, nn.LSTMCell):
                  h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                  h_t = self.norm_h(h_t)
                  c_t = self.norm_c(c_t)
                  prev_c[i] = c_t
              else:
                  h_t = layer(input, prev_hidden[i])
                  h_t = self.norm_h(h_t)
              prev_hidden[i] = h_t


          #step_logits = F.log_softmax(self.agent_receiver(h_t,None), dim=1)
          agent_output = F.log_softmax(F.linear(h_t,weight=self.FC_features.weight.T), dim=1)
          log = torch.zeros(agent_output.size(0)).to(agent_output.device)
          ent = log

          #distr = Categorical(logits=step_logits)
          #entropy.append(distr.entropy())
          #x=step_logits.argmax(dim=1)

          logits.append(log)
          entropy.append(ent)
          sequence.append(agent_output)

      sequence = torch.stack(sequence).permute(1, 0, 2)
      logits = torch.stack(logits).permute(1, 0)
      entropy = torch.stack(entropy).permute(1, 0)

      # Here choose EOS
      #sequence=sequence[:,-1,:]
      #logits=logits[:,-1]
      #entropy=entropy[:,-1]

      output=[]
      for j in range(sequence.size(0)):
        output.append(sequence[j,message_lengths[j]-1,:])

      output=torch.stack(output)
      logits=logits[:,-1]
      entropy=entropy[:,-1]

      return output, logits, entropy

    def imitate(self,x):

      prev_hidden = [self.FC_features(x)]
      prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers_sender - 1)])

      prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers_sender)]  # only used for LSTM

      input = torch.stack([self.sos_embedding] * x.size(0))

      sequence = []
      logits = []
      entropy = []

      for step in range(self.max_len):
          for i, layer in enumerate(self.cells_sender):
              if isinstance(layer, nn.LSTMCell):
                  h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                  h_t = self.norm_h(h_t)
                  c_t = self.norm_c(c_t)
                  prev_c[i] = c_t
              else:
                  h_t = layer(input, prev_hidden[i])
                  h_t = self.norm_h(h_t)
              prev_hidden[i] = h_t
              input = h_t


          step_logits = F.log_softmax(self.FC_vocabulary(h_t), dim=1)

          distr = Categorical(logits=step_logits)
          entropy.append(distr.entropy())

          if self.training:
              x = distr.sample()
          else:
              x = step_logits.argmax(dim=1)

          logits.append(distr.probs)


          #input = F.embedding(x,weight=self.FC_vocabulary.weight)
          input = self.embedding_speaker(x)
          sequence.append(x)

      sequence = torch.stack(sequence).permute(1, 0)

      logits = torch.stack(logits).permute(1,2, 0)

      entropy = torch.stack(entropy).permute(1, 0)

      if self.force_eos:
          zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

          sequence = torch.cat([sequence, zeros.long()], dim=1)
          logits = torch.cat([logits, zeros], dim=1)
          entropy = torch.cat([entropy, zeros], dim=1)

      return sequence, logits, entropy


class DialogReinforce(nn.Module):

    """
    DialogReinforce implements the Dialog game
    """

    def __init__(self,
                 Agent_1,
                 Agent_2,
                 loss_understanding,
                 loss_imitation,
                 optim_params,
                 loss_weights,
                 device,
                 baseline_mode="new",
                 reward_mode="neg_loss"):
        """
        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":0.,
                      "receiver_entropy_coeff_1":0.,
                      "sender_entropy_coeff_2":0.,
                      "receiver_entropy_coeff_2":0.}

        loss_weights={"self":1.,
                      "cross":1.,
                      "imitation":1.,
                      "length_regularization":0.,
                      "entropy_regularization":1.}
        """
        super(DialogReinforce, self).__init__()
        self.agent_1 = Agent_1
        self.agent_2 = Agent_2
        self.optim_params = optim_params
        self.loss_understanding = loss_understanding
        self.loss_message_imitation = loss_imitation
        self.loss_weights = loss_weights
        self.baseline_mode=baseline_mode
        self.reward_mode=reward_mode
        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.device=device
        self.agent_1.to(self.device)
        self.agent_2.to(self.device)

    def forward(self,
                sender_input,
                unused_labels,
                direction,
                receiver_input=None):

        """
        Inputs:
        - direction : "1->2" or "2->1"
        """

        sender_input=sender_input.to(self.device)

        if direction=="1->2":
            agent_sender=self.agent_1
            agent_receiver=self.agent_2
            sender_id=1
            receiver_id=2
        else:
            agent_sender=self.agent_2
            agent_receiver=self.agent_1
            sender_id=2
            receiver_id=1

        " 1. Agent actions "
        # Message sending
        message, log_prob_s,whole_log_prob_s, entropy_s = agent_sender.send(sender_input,return_policies=True)
        message_lengths = find_lengths(message)
        # Cross listening
        receiver_output_cross, log_prob_r_cross, entropy_r_cross = agent_receiver.receive_2(message, receiver_input, message_lengths)
        # Self listening
        receiver_output_self, log_prob_r_self, entropy_r_self = agent_sender.receive_2(message, receiver_input, message_lengths)
        # Imitation
        #candidates_cross=receiver_output_cross.argmax(dim=1)
        #message_reconstruction, prob_reconstruction, _ = agent_receiver.imitate(sender_input)
        message_to_imitate, _, _ = agent_receiver.send(sender_input,eval=True)
        message_to_imitate_lengths = find_lengths(message_to_imitate)
        send_output, _, _ = agent_sender.receive_2(message_to_imitate, receiver_input, message_to_imitate_lengths)
        message_reconstruction, prob_reconstruction, _ = agent_sender.imitate(sender_input)

        "2. Losses computation"
        loss_self, rest_self = self.loss_understanding(sender_input,receiver_output_self)
        loss_cross, rest_cross = self.loss_understanding(sender_input,receiver_output_cross)
        #loss_imitation, rest_imitation = self.loss_message_imitation(message,prob_reconstruction,message_lengths)
        loss_imitation, rest_imitation = self.loss_message_imitation(message_to_imitate,prob_reconstruction,message_to_imitate_lengths)
        _, rest_und_cross = self.loss_understanding(sender_input,send_output)
        prob_conf=torch.exp((sender_input*F.log_softmax(send_output,dim=1)).sum(1))
        loss_imitation=loss_imitation*prob_conf

        # Average loss. Rk. Sortir loss_imitation de cette somme
        loss = self.loss_weights["self"]*loss_self + self.loss_weights["cross"]*loss_cross + self.loss_weights["imitation"]*loss_imitation
        loss /= (self.loss_weights["self"]+self.loss_weights["cross"]+self.loss_weights["imitation"])

        # Reward
        if self.reward_mode=="neg_loss":
            reward_self = -loss_self.detach()
            reward_cross = -loss_cross.detach()
        elif self.reward_mode=="proba":
            reward_self = torch.exp(-loss_self.detach())
            reward_cross = torch.exp(-loss_cross.detach())

        "3. Entropy + length Regularization"

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r_self)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r_self)


        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.optim_params["sender_entropy_coeff_{}".format(sender_id)] #+ entropy_r_12.mean() * self.receiver_entropy_coeff_1

        log_prob = effective_log_prob_s #+ log_prob_r_12

        length_loss = message_lengths.float() * self.optim_params["length_cost"]

        "4. Variance reduction"
        if self.baseline_mode=="original":
            policy_loss_self = -((loss_self.detach() - self.mean_baseline['loss_self_{}'.format(sender_id)]) * log_prob).mean()
            policy_loss_cross = -((loss_cross.detach() - self.mean_baseline['loss_cross_{}'.format(sender_id)]) * log_prob).mean()
            policy_loss_imitation = ((loss_imitation.detach() - self.mean_baseline['loss_imitation_{}'.format(sender_id)]) * log_prob).mean()
            policy_length_loss = ((length_loss.float() - self.mean_baseline['length_{}'.format(sender_id)]) * effective_log_prob_s).mean()

        elif self.baseline_mode=="new":

            policy_loss_self = -((reward_self - reward_self.mean())/(reward_self.std()) * log_prob).mean()
            policy_loss_cross = -((reward_cross - reward_cross.mean())/(reward_cross.std())  * log_prob).mean()
            policy_loss_imitation = ((loss_imitation.detach() - loss_imitation.detach().mean())  * log_prob).mean()
            policy_length_loss = ((length_loss.float() - length_loss.float().mean())  * effective_log_prob_s).mean()

        " 5. Final loss"
        policy_loss = self.loss_weights["self"]*policy_loss_self + self.loss_weights["cross"]*policy_loss_cross + self.loss_weights["imitation"]*policy_loss_imitation
        policy_loss /= (self.loss_weights["self"]+self.loss_weights["cross"]+self.loss_weights["imitation"])

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss_self_{}'.format(sender_id), loss_self)
            self.update_baseline('loss_cross_{}'.format(sender_id), loss_cross)
            self.update_baseline('loss_imitation_{}'.format(sender_id), loss_imitation)
            self.update_baseline('length_{}'.format(sender_id), length_loss)

        "6. Store results"
        rest={}
        rest['loss'] = optimized_loss.detach().item()
        rest['loss_{}'.format(sender_id)] = optimized_loss.detach().item()
        rest['sender_entropy_{}'.format(sender_id)] = entropy_s.mean().item()
        rest['mean_length_{}'.format(sender_id)] = message_lengths.float().mean().item()
        rest['loss_self_{}{}'.format(sender_id,sender_id)] = loss_self.mean().item()
        rest['loss_cross_{}{}'.format(sender_id,receiver_id)] = loss_cross.mean().item()
        rest['loss_imitation_{}{}'.format(receiver_id,sender_id)] = loss_imitation.mean().item()
        rest['acc_self_{}{}'.format(sender_id,sender_id)]=rest_self['acc'].mean().item()
        rest['acc_cross_{}{}'.format(sender_id,receiver_id)]=rest_cross['acc'].mean().item()
        rest['acc_imitation_{}{}'.format(receiver_id,sender_id)]=rest_imitation['acc_imitation'].mean().item()
        rest['reinforce_term_{}'.format(sender_id)]=policy_loss.detach().item()
        rest['baseline_term_{}'.format(sender_id)]=(policy_loss/log_prob.mean()).detach().item()
        rest['policy_{}'.format(sender_id)]=whole_log_prob_s.detach()

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]

class DialogReinforceSingleListener(nn.Module):

    """
    DialogReinforce implements the Dialog game
    """

    def __init__(self,
                 Agent_1,
                 Agent_2,
                 loss_understanding,
                 loss_imitation,
                 optim_params,
                 loss_weights,
                 device,
                 baseline_mode="new",
                 reward_mode="neg_loss"):
        """
        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":0.,
                      "receiver_entropy_coeff_1":0.,
                      "sender_entropy_coeff_2":0.,
                      "receiver_entropy_coeff_2":0.}

        loss_weights={"self":1.,
                      "cross":1.,
                      "imitation":1.,
                      "length_regularization":0.,
                      "entropy_regularization":1.}
        """
        super(DialogReinforceSingleListener, self).__init__()
        self.agent_1 = Agent_1
        self.agent_2 = Agent_2
        self.optim_params = optim_params
        self.loss_understanding = loss_understanding
        self.loss_message_imitation = loss_imitation
        self.loss_weights = loss_weights
        self.baseline_mode=baseline_mode
        self.reward_mode=reward_mode
        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.device=device
        self.agent_1.to(self.device)
        self.agent_2.to(self.device)

    def forward(self,
                sender_input,
                unused_labels,
                direction,
                receiver_input=None):

        """
        Inputs:
        - direction : "1->2" or "2->1"
        """

        sender_input=sender_input.to(self.device)

        if direction=="1->2":
            agent_sender=self.agent_1
            agent_receiver=self.agent_2
            sender_id=1
            receiver_id=2
            self.loss_weights={"self":1.,"cross":0.,"imitation":0.}
        else:
            agent_sender=self.agent_2
            agent_receiver=self.agent_1
            sender_id=2
            receiver_id=1
            self.loss_weights={"self":0.,"cross":1.,"imitation":0.}

        " 1. Agent actions "
        # Message sending
        message, log_prob_s,whole_log_prob_s, entropy_s = agent_sender.send(sender_input,return_policies=True)
        message_lengths = find_lengths(message)
        # Cross listening
        receiver_output_cross, log_prob_r_cross, entropy_r_cross = agent_receiver.receive_2(message, receiver_input, message_lengths)
        # Self listening
        receiver_output_self, log_prob_r_self, entropy_r_self = agent_sender.receive_2(message, receiver_input, message_lengths)
        # Imitation
        #candidates_cross=receiver_output_cross.argmax(dim=1)
        #message_reconstruction, prob_reconstruction, _ = agent_receiver.imitate(sender_input)
        message_to_imitate, _, _ = agent_receiver.send(sender_input,eval=True)
        message_to_imitate_lengths = find_lengths(message_to_imitate)
        send_output, _, _ = agent_sender.receive_2(message_to_imitate, receiver_input, message_to_imitate_lengths)
        message_reconstruction, prob_reconstruction, _ = agent_sender.imitate(sender_input)

        "2. Losses computation"
        loss_self, rest_self = self.loss_understanding(sender_input,receiver_output_self)
        loss_cross, rest_cross = self.loss_understanding(sender_input,receiver_output_cross)
        #loss_imitation, rest_imitation = self.loss_message_imitation(message,prob_reconstruction,message_lengths)
        loss_imitation, rest_imitation = self.loss_message_imitation(message_to_imitate,prob_reconstruction,message_to_imitate_lengths)
        _, rest_und_cross = self.loss_understanding(sender_input,send_output)
        prob_conf=torch.exp((sender_input*F.log_softmax(send_output,dim=1)).sum(1))
        loss_imitation=loss_imitation*prob_conf

        # Average loss. Rk. Sortir loss_imitation de cette somme
        loss = self.loss_weights["self"]*loss_self + self.loss_weights["cross"]*loss_cross + self.loss_weights["imitation"]*loss_imitation
        loss /= (self.loss_weights["self"]+self.loss_weights["cross"]+self.loss_weights["imitation"])

        # Reward
        if self.reward_mode=="neg_loss":
            reward_self = -loss_self.detach()
            reward_cross = -loss_cross.detach()
        elif self.reward_mode=="proba":
            reward_self = torch.exp(-loss_self.detach())
            reward_cross = torch.exp(-loss_cross.detach())

        "3. Entropy + length Regularization"

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r_self)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r_self)


        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.optim_params["sender_entropy_coeff_{}".format(sender_id)] #+ entropy_r_12.mean() * self.receiver_entropy_coeff_1

        log_prob = effective_log_prob_s #+ log_prob_r_12

        length_loss = message_lengths.float() * self.optim_params["length_cost"]

        "4. Variance reduction"
        if self.baseline_mode=="original":
            policy_loss_self = -((loss_self.detach() - self.mean_baseline['loss_self_{}'.format(sender_id)]) * log_prob).mean()
            policy_loss_cross = -((loss_cross.detach() - self.mean_baseline['loss_cross_{}'.format(sender_id)]) * log_prob).mean()
            policy_loss_imitation = ((loss_imitation.detach() - self.mean_baseline['loss_imitation_{}'.format(sender_id)]) * log_prob).mean()
            policy_length_loss = ((length_loss.float() - self.mean_baseline['length_{}'.format(sender_id)]) * effective_log_prob_s).mean()

        elif self.baseline_mode=="new":

            policy_loss_self = -((reward_self - reward_self.mean())/(reward_self.std()) * log_prob).mean()
            policy_loss_cross = -((reward_cross - reward_cross.mean())/(reward_cross.std())  * log_prob).mean()
            policy_loss_imitation = ((loss_imitation.detach() - loss_imitation.detach().mean())  * log_prob).mean()
            policy_length_loss = ((length_loss.float() - length_loss.float().mean())  * effective_log_prob_s).mean()

        " 5. Final loss"
        policy_loss = self.loss_weights["self"]*policy_loss_self + self.loss_weights["cross"]*policy_loss_cross + self.loss_weights["imitation"]*policy_loss_imitation
        policy_loss /= (self.loss_weights["self"]+self.loss_weights["cross"]+self.loss_weights["imitation"])

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss_self_{}'.format(sender_id), loss_self)
            self.update_baseline('loss_cross_{}'.format(sender_id), loss_cross)
            self.update_baseline('loss_imitation_{}'.format(sender_id), loss_imitation)
            self.update_baseline('length_{}'.format(sender_id), length_loss)

        "6. Store results"
        rest={}
        rest['loss'] = optimized_loss.detach().item()
        rest['loss_{}'.format(sender_id)] = optimized_loss.detach().item()
        rest['sender_entropy_{}'.format(sender_id)] = entropy_s.mean().item()
        rest['mean_length_{}'.format(sender_id)] = message_lengths.float().mean().item()
        rest['loss_self_{}{}'.format(sender_id,sender_id)] = loss_self.mean().item()
        rest['loss_cross_{}{}'.format(sender_id,receiver_id)] = loss_cross.mean().item()
        rest['loss_imitation_{}{}'.format(receiver_id,sender_id)] = loss_imitation.mean().item()
        rest['acc_self_{}{}'.format(sender_id,sender_id)]=rest_self['acc'].mean().item()
        rest['acc_cross_{}{}'.format(sender_id,receiver_id)]=rest_cross['acc'].mean().item()
        rest['acc_imitation_{}{}'.format(receiver_id,sender_id)]=rest_imitation['acc_imitation'].mean().item()
        rest['reinforce_term_{}'.format(sender_id)]=policy_loss.detach().item()
        rest['baseline_term_{}'.format(sender_id)]=(policy_loss/log_prob.mean()).detach().item()
        rest['policy_{}'.format(sender_id)]=whole_log_prob_s.detach()

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]

class DialogReinforceCompositionality(nn.Module):

    """
    DialogReinforce implements the Dialog game
    """

    def __init__(self,
                 Agent_1,
                 Agent_2,
                 n_attributes,
                 n_values,
                 loss_understanding,
                 optim_params,
                 loss_weights,
                 device,
                 baseline_mode="new",
                 reward_mode="neg_loss"):
        """
        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":0.,
                      "receiver_entropy_coeff_1":0.,
                      "sender_entropy_coeff_2":0.,
                      "receiver_entropy_coeff_2":0.}

        loss_weights={"self":1.,
                      "cross":1.,
                      "imitation":1.,
                      "length_regularization":0.,
                      "entropy_regularization":1.}
        """
        super(DialogReinforceCompositionality, self).__init__()
        self.agent_1 = Agent_1
        self.agent_2 = Agent_2
        self.n_attributes=n_attributes
        self.n_values=n_values
        self.optim_params = optim_params
        self.loss_understanding = loss_understanding
        self.loss_weights = loss_weights
        self.baseline_mode=baseline_mode
        self.reward_mode=reward_mode
        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.device=device
        self.agent_1.to(self.device)
        self.agent_2.to(self.device)

    def forward(self,
                sender_input,
                unused_labels,
                direction,
                receiver_input=None):

        """
        Inputs:
        - direction : "1->2" or "2->1"
        """

        sender_input=sender_input.to(self.device)

        if direction=="1->2":
            agent_sender=self.agent_1
            agent_receiver=self.agent_2
            sender_id=1
            receiver_id=2
        else:
            agent_sender=self.agent_2
            agent_receiver=self.agent_1
            sender_id=2
            receiver_id=1

        " 1. Agent actions "
        # Message sending
        message, log_prob_s,whole_log_prob_s, entropy_s = agent_sender.send(sender_input,return_policies=True)
        message_lengths = find_lengths(message)
        # Cross listening
        receiver_output_cross, log_prob_r_cross, entropy_r_cross = agent_receiver.receive(message, receiver_input, message_lengths)
        # Self listening
        receiver_output_self, log_prob_r_self, entropy_r_self = agent_sender.receive(message, receiver_input, message_lengths)
        # Imitation
        # NO IMITATION


        "2. Losses computation"
        loss_cross, rest_cross = self.loss_understanding(sender_input, receiver_output_cross,self.n_attributes,self.n_values)

        loss_self, rest_self = self.loss_understanding(sender_input, receiver_output_self,self.n_attributes,self.n_values)

        # Average loss. Rk. Sortir loss_imitation de cette somme
        loss = self.loss_weights["self"]*loss_self + self.loss_weights["cross"]*loss_cross
        loss /= (self.loss_weights["self"]+self.loss_weights["cross"])

        # Reward
        if self.reward_mode=="neg_loss":
            reward_self = -loss_self.detach()
            reward_cross = -loss_cross.detach()
        elif self.reward_mode=="proba":
            reward_self = torch.exp(-loss_self.detach())
            reward_cross = torch.exp(-loss_cross.detach())
        elif self.reward_mode=="dense":
            reward_self = 1.*(rest_self["acc"].sum(1)==self.n_attributes).detach()
            reward_cross = 1.*(rest_cross["acc"].sum(1)==self.n_attributes).detach()

        "3. Entropy + length Regularization"

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r_self.mean(1))

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r_self.mean(1))


        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.optim_params["sender_entropy_coeff_{}".format(sender_id)] #+ entropy_r_12.mean() * self.receiver_entropy_coeff_1

        log_prob = effective_log_prob_s #+ log_prob_r_12

        length_loss = message_lengths.float() * self.optim_params["length_cost"]

        "4. Variance reduction"
        if self.baseline_mode=="original":
            policy_loss_self = -((loss_self.detach() - self.mean_baseline['loss_self_{}'.format(sender_id)]) * log_prob).mean()
            policy_loss_cross = -((loss_cross.detach() - self.mean_baseline['loss_cross_{}'.format(sender_id)]) * log_prob).mean()
            policy_length_loss = ((length_loss.float() - self.mean_baseline['length_{}'.format(sender_id)]) * effective_log_prob_s).mean()

        elif self.baseline_mode=="new":

            policy_loss_self = -((reward_self - reward_self.mean())/(reward_self.std()) * log_prob).mean()
            policy_loss_cross = -((reward_cross - reward_cross.mean())/(reward_cross.std())  * log_prob).mean()
            policy_length_loss = ((length_loss.float() - length_loss.float().mean())  * effective_log_prob_s).mean()

        " 5. Final loss"
        policy_loss = self.loss_weights["self"]*policy_loss_self + self.loss_weights["cross"]*policy_loss_cross
        policy_loss /= (self.loss_weights["self"]+self.loss_weights["cross"])

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss_self_{}'.format(sender_id), loss_self)
            self.update_baseline('loss_cross_{}'.format(sender_id), loss_cross)
            self.update_baseline('length_{}'.format(sender_id), length_loss)

        "6. Store results"
        rest={}
        rest['loss'] = optimized_loss.detach().item()
        rest['loss_{}'.format(sender_id)] = optimized_loss.detach().item()
        rest['sender_entropy_{}'.format(sender_id)] = entropy_s.mean().item()
        rest['mean_length_{}'.format(sender_id)] = message_lengths.float().mean().item()
        rest['loss_self_{}{}'.format(sender_id,sender_id)] = loss_self.mean().item()
        rest['loss_cross_{}{}'.format(sender_id,receiver_id)] = loss_cross.mean().item()
        rest['acc_self_{}{}'.format(sender_id,sender_id)]=rest_self['acc'].mean().item()
        rest['acc_cross_{}{}'.format(sender_id,receiver_id)]=rest_cross['acc'].mean().item()
        rest['reinforce_term_{}'.format(sender_id)]=policy_loss.detach().item()
        rest['baseline_term_{}'.format(sender_id)]=(policy_loss/log_prob.mean()).detach().item()
        rest['policy_{}'.format(sender_id)]=whole_log_prob_s.detach()

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]


class DialogReinforceCompositionalitySingleListener(nn.Module):

    """
    DialogReinforce implements the Dialog game
    """

    def __init__(self,
                 Agent_1,
                 Agent_2,
                 n_attributes,
                 n_values,
                 loss_understanding,
                 optim_params,
                 loss_weights,
                 device,
                 baseline_mode="new",
                 reward_mode="neg_loss"):
        """
        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":0.,
                      "receiver_entropy_coeff_1":0.,
                      "sender_entropy_coeff_2":0.,
                      "receiver_entropy_coeff_2":0.}

        loss_weights={"self":1.,
                      "cross":1.,
                      "imitation":1.,
                      "length_regularization":0.,
                      "entropy_regularization":1.}
        """
        super(DialogReinforceCompositionalitySingleListener, self).__init__()
        self.agent_1 = Agent_1
        self.agent_2 = Agent_2
        self.n_attributes=n_attributes
        self.n_values=n_values
        self.optim_params = optim_params
        self.loss_understanding = loss_understanding
        self.loss_weights = loss_weights
        self.baseline_mode=baseline_mode
        self.reward_mode=reward_mode
        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.device=device
        self.agent_1.to(self.device)
        self.agent_2.to(self.device)

    def forward(self,
                sender_input,
                unused_labels,
                direction,
                receiver_input=None):

        """
        Inputs:
        - direction : "1->2" or "2->1"
        """

        sender_input=sender_input.to(self.device)

        if direction=="1->2":
            agent_sender=self.agent_1
            agent_receiver=self.agent_2
            sender_id=1
            receiver_id=2
            self.loss_weights={"self":1.,"cross":0.,"imitation":1.}
        else:
            agent_sender=self.agent_2
            agent_receiver=self.agent_1
            sender_id=2
            receiver_id=1
            self.loss_weights={"self":0.,"cross":1.,"imitation":1.}

        " 1. Agent actions "
        # Message sending
        message, log_prob_s,whole_log_prob_s, entropy_s = agent_sender.send(sender_input,return_policies=True)
        message_lengths = find_lengths(message)
        # Cross listening
        receiver_output_cross, log_prob_r_cross, entropy_r_cross = agent_receiver.receive(message, receiver_input, message_lengths)
        # Self listening
        receiver_output_self, log_prob_r_self, entropy_r_self = agent_sender.receive(message, receiver_input, message_lengths)
        # Imitation
        # NO IMITATION


        "2. Losses computation"
        loss_cross, rest_cross = self.loss_understanding(sender_input, receiver_output_cross,self.n_attributes,self.n_values)

        loss_self, rest_self = self.loss_understanding(sender_input, receiver_output_self,self.n_attributes,self.n_values)

        # Average loss. Rk. Sortir loss_imitation de cette somme
        loss = self.loss_weights["self"]*loss_self + self.loss_weights["cross"]*loss_cross
        loss /= (self.loss_weights["self"]+self.loss_weights["cross"])

        # Reward
        if self.reward_mode=="neg_loss":
            reward_self = -loss_self.detach()
            reward_cross = -loss_cross.detach()
        elif self.reward_mode=="proba":
            reward_self = torch.exp(-loss_self.detach())
            reward_cross = torch.exp(-loss_cross.detach())
        elif self.reward_mode=="dense":
            reward_self = 1.*(rest_self["acc"].sum(1)==self.n_attributes).detach()
            reward_cross = 1.*(rest_cross["acc"].sum(1)==self.n_attributes).detach()

        "3. Entropy + length Regularization"

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r_self.mean(1))

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r_self.mean(1))


        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.optim_params["sender_entropy_coeff_{}".format(sender_id)] #+ entropy_r_12.mean() * self.receiver_entropy_coeff_1

        log_prob = effective_log_prob_s #+ log_prob_r_12

        length_loss = message_lengths.float() * self.optim_params["length_cost"]

        "4. Variance reduction"
        if self.baseline_mode=="original":
            policy_loss_self = -((loss_self.detach() - self.mean_baseline['loss_self_{}'.format(sender_id)]) * log_prob).mean()
            policy_loss_cross = -((loss_cross.detach() - self.mean_baseline['loss_cross_{}'.format(sender_id)]) * log_prob).mean()
            policy_length_loss = ((length_loss.float() - self.mean_baseline['length_{}'.format(sender_id)]) * effective_log_prob_s).mean()

        elif self.baseline_mode=="new":

            policy_loss_self = -((reward_self - reward_self.mean())/(reward_self.std()) * log_prob).mean()
            policy_loss_cross = -((reward_cross - reward_cross.mean())/(reward_cross.std())  * log_prob).mean()
            policy_length_loss = ((length_loss.float() - length_loss.float().mean())  * effective_log_prob_s).mean()

        " 5. Final loss"
        policy_loss = self.loss_weights["self"]*policy_loss_self + self.loss_weights["cross"]*policy_loss_cross
        policy_loss /= (self.loss_weights["self"]+self.loss_weights["cross"])

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss_self_{}'.format(sender_id), loss_self)
            self.update_baseline('loss_cross_{}'.format(sender_id), loss_cross)
            self.update_baseline('length_{}'.format(sender_id), length_loss)

        "6. Store results"
        rest={}
        rest['loss'] = optimized_loss.detach().item()
        rest['loss_{}'.format(sender_id)] = optimized_loss.detach().item()
        rest['sender_entropy_{}'.format(sender_id)] = entropy_s.mean().item()
        rest['mean_length_{}'.format(sender_id)] = message_lengths.float().mean().item()
        rest['loss_self_{}{}'.format(sender_id,sender_id)] = loss_self.mean().item()
        rest['loss_cross_{}{}'.format(sender_id,receiver_id)] = loss_cross.mean().item()
        rest['acc_self_{}{}'.format(sender_id,sender_id)]=rest_self['acc'].mean().item()
        rest['acc_cross_{}{}'.format(sender_id,receiver_id)]=rest_cross['acc'].mean().item()
        rest['reinforce_term_{}'.format(sender_id)]=policy_loss.detach().item()
        rest['baseline_term_{}'.format(sender_id)]=(policy_loss/log_prob.mean()).detach().item()
        rest['policy_{}'.format(sender_id)]=whole_log_prob_s.detach()

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]


class DialogReinforceCompositionalityMultiAgent(nn.Module):

    """
    DialogReinforce implements the Dialog game
    """

    def __init__(self,
                 Agents,
                 n_attributes,
                 n_values,
                 loss_understanding,
                 optim_params,
                 loss_weights,
                 device,
                 baseline_mode="new",
                 reward_mode="neg_loss"):
        """
        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":0.,
                      "receiver_entropy_coeff_1":0.,
                      "sender_entropy_coeff_2":0.,
                      "receiver_entropy_coeff_2":0.}

        loss_weights={"self":1.,
                      "cross":1.,
                      "imitation":1.,
                      "length_regularization":0.,
                      "entropy_regularization":1.}
        """
        super(DialogReinforceCompositionalityMultiAgent, self).__init__()
        self.agents = Agents
        self.n_attributes=n_attributes
        self.n_values=n_values
        self.optim_params = optim_params
        self.loss_understanding = loss_understanding
        self.loss_weights = loss_weights
        self.baseline_mode=baseline_mode
        self.reward_mode=reward_mode
        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.device=device
        for agent in self.agents:
          self.agents[agent].to(self.device)

    def forward(self,
                sender_input,
                unused_labels,
                sender_id,
                receiver_ids,
                receiver_input=None,
                save_probs=None):

        """
        Inputs:
        - direction : N means "N->0"
        """

        sender_input=sender_input.to(self.device)

        "0. Get sender and receiver (id + optim info) for playing the game"
        # Get sender_id and sender information
        agent_sender = self.agents["agent_{}".format(sender_id)]
        loss_weights_sender = self.loss_weights["agent_{}".format(sender_id)]
        optim_params_sender = self.optim_params["agent_{}".format(sender_id)]

        # Get receiver information (receiver_id always 0)
        agent_receivers={"agent_{}".format(receiver_id):self.agents["agent_{}".format(receiver_id)] for receiver_id in receiver_ids}

        " 1. Agent actions and loss"
        # Message sending
        message, log_prob_s,whole_log_prob_s, entropy_s = agent_sender.send(sender_input,return_policies=True)
        message_lengths = find_lengths(message)

        # Self listening
        receiver_output_self, log_prob_r_self, entropy_r_self = agent_sender.receive(message, receiver_input, message_lengths)
        loss_self, rest_self = self.loss_understanding(sender_input, receiver_output_self,self.n_attributes,self.n_values)

        # Cross listening
        losses_cross={}
        restes_cross = {}
        if self.reward_mode=="dense":
            samples = {}
        for agent in agent_receivers:
            if self.reward_mode=="dense":
                sample, receiver_output_cross, log_prob_r_cross,whole_log_prob_r_cross, entropy_r_cross = agent_receivers[agent].receive(message, receiver_input, message_lengths,return_sample=True)
                samples[agent] = sample
            else:
                receiver_output_cross, log_prob_r_cross,whole_log_prob_r_cross, entropy_r_cross = agent_receivers[agent].receive(message, receiver_input, message_lengths,return_policies=True)
            loss_cross, rest_cross = self.loss_understanding(sender_input, receiver_output_cross,self.n_attributes,self.n_values)
            losses_cross[agent] = loss_cross
            restes_cross[agent] = rest_cross

            if save_probs:
                np.save(save_probs+"_receiver_probs_"+agent+".npy",whole_log_prob_r_cross.cpu().numpy())
        # Imitation
        # NO IMITATION


        "2. Reward computation"

        loss_cross= torch.stack([losses_cross[agent] for agent in losses_cross]).mean(0)# MEAN ACROSS AXIS

        # Average loss. Rk. Sortir loss_imitation de cette somme
        loss = loss_weights_sender["self"]*loss_self + loss_weights_sender["cross"]*loss_cross
        loss /= (loss_weights_sender["self"]+loss_weights_sender["cross"])

        # Reward
        if self.reward_mode=="neg_loss":
            reward_self = -loss_self.detach()
            reward_cross = -loss_cross.detach()
        elif self.reward_mode=="proba":
            reward_self = torch.exp(-loss_self.detach())
            reward_cross = torch.exp(-loss_cross.detach())
        elif self.reward_mode=="dense":
            reward_self = 1.*(rest_self["acc"].sum(1)==self.n_attributes).detach()
            reward_cross=[]
            #for agent in agent_receivers:
                #reward_cross.append(1.*(restes_cross[agent]["acc"].sum(1)==self.n_attributes).detach())
            #reward_cross=torch.stack(reward_cross)
            #reward_cross=reward_cross.mean(0)
            for agent in agent_receivers:
                acc = 1*(samples[agent] == sender_input.reshape([sample.size(0),sample.size(1),sender_input.size(1)//sample.size(1)]).argmax(2)).float().mean(1).detach()
                acc = 1*(acc==1).float()
                reward_cross.append(acc)
            reward_cross=torch.stack(reward_cross)
            reward_cross=reward_cross.mean(0)
        elif self.reward_mode=="discrete":
            reward_self = rest_self["acc"].sum(1).detach()
            reward_cross = rest_cross["acc"].sum(1).detach()

        "3. Entropy + length Regularization"

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r_self.mean(1))

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r_self.mean(1))


        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * optim_params_sender["sender_entropy_coeff"] #+ entropy_r_12.mean() * self.receiver_entropy_coeff_1

        log_prob = effective_log_prob_s #+ log_prob_r_12

        length_loss = message_lengths.float() * optim_params_sender["length_cost"]

        "4. Variance reduction"
        if self.baseline_mode=="original":
            policy_loss_self = -((loss_self.detach() - self.mean_baseline['loss_self_{}'.format(sender_id)]) * log_prob).mean()
            policy_loss_cross = -((loss_cross.detach() - self.mean_baseline['loss_cross_{}'.format(sender_id)]) * log_prob).mean()
            policy_length_loss = ((length_loss.float() - self.mean_baseline['length_{}'.format(sender_id)]) * effective_log_prob_s).mean()

        elif self.baseline_mode=="new":
            eps=1e-16
            policy_loss_self = -((reward_self - reward_self.mean())/(reward_self.std()+eps) * log_prob).mean()
            policy_loss_cross = -((reward_cross - reward_cross.mean())/(reward_cross.std()+eps)  * log_prob).mean()
            policy_length_loss = ((length_loss.float() - length_loss.float().mean())  * effective_log_prob_s).mean()

        " 5. Final loss"
        policy_loss = loss_weights_sender["self"]*policy_loss_self + loss_weights_sender["cross"]*policy_loss_cross
        policy_loss /= (loss_weights_sender["self"]+loss_weights_sender["cross"])

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy 
        speaker_loss = optimized_loss.detach().item()

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss_self_{}'.format(sender_id), loss_self)
            self.update_baseline('loss_cross_{}'.format(sender_id), loss_cross)
            self.update_baseline('length_{}'.format(sender_id), length_loss)

        "6. Store results"
        rest={}
        rest['loss'] = optimized_loss.detach().item()
        rest['loss_speaker'] = speaker_loss
        rest['loss_{}'.format(sender_id)] = optimized_loss.detach().item()
        rest['sender_entropy_{}'.format(sender_id)] = entropy_s.mean().item()
        rest['mean_length_{}'.format(sender_id)] = message_lengths.float().mean().item()
        rest['loss_self_{}{}'.format(sender_id,sender_id)] = loss_self.mean().item()
        for receiver_id in receiver_ids:
            rest['loss_cross_{}{}'.format(sender_id,receiver_id)] = losses_cross["agent_{}".format(receiver_id)].mean().item()
        rest['acc_self_{}{}'.format(sender_id,sender_id)]=rest_self['acc'].mean().item()
        for receiver_id in receiver_ids:
            rest['acc_cross_{}{}'.format(sender_id,receiver_id)]=restes_cross["agent_{}".format(receiver_id)]['acc'].mean().item()
        rest['reinforce_term_{}'.format(sender_id)]=policy_loss.detach().item()
        rest['baseline_term_{}'.format(sender_id)]=(policy_loss/log_prob.mean()).detach().item()
        rest['policy_{}'.format(sender_id)]=whole_log_prob_s.detach()

        "7. Save probs"
        if save_probs:
            np.save(save_probs+"_sender_input.npy",sender_input.cpu().numpy())
            np.save(save_probs+"_message.npy",message.cpu().numpy())
            np.save(save_probs+"_sender_probs.npy",whole_log_prob_s.cpu().numpy())

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]


class ForwardPassSpeakerMultiAgent(nn.Module):

    def __init__(self,
                 Agents,
                 n_attributes,
                 n_values,
                 loss_imitation,
                 optim_params,
                 message_to_imitate):
        """
        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":0.,
                      "receiver_entropy_coeff_1":0.,
                      "sender_entropy_coeff_2":0.,
                      "receiver_entropy_coeff_2":0.}

        loss_weights={"self":1.,
                      "cross":1.,
                      "imitation":1.,
                      "length_regularization":0.,
                      "entropy_regularization":1.}
        """
        super(ForwardPassSpeakerMultiAgent, self).__init__()
        self.agents = Agents
        self.n_attributes=n_attributes
        self.n_values=n_values
        self.optim_params = optim_params
        self.loss_imitation = loss_imitation
        self.loss_weights = loss_weights
        self.baseline_mode=baseline_mode
        self.reward_mode=reward_mode
        self.message_to_imitate = message_to_imitate
        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.device=device
        for agent in self.agents:
          self.agents[agent].to(self.device)

    def forward(self,
                sender_input,
                unused_labels,
                sender_id,
                receiver_ids,
                receiver_input=None,
                save_probs=None):

        """
        Inputs:
        - direction : N means "N->0"
        """

        sender_input=sender_input.to(self.device)

        "0. Get sender and receiver (id + optim info) for playing the game"
        # Get sender_id and sender information
        agent_sender = self.agents["agent_{}".format(sender_id)]
        optim_params_sender = self.optim_params["agent_{}".format(sender_id)]

        " 1. Agent actions and loss"
        # Message sending
        message, log_prob_s,whole_log_prob_s, entropy_s = agent_sender.send(sender_input,return_policies=True)
        message_lengths = find_lengths(message)
        message_to_imitate_lengths = find_lengths(self.message_to_imitate)

        loss_imitation, rest_imitation = self.loss_imitation(self.message_to_imitate,whole_log_prob_s,self.message_to_imitate_lengths)

        "6. Store results"
        rest={}
        rest['loss'] = loss_imitation.detach().item()
        rest['loss_{}'.format(sender_id)] = loss_imitation.detach().item()
        rest['mean_length_{}'.format(sender_id)] = message_lengths.float().mean().item()

        "7. Save probs"
        if save_probs:
            np.save(save_probs+"_sender_input.npy",sender_input.cpu().numpy())
            np.save(save_probs+"_message.npy",message.cpu().numpy())
            np.save(save_probs+"_sender_probs.npy",whole_log_prob_s.cpu().numpy())

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]


class DialogReinforceMemory(nn.Module):

    """
    DialogReinforce implements the Dialog game
    """

    def __init__(self,
                 Agent_1,
                 Agent_2,
                 loss_understanding,
                 loss_imitation,
                 optim_params,
                 loss_weights,
                 vocab_size,
                 max_len,
                 n_features,
                 device):
        """
        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":0.,
                      "receiver_entropy_coeff_1":0.,
                      "sender_entropy_coeff_2":0.,
                      "receiver_entropy_coeff_2":0.}

        loss_weights={"self":1.,
                      "cross":1.,
                      "imitation":1.,
                      "length_regularization":0.,
                      "entropy_regularization":1.}
        """
        super(DialogReinforceMemory, self).__init__()
        self.agent_1 = Agent_1
        self.agent_2 = Agent_2
        self.optim_params = optim_params
        self.loss_understanding = loss_understanding
        self.loss_message_imitation = loss_imitation
        self.loss_weights = loss_weights
        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.max_len=max_len
        self.vocab_size=vocab_size
        self.n_features=n_features
        self.device=device

    def forward(self,
                sender_input,
                unused_labels,
                direction,
                receiver_input=None):

        """
        Inputs:
        - direction : "1->2" or "2->1"
        """

        sender_input=sender_input.to(self.device)

        if direction=="1->2":
            agent_sender=self.agent_1
            agent_receiver=self.agent_2
            sender_id=1
            receiver_id=2
        else:
            agent_sender=self.agent_2
            agent_receiver=self.agent_1
            sender_id=2
            receiver_id=1

        " 1. Agent actions "
        # Message sending
        message, log_prob_s, entropy_s = agent_sender.send(sender_input)
        message_lengths = find_lengths(message)
        # Cross listening
        receiver_output_cross, log_prob_r_cross, entropy_r_cross = agent_receiver.receive(message, receiver_input, message_lengths)
        # Self listening
        receiver_output_self, log_prob_r_self, entropy_r_self = agent_sender.receive(message, receiver_input, message_lengths)
        # Imitation
        #candidates_cross=receiver_output_cross.argmax(dim=1)
        #message_reconstruction, prob_reconstruction, _ = agent_receiver.imitate(sender_input)
        message_to_imitate, _, _ = agent_receiver.send(sender_input,eval=True)
        message_to_imitate_lengths = find_lengths(message_to_imitate)
        send_output, _, _ = agent_sender.receive(message_to_imitate, receiver_input, message_to_imitate_lengths)

        i_hat=send_output.argmax(1).cpu().numpy()
        policy_prob=torch.exp(send_output.max(1).values)
        for j in range(send_output.size(0)):
            m=message_to_imitate[j]
            m_dense=torch.zeros([self.max_len,self.vocab_size]).to("cuda")
            for i in range(len(m)):
                m_dense[i,m[i]]=1.
            agent_sender.mem[i_hat[j]].append(m_dense)
            agent_sender.w_mem[i_hat[j]].append(torch.exp(policy_prob[j]))

        for i in agent_sender.mem:
          if len(agent_sender.mem[i])>0:
              agent_sender.est_policy[i]=(torch.stack(agent_sender.mem[i])*torch.stack(agent_sender.w_mem[i]).unsqueeze(1).unsqueeze(2)).sum(0)
              agent_sender.est_policy[i]/=torch.stack(agent_sender.w_mem[i]).sum(0)

        policy_receiver=[]

        for i in range(sender_input.size(0)):
          policy_receiver.append(agent_sender.est_policy[int(sender_input.argmax(1)[i].cpu().numpy())])

        policy_receiver=torch.stack(policy_receiver)


        "2. Losses computation"
        loss_self, rest_self = self.loss_understanding(sender_input,receiver_output_self)
        loss_cross, rest_cross = self.loss_understanding(sender_input,receiver_output_cross)
        loss_imitation=torch.zeros([1024]).to("cuda")
        rest_imitation={"acc_imitation":torch.tensor([0.])}

        # Average loss. Rk. Sortir loss_imitation de cette somme
        loss = self.loss_weights["self"]*loss_self + self.loss_weights["cross"]*loss_cross
        loss /= (self.loss_weights["self"]+self.loss_weights["cross"])

        "3. Entropy + length Regularization"

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r_self)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r_self)


        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.optim_params["sender_entropy_coeff_{}".format(sender_id)] #+ entropy_r_12.mean() * self.receiver_entropy_coeff_1

        log_prob = effective_log_prob_s #+ log_prob_r_12

        length_loss = message_lengths.float() * self.optim_params["length_cost"]

        "4. Variance reduction"

        policy_loss_self = ((loss_self.detach() - self.mean_baseline['loss_self_{}'.format(sender_id)]) * log_prob).mean()
        policy_loss_cross = ((loss_cross.detach() - self.mean_baseline['loss_cross_{}'.format(sender_id)]) * log_prob).mean()
        policy_length_loss = ((length_loss.float() - self.mean_baseline['length_{}'.format(sender_id)]) * effective_log_prob_s).mean()

        " 5. Final loss"
        policy_loss = self.loss_weights["self"]*policy_loss_self + self.loss_weights["cross"]*policy_loss_cross
        policy_loss /= (self.loss_weights["self"]+self.loss_weights["cross"])

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss_self_{}'.format(sender_id), loss_self)
            self.update_baseline('loss_cross_{}'.format(sender_id), loss_cross)
            self.update_baseline('loss_imitation_{}'.format(sender_id), loss_imitation)
            self.update_baseline('length_{}'.format(sender_id), length_loss)

        "6. Store results"
        rest={}
        rest['loss'] = optimized_loss.detach().item()
        rest['loss_{}'.format(sender_id)] = optimized_loss.detach().item()
        rest['sender_entropy_{}'.format(sender_id)] = entropy_s.mean().item()
        rest['mean_length_{}'.format(sender_id)] = message_lengths.float().mean().item()
        rest['loss_self_{}{}'.format(sender_id,sender_id)] = loss_self.mean().item()
        rest['loss_cross_{}{}'.format(sender_id,receiver_id)] = loss_cross.mean().item()
        rest['loss_imitation_{}{}'.format(receiver_id,sender_id)] = loss_imitation.mean().item()
        rest['acc_self_{}{}'.format(sender_id,sender_id)]=rest_self['acc'].mean().item()
        rest['acc_cross_{}{}'.format(sender_id,receiver_id)]=rest_cross['acc'].mean().item()
        rest['acc_imitation_{}{}'.format(receiver_id,sender_id)]=rest_imitation['acc_imitation'].mean().item()

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]

    def to_dense(self,m):
        m_dense=torch.zeros([self.max_len,self.vocab_size])
        for i in range(len(m)):
            m_dense[i,m[i]]=1.

class DialogReinforceBis(nn.Module):

    """
    DialogReinforce implements the Dialog game
    """

    def __init__(self,
                 Agent_1,
                 Agent_2,
                 loss_understanding,
                 loss_imitation,
                 optim_params,
                 loss_weights,
                 n_features,
                 max_len,
                 batch_size,
                 device):
        """
        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":0.,
                      "receiver_entropy_coeff_1":0.,
                      "sender_entropy_coeff_2":0.,
                      "receiver_entropy_coeff_2":0.}

        loss_weights={"self":1.,
                      "cross":1.,
                      "imitation":1.,
                      "length_regularization":0.,
                      "entropy_regularization":1.}
        """
        super(DialogReinforceBis, self).__init__()
        self.agent_1 = Agent_1
        self.agent_2 = Agent_2
        self.optim_params = optim_params
        self.loss_understanding = loss_understanding
        self.loss_message_imitation = loss_imitation
        self.loss_weights = loss_weights
        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.device=device
        self.batch_size=batch_size
        self.last_messages_train=torch.zeros([batch_size,max_len],dtype=int).to("cuda")
        self.last_messages_eval=torch.zeros([n_features,max_len],dtype=int).to("cuda")
        self.last_input_train=torch.zeros([batch_size,n_features]).to("cuda")
        self.last_input_eval=torch.zeros([n_features,n_features]).to("cuda")


    def forward(self,
                sender_input,
                unused_labels,
                direction,
                receiver_input=None):

        """
        Inputs:
        - direction : "1->2" or "2->1"
        """

        sender_input=sender_input.to(self.device)

        if direction=="1->2":
            agent_sender=self.agent_1
            agent_receiver=self.agent_2
            sender_id=1
            receiver_id=2
        else:
            agent_sender=self.agent_2
            agent_receiver=self.agent_1
            sender_id=2
            receiver_id=1

        " 1. Agent actions "
        # Message sending
        message, log_prob_s, entropy_s = agent_sender.send(sender_input)
        message_lengths = find_lengths(message)
        # Cross listening
        receiver_output_cross, log_prob_r_cross, entropy_r_cross = agent_receiver.receive(message, receiver_input, message_lengths)
        # Self listening
        receiver_output_self, log_prob_r_self, entropy_r_self = agent_sender.receive(message, receiver_input, message_lengths)
        # Imitation

        if sender_input.size(0)==self.batch_size:
          message_to_imitate=self.last_messages_train
          last_input=self.last_input_train
        else:
          message_to_imitate=self.last_messages_eval
          last_input=self.last_input_eval

        message_to_imitate_lengths = find_lengths(message_to_imitate)
        send_output, _, _ = agent_sender.receive(message_to_imitate, receiver_input, message_to_imitate_lengths)

        one_hots=torch.eye(100)
        inp_to_imitate=[]
        for i in range(send_output.size(0)):
          inp_to_imitate.append(one_hots[send_output.argmax(1)[i]])

        inp_to_imitate=torch.stack(inp_to_imitate).to("cuda")

        message_reconstruction, prob_reconstruction, _ = agent_sender.imitate(last_input)

        "2. Losses computation"
        loss_self, rest_self = self.loss_understanding(sender_input,receiver_output_self)
        loss_cross, rest_cross = self.loss_understanding(sender_input,receiver_output_cross)
        loss_imitation, rest_imitation = self.loss_message_imitation(message_to_imitate,prob_reconstruction,message_to_imitate_lengths)
        _, rest_und_cross = self.loss_understanding(last_input,send_output)
        loss_imitation=loss_imitation*rest_und_cross["acc"]
        prob_conf=torch.exp((last_input*F.log_softmax(send_output,dim=1)).sum(1))
        #loss_imitation=loss_imitation-(sender_input*F.log_softmax(send_output,dim=1)).sum(1)
        loss_imitation*=prob_conf

        if sender_input.size(0)==self.batch_size:
          self.last_messages_train=message
          self.last_input_train=sender_input
        else:
          self.last_messages_eval=message
          self.last_input_eval=sender_input

        #print(torch.exp((sender_input*F.log_softmax(send_output,dim=1)).sum(1)))

        # Average loss. Rk. Sortir loss_imitation de cette somme
        loss = self.loss_weights["self"]*loss_self + self.loss_weights["cross"]*loss_cross + self.loss_weights["imitation"]*loss_imitation
        loss /= (self.loss_weights["self"]+self.loss_weights["cross"]+self.loss_weights["imitation"])

        "3. Entropy + length Regularization"

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r_self)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r_self)


        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.optim_params["sender_entropy_coeff_{}".format(sender_id)] #+ entropy_r_12.mean() * self.receiver_entropy_coeff_1

        log_prob = effective_log_prob_s #+ log_prob_r_12

        length_loss = message_lengths.float() * self.optim_params["length_cost"]

        "4. Variance reduction"

        policy_loss_self = ((loss_self.detach() - self.mean_baseline['loss_self_{}'.format(sender_id)]) * log_prob).mean()
        policy_loss_cross = ((loss_cross.detach() - self.mean_baseline['loss_cross_{}'.format(sender_id)]) * log_prob).mean()
        policy_loss_imitation = ((loss_imitation.detach() - self.mean_baseline['loss_imitation_{}'.format(sender_id)]) * log_prob).mean()
        policy_length_loss = ((length_loss.float() - self.mean_baseline['length_{}'.format(sender_id)]) * effective_log_prob_s).mean()

        " 5. Final loss"
        policy_loss = self.loss_weights["self"]*policy_loss_self + self.loss_weights["cross"]*policy_loss_cross + self.loss_weights["imitation"]*policy_loss_imitation
        policy_loss /= (self.loss_weights["self"]+self.loss_weights["cross"]+self.loss_weights["imitation"])

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss_self_{}'.format(sender_id), loss_self)
            self.update_baseline('loss_cross_{}'.format(sender_id), loss_cross)
            self.update_baseline('loss_imitation_{}'.format(sender_id), loss_imitation)
            self.update_baseline('length_{}'.format(sender_id), length_loss)

        "6. Store results"
        rest={}
        rest['loss'] = optimized_loss.detach().item()
        rest['loss_{}'.format(sender_id)] = optimized_loss.detach().item()
        rest['sender_entropy_{}'.format(sender_id)] = entropy_s.mean().item()
        rest['mean_length_{}'.format(sender_id)] = message_lengths.float().mean().item()
        rest['loss_self_{}{}'.format(sender_id,sender_id)] = loss_self.mean().item()
        rest['loss_cross_{}{}'.format(sender_id,receiver_id)] = loss_cross.mean().item()
        rest['loss_imitation_{}{}'.format(receiver_id,sender_id)] = loss_imitation.mean().item()
        rest['acc_self_{}{}'.format(sender_id,sender_id)]=rest_self['acc'].mean().item()
        rest['acc_cross_{}{}'.format(sender_id,receiver_id)]=rest_cross['acc'].mean().item()
        rest['acc_imitation_{}{}'.format(receiver_id,sender_id)]=rest_imitation['acc_imitation'].mean().item()

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]


class DialogReinforceKL(nn.Module):

    """
    DialogReinforce implements the Dialog game
    """

    def __init__(self,
                 Agent_1,
                 Agent_2,
                 loss_understanding,
                 loss_imitation,
                 optim_params,
                 loss_weights,
                 device):
        """
        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":0.,
                      "receiver_entropy_coeff_1":0.,
                      "sender_entropy_coeff_2":0.,
                      "receiver_entropy_coeff_2":0.}

        loss_weights={"self":1.,
                      "cross":1.,
                      "imitation":1.,
                      "length_regularization":0.,
                      "entropy_regularization":1.}
        """
        super(DialogReinforceKL, self).__init__()
        self.agent_1 = Agent_1
        self.agent_2 = Agent_2
        self.optim_params = optim_params
        self.loss_understanding = loss_understanding
        self.loss_message_imitation = loss_imitation
        self.loss_weights = loss_weights
        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.device=device

    def forward(self,
                sender_input,
                unused_labels,
                direction,
                receiver_input=None):

        """
        Inputs:
        - direction : "1->2" or "2->1"
        """

        sender_input=sender_input.to(self.device)

        if direction=="1->2":
            agent_sender=self.agent_1
            agent_receiver=self.agent_2
            sender_id=1
            receiver_id=2
        else:
            agent_sender=self.agent_2
            agent_receiver=self.agent_1
            sender_id=2
            receiver_id=1

        " 1. Agent actions "
        # Message sending
        message, log_prob_s,whole_log_prob_s, entropy_s = agent_sender.send(sender_input)
        message_lengths = find_lengths(message)
        # Cross listening
        receiver_output_cross, log_prob_r_cross, entropy_r_cross = agent_receiver.receive(message, receiver_input, message_lengths)
        # Self listening
        receiver_output_self, log_prob_r_self, entropy_r_self = agent_sender.receive(message, receiver_input, message_lengths)
        # Imitation
        #candidates_cross=receiver_output_cross.argmax(dim=1)
        #message_reconstruction, prob_reconstruction, _ = agent_receiver.imitate(sender_input)
        message_other, other_log_prob_s,other_whole_log_prob_s, _ = agent_receiver.send(sender_input,eval=True)
        message_other_lengths = find_lengths(message_other)
        send_output, _, _ = agent_sender.receive(message_other, receiver_input, message_other_lengths)

        other_log_prob_s = send_output.max()

        "2. Losses computation"
        loss_self, rest_self = self.loss_understanding(sender_input,receiver_output_self)
        loss_cross, rest_cross = self.loss_understanding(sender_input,receiver_output_cross)
        #loss_imitation, rest_imitation = self.loss_message_imitation(message,prob_reconstruction,message_lengths)
        #loss_imitation, rest_imitation = self.loss_message_imitation(message_to_imitate,prob_reconstruction,message_to_imitate_lengths)
        #_, rest_und_cross = self.loss_understanding(sender_input,send_output)
        #loss_imitation=loss_imitation*rest_und_cross["acc"]
        prob_conf=torch.exp((sender_input*F.log_softmax(send_output,dim=1)).sum(1))
        KL_div=torch.nn.KLDivLoss(reduce=False)
        #loss_imitation=KL_div(whole_log_prob_s.reshape(whole_log_prob_s.size(0)*whole_log_prob_s.size(1)*whole_log_prob_s.size(2)),other_whole_log_prob_s.reshape(other_whole_log_prob_s.size(0)*other_whole_log_prob_s.size(1)*other_whole_log_prob_s.size(2)))
        loss_imitation = KL_div(torch.exp(whole_log_prob_s.reshape([whole_log_prob_s.size(0)*whole_log_prob_s.size(1),whole_log_prob_s.size(2)])),torch.exp(other_whole_log_prob_s.reshape([other_whole_log_prob_s.size(0)*other_whole_log_prob_s.size(1),other_whole_log_prob_s.size(2)])))
        loss_imitation=loss_imitation.reshape([whole_log_prob_s.size(0),whole_log_prob_s.size(1),whole_log_prob_s.size(2)])

        loss_imitation=loss_imitation.sum(2).sum(1)
        rest_imitation={"acc_imitation":torch.tensor([0.])}

        loss_imitation*=prob_conf

        # Average loss. Rk. Sortir loss_imitation de cette somme
        loss = self.loss_weights["self"]*loss_self + self.loss_weights["cross"]*loss_cross + self.loss_weights["imitation"]*loss_imitation
        loss /= (self.loss_weights["self"]+self.loss_weights["cross"]+self.loss_weights["imitation"])

        "3. Entropy + length Regularization"

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r_self)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r_self)


        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.optim_params["sender_entropy_coeff_{}".format(sender_id)] #+ entropy_r_12.mean() * self.receiver_entropy_coeff_1

        log_prob = effective_log_prob_s #+ log_prob_r_12

        length_loss = message_lengths.float() * self.optim_params["length_cost"]

        "4. Variance reduction"

        policy_loss_self = ((loss_self.detach() - self.mean_baseline['loss_self_{}'.format(sender_id)]) * log_prob).mean()
        policy_loss_cross = ((loss_cross.detach() - self.mean_baseline['loss_cross_{}'.format(sender_id)]) * log_prob).mean()
        policy_loss_imitation = ((loss_imitation.detach() - self.mean_baseline['loss_imitation_{}'.format(sender_id)]) * log_prob).mean()
        policy_length_loss = ((length_loss.float() - self.mean_baseline['length_{}'.format(sender_id)]) * effective_log_prob_s).mean()

        " 5. Final loss"
        policy_loss = self.loss_weights["self"]*policy_loss_self + self.loss_weights["cross"]*policy_loss_cross #+ self.loss_weights["imitation"]*policy_loss_imitation
        policy_loss /= (self.loss_weights["self"]+self.loss_weights["cross"])#+self.loss_weights["imitation"])

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss_self_{}'.format(sender_id), loss_self)
            self.update_baseline('loss_cross_{}'.format(sender_id), loss_cross)
            self.update_baseline('loss_imitation_{}'.format(sender_id), loss_imitation)
            self.update_baseline('length_{}'.format(sender_id), length_loss)

        "6. Store results"
        rest={}
        rest['loss'] = optimized_loss.detach().item()
        rest['loss_{}'.format(sender_id)] = optimized_loss.detach().item()
        rest['sender_entropy_{}'.format(sender_id)] = entropy_s.mean().item()
        rest['mean_length_{}'.format(sender_id)] = message_lengths.float().mean().item()
        rest['loss_self_{}{}'.format(sender_id,sender_id)] = loss_self.mean().item()
        rest['loss_cross_{}{}'.format(sender_id,receiver_id)] = loss_cross.mean().item()
        rest['loss_imitation_{}{}'.format(receiver_id,sender_id)] = loss_imitation.mean().item()
        rest['acc_self_{}{}'.format(sender_id,sender_id)]=rest_self['acc'].mean().item()
        rest['acc_cross_{}{}'.format(sender_id,receiver_id)]=rest_cross['acc'].mean().item()
        rest['acc_imitation_{}{}'.format(receiver_id,sender_id)]=rest_imitation['acc_imitation'].mean().item()

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]

class DialogReinforceBaseline(nn.Module):

    """
    DialogReinforce implements the Dialog game
    """

    def __init__(self,
                 Agent_1,
                 Agent_2,
                 loss,
                 sender_entropy_coeff_1,
                 receiver_entropy_coeff_1,
                 sender_entropy_coeff_2,
                 receiver_entropy_coeff_2,
                 device,
                 loss_weights=[0.5,0.5],
                 length_cost=0.0,
                 unigram_penalty=0.0,
                 reg=False):
        """

        """
        super(DialogReinforceBaseline, self).__init__()
        self.agent_1 = Agent_1
        self.agent_2 = Agent_2
        self.sender_entropy_coeff_1 = sender_entropy_coeff_1
        self.receiver_entropy_coeff_1 = receiver_entropy_coeff_1
        self.sender_entropy_coeff_2 = sender_entropy_coeff_2
        self.receiver_entropy_coeff_2 = receiver_entropy_coeff_2
        self.loss = loss
        self.loss_weights = loss_weights
        self.length_cost = length_cost
        self.unigram_penalty = unigram_penalty
        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.reg=reg
        self.device=device

    def forward(self, sender_input, labels, receiver_input=None):

        sender_input=sender_input.to(self.device)

        "1. Agent_1 -> Agent_2"

        message_1, log_prob_s_1, entropy_s_1 = self.agent_1.sender(sender_input)
        message_lengths_1 = find_lengths(message_1)

        receiver_output_1, log_prob_r_1, entropy_r_1 = self.agent_2.receiver(message_1, receiver_input, message_lengths_1)

        loss_1, rest_1 = self.loss(sender_input, message_1, receiver_input, receiver_output_1, labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s_1 = torch.zeros_like(entropy_r_1)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s_1 = torch.zeros_like(log_prob_r_1)

        for i in range(message_1.size(1)):
            not_eosed_1 = (i < message_lengths_1).float()
            effective_entropy_s_1 += entropy_s_1[:, i] * not_eosed_1
            effective_log_prob_s_1 += log_prob_s_1[:, i] * not_eosed_1
        effective_entropy_s_1 = effective_entropy_s_1 / message_lengths_1.float()

        weighted_entropy_1 = effective_entropy_s_1.mean() * self.sender_entropy_coeff_1 + \
                entropy_r_1.mean() * self.receiver_entropy_coeff_1

        log_prob_1 = effective_log_prob_s_1 + log_prob_r_1

        length_loss_1 = message_lengths_1.float() * self.length_cost

        policy_length_loss_1 = ((length_loss_1.float() - self.mean_baseline['length_1']) * effective_log_prob_s_1).mean()
        policy_loss_1 = ((loss_1.detach() - self.mean_baseline['loss_1']) * log_prob_1).mean()

        optimized_loss_1 = policy_length_loss_1 + policy_loss_1 - weighted_entropy_1

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_1 += loss_1.mean()

        if self.training:
            self.update_baseline('loss_1', loss_1)
            self.update_baseline('length_1', length_loss_1)

        for k, v in rest_1.items():
            rest_1[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_1['loss'] = optimized_loss_1.detach().item()
        rest_1['sender_entropy'] = entropy_s_1.mean().item()
        rest_1['receiver_entropy'] = entropy_r_1.mean().item()
        rest_1['original_loss'] = loss_1.mean().item()
        rest_1['mean_length'] = message_lengths_1.float().mean().item()


        "2. Agent_2 -> Agent_1"

        message_2, log_prob_s_2, entropy_s_2 = self.agent_2.sender(sender_input)
        message_lengths_2 = find_lengths(message_2)

        receiver_output_2, log_prob_r_2, entropy_r_2 = self.agent_1.receiver(message_2, receiver_input, message_lengths_2)

        loss_2, rest_2 = self.loss(sender_input, message_2, receiver_input, receiver_output_2, labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s_2 = torch.zeros_like(entropy_r_2)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s_2 = torch.zeros_like(log_prob_r_2)

        for i in range(message_2.size(1)):
            not_eosed_2 = (i < message_lengths_2).float()
            effective_entropy_s_2 += entropy_s_2[:, i] * not_eosed_2
            effective_log_prob_s_2 += log_prob_s_2[:, i] * not_eosed_2
        effective_entropy_s_2 = effective_entropy_s_2 / message_lengths_2.float()

        weighted_entropy_2 = effective_entropy_s_2.mean() * self.sender_entropy_coeff_2 + \
                entropy_r_2.mean() * self.receiver_entropy_coeff_2

        log_prob_2 = effective_log_prob_s_2 + log_prob_r_2

        length_loss_2 = message_lengths_2.float() * self.length_cost

        policy_length_loss_2 = ((length_loss_2.float() - self.mean_baseline['length_2']) * effective_log_prob_s_2).mean()
        policy_loss_2 = ((loss_2.detach() - self.mean_baseline['loss_2']) * log_prob_2).mean()

        optimized_loss_2 = policy_length_loss_2 + policy_loss_2 - weighted_entropy_2

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_2 += loss_2.mean()

        if self.training:
            self.update_baseline('loss_2', loss_2)
            self.update_baseline('length_2', length_loss_2)

        for k, v in rest_2.items():
            rest_2[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_2['loss'] = optimized_loss_2.detach().item()
        rest_2['sender_entropy'] = entropy_s_2.mean().item()
        rest_2['receiver_entropy'] = entropy_r_2.mean().item()
        rest_2['original_loss'] = loss_2.mean().item()
        rest_2['mean_length'] = message_lengths_2.float().mean().item()

        "3. Average loss"

        optimized_loss = self.loss_weights[0]*optimized_loss_1 + self.loss_weights[1]*optimized_loss_2

        rest={}
        rest['loss']=self.loss_weights[0]*rest_1['loss'] + self.loss_weights[1]* rest_2['loss']
        rest['sender_entropy']=self.loss_weights[0]*rest_1['sender_entropy'] + self.loss_weights[1]* rest_2['sender_entropy']
        rest['receiver_entropy']=self.loss_weights[0]*rest_1['receiver_entropy'] + self.loss_weights[1]* rest_2['receiver_entropy']
        rest['original_loss']=self.loss_weights[0]*rest_1['original_loss'] + self.loss_weights[1]* rest_2['original_loss']
        rest['mean_length']=self.loss_weights[0]*rest_1['mean_length'] + self.loss_weights[1]* rest_2['mean_length']
        rest['acc']=self.loss_weights[0]*rest_1['acc'] + self.loss_weights[1]* rest_2['acc']

        return optimized_loss_1, optimized_loss_2, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]

class DialogReinforceModel1(nn.Module):

    """
    DialogReinforce implements the Dialog game
    """

    def __init__(self,
                 Agent_1,
                 Agent_2,
                 loss,
                 sender_entropy_coeff_1,
                 receiver_entropy_coeff_1,
                 sender_entropy_coeff_2,
                 receiver_entropy_coeff_2,
                 device,
                 loss_weights=[[0.25,0.25],[0.25,0.25]],
                 length_cost=0.0,
                 unigram_penalty=0.0,
                 reg=False):
        """

        """
        super(DialogReinforceModel1, self).__init__()
        self.agent_1 = Agent_1
        self.agent_2 = Agent_2
        self.sender_entropy_coeff_1 = sender_entropy_coeff_1
        self.receiver_entropy_coeff_1 = receiver_entropy_coeff_1
        self.sender_entropy_coeff_2 = sender_entropy_coeff_2
        self.receiver_entropy_coeff_2 = receiver_entropy_coeff_2
        self.loss = loss
        self.loss_weights = loss_weights
        self.length_cost = length_cost
        self.unigram_penalty = unigram_penalty
        self.device=device
        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.reg=reg

    def forward(self, sender_input, labels, receiver_input=None):

        sender_input=sender_input.to(self.device)

        "1. Agent 1"
        message_1, log_prob_s_1, entropy_s_1 = self.agent_1.sender(sender_input)
        message_lengths_1 = find_lengths(message_1)



        "1.2 Agent_1 -> Agent_2"

        #message_12, log_prob_s_12, entropy_s_12 = message_1, log_prob_s_1, entropy_s_1

        receiver_output_12, log_prob_r_12, entropy_r_12 = self.agent_2.receiver(message_1, receiver_input, message_lengths_1)

        loss_12, rest_12 = self.loss(sender_input, message_1, receiver_input, receiver_output_12, labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s_1 = torch.zeros_like(entropy_r_12)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s_1 = torch.zeros_like(log_prob_r_12)


        for i in range(message_1.size(1)):
            not_eosed_1 = (i < message_lengths_1).float()
            effective_entropy_s_1 += entropy_s_1[:, i] * not_eosed_1
            effective_log_prob_s_1 += log_prob_s_1[:, i] * not_eosed_1
        effective_entropy_s_1 = effective_entropy_s_1 / message_lengths_1.float()

        weighted_entropy_12 = effective_entropy_s_1.mean() * self.sender_entropy_coeff_1 + \
                entropy_r_12.mean() * self.receiver_entropy_coeff_1

        log_prob_12 = effective_log_prob_s_1 + log_prob_r_12

        length_loss_12 = message_lengths_1.float() * self.length_cost

        policy_length_loss_12 = ((length_loss_12.float() - self.mean_baseline['length_1']) * effective_log_prob_s_1).mean()
        policy_loss_12 = ((loss_12.detach() - self.mean_baseline['loss_12']) * log_prob_12).mean()

        optimized_loss_12 = policy_length_loss_12 + policy_loss_12 - weighted_entropy_12

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_12 += loss_12.mean()

        if self.training:
            self.update_baseline('loss_12', loss_12)
            self.update_baseline('length_12', length_loss_12)

        for k, v in rest_12.items():
            rest_12[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_12['loss'] = optimized_loss_12.detach().item()
        rest_12['sender_entropy'] = entropy_s_1.mean().item()
        rest_12['receiver_entropy'] = entropy_r_12.mean().item()
        rest_12['original_loss'] = loss_12.mean().item()
        rest_12['mean_length'] = message_lengths_1.float().mean().item()

        "1.1 Agent_1 -> Agent_1"

        #message_11, log_prob_s_11, entropy_s_11 = message_1, log_prob_s_1, entropy_s_1

        receiver_output_11, log_prob_r_11, entropy_r_11 = self.agent_1.receiver(message_1, receiver_input, message_lengths_1)

        loss_11, rest_11 = self.loss(sender_input, message_1, receiver_input, receiver_output_11, labels)


        weighted_entropy_11 = effective_entropy_s_1.mean() * self.sender_entropy_coeff_1 + \
                entropy_r_11.mean() * self.receiver_entropy_coeff_1

        log_prob_11 = effective_log_prob_s_1 + log_prob_r_11

        length_loss_11 = message_lengths_1.float() * self.length_cost

        policy_length_loss_11 = ((length_loss_11.float() - self.mean_baseline['length_1']) * effective_log_prob_s_1).mean()
        policy_loss_11 = ((loss_11.detach() - self.mean_baseline['loss_11']) * log_prob_11).mean()

        optimized_loss_11 = policy_length_loss_11 + policy_loss_11 - weighted_entropy_11

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_11 += loss_11.mean()

        if self.training:
            self.update_baseline('loss_11', loss_11)
            self.update_baseline('length_11', length_loss_11)

        for k, v in rest_11.items():
            rest_11[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_11['loss'] = optimized_loss_11.detach().item()
        rest_11['sender_entropy'] = entropy_s_1.mean().item()
        rest_11['receiver_entropy'] = entropy_r_11.mean().item()
        rest_11['original_loss'] = loss_11.mean().item()
        rest_11['mean_length'] = message_lengths_1.float().mean().item()


        "2. Agent 2"
        message_2, log_prob_s_2, entropy_s_2 = self.agent_2.sender(sender_input)
        message_lengths_2 = find_lengths(message_2)

        "2. Agent_2 -> Agent_1"

        #message_21, log_prob_s_21, entropy_s_21 = message_2, log_prob_s_2, entropy_s_2

        receiver_output_21, log_prob_r_21, entropy_r_21 = self.agent_1.receiver(message_2, receiver_input, message_lengths_2)

        loss_21, rest_21 = self.loss(sender_input, message_2, receiver_input, receiver_output_21, labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s_2 = torch.zeros_like(entropy_r_21)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s_2 = torch.zeros_like(log_prob_r_21)

        for i in range(message_2.size(1)):
            not_eosed_2 = (i < message_lengths_2).float()
            effective_entropy_s_2 += entropy_s_2[:, i] * not_eosed_2
            effective_log_prob_s_2 += log_prob_s_2[:, i] * not_eosed_2
        effective_entropy_s_2 = effective_entropy_s_2 / message_lengths_2.float()

        weighted_entropy_21 = effective_entropy_s_2.mean() * self.sender_entropy_coeff_2 + \
                entropy_r_21.mean() * self.receiver_entropy_coeff_2

        log_prob_21 = effective_log_prob_s_2 + log_prob_r_21

        length_loss_21 = message_lengths_2.float() * self.length_cost

        policy_length_loss_21 = ((length_loss_21.float() - self.mean_baseline['length_21']) * effective_log_prob_s_2).mean()
        policy_loss_21 = ((loss_21.detach() - self.mean_baseline['loss_21']) * log_prob_21).mean()

        optimized_loss_21 = policy_length_loss_21 + policy_loss_21 - weighted_entropy_21

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_21 += loss_21.mean()

        if self.training:
            self.update_baseline('loss_21', loss_21)
            self.update_baseline('length_21', length_loss_21)

        for k, v in rest_21.items():
            rest_21[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_21['loss'] = optimized_loss_21.detach().item()
        rest_21['sender_entropy'] = entropy_s_2.mean().item()
        rest_21['receiver_entropy'] = entropy_r_21.mean().item()
        rest_21['original_loss'] = loss_21.mean().item()
        rest_21['mean_length'] = message_lengths_2.float().mean().item()

        "2. Agent_2 -> Agent_2"

        #message_22, log_prob_s_22, entropy_s_22 = message_2, log_prob_s_2, entropy_s_2

        #message_lengths_22 = find_lengths(message_22)

        receiver_output_22, log_prob_r_22, entropy_r_22 = self.agent_2.receiver(message_2, receiver_input, message_lengths_2)

        loss_22, rest_22 = self.loss(sender_input, message_2, receiver_input, receiver_output_22, labels)


        weighted_entropy_22 = effective_entropy_s_2.mean() * self.sender_entropy_coeff_2 + \
                entropy_r_22.mean() * self.receiver_entropy_coeff_2

        log_prob_22 = effective_log_prob_s_2 + log_prob_r_22

        length_loss_22 = message_lengths_2.float() * self.length_cost

        policy_length_loss_22 = ((length_loss_22.float() - self.mean_baseline['length_22']) * effective_log_prob_s_2).mean()
        policy_loss_22 = ((loss_22.detach() - self.mean_baseline['loss_22']) * log_prob_22).mean()

        optimized_loss_22 = policy_length_loss_22 + policy_loss_22 - weighted_entropy_22

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_22 += loss_22.mean()

        if self.training:
            self.update_baseline('loss_22', loss_22)
            self.update_baseline('length_22', length_loss_22)

        for k, v in rest_22.items():
            rest_22[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_22['loss'] = optimized_loss_22.detach().item()
        rest_22['sender_entropy'] = entropy_s_2.mean().item()
        rest_22['receiver_entropy'] = entropy_r_22.mean().item()
        rest_22['original_loss'] = loss_22.mean().item()
        rest_22['mean_length'] = message_lengths_2.float().mean().item()

        "3. Average loss"

        optimized_loss_1 = self.loss_weights[0][0]*optimized_loss_11 + self.loss_weights[0][1]*optimized_loss_12
        optimized_loss_2 = self.loss_weights[1][0]*optimized_loss_21 + self.loss_weights[1][1]*optimized_loss_22

        optimized_loss = self.loss_weights[0][0]*optimized_loss_11 + self.loss_weights[0][1]*optimized_loss_12+ \
                         self.loss_weights[1][0]*optimized_loss_21 + self.loss_weights[1][1]*optimized_loss_22

        rest={}
        rest['loss']=self.loss_weights[0][0]*rest_11['loss'] + self.loss_weights[0][1]*rest_12['loss']+ \
                         self.loss_weights[1][0]*rest_21['loss'] + self.loss_weights[1][1]*rest_22['loss']
        rest['sender_entropy']=self.loss_weights[0][0]*rest_11['sender_entropy'] + self.loss_weights[0][1]*rest_12['sender_entropy']+ \
                                self.loss_weights[1][0]*rest_21['sender_entropy'] + self.loss_weights[1][1]*rest_22['sender_entropy']
        rest['receiver_entropy']=self.loss_weights[0][0]*rest_11['receiver_entropy'] + self.loss_weights[0][1]*rest_12['receiver_entropy']+ \
                                 self.loss_weights[1][0]*rest_21['receiver_entropy'] + self.loss_weights[1][1]*rest_22['receiver_entropy']
        rest['original_loss']=self.loss_weights[0][0]*rest_11['original_loss'] + self.loss_weights[0][1]*rest_12['original_loss']+ \
                                self.loss_weights[1][0]*rest_21['original_loss'] + self.loss_weights[1][1]*rest_22['original_loss']
        rest['mean_length']=self.loss_weights[0][0]*rest_11['mean_length'] + self.loss_weights[0][1]*rest_12['mean_length']+ \
                            self.loss_weights[1][0]*rest_21['mean_length'] + self.loss_weights[1][1]*rest_22['mean_length']
        rest['acc']=self.loss_weights[0][0]*rest_11['acc'] + self.loss_weights[0][1]*rest_12['acc']+ \
                         self.loss_weights[1][0]*rest_21['acc'] + self.loss_weights[1][1]*rest_22['acc']

        return optimized_loss_11, optimized_loss_12, optimized_loss_21, optimized_loss_22, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]


class DialogReinforceModel2(nn.Module):

    """
    DialogReinforce implements the Dialog game
    """

    def __init__(self,
                 Agent_1,
                 Agent_2,
                 loss,
                 sender_entropy_coeff_1,
                 receiver_entropy_coeff_1,
                 sender_entropy_coeff_2,
                 receiver_entropy_coeff_2,
                 device,
                 loss_weights=[0.5,0.5],
                 length_cost=0.0,
                 unigram_penalty=0.0,
                 reg=False):
        """

        """
        super(DialogReinforceModel2, self).__init__()
        self.agent_1 = Agent_1
        self.agent_2 = Agent_2
        self.sender_entropy_coeff_1 = sender_entropy_coeff_1
        self.receiver_entropy_coeff_1 = receiver_entropy_coeff_1
        self.sender_entropy_coeff_2 = sender_entropy_coeff_2
        self.receiver_entropy_coeff_2 = receiver_entropy_coeff_2
        self.loss = loss
        self.loss_weights = loss_weights
        self.length_cost = length_cost
        self.unigram_penalty = unigram_penalty
        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.reg=reg
        self.device=device

    def forward(self, sender_input, labels, receiver_input=None):

        sender_input=sender_input.to(self.device)

        "1. Agent_1 -> Agent_2"

        message_1, log_prob_s_1, entropy_s_1 = self.agent_1.send(sender_input)
        message_lengths_1 = find_lengths(message_1)

        receiver_output_1, log_prob_r_1, entropy_r_1, sequence_lm, log_probs_lm = self.agent_2.receive(message_1, receiver_input, message_lengths_1)

        # Take only the last => change to EOS position
        log_prob_r_1=log_prob_r_1[:,-1]
        entropy_r_1=entropy_r_1[:,-1]

        loss_1, loss_lm_1, rest_1 = self.loss(sender_input, message_1, message_lengths_1, receiver_input, receiver_output_1, sequence_lm , labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s_1 = torch.zeros_like(entropy_r_1)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s_1 = torch.zeros_like(log_prob_r_1)

        for i in range(message_1.size(1)):
            not_eosed_1 = (i < message_lengths_1).float()
            effective_entropy_s_1 += entropy_s_1[:, i] * not_eosed_1
            effective_log_prob_s_1 += log_prob_s_1[:, i] * not_eosed_1
        effective_entropy_s_1 = effective_entropy_s_1 / message_lengths_1.float()

        weighted_entropy_1 = effective_entropy_s_1.mean() * self.sender_entropy_coeff_1 + \
                entropy_r_1.mean() * self.receiver_entropy_coeff_1

        log_prob_1 = effective_log_prob_s_1 + log_prob_r_1

        length_loss_1 = message_lengths_1.float() * self.length_cost

        policy_length_loss_1 = ((length_loss_1.float() - self.mean_baseline['length_1']) * effective_log_prob_s_1).mean()
        policy_loss_1 = ((loss_1.detach() - self.mean_baseline['loss_1']) * log_prob_1).mean()

        optimized_loss_1 = policy_length_loss_1 + policy_loss_1 - weighted_entropy_1

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_1 += loss_1.mean()

        # Average between task and imitation loss
        optimized_loss_1 = 0.5*(optimized_loss_1 + loss_lm_1.mean())


        if self.training:
            self.update_baseline('loss_1', loss_1)
            self.update_baseline('length_1', length_loss_1)

        for k, v in rest_1.items():
            rest_1[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_1['loss'] = optimized_loss_1.detach().item()
        rest_1['sender_entropy'] = entropy_s_1.mean().item()
        rest_1['receiver_entropy'] = entropy_r_1.mean().item()
        rest_1['original_loss'] = loss_1.mean().item()
        rest_1['mean_length'] = message_lengths_1.float().mean().item()


        "2. Agent_2 -> Agent_1"

        message_2, log_prob_s_2, entropy_s_2 = self.agent_2.send(sender_input)

        message_lengths_2 = find_lengths(message_2)

        receiver_output_2, log_prob_r_2, entropy_r_2,sequence_lm, logits_lm = self.agent_1.receive(message_2, receiver_input, message_lengths_2)

        # Take only the last => change to EOS position
        log_prob_r_2=log_prob_r_2[:,-1]
        entropy_r_2=entropy_r_2[:,-1]

        loss_2, loss_lm_2, rest_2 = self.loss(sender_input, message_2, message_lengths_2, receiver_input, receiver_output_2, sequence_lm , labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s_2 = torch.zeros_like(entropy_r_2)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s_2 = torch.zeros_like(log_prob_r_2)

        for i in range(message_2.size(1)):
            not_eosed_2 = (i < message_lengths_2).float()
            effective_entropy_s_2 += entropy_s_2[:, i] * not_eosed_2
            effective_log_prob_s_2 += log_prob_s_2[:, i] * not_eosed_2
        effective_entropy_s_2 = effective_entropy_s_2 / message_lengths_2.float()

        weighted_entropy_2 = effective_entropy_s_2.mean() * self.sender_entropy_coeff_2 + \
                entropy_r_2.mean() * self.receiver_entropy_coeff_2

        log_prob_2 = effective_log_prob_s_2 + log_prob_r_2

        length_loss_2 = message_lengths_2.float() * self.length_cost

        policy_length_loss_2 = ((length_loss_2.float() - self.mean_baseline['length_2']) * effective_log_prob_s_2).mean()
        policy_loss_2 = ((loss_2.detach() - self.mean_baseline['loss_2']) * log_prob_2).mean()

        optimized_loss_2 = policy_length_loss_2 + policy_loss_2 - weighted_entropy_2

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_2 += loss_2.mean()

        optimized_loss_2 = 0.5*(optimized_loss_2 + loss_lm_2.mean())

        if self.training:
            self.update_baseline('loss_2', loss_2)
            self.update_baseline('length_2', length_loss_2)

        for k, v in rest_2.items():
            rest_2[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_2['loss'] = optimized_loss_2.detach().item()
        rest_2['sender_entropy'] = entropy_s_2.mean().item()
        rest_2['receiver_entropy'] = entropy_r_2.mean().item()
        rest_2['original_loss'] = loss_2.mean().item()
        rest_2['mean_length'] = message_lengths_2.float().mean().item()

        "3. Average loss"

        optimized_loss = self.loss_weights[0]*optimized_loss_1 + self.loss_weights[1]*optimized_loss_2

        rest={}
        rest['loss']=self.loss_weights[0]*rest_1['loss'] + self.loss_weights[1]* rest_2['loss']
        rest['sender_entropy']=self.loss_weights[0]*rest_1['sender_entropy'] + self.loss_weights[1]* rest_2['sender_entropy']
        rest['receiver_entropy']=self.loss_weights[0]*rest_1['receiver_entropy'] + self.loss_weights[1]* rest_2['receiver_entropy']
        rest['original_loss']=self.loss_weights[0]*rest_1['original_loss'] + self.loss_weights[1]* rest_2['original_loss']
        rest['mean_length']=self.loss_weights[0]*rest_1['mean_length'] + self.loss_weights[1]* rest_2['mean_length']
        rest['acc']=self.loss_weights[0]*rest_1['acc'] + self.loss_weights[1]* rest_2['acc']

        return optimized_loss_1, optimized_loss_2, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]


class DialogReinforceModel3(nn.Module):

    """
    DialogReinforce implements the Dialog game
    """

    def __init__(self,
                 Agent_1,
                 Agent_2,
                 loss,
                 sender_entropy_coeff_1,
                 receiver_entropy_coeff_1,
                 sender_entropy_coeff_2,
                 receiver_entropy_coeff_2,
                 device,
                 loss_weights=[0.5,0.5],
                 length_cost=0.0,
                 unigram_penalty=0.0,
                 reg=False):
        """

        """
        super(DialogReinforceModel3, self).__init__()
        self.agent_1 = Agent_1
        self.agent_2 = Agent_2
        self.sender_entropy_coeff_1 = sender_entropy_coeff_1
        self.receiver_entropy_coeff_1 = receiver_entropy_coeff_1
        self.sender_entropy_coeff_2 = sender_entropy_coeff_2
        self.receiver_entropy_coeff_2 = receiver_entropy_coeff_2
        self.loss = loss
        self.loss_weights = loss_weights
        self.length_cost = length_cost
        self.unigram_penalty = unigram_penalty
        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.reg=reg
        self.device=device

    def forward(self, sender_input, labels, receiver_input=None):

        sender_input=sender_input.to(self.device)

        "1. Agent_1 -> Agent_2"

        message_1, log_prob_s_1, entropy_s_1 = self.agent_1.send(sender_input)
        message_lengths_1 = find_lengths(message_1)

        receiver_output_1, prob_r_1, _ , log_prob_r_1, entropy_r_1 = self.agent_2.receive(message_1, receiver_input, message_lengths_1,imitate=True)

        candidates_1=receiver_output_1.argmax(dim=1)

        message_reconstruction_1, prob_reconstruction_1, _ = self.agent_2.imitate(sender_input,imitate=True)

        loss_1_comm, loss_1_imitation, rest_1 = self.loss(sender_input, message_1, receiver_input, receiver_output_1,message_reconstruction_1,prob_reconstruction_1, labels)

        # Imitation loss weighted by likelihood of candidate
        loss_1_imitation = loss_1_imitation #* prob_r_1.max(1).values
        loss_1_imitation=loss_1_imitation.mean()

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s_1 = torch.zeros_like(entropy_r_1)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s_1 = torch.zeros_like(log_prob_r_1)

        for i in range(message_1.size(1)):
            not_eosed_1 = (i < message_lengths_1).float()
            effective_entropy_s_1 += entropy_s_1[:, i] * not_eosed_1
            effective_log_prob_s_1 += log_prob_s_1[:, i] * not_eosed_1
        effective_entropy_s_1 = effective_entropy_s_1 / message_lengths_1.float()

        weighted_entropy_1 = effective_entropy_s_1.mean() * self.sender_entropy_coeff_1 + \
                entropy_r_1.mean() * self.receiver_entropy_coeff_1

        log_prob_1 = effective_log_prob_s_1 + log_prob_r_1

        length_loss_1 = message_lengths_1.float() * self.length_cost

        policy_length_loss_1 = ((length_loss_1.float() - self.mean_baseline['length_1']) * effective_log_prob_s_1).mean()
        policy_loss_1 = ((loss_1_comm.detach() - self.mean_baseline['loss_1']) * log_prob_1).mean()

        optimized_loss_1 = policy_length_loss_1 + policy_loss_1 - weighted_entropy_1

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_1 += loss_1_comm.mean()

        if self.training:
            self.update_baseline('loss_1', loss_1_comm)
            self.update_baseline('length_1', length_loss_1)

        for k, v in rest_1.items():
            rest_1[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_1['loss'] = optimized_loss_1.detach().item()
        rest_1['sender_entropy'] = entropy_s_1.mean().item()
        rest_1['receiver_entropy'] = entropy_r_1.mean().item()
        rest_1['original_loss'] = loss_1_comm.mean().item()
        rest_1['mean_length'] = message_lengths_1.float().mean().item()


        "2. Agent_2 -> Agent_1"

        message_2, log_prob_s_2, entropy_s_2 = self.agent_2.send(sender_input)
        message_lengths_2 = find_lengths(message_2)

        receiver_output_2, prob_r_2, _ , log_prob_r_2, entropy_r_2 = self.agent_1.receive(message_2, receiver_input, message_lengths_2,imitate=True)

        candidates_2=receiver_output_2.argmax(dim=1)

        message_reconstruction_2, prob_reconstruction_2, _ = self.agent_1.imitate(sender_input,imitate=True)

        loss_2_comm, loss_2_imitation, rest_2 = self.loss(sender_input, message_2, receiver_input, receiver_output_2,message_reconstruction_2,prob_reconstruction_2, labels)

        # Imitation loss weighted by likelihood of candidate
        loss_2_imitation = loss_2_imitation #* prob_r_2.max(1).values
        loss_2_imitation=loss_2_imitation.mean()

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s_2 = torch.zeros_like(entropy_r_2)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s_2 = torch.zeros_like(log_prob_r_2)

        for i in range(message_2.size(1)):
            not_eosed_2 = (i < message_lengths_2).float()
            effective_entropy_s_2 += entropy_s_2[:, i] * not_eosed_2
            effective_log_prob_s_2 += log_prob_s_2[:, i] * not_eosed_2
        effective_entropy_s_2 = effective_entropy_s_2 / message_lengths_2.float()

        weighted_entropy_2 = effective_entropy_s_2.mean() * self.sender_entropy_coeff_2 + \
                entropy_r_2.mean() * self.receiver_entropy_coeff_2

        log_prob_2 = effective_log_prob_s_2 + log_prob_r_2

        length_loss_2 = message_lengths_2.float() * self.length_cost

        policy_length_loss_2 = ((length_loss_2.float() - self.mean_baseline['length_2']) * effective_log_prob_s_2).mean()
        policy_loss_2 = ((loss_2_comm.detach() - self.mean_baseline['loss_2']) * log_prob_2).mean()

        optimized_loss_2 = policy_length_loss_2 + policy_loss_2 - weighted_entropy_2

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_2 += loss_2_comm.mean()

        if self.training:
            self.update_baseline('loss_2', loss_2_comm)
            self.update_baseline('length_2', length_loss_2)

        for k, v in rest_2.items():
            rest_2[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_2['loss'] = optimized_loss_2.detach().item()
        rest_2['sender_entropy'] = entropy_s_2.mean().item()
        rest_2['receiver_entropy'] = entropy_r_2.mean().item()
        rest_2['original_loss'] = loss_2_comm.mean().item()
        rest_2['mean_length'] = message_lengths_2.float().mean().item()

        "3. Average loss"

        optimized_loss = self.loss_weights[0]*optimized_loss_1 + self.loss_weights[1]*optimized_loss_2

        rest={}
        rest['loss']=self.loss_weights[0]*rest_1['loss'] + self.loss_weights[1]* rest_2['loss']
        rest['sender_entropy']=self.loss_weights[0]*rest_1['sender_entropy'] + self.loss_weights[1]* rest_2['sender_entropy']
        rest['receiver_entropy']=self.loss_weights[0]*rest_1['receiver_entropy'] + self.loss_weights[1]* rest_2['receiver_entropy']
        rest['original_loss']=self.loss_weights[0]*rest_1['original_loss'] + self.loss_weights[1]* rest_2['original_loss']
        rest['mean_length']=self.loss_weights[0]*rest_1['mean_length'] + self.loss_weights[1]* rest_2['mean_length']
        rest['acc']=self.loss_weights[0]*rest_1['acc'] + self.loss_weights[1]* rest_2['acc']

        return optimized_loss_1,loss_1_imitation, optimized_loss_2, loss_2_imitation, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]

class DialogReinforceModel4(nn.Module):

    """
    DialogReinforce implements the Dialog game
    """

    def __init__(self,
                 Agent_1,
                 Agent_2,
                 loss,
                 sender_entropy_coeff_1,
                 receiver_entropy_coeff_1,
                 sender_entropy_coeff_2,
                 receiver_entropy_coeff_2,
                 device,
                 loss_weights=[[0.25,0.25],[0.25,0.25]],
                 length_cost=0.0,
                 unigram_penalty=0.0,
                 reg=False):
        """

        """
        super(DialogReinforceModel4, self).__init__()
        self.agent_1 = Agent_1
        self.agent_2 = Agent_2
        self.sender_entropy_coeff_1 = sender_entropy_coeff_1
        self.receiver_entropy_coeff_1 = receiver_entropy_coeff_1
        self.sender_entropy_coeff_2 = sender_entropy_coeff_2
        self.receiver_entropy_coeff_2 = receiver_entropy_coeff_2
        self.loss = loss
        self.loss_weights = loss_weights
        self.length_cost = length_cost
        self.unigram_penalty = unigram_penalty
        self.device=device
        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.reg=reg

    def forward(self, sender_input, labels, receiver_input=None):

        sender_input=sender_input.to(self.device)

        "1. Agent 1"
        message_1, log_prob_s_1, entropy_s_1 = self.agent_1.sender(sender_input)
        message_lengths_1 = find_lengths(message_1)

        a_self=3.
        a_cross=1.
        a_im=1.

        "1.2 Agent_1 -> Agent_2"

        #message_12, log_prob_s_12, entropy_s_12 = message_1, log_prob_s_1, entropy_s_1

        receiver_output_12, prob_r_12, _ , log_prob_r_12, entropy_r_12 = self.agent_2.receive(message_1, receiver_input, message_lengths_1,imitate=True)

        candidates_12=receiver_output_12.argmax(dim=1)

        message_reconstruction_12, prob_reconstruction_12, _ = self.agent_2.imitate(sender_input,imitate=True)

        loss_12_comm, loss_12_imitation, rest_12 = self.loss(sender_input, message_1, receiver_input, receiver_output_12,message_reconstruction_12,prob_reconstruction_12, labels,message_lengths_1)

        # Imitation loss weighted by likelihood of candidate
        loss_12_imitation = loss_12_imitation #* prob_r_12.max(1).values
        #loss_12_imitation=loss_12_imitation.mean()

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s_1 = torch.zeros_like(entropy_r_12)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s_1 = torch.zeros_like(log_prob_r_12)


        for i in range(message_1.size(1)):
            not_eosed_1 = (i < message_lengths_1).float()
            effective_entropy_s_1 += entropy_s_1[:, i] * not_eosed_1
            effective_log_prob_s_1 += log_prob_s_1[:, i] * not_eosed_1
        effective_entropy_s_1 = effective_entropy_s_1 / message_lengths_1.float()

        weighted_entropy_12 = effective_entropy_s_1.mean() * self.sender_entropy_coeff_1 + \
                entropy_r_12.mean() * self.receiver_entropy_coeff_1

        log_prob_12 = effective_log_prob_s_1 + log_prob_r_12

        length_loss_12 = message_lengths_1.float() * self.length_cost

        policy_length_loss_12 = ((length_loss_12.float() - self.mean_baseline['length_12']) * effective_log_prob_s_1).mean()
        policy_loss_12 = ((loss_12_comm.detach() - self.mean_baseline['loss_12']) * log_prob_12).mean()

        optimized_loss_12 = policy_length_loss_12 + policy_loss_12 - weighted_entropy_12

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_12 += loss_12_comm.mean()

        if self.training:
            self.update_baseline('loss_12', loss_12_comm)
            self.update_baseline('length_12', length_loss_12)

        for k, v in rest_12.items():
            rest_12[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_12['loss'] = optimized_loss_12.detach().item()
        rest_12['sender_entropy'] = entropy_s_1.mean().item()
        rest_12['receiver_entropy'] = entropy_r_12.mean().item()
        rest_12['original_loss'] = loss_12_comm.mean().item()
        rest_12['mean_length'] = message_lengths_1.float().mean().item()

        "1.1 Agent_1 -> Agent_1"

        #message_11, log_prob_s_11, entropy_s_11 = message_1, log_prob_s_1, entropy_s_1

        receiver_output_11, prob_r_11, _ , log_prob_r_11, entropy_r_11 = self.agent_1.receive(message_1, receiver_input, message_lengths_1,imitate=True)

        candidates_11=receiver_output_11.argmax(dim=1)

        message_reconstruction_11, prob_reconstruction_11, _ = self.agent_1.imitate(sender_input,imitate=True)

        loss_11_comm, loss_11_imitation, rest_11 = self.loss(sender_input, message_1, receiver_input, receiver_output_11,message_reconstruction_11,prob_reconstruction_11, labels,message_lengths_1)

        # Imitation loss weighted by likelihood of candidate
        loss_11_imitation = loss_11_imitation #* prob_r_11.max(1).values
        #loss_11_imitation=loss_11_imitation.mean()

        loss_11_comm=a_self*loss_11_comm+a_cross*loss_12_comm+a_im*loss_12_imitation

        weighted_entropy_11 = effective_entropy_s_1.mean() * self.sender_entropy_coeff_1 + \
                entropy_r_11.mean() * self.receiver_entropy_coeff_1

        log_prob_11 = effective_log_prob_s_1 + log_prob_r_11

        length_loss_11 = message_lengths_1.float() * self.length_cost

        policy_length_loss_11 = ((length_loss_11.float() - self.mean_baseline['length_1']) * effective_log_prob_s_1).mean()
        policy_loss_11 = ((loss_11_comm.detach() - self.mean_baseline['loss_11']) * log_prob_11).mean()

        optimized_loss_11 = policy_length_loss_11 + policy_loss_11 - weighted_entropy_11

        # if the receiver is deterministic/differentiable, we apply the actual loss

        optimized_loss_11 += loss_11_comm.mean()

        if self.training:
            self.update_baseline('loss_11', loss_11_comm)
            self.update_baseline('length_11', length_loss_11)

        for k, v in rest_11.items():
            rest_11[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_11['loss'] = optimized_loss_11.detach().item()
        rest_11['sender_entropy'] = entropy_s_1.mean().item()
        rest_11['receiver_entropy'] = entropy_r_11.mean().item()
        rest_11['original_loss'] = loss_11_comm.mean().item()
        rest_11['mean_length'] = message_lengths_1.float().mean().item()


        "2. Agent 2"
        message_2, log_prob_s_2, entropy_s_2 = self.agent_2.sender(sender_input)
        message_lengths_2 = find_lengths(message_2)

        "2. Agent_2 -> Agent_1"

        #message_21, log_prob_s_21, entropy_s_21 = message_2, log_prob_s_2, entropy_s_2

        receiver_output_21, prob_r_21, _ , log_prob_r_21, entropy_r_21 = self.agent_1.receive(message_2, receiver_input, message_lengths_2,imitate=True)

        candidates_21=receiver_output_21.argmax(dim=1)

        message_reconstruction_21, prob_reconstruction_21, _ = self.agent_1.imitate(sender_input,imitate=True)

        loss_21_comm, loss_21_imitation, rest_21 = self.loss(sender_input, message_2, receiver_input, receiver_output_21,message_reconstruction_21,prob_reconstruction_21, labels,message_lengths_2)

        # Imitation loss weighted by likelihood of candidate
        loss_21_imitation = loss_21_imitation #* prob_r_21.max(1).values
        #loss_21_imitation=loss_21_imitation.mean()

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s_2 = torch.zeros_like(entropy_r_21)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s_2 = torch.zeros_like(log_prob_r_21)

        for i in range(message_2.size(1)):
            not_eosed_2 = (i < message_lengths_2).float()
            effective_entropy_s_2 += entropy_s_2[:, i] * not_eosed_2
            effective_log_prob_s_2 += log_prob_s_2[:, i] * not_eosed_2
        effective_entropy_s_2 = effective_entropy_s_2 / message_lengths_2.float()

        weighted_entropy_21 = effective_entropy_s_2.mean() * self.sender_entropy_coeff_2 + \
                entropy_r_21.mean() * self.receiver_entropy_coeff_2

        log_prob_21 = effective_log_prob_s_2 + log_prob_r_21

        length_loss_21 = message_lengths_2.float() * self.length_cost

        policy_length_loss_21 = ((length_loss_21.float() - self.mean_baseline['length_21']) * effective_log_prob_s_2).mean()
        policy_loss_21 = ((loss_21_comm.detach() - self.mean_baseline['loss_21']) * log_prob_21).mean()

        optimized_loss_21 = policy_length_loss_21 + policy_loss_21 - weighted_entropy_21

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_21 += loss_21_comm.mean()

        if self.training:
            self.update_baseline('loss_21', loss_21_comm)
            self.update_baseline('length_21', length_loss_21)

        for k, v in rest_21.items():
            rest_21[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_21['loss'] = optimized_loss_21.detach().item()
        rest_21['sender_entropy'] = entropy_s_2.mean().item()
        rest_21['receiver_entropy'] = entropy_r_21.mean().item()
        rest_21['original_loss'] = loss_21_comm.mean().item()
        rest_21['mean_length'] = message_lengths_2.float().mean().item()

        "2. Agent_2 -> Agent_2"

        #message_22, log_prob_s_22, entropy_s_22 = message_2, log_prob_s_2, entropy_s_2

        #message_lengths_22 = find_lengths(message_22)

        receiver_output_22, prob_r_22, _ , log_prob_r_22, entropy_r_22 = self.agent_2.receive(message_2, receiver_input, message_lengths_2,imitate=True)

        candidates_22=receiver_output_22.argmax(dim=1)

        message_reconstruction_22, prob_reconstruction_22, _ = self.agent_2.imitate(sender_input,imitate=True)

        loss_22_comm, loss_22_imitation, rest_22 = self.loss(sender_input, message_2, receiver_input, receiver_output_22,message_reconstruction_22,prob_reconstruction_22, labels, message_lengths_2)

        # Imitation loss weighted by likelihood of candidate
        loss_22_imitation = loss_22_imitation #* prob_r_22.max(1).values
        #loss_22_imitation=loss_22_imitation.mean()

        loss_22_comm=a_self*loss_22_comm+a_cross*loss_21_comm+a_im*loss_21_imitation


        weighted_entropy_22 = effective_entropy_s_2.mean() * self.sender_entropy_coeff_2 + \
                entropy_r_22.mean() * self.receiver_entropy_coeff_2

        log_prob_22 = effective_log_prob_s_2 + log_prob_r_22

        length_loss_22 = message_lengths_2.float() * self.length_cost

        policy_length_loss_22 = ((length_loss_22.float() - self.mean_baseline['length_22']) * effective_log_prob_s_2).mean()
        policy_loss_22 = ((loss_22_comm.detach() - self.mean_baseline['loss_22']) * log_prob_22).mean()

        optimized_loss_22 = policy_length_loss_22 + policy_loss_22 - weighted_entropy_22

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_22 += loss_22_comm.mean()

        if self.training:
            self.update_baseline('loss_22', loss_22_comm)
            self.update_baseline('length_22', length_loss_22)

        for k, v in rest_22.items():
            rest_22[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_22['loss'] = optimized_loss_22.detach().item()
        rest_22['sender_entropy'] = entropy_s_2.mean().item()
        rest_22['receiver_entropy'] = entropy_r_22.mean().item()
        rest_22['original_loss'] = loss_22_comm.mean().item()
        rest_22['mean_length'] = message_lengths_2.float().mean().item()

        "3. Average loss"

        optimized_loss_1 = self.loss_weights[0][0]*optimized_loss_11 + self.loss_weights[0][1]*optimized_loss_12
        optimized_loss_2 = self.loss_weights[1][0]*optimized_loss_21 + self.loss_weights[1][1]*optimized_loss_22

        optimized_loss = self.loss_weights[0][0]*optimized_loss_11 + self.loss_weights[0][1]*optimized_loss_12+ \
                         self.loss_weights[1][0]*optimized_loss_21 + self.loss_weights[1][1]*optimized_loss_22

        rest={}
        rest['loss']=self.loss_weights[0][0]*rest_11['loss'] + self.loss_weights[0][1]*rest_12['loss']+ \
                         self.loss_weights[1][0]*rest_21['loss'] + self.loss_weights[1][1]*rest_22['loss']
        rest['sender_entropy']=self.loss_weights[0][0]*rest_11['sender_entropy'] + self.loss_weights[0][1]*rest_12['sender_entropy']+ \
                                self.loss_weights[1][0]*rest_21['sender_entropy'] + self.loss_weights[1][1]*rest_22['sender_entropy']
        rest['receiver_entropy']=self.loss_weights[0][0]*rest_11['receiver_entropy'] + self.loss_weights[0][1]*rest_12['receiver_entropy']+ \
                                 self.loss_weights[1][0]*rest_21['receiver_entropy'] + self.loss_weights[1][1]*rest_22['receiver_entropy']
        rest['original_loss']=self.loss_weights[0][0]*rest_11['original_loss'] + self.loss_weights[0][1]*rest_12['original_loss']+ \
                                self.loss_weights[1][0]*rest_21['original_loss'] + self.loss_weights[1][1]*rest_22['original_loss']
        rest['mean_length']=self.loss_weights[0][0]*rest_11['mean_length'] + self.loss_weights[0][1]*rest_12['mean_length']+ \
                            self.loss_weights[1][0]*rest_21['mean_length'] + self.loss_weights[1][1]*rest_22['mean_length']
        rest['acc']=self.loss_weights[0][0]*rest_11['acc'] + self.loss_weights[0][1]*rest_12['acc']+ \
                         self.loss_weights[1][0]*rest_21['acc'] + self.loss_weights[1][1]*rest_22['acc']
        rest["acc_11"]=rest_11["acc"]
        rest["acc_12"]=rest_12["acc"]
        rest["acc_21"]=rest_21["acc"]
        rest["acc_22"]=rest_22["acc"]

        return optimized_loss_11,loss_11_imitation, optimized_loss_12,loss_12_imitation, optimized_loss_21,loss_21_imitation, optimized_loss_22,loss_22_imitation, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]


class PretrainAgent(nn.Module):

    """
    DialogReinforce implements the Dialog game
    """

    def __init__(self,
                 Agent_1,
                 loss,
                 pretrained_messages,
                 sender_entropy_coeff_1,
                 receiver_entropy_coeff_1,
                 device,
                 n_features,
                 length_cost=0.0,
                 unigram_penalty=0.0,
                 reg=False):
        """

        """
        super(PretrainAgent, self).__init__()
        self.agent_1 = Agent_1
        self.sender_entropy_coeff_1 = sender_entropy_coeff_1
        self.receiver_entropy_coeff_1 = receiver_entropy_coeff_1
        self.pretrained_messages=pretrained_messages
        if self.pretrained_messages is not None:
            self.pretrained_messages=self.pretrained_messages.to(device)
        self.loss = loss
        self.n_features=n_features
        self.length_cost = length_cost
        self.unigram_penalty = unigram_penalty
        self.device=device
        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.reg=reg

    def forward(self, sender_input, labels, receiver_input=None):

        sender_input=sender_input.to(self.device)

        message_1, log_prob_s_1, entropy_s_1 = self.agent_1.sender(sender_input)
        message_lengths_1 = find_lengths(message_1)

        "1.1 Agent_1 -> Agent_1"

        #message_11, log_prob_s_11, entropy_s_11 = message_1, log_prob_s_1, entropy_s_1

        receiver_output_11, prob_r_11, _ , log_prob_r_11, entropy_r_11 = self.agent_1.receive(message_1, receiver_input, message_lengths_1,imitate=True)

        if self.pretrained_messages is not None:
            pretrained_sender_input = torch.eye(self.n_features).to(self.device)

            message_reconstruction_11, prob_reconstruction_11, _ = self.agent_1.imitate(pretrained_sender_input,imitate=True)
        else:
            message_reconstruction_11=None
            prob_reconstruction_11=None

        loss_11_comm, loss_11_imitation, rest_11 = self.loss(sender_input, message_1,self.pretrained_messages, receiver_input, receiver_output_11,message_reconstruction_11,prob_reconstruction_11, labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s_1 = torch.zeros_like(entropy_r_11)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s_1 = torch.zeros_like(log_prob_r_11)


        for i in range(message_1.size(1)):
            not_eosed_1 = (i < message_lengths_1).float()
            effective_entropy_s_1 += entropy_s_1[:, i] * not_eosed_1
            effective_log_prob_s_1 += log_prob_s_1[:, i] * not_eosed_1
        effective_entropy_s_1 = effective_entropy_s_1 / message_lengths_1.float()


        weighted_entropy_11 = effective_entropy_s_1.mean() * self.sender_entropy_coeff_1 + \
                entropy_r_11.mean() * self.receiver_entropy_coeff_1

        log_prob_11 = effective_log_prob_s_1 + log_prob_r_11

        length_loss_11 = message_lengths_1.float() * self.length_cost

        policy_length_loss_11 = ((length_loss_11.float() - self.mean_baseline['length_1']) * effective_log_prob_s_1).mean()
        policy_loss_11 = ((loss_11_comm.detach() - self.mean_baseline['loss_11']) * log_prob_11).mean()

        optimized_loss_11 = policy_length_loss_11 + policy_loss_11 - weighted_entropy_11

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_11 += loss_11_comm.mean()

        if self.training:
            self.update_baseline('loss_11', loss_11_comm)
            self.update_baseline('length_11', length_loss_11)

        for k, v in rest_11.items():
            rest_11[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_11['loss'] = optimized_loss_11.detach().item()
        rest_11['sender_entropy'] = entropy_s_1.mean().item()
        rest_11['receiver_entropy'] = entropy_r_11.mean().item()
        rest_11['original_loss'] = loss_11_comm.mean().item()
        rest_11['mean_length'] = message_lengths_1.float().mean().item()

        return optimized_loss_11,loss_11_imitation, rest_11

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]


class DialogReinforceModel6(nn.Module):

    """
    DialogReinforce implements the Dialog game
    """

    def __init__(self,
                 Agent_1,
                 Agent_2,
                 loss,
                 sender_entropy_coeff_1,
                 receiver_entropy_coeff_1,
                 sender_entropy_coeff_2,
                 receiver_entropy_coeff_2,
                 imitate,
                 device,
                 loss_weights=[[0.25,0.25],[0.25,0.25]],
                 length_cost=0.0,
                 unigram_penalty=0.0,
                 reg=False):
        """

        """
        super(DialogReinforceModel6, self).__init__()
        self.agent_1 = Agent_1
        self.agent_2 = Agent_2
        self.sender_entropy_coeff_1 = sender_entropy_coeff_1
        self.receiver_entropy_coeff_1 = receiver_entropy_coeff_1
        self.sender_entropy_coeff_2 = sender_entropy_coeff_2
        self.receiver_entropy_coeff_2 = receiver_entropy_coeff_2
        self.loss = loss
        self.loss_weights = loss_weights
        self.length_cost = length_cost
        self.unigram_penalty = unigram_penalty
        self.device=device
        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.reg=reg
        self.imitate=imitate

    def forward(self, sender_input, labels, receiver_input=None):

        sender_input=sender_input.to(self.device)

        "1. Agent 1"
        message_1, log_prob_s_1, entropy_s_1 = self.agent_1.send(sender_input)
        message_lengths_1 = find_lengths(message_1)

        a_self=3.
        a_cross=1.
        a_im=1.

        "1.2 Agent_1 -> Agent_2"

        #message_12, log_prob_s_12, entropy_s_12 = message_1, log_prob_s_1, entropy_s_1

        receiver_output_12, log_prob_r_12, entropy_r_12 = self.agent_2.receive(message_1, receiver_input, message_lengths_1)

        if self.imitate:
          candidates_12=receiver_output_12.argmax(dim=1)
          message_reconstruction_12, prob_reconstruction_12, _ = self.agent_2.imitate(sender_input)
          loss_12, loss_12_imitation, rest_12 = self.loss(sender_input, message_1, receiver_input, receiver_output_12,message_reconstruction_12,prob_reconstruction_12, labels,message_lengths_1)
          #loss_12_imitation=loss_12_imitation.mean()
        else:
          loss_12, rest_12 = self.loss(sender_input, message_1, receiver_input, receiver_output_12, labels)


        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s_1 = torch.zeros_like(entropy_r_12)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s_1 = torch.zeros_like(log_prob_r_12)


        for i in range(message_1.size(1)):
            not_eosed_1 = (i < message_lengths_1).float()
            effective_entropy_s_1 += entropy_s_1[:, i] * not_eosed_1
            effective_log_prob_s_1 += log_prob_s_1[:, i] * not_eosed_1
        effective_entropy_s_1 = effective_entropy_s_1 / message_lengths_1.float()

        weighted_entropy_12 = effective_entropy_s_1.mean() * self.sender_entropy_coeff_1 + \
                entropy_r_12.mean() * self.receiver_entropy_coeff_1

        log_prob_12 = effective_log_prob_s_1 + log_prob_r_12

        length_loss_12 = message_lengths_1.float() * self.length_cost

        policy_length_loss_12 = ((length_loss_12.float() - self.mean_baseline['length_1']) * effective_log_prob_s_1).mean()
        policy_loss_12 = ((loss_12.detach() - self.mean_baseline['loss_12']) * log_prob_12).mean()

        optimized_loss_12 = policy_length_loss_12 + policy_loss_12 - weighted_entropy_12

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_12 += loss_12.mean()

        if self.training:
            self.update_baseline('loss_12', loss_12)
            self.update_baseline('length_12', length_loss_12)

        for k, v in rest_12.items():
            rest_12[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_12['loss'] = optimized_loss_12.detach().item()
        rest_12['sender_entropy'] = entropy_s_1.mean().item()
        rest_12['receiver_entropy'] = entropy_r_12.mean().item()
        rest_12['original_loss'] = loss_12.mean().item()
        rest_12['mean_length'] = message_lengths_1.float().mean().item()

        "1.1 Agent_1 -> Agent_1"

        #message_11, log_prob_s_11, entropy_s_11 = message_1, log_prob_s_1, entropy_s_1

        receiver_output_11, log_prob_r_11, entropy_r_11 = self.agent_1.receive(message_1, receiver_input, message_lengths_1)

        if self.imitate:
          candidates_11=receiver_output_11.argmax(dim=1)
          message_reconstruction_11, prob_reconstruction_11, _ = self.agent_1.imitate(sender_input)
          loss_11, loss_11_imitation, rest_11 = self.loss(sender_input, message_1, receiver_input, receiver_output_11,message_reconstruction_11,prob_reconstruction_11, labels,message_lengths_1)
          #loss_11_imitation=loss_11_imitation.mean()
          loss_11=a_self*loss_11+a_cross*loss_12+a_im*loss_12_imitation
        else:
          loss_11, rest_11 = self.loss(sender_input, message_1, receiver_input, receiver_output_11, labels)


        weighted_entropy_11 = effective_entropy_s_1.mean() * self.sender_entropy_coeff_1 + \
                entropy_r_11.mean() * self.receiver_entropy_coeff_1

        log_prob_11 = effective_log_prob_s_1 + log_prob_r_11

        length_loss_11 = message_lengths_1.float() * self.length_cost

        policy_length_loss_11 = ((length_loss_11.float() - self.mean_baseline['length_1']) * effective_log_prob_s_1).mean()
        policy_loss_11 = ((loss_11.detach() - self.mean_baseline['loss_11']) * log_prob_11).mean()

        optimized_loss_11 = policy_length_loss_11 + policy_loss_11 - weighted_entropy_11

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_11 += loss_11.mean()

        if self.training:
            self.update_baseline('loss_11', loss_11)
            self.update_baseline('length_11', length_loss_11)

        for k, v in rest_11.items():
            rest_11[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_11['loss'] = optimized_loss_11.detach().item()
        rest_11['sender_entropy'] = entropy_s_1.mean().item()
        rest_11['receiver_entropy'] = entropy_r_11.mean().item()
        rest_11['original_loss'] = loss_11.mean().item()
        rest_11['mean_length'] = message_lengths_1.float().mean().item()


        "2. Agent 2"
        message_2, log_prob_s_2, entropy_s_2 = self.agent_2.send(sender_input)
        message_lengths_2 = find_lengths(message_2)

        "2. Agent_2 -> Agent_1"

        #message_21, log_prob_s_21, entropy_s_21 = message_2, log_prob_s_2, entropy_s_2

        receiver_output_21, log_prob_r_21, entropy_r_21 = self.agent_1.receive(message_2, receiver_input, message_lengths_2)

        if self.imitate:
          candidates_21=receiver_output_21.argmax(dim=1)
          message_reconstruction_21, prob_reconstruction_21, _ = self.agent_1.imitate(sender_input)
          loss_21, loss_21_imitation, rest_21 = self.loss(sender_input, message_2, receiver_input, receiver_output_21,message_reconstruction_21,prob_reconstruction_21, labels,message_lengths_2)
          #loss_21_imitation=loss_21_imitation.mean()

        else:
          loss_21, rest_21 = self.loss(sender_input, message_2, receiver_input, receiver_output_21, labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s_2 = torch.zeros_like(entropy_r_21)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s_2 = torch.zeros_like(log_prob_r_21)

        for i in range(message_2.size(1)):
            not_eosed_2 = (i < message_lengths_2).float()
            effective_entropy_s_2 += entropy_s_2[:, i] * not_eosed_2
            effective_log_prob_s_2 += log_prob_s_2[:, i] * not_eosed_2
        effective_entropy_s_2 = effective_entropy_s_2 / message_lengths_2.float()

        weighted_entropy_21 = effective_entropy_s_2.mean() * self.sender_entropy_coeff_2 + \
                entropy_r_21.mean() * self.receiver_entropy_coeff_2

        log_prob_21 = effective_log_prob_s_2 + log_prob_r_21

        length_loss_21 = message_lengths_2.float() * self.length_cost

        policy_length_loss_21 = ((length_loss_21.float() - self.mean_baseline['length_21']) * effective_log_prob_s_2).mean()
        policy_loss_21 = ((loss_21.detach() - self.mean_baseline['loss_21']) * log_prob_21).mean()

        optimized_loss_21 = policy_length_loss_21 + policy_loss_21 - weighted_entropy_21

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_21 += loss_21.mean()

        if self.training:
            self.update_baseline('loss_21', loss_21)
            self.update_baseline('length_21', length_loss_21)

        for k, v in rest_21.items():
            rest_21[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_21['loss'] = optimized_loss_21.detach().item()
        rest_21['sender_entropy'] = entropy_s_2.mean().item()
        rest_21['receiver_entropy'] = entropy_r_21.mean().item()
        rest_21['original_loss'] = loss_21.mean().item()
        rest_21['mean_length'] = message_lengths_2.float().mean().item()

        "2. Agent_2 -> Agent_2"

        #message_22, log_prob_s_22, entropy_s_22 = message_2, log_prob_s_2, entropy_s_2

        #message_lengths_22 = find_lengths(message_22)

        receiver_output_22, log_prob_r_22, entropy_r_22 = self.agent_2.receive(message_2, receiver_input, message_lengths_2)

        if self.imitate:
          candidates_22=receiver_output_22.argmax(dim=1)
          message_reconstruction_22, prob_reconstruction_22, _ = self.agent_2.imitate(sender_input)
          loss_22, loss_22_imitation, rest_22 = self.loss(sender_input, message_2, receiver_input, receiver_output_22,message_reconstruction_22,prob_reconstruction_22, labels,message_lengths_2)
          #loss_22_imitation=loss_22_imitation.mean()
          loss_22=a_self*loss_22+a_cross*loss_21+a_im*loss_21_imitation
        else:
          loss_22, rest_22 = self.loss(sender_input, message_2, receiver_input, receiver_output_22, labels)

        weighted_entropy_22 = effective_entropy_s_2.mean() * self.sender_entropy_coeff_2 + \
                entropy_r_22.mean() * self.receiver_entropy_coeff_2

        log_prob_22 = effective_log_prob_s_2 + log_prob_r_22

        length_loss_22 = message_lengths_2.float() * self.length_cost

        policy_length_loss_22 = ((length_loss_22.float() - self.mean_baseline['length_22']) * effective_log_prob_s_2).mean()
        policy_loss_22 = ((loss_22.detach() - self.mean_baseline['loss_22']) * log_prob_22).mean()

        optimized_loss_22 = policy_length_loss_22 + policy_loss_22 - weighted_entropy_22

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss_22 += loss_22.mean()

        if self.training:
            self.update_baseline('loss_22', loss_22)
            self.update_baseline('length_22', length_loss_22)

        for k, v in rest_22.items():
            rest_22[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest_22['loss'] = optimized_loss_22.detach().item()
        rest_22['sender_entropy'] = entropy_s_2.mean().item()
        rest_22['receiver_entropy'] = entropy_r_22.mean().item()
        rest_22['original_loss'] = loss_22.mean().item()
        rest_22['mean_length'] = message_lengths_2.float().mean().item()

        "3. Average loss"

        optimized_loss_1 = self.loss_weights[0][0]*optimized_loss_11 + self.loss_weights[0][1]*optimized_loss_12
        optimized_loss_2 = self.loss_weights[1][0]*optimized_loss_21 + self.loss_weights[1][1]*optimized_loss_22

        optimized_loss = self.loss_weights[0][0]*optimized_loss_11 + self.loss_weights[0][1]*optimized_loss_12+ \
                         self.loss_weights[1][0]*optimized_loss_21 + self.loss_weights[1][1]*optimized_loss_22

        rest={}
        rest['loss']=self.loss_weights[0][0]*rest_11['loss'] + self.loss_weights[0][1]*rest_12['loss']+ \
                         self.loss_weights[1][0]*rest_21['loss'] + self.loss_weights[1][1]*rest_22['loss']
        rest['sender_entropy']=self.loss_weights[0][0]*rest_11['sender_entropy'] + self.loss_weights[0][1]*rest_12['sender_entropy']+ \
                                self.loss_weights[1][0]*rest_21['sender_entropy'] + self.loss_weights[1][1]*rest_22['sender_entropy']
        rest['receiver_entropy']=self.loss_weights[0][0]*rest_11['receiver_entropy'] + self.loss_weights[0][1]*rest_12['receiver_entropy']+ \
                                 self.loss_weights[1][0]*rest_21['receiver_entropy'] + self.loss_weights[1][1]*rest_22['receiver_entropy']
        rest['original_loss']=self.loss_weights[0][0]*rest_11['original_loss'] + self.loss_weights[0][1]*rest_12['original_loss']+ \
                                self.loss_weights[1][0]*rest_21['original_loss'] + self.loss_weights[1][1]*rest_22['original_loss']
        rest['mean_length']=self.loss_weights[0][0]*rest_11['mean_length'] + self.loss_weights[0][1]*rest_12['mean_length']+ \
                            self.loss_weights[1][0]*rest_21['mean_length'] + self.loss_weights[1][1]*rest_22['mean_length']
        rest['acc']=self.loss_weights[0][0]*rest_11['acc'] + self.loss_weights[0][1]*rest_12['acc']+ \
                         self.loss_weights[1][0]*rest_21['acc'] + self.loss_weights[1][1]*rest_22['acc']

        rest['acc_21']=rest_21['acc']
        rest['acc_12']=rest_12['acc']

        if not self.imitate:
            return optimized_loss_11, optimized_loss_12, optimized_loss_21, optimized_loss_22, rest
        else:
            return optimized_loss_11, optimized_loss_12, optimized_loss_21, optimized_loss_22,loss_12_imitation,loss_21_imitation, rest


    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]



class SenderReceiverRnnReinforce(nn.Module):
    """
    Implements Sender/Receiver game with training done via Reinforce. Both agents are supposed to
    return 3-tuples of (output, log-prob of the output, entropy).
    The game implementation is responsible for handling the end-of-sequence term, so that the optimized loss
    corresponds either to the position of the eos term (assumed to be 0) or the end of sequence.

    Sender and Receiver can be obtained by applying the corresponding wrappers.
    `SenderReceiverRnnReinforce` also applies the mean baseline to the loss function to reduce the variance of the
    gradient estimate.

    >>> sender = nn.Linear(3, 10)
    >>> sender = RnnSenderReinforce(sender, vocab_size=15, embed_dim=5, hidden_size=10, max_len=10, cell='lstm')

    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> receiver = RnnReceiverDeterministic(Receiver(), vocab_size=15, embed_dim=10, hidden_size=5)
    >>> def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    ...     return F.mse_loss(sender_input, receiver_output, reduction='none').mean(dim=1), {'aux': 5.0}

    >>> game = SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0,
    ...                                   length_cost=1e-2)
    >>> input = torch.zeros((16, 3)).normal_()
    >>> optimized_loss, aux_info = game(input, labels=None)
    >>> sorted(list(aux_info.keys()))  # returns some debug info, such as entropies of the agents, message length etc
    ['aux', 'loss', 'mean_length', 'original_loss', 'receiver_entropy', 'sender_entropy']
    >>> aux_info['aux']
    5.0
    """
    def __init__(self, sender, receiver, loss, sender_entropy_coeff, receiver_entropy_coeff,
                 length_cost=0.0,unigram_penalty=0.0,reg=False):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param reg: apply the regularization scheduling (Lazy Speaker)
        """
        super(SenderReceiverRnnReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost
        self.unigram_penalty = unigram_penalty

        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)
        self.reg=reg

    def forward(self, sender_input, labels, receiver_input=None):
        message, log_prob_s, entropy_s = self.sender(sender_input)
        message_lengths = find_lengths(message)

        receiver_output, log_prob_r, entropy_r = self.receiver(message, receiver_input, message_lengths)

        loss, rest = self.loss(sender_input, message, receiver_input, receiver_output, labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.sender_entropy_coeff + \
                entropy_r.mean() * self.receiver_entropy_coeff

        log_prob = effective_log_prob_s + log_prob_r

        if self.reg:

          sc=rest["acc"].sum()/rest["acc"].size(0)

          # Pour n_features=100
          self.length_cost= sc**(45) / 5

          #self.length_cost= sc**(45) / 10
          #if sc>0.99:
              #self.length_cost=(sc-0.99)*100 +0.01
          #else:
              #self.length_cost=0.
          #if sc>0.995:
              #self.length_cost+=0.01
              #if self.length_cost==0.3:
            #      self.length_cost-=0.01
              #print(self.length_cost)

          #if sc<0.98:
              #self.length_cost=0.


        length_loss = message_lengths.float() * self.length_cost

        policy_length_loss = ((length_loss.float() - self.mean_baseline['length']) * effective_log_prob_s).mean()
        policy_loss = ((loss.detach() - self.mean_baseline['loss']) * log_prob).mean()

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss', loss)
            self.update_baseline('length', length_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]

class SenderImpatientReceiverRnnReinforce(nn.Module):
    """
    Implements Sender/ Impatient Receiver game with training done via Reinforce.
    It is equivalent to SenderReceiverRnnReinforce but takes into account the intermediate predictions of Impatient Listener:
    - the Impatient loss is used
    - tensor shapes are adapted for variance reduction.

    When reg is set to True, the regularization scheduling is applied (Lazy Speaker).
    """
    def __init__(self, sender, receiver, loss, sender_entropy_coeff, receiver_entropy_coeff,
                 length_cost=0.0,unigram_penalty=0.0,reg=False):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param reg: apply the regularization scheduling (Lazy Speaker)
        """
        super(SenderImpatientReceiverRnnReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost
        self.unigram_penalty = unigram_penalty
        self.reg=reg

        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)

    def forward(self, sender_input, labels, receiver_input=None):
        message, log_prob_s, entropy_s = self.sender(sender_input)
        message_lengths = find_lengths(message)

        # If impatient 1
        receiver_output, log_prob_r, entropy_r = self.receiver(message, receiver_input, message_lengths)

        """ NOISE VERSION

        # Randomly takes a position
        rand_length=np.random.randint(0,message.size(1))

        # Loss by output
        loss, rest = self.loss(sender_input, message, receiver_input, receiver_output[:,rand_length,:], labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r[:,rand_length])

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r[:,rand_length])
        """

        #Loss
        loss, rest, crible_acc = self.loss(sender_input, message, message_lengths, receiver_input, receiver_output, labels)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r.mean(1))

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r.mean(1))

        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.sender_entropy_coeff + \
                entropy_r.mean() * self.receiver_entropy_coeff

        log_prob = effective_log_prob_s + log_prob_r.mean(1)

        if self.reg:
            sc=0.

            for i in range(message_lengths.size(0)):
              sc+=crible_acc[i,message_lengths[i]-1]
            sc/=message_lengths.size(0)

            # Regularization scheduling paper
            #self.length_cost= sc**(45) / 10

            # Pour n_features=100
            self.length_cost= sc**(45) / 5

        length_loss = message_lengths.float() * self.length_cost

        policy_length_loss = ((length_loss.float() - self.mean_baseline['length']) * effective_log_prob_s).mean()
        policy_loss = ((loss.detach() - self.mean_baseline['loss']) * log_prob).mean()

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss', loss)
            self.update_baseline('length', length_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]

class CompositionalitySenderReceiverRnnReinforce(nn.Module):

    """
    Adaptation of SenderReceiverRnnReinforce to inputs with several attributes.
    """
    def __init__(self, sender, receiver, loss, sender_entropy_coeff, receiver_entropy_coeff,n_attributes,n_values,
                 length_cost=0.0,unigram_penalty=0.0,reg=False):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        """
        super(CompositionalitySenderReceiverRnnReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost
        self.unigram_penalty = unigram_penalty
        self.reg=reg
        self.n_attributes=n_attributes
        self.n_values=n_values

        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)

    def forward(self, sender_input, labels, receiver_input=None):

        message, log_prob_s, entropy_s = self.sender(sender_input)
        message_lengths = find_lengths(message)

        # Noisy channel
        noise_level=0.
        noise_map=torch.from_numpy(1*(np.random.rand(message.size(0),message.size(1))<noise_level)).to("cuda")
        noise=torch.from_numpy(np.random.randint(1,self.sender.vocab_size,size=(message.size(0),message.size(1)))).to("cuda") # random symbols

        message_noise=message*(1-noise_map) + noise_map* noise

        # Receiver normal
        receiver_output_all_att, log_prob_r_all_att, entropy_r_all_att = self.receiver(message_noise, receiver_input, message_lengths)
        #dim=[batch_size,n_att,n_val]

        # reg
        sc=0.

        loss, rest, crible_acc = self.loss(sender_input, message, message_lengths, receiver_input, receiver_output_all_att, labels,self.n_attributes,self.n_values)

        #if self.reg:
        #     for i in range(message_lengths.size(0)):
        #      sc+=crible_acc[i,message_lengths[i]-1]


        log_prob_r=log_prob_r_all_att.mean(1)
        entropy_r=entropy_r_all_att.mean(1)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.sender_entropy_coeff + \
                entropy_r.mean() * self.receiver_entropy_coeff

        log_prob = effective_log_prob_s + log_prob_r

        #if self.reg:
        #    sc/=message_lengths.size(0)

        #    if sc>0.98:
        #    	self.length_cost+=0.1
        #    else:
        #    	self.length_cost=0.
            #self.length_cost= sc**(60) / 2

        length_loss = message_lengths.float() * self.length_cost

        policy_length_loss = ((length_loss.float() - self.mean_baseline['length']) * effective_log_prob_s).mean()
        policy_loss = ((loss.detach() - self.mean_baseline['loss']) * log_prob).mean()

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss', loss)
            self.update_baseline('length', length_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]



class CompositionalitySenderImpatientReceiverRnnReinforce(nn.Module):
    """
    Implements Sender/Receiver game with training done via Reinforce. Both agents are supposed to
    return 3-tuples of (output, log-prob of the output, entropy).
    The game implementation is responsible for handling the end-of-sequence term, so that the optimized loss
    corresponds either to the position of the eos term (assumed to be 0) or the end of sequence.

    Sender and Receiver can be obtained by applying the corresponding wrappers.
    `SenderReceiverRnnReinforce` also applies the mean baseline to the loss function to reduce the variance of the
    gradient estimate.

    >>> sender = nn.Linear(3, 10)
    >>> sender = RnnSenderReinforce(sender, vocab_size=15, embed_dim=5, hidden_size=10, max_len=10, cell='lstm')

    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> receiver = RnnReceiverDeterministic(Receiver(), vocab_size=15, embed_dim=10, hidden_size=5)
    >>> def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    ...     return F.mse_loss(sender_input, receiver_output, reduction='none').mean(dim=1), {'aux': 5.0}

    >>> game = SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0,
    ...                                   length_cost=1e-2)
    >>> input = torch.zeros((16, 3)).normal_()
    >>> optimized_loss, aux_info = game(input, labels=None)
    >>> sorted(list(aux_info.keys()))  # returns some debug info, such as entropies of the agents, message length etc
    ['aux', 'loss', 'mean_length', 'original_loss', 'receiver_entropy', 'sender_entropy']
    >>> aux_info['aux']
    5.0
    """
    def __init__(self, sender, receiver, loss, sender_entropy_coeff, receiver_entropy_coeff,n_attributes,n_values,att_weights,
                 length_cost=0.0,unigram_penalty=0.0,reg=False):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        """
        super(CompositionalitySenderImpatientReceiverRnnReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost
        self.unigram_penalty = unigram_penalty
        self.reg=reg
        self.n_attributes=n_attributes
        self.n_values=n_values
        self.att_weights=att_weights

        self.mean_baseline = defaultdict(float)
        self.n_points = defaultdict(float)

    def forward(self, sender_input, labels, receiver_input=None):

        #print(sender_input[:,11:-1])
        message, log_prob_s, entropy_s = self.sender(torch.floor(sender_input))
        message_lengths = find_lengths(message)

        # If impatient 1
        receiver_output_all_att, log_prob_r_all_att, entropy_r_all_att = self.receiver(message, receiver_input, message_lengths)

        # reg
        sc=0.

        # Version de base
        #loss, rest, crible_acc = self.loss(sender_input, message, message_lengths, receiver_input, receiver_output_all_att, labels,self.n_attributes,self.n_values,self.att_weights)

        # Take into account the fact that an attribute is not sampled
        loss, rest, crible_acc = self.loss(sender_input, message, message_lengths, receiver_input, receiver_output_all_att, labels,self.n_attributes,self.n_values,self.att_weights)


        if self.reg:
            for i in range(message_lengths.size(0)):
              sc+=crible_acc[i,message_lengths[i]-1]


        log_prob_r=log_prob_r_all_att.mean(1).mean(1)
        entropy_r=entropy_r_all_att.mean(1).mean(1)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        for i in range(message.size(1)):
            not_eosed = (i < message_lengths).float()
            effective_entropy_s += entropy_s[:, i] * not_eosed
            effective_log_prob_s += log_prob_s[:, i] * not_eosed
        effective_entropy_s = effective_entropy_s / message_lengths.float()

        weighted_entropy = effective_entropy_s.mean() * self.sender_entropy_coeff + \
                entropy_r.mean() * self.receiver_entropy_coeff

        log_prob = effective_log_prob_s + log_prob_r

        if self.reg:
            sc/=message_lengths.size(0)

            if sc>0.9 and sc<0.99:
                self.length_cost=0.
            if sc>0.99:
                self.length_cost+=0.01
            #if sc<0.9:
            #   self.length_cost=-0.1
            #self.length_cost= sc**(60) / 2

        length_loss = message_lengths.float() * self.length_cost

        # Penalty redundancy
        #counts_unigram=((message[:,1:]-message[:,:-1])==0).sum(axis=1).sum(axis=0)
        #unigram_loss = self.unigram_penalty*counts_unigram

        policy_length_loss = ((length_loss.float() - self.mean_baseline['length']) * effective_log_prob_s).mean()
        policy_loss = ((loss.detach() - self.mean_baseline['loss']) * log_prob).mean()

        optimized_loss = policy_length_loss + policy_loss - weighted_entropy

        # if the receiver is deterministic/differentiable, we apply the actual loss
        optimized_loss += loss.mean()

        if self.training:
            self.update_baseline('loss', loss)
            self.update_baseline('length', length_loss)

        for k, v in rest.items():
            rest[k] = v.mean().item() if hasattr(v, 'mean') else v
        rest['loss'] = optimized_loss.detach().item()
        rest['sender_entropy'] = entropy_s.mean().item()
        rest['receiver_entropy'] = entropy_r.mean().item()
        rest['original_loss'] = loss.mean().item()
        rest['mean_length'] = message_lengths.float().mean().item()

        return optimized_loss, rest

    def update_baseline(self, name, value):
        self.n_points[name] += 1
        self.mean_baseline[name] += (value.detach().mean().item() - self.mean_baseline[name]) / self.n_points[name]


class TransformerReceiverDeterministic(nn.Module):
    def __init__(self, agent, vocab_size, max_len, embed_dim, num_heads, hidden_size, num_layers, positional_emb=True,
                causal=True):
        super(TransformerReceiverDeterministic, self).__init__()
        self.agent = agent
        self.encoder = TransformerEncoder(vocab_size=vocab_size,
                                          max_len=max_len,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          num_layers=num_layers,
                                          hidden_size=hidden_size,
                                          positional_embedding=positional_emb,
                                          causal=causal)

    def forward(self, message, input=None, lengths=None):
        if lengths is None:
            lengths = find_lengths(message)

        transformed = self.encoder(message, lengths)
        agent_output = self.agent(transformed, input)

        logits = torch.zeros(agent_output.size(0)).to(agent_output.device)
        entropy = logits

        return agent_output, logits, entropy


class TransformerSenderReinforce(nn.Module):
    def __init__(self, agent, vocab_size, embed_dim, max_len, num_layers, num_heads, hidden_size,
                 generate_style='standard', causal=True, force_eos=True):
        """
        :param agent: the agent to be wrapped, returns the "encoder" state vector, which is the unrolled into a message
        :param vocab_size: vocab size of the message
        :param embed_dim: embedding dimensions
        :param max_len: maximal length of the message (including <eos>)
        :param num_layers: number of transformer layers
        :param num_heads: number of attention heads
        :param hidden_size: size of the FFN layers
        :param causal: whether embedding of a particular symbol should only depend on the symbols to the left
        :param generate_style: Two alternatives: 'standard' and 'in-place'. Suppose we are generating 4th symbol,
            after three symbols [s1 s2 s3] were generated.
            Then,
            'standard': [s1 s2 s3] -> embeddings [[e1] [e2] [e3]] -> (s4 = argmax(linear(e3)))
            'in-place': [s1 s2 s3] -> [s1 s2 s3 <need-symbol>] -> embeddings [[e1] [e2] [e3] [e4]] -> (s4 = argmax(linear(e4)))
        :param force_eos: <eos> added to the end of each sequence
        """
        super(TransformerSenderReinforce, self).__init__()
        self.agent = agent

        self.force_eos = force_eos
        assert generate_style in ['standard', 'in-place']
        self.generate_style = generate_style
        self.causal = causal

        self.max_len = max_len

        if force_eos:
            self.max_len -= 1

        self.transformer = TransformerDecoder(embed_dim=embed_dim,
                                              max_len=max_len, num_layers=num_layers,
                                              num_heads=num_heads, hidden_size=hidden_size)

        self.embedding_to_vocab = nn.Linear(embed_dim, vocab_size)

        self.special_symbol_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.embed_tokens = torch.nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=self.embed_dim ** -0.5)
        self.embed_scale = math.sqrt(embed_dim)

    def generate_standard(self, encoder_state):
        batch_size = encoder_state.size(0)
        device = encoder_state.device

        sequence = []
        logits = []
        entropy = []

        special_symbol = self.special_symbol_embedding.expand(batch_size, -1).unsqueeze(1).to(device)
        input = special_symbol

        for step in range(self.max_len):
            if self.causal:
                attn_mask = torch.triu(torch.ones(step+1, step+1).byte(), diagonal=1).to(device)
                attn_mask = attn_mask.float().masked_fill(attn_mask == 1, float('-inf'))
            else:
                attn_mask = None
            output = self.transformer(embedded_input=input, encoder_out=encoder_state, attn_mask=attn_mask)
            step_logits = F.log_softmax(self.embedding_to_vocab(output[:, -1, :]), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())
            if self.training:
                symbols = distr.sample()
            else:
                symbols = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(symbols))
            sequence.append(symbols)

            new_embedding = self.embed_tokens(symbols) * self.embed_scale
            input = torch.cat([input, new_embedding.unsqueeze(dim=1)], dim=1)

        return sequence, logits, entropy

    def generate_inplace(self, encoder_state):
        batch_size = encoder_state.size(0)
        device = encoder_state.device

        sequence = []
        logits = []
        entropy = []

        special_symbol = self.special_symbol_embedding.expand(batch_size, -1).unsqueeze(1).to(encoder_state.device)
        output = []
        for step in range(self.max_len):
            input = torch.cat(output + [special_symbol], dim=1)
            if self.causal:
                attn_mask = torch.triu(torch.ones(step+1, step+1).byte(), diagonal=1).to(device)
                attn_mask = attn_mask.float().masked_fill(attn_mask == 1, float('-inf'))
            else:
                attn_mask = None

            embedded = self.transformer(embedded_input=input, encoder_out=encoder_state, attn_mask=attn_mask)
            step_logits = F.log_softmax(self.embedding_to_vocab(embedded[:, -1, :]), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())
            if self.training:
                symbols = distr.sample()
            else:
                symbols = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(symbols))
            sequence.append(symbols)

            new_embedding = self.embed_tokens(symbols) * self.embed_scale
            output.append(new_embedding.unsqueeze(dim=1))

        return sequence, logits, entropy

    def forward(self, x):
        encoder_state = self.agent(x)

        if self.generate_style == 'standard':
            sequence, logits, entropy = self.generate_standard(encoder_state)
        elif self.generate_style == 'in-place':
            sequence, logits, entropy = self.generate_inplace(encoder_state)
        else:
            assert False, 'Unknown generate style'

        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy
