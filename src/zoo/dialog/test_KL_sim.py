# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import numpy as np
import os
from os import path
import itertools
from scipy.stats import entropy
import collections
import torch.utils.data
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import src.core as core
#from scipy.stats import entropy
from src.core import EarlyStopperAccuracy
from src.zoo.dialog.features import OneHotLoader, UniformLoader, OneHotLoaderCompositionality, TestLoaderCompositionality
from src.zoo.dialog.archs import Sender, Receiver
from src.core.reinforce_wrappers import RnnReceiverImpatient, RnnReceiverImpatientCompositionality, RnnReceiverCompositionality
from src.core.reinforce_wrappers import SenderImpatientReceiverRnnReinforce, CompositionalitySenderImpatientReceiverRnnReinforce, CompositionalitySenderReceiverRnnReinforce
from src.core.util import dump_dialog_compositionality ,levenshtein, convert_messages_to_numpy

#Dialog
from src.core.reinforce_wrappers import RnnReceiverWithHiddenStates,RnnSenderReinforceModel3
from src.core.reinforce_wrappers import  AgentBaseline,AgentModel2,AgentModel3
from src.core.reinforce_wrappers import DialogReinforceBaseline,DialogReinforceModel1,DialogReinforceModel2, DialogReinforceModel3,DialogReinforceModel4,PretrainAgent,DialogReinforceModel6
from src.core.util import dump_sender_receiver_dialog,dump_sender_receiver_dialog_model_1,dump_sender_receiver_dialog_model_2,dump_pretraining_u,dump_sender_receiver_dialog_model_6
from src.core.trainers import TrainerDialogModel1, TrainerDialogModel2, TrainerDialogModel3,TrainerDialogModel4,TrainerDialogModel5,TrainerPretraining,TrainerDialogModel6

# Compo
from src.core.reinforce_wrappers import AgentBaseline2,AgentSharedRNN,AgentSharedEmbedding,AgentBaselineKL,AgentPol
from src.core.reinforce_wrappers import DialogReinforce,DialogReinforceBis,DialogReinforceKL,DialogReinforceMemory
from src.core.reinforce_wrappers import DialogReinforceCompositionality, AgentBaselineCompositionality
from src.core.trainers import CompoTrainer,TrainerDialogCompositionality
from src.core.util import sample_messages,dump_sender_receiver_with_noise


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=10,
                        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--batches_per_epoch', type=int, default=1000,
                        help='Number of batches per epoch (default: 1000)')
    parser.add_argument('--dim_dataset', type=int, default=10240,
                        help='Dim of constructing the data (default: 10240)')
    parser.add_argument('--force_eos', type=int, default=0,
                        help='Force EOS at the end of the messages (default: 0)')

    parser.add_argument('--sender_hidden', type=int, default=10,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument('--receiver_num_layers', type=int, default=1,
                        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument('--sender_num_layers', type=int, default=1,
                        help='Number hidden layers of receiver. Only in reinforce (default: 1)')
    parser.add_argument('--receiver_num_heads', type=int, default=8,
                        help='Number of attention heads for Transformer Receiver (default: 8)')
    parser.add_argument('--sender_num_heads', type=int, default=8,
                        help='Number of self-attention heads for Transformer Sender (default: 8)')
    parser.add_argument('--sender_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 10)')
    parser.add_argument('--receiver_embedding', type=int, default=10,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 10)')

    parser.add_argument('--causal_sender', default=False, action='store_true')
    parser.add_argument('--causal_receiver', default=False, action='store_true')

    parser.add_argument('--sender_generate_style', type=str, default='in-place', choices=['standard', 'in-place'],
                        help='How the next symbol is generated within the TransformerDecoder (default: in-place)')

    parser.add_argument('--sender_cell', type=str, default='rnn',
                        help='Type of the cell used for Sender {rnn, gru, lstm, transformer} (default: rnn)')
    parser.add_argument('--receiver_cell', type=str, default='rnn',
                        help='Type of the model used for Receiver {rnn, gru, lstm, transformer} (default: rnn)')

    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient for Sender (default: 1e-1)')
    parser.add_argument('--receiver_entropy_coeff', type=float, default=1e-1,
                        help='The entropy regularisation coefficient for Receiver (default: 1e-1)')

    parser.add_argument('--probs', type=str, default='uniform',
                        help="Prior distribution over the concepts (default: uniform)")
    parser.add_argument('--length_cost', type=float, default=0.0,
                        help="Penalty for the message length, each symbol would before <EOS> would be "
                             "penalized by this cost (default: 0.0)")
    parser.add_argument('--name', type=str, default='model',
                        help="Name for your checkpoint (default: model)")
    parser.add_argument('--early_stopping_thr', type=float, default=0.9999,
                        help="Early stopping threshold on accuracy (default: 0.9999)")

    # AJOUT
    parser.add_argument('--dir_save', type=str, default="expe_1",
                        help="Directory in which we will save the information")
    parser.add_argument('--unigram_pen', type=float, default=0.0,
                        help="Add a penalty for redundancy")
    parser.add_argument('--impatient', type=bool, default=False,
                        help="Impatient listener")
    parser.add_argument('--print_message', type=bool, default=False,
                        help='Print message ?')
    parser.add_argument('--reg', type=bool, default=False,
                        help='Add regularization ?')

    # Compositionality
    parser.add_argument('--n_attributes', type=int, default=3,
                        help='Number of attributes (default: 2)')
    parser.add_argument('--n_values', type=int, default=3,
                        help='Number of values by attribute')
    parser.add_argument('--probs_attributes', type=str, default="uniform",
                        help='Sampling prob for each att')

    # Propre
    parser.add_argument('--self_weight', type=float, default=1.,help='Weight for self')
    parser.add_argument('--cross_weight', type=float, default=1.,help='Weight for cross')
    parser.add_argument('--imitation_weight', type=float, default=1.,help='Weight for imitation')
    parser.add_argument('--optim_mode', type=str, default="cross",help='Choice for losses')

    # Baseline/reward mode
    parser.add_argument('--reward_mode', type=str, default="neg_loss",help='Choice for reward')
    parser.add_argument('--baseline_mode', type=str, default="new",help='Choice for baseline')

    # Split
    parser.add_argument('--split_proportion', type=float, default=0.8,help='Train/test split prop')

    # Test
    parser.add_argument('--agent_1_weights', type=str,help='Path to agent weights')
    parser.add_argument('--agent_2_weights', type=str,help='Path to agent weights')
    parser.add_argument('--compositionality', type=bool,default=False,help='Compositionality game ?')
    parser.add_argument('--n_sampling', type=int,default=1000,help='Number of message sampling for estimation')
    parser.add_argument('--noise_prob', type=float,default=0.1,help='Noise level')


    args = core.init(parser, params)

    return args

def build_compo_dataset(n_values,n_attributes):
    one_hots = np.eye(n_values)

    val=np.arange(n_values)
    combination=list(itertools.product(val,repeat=n_attributes))

    dataset=[]

    for i in range(len(combination)):
      new_input=np.zeros(0)
      for j in combination[i]:
        new_input=np.concatenate((new_input,one_hots[j]))
      dataset.append(new_input)

    return dataset

def estimate_policy(agent,n_sampling,n_features,vocab_size,max_len,device):

    """
    Estimate agent (speaker module) policies based on message samples
    """

    dataset = [[torch.eye(n_features).to(device), None]]

    onehots=torch.eye(vocab_size)

    policy=torch.zeros(n_features,max_len,vocab_size)

    for _ in range(n_sampling):
        messages = sample_messages(agent,dataset,device)

        for i in range(messages.size(0)):
            for j in range(messages.size(1)):
                policy[i,j,:]+=onehots[messages[i,j]]

    policy/=n_sampling

    return policy


def main(params):
    opts = get_params(params)
    device = opts.device

    force_eos = opts.force_eos == 1

    if opts.probs=="uniform":
        probs=[]
        probs_by_att = np.ones(opts.n_values)
        probs_by_att /= probs_by_att.sum()
        for i in range(opts.n_attributes):
            probs.append(probs_by_att)
        probs_attributes=[1]*opts.n_attributes

    agent_1=AgentBaseline2(vocab_size=opts.vocab_size,
                        n_features=opts.n_features,
                        max_len=opts.max_len,
                        embed_dim=opts.sender_embedding,
                        sender_hidden_size=opts.sender_hidden,
                        receiver_hidden_size=opts.receiver_hidden,
                        sender_cell=opts.sender_cell,
                        receiver_cell=opts.receiver_cell,
                        sender_num_layers=opts.sender_num_layers,
                        receiver_num_layers=opts.receiver_num_layers,
                        force_eos=force_eos)


    agent_1.load_state_dict(torch.load(opts.agent_1_weights,map_location=torch.device('cpu')))
    agent_1.to(device)

    agent_2=AgentBaseline2(vocab_size=opts.vocab_size,
                        n_features=opts.n_features,
                        max_len=opts.max_len,
                        embed_dim=opts.sender_embedding,
                        sender_hidden_size=opts.sender_hidden,
                        receiver_hidden_size=opts.receiver_hidden,
                        sender_cell=opts.sender_cell,
                        receiver_cell=opts.receiver_cell,
                        sender_num_layers=opts.sender_num_layers,
                        receiver_num_layers=opts.receiver_num_layers,
                        force_eos=force_eos)


    agent_2.load_state_dict(torch.load(opts.agent_2_weights,map_location=torch.device('cpu')))
    agent_2.to(device)

    policy_1 = estimate_policy(agent_1,opts.n_sampling,opts.n_features,opts.vocab_size,opts.max_len,device)
    policy_2 = estimate_policy(agent_2,opts.n_sampling,opts.n_features,opts.vocab_size,opts.max_len,device)

    #def L2_sim(p, q):
    #    l2=(p-q)**2
    #    l2=np.sum(l2,axis=2)
    #    return np.mean(l2)

    #l2=L2_sim(policy_1.cpu().numpy(),policy_2.cpu().numpy())
    np.save(opts.dir_save+'/training_info/policy_1.npy',policy_1.cpu().numpy())
    np.save(opts.dir_save+'/training_info/policy_2.npy',policy_2.cpu().numpy())

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
