# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from os import path
import json
import argparse
import numpy as np
import torch.utils.data
import torch.nn.functional as F
import src.core as core
from src.core import EarlyStopperAccuracy
from src.zoo.dialog.features import OneHotLoader, UniformLoader
from src.zoo.dialog.archs import Sender, Receiver
from src.core.reinforce_wrappers import RnnReceiverImpatient
from src.core.reinforce_wrappers import SenderImpatientReceiverRnnReinforce
from src.core.util import dump_sender_receiver_impatient,levenshtein, convert_messages_to_numpy
#Dialog
from src.core.reinforce_wrappers import RnnReceiverWithHiddenStates,RnnSenderReinforceModel3
from src.core.reinforce_wrappers import  AgentBaseline,AgentModel2,AgentModel3
from src.core.reinforce_wrappers import DialogReinforceBaseline,DialogReinforceModel1,DialogReinforceModel2, DialogReinforceModel3,DialogReinforceModel4,PretrainAgent,DialogReinforceModel6
from src.core.util import dump_sender_receiver_dialog,dump_sender_receiver_dialog_model_1,dump_sender_receiver_dialog_model_2,dump_pretraining_u,dump_sender_receiver_dialog_model_6
from src.core.util import test_receiver_evolution_core
from src.core.trainers import TrainerDialogModel1, TrainerDialogModel2, TrainerDialogModel3,TrainerDialogModel4,TrainerDialogModel5,TrainerPretraining,TrainerDialogModel6

# Propre
from src.core.reinforce_wrappers import AgentBaseline2,AgentSharedRNN,AgentSharedEmbedding,AgentBaselineKL,AgentPol
from src.core.reinforce_wrappers import DialogReinforce,DialogReinforceBis,DialogReinforceKL,DialogReinforceMemory,DialogReinforceSingleListener
from src.core.trainers import TrainerDialog,TrainerDialogAsymLR,TrainerDialog4Optim,TrainerDialogAsymStep

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
    # Dialog
    parser.add_argument('--dialog', type=bool, default=True,
                        help='if dialog game')
    parser.add_argument('--model', type=str, default="baseline",
                        help='dialog agents model')
    # If entropy scheduling
    parser.add_argument('--entropy_scheduling', type=bool, default=False,
                        help='Schedule entropy coefficient')
    parser.add_argument('--pretrain_agent_1', type=bool, default=False,
                        help='Schedule entropy coefficient')
    parser.add_argument('--pretrain_agent_2', type=bool, default=False,
                        help='Schedule entropy coefficient')
    parser.add_argument('--imitate', type=bool, default=False,
                        help='Imitation')

    # Propre
    parser.add_argument('--self_weight', type=float, default=1.,help='Weight for self')
    parser.add_argument('--cross_weight', type=float, default=1.,help='Weight for cross')
    parser.add_argument('--imitation_weight', type=float, default=1.,help='Weight for imitation')
    parser.add_argument('--optim_mode', type=str, default="cross",help='Choice for losses')

    # Baseline/reward mode
    parser.add_argument('--reward_mode', type=str, default="neg_loss",help='Choice for reward')
    parser.add_argument('--baseline_mode', type=str, default="new",help='Choice for baseline')

    # Asymmetric lr
    parser.add_argument('--sender_lr', type=float, default=0.0005,help='Lr for senders (for asymmetric expe)')
    parser.add_argument('--receiver_lr', type=float, default=0.0005,help='Lr for receivers (for asymmetric expe)')

    # Asym learning
    parser.add_argument('--N_speaker', type=float, default=10,help='Number of speaker training step')
    parser.add_argument('--N_listener', type=float, default=10,help='Number of listener training step')

    args = core.init(parser, params)

    return args


def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    acc = (receiver_output.argmax(dim=1) == sender_input.argmax(dim=1)).detach().float()
    loss = F.cross_entropy(receiver_output, sender_input.argmax(dim=1), reduction="none")
    return loss, {'acc': acc}

def loss_impatient(sender_input, _message, message_length, _receiver_input, receiver_output, _labels):

    """
    Compute the loss function for the Impatient Listener.
    It is equal to the average cross entropy of all the intermediate predictions

    Params:
    - sender_input: ground truth 1-hot vector | size=(batch_size,n_features)
    - receiver_output: receiver predictions | size=(batch_size,max_len,n_features)
    - message_lengh: message length | size=(batch_size)

    Returns:
    - loss: |  size= ????
    - {acc:acc}: mean accuracy | size=(batch_size)
    - crible_acc: accuracy by position | size=(batch_size,max_len)
    """

    # 1. len_mask selects only the symbols before EOS-token
    to_onehot=torch.eye(_message.size(1)).to("cuda")
    to_onehot=torch.cat((to_onehot,torch.zeros((1,_message.size(1))).to("cuda")),0)
    len_mask=[]
    for i in range(message_length.size(0)):
      len_mask.append(to_onehot[message_length[i]])
    len_mask=torch.stack(len_mask,dim=0)

    len_mask=torch.cumsum(len_mask,dim=1)
    len_mask=torch.ones(len_mask.size()).to("cuda").add_(-len_mask)

    # 2. coef applies weights on each position. By default it is equal
    coef=(1/message_length.to(float)).repeat(_message.size(1),1).transpose(1,0) # useless ?
    len_mask.mul_((coef))
    len_mask.mul_((1/len_mask.sum(1)).repeat((_message.size(1),1)).transpose(1,0))

    # Test: change positional wieghts
    #coef2=coef*torch.arange(_message.size(1),0,-1).repeat(_message.size(0),1).to("cuda")


    # 3. crible_acc gathers accuracy for each input/position, crible_loss gathers losses for each input/position
    crible_acc=torch.zeros(size=_message.size()).to("cuda")
    crible_loss=torch.zeros(size=_message.size()).to("cuda")

    for i in range(receiver_output.size(1)):
      crible_acc[:,i].add_((receiver_output[:,i,:].argmax(dim=1) == sender_input.argmax(dim=1)).detach().float())
      crible_loss[:,i].add_(F.cross_entropy(receiver_output[:,i,:], sender_input.argmax(dim=1), reduction="none"))

    # 4. Apply mask to remove the positions after EOS-token
    acc=crible_acc*len_mask
    loss=crible_loss*len_mask

    acc = acc.sum(1)
    loss= loss.sum(1)

    return loss, {'acc': acc}, crible_acc

def loss_understanding(sender_input, receiver_output):
    acc = (receiver_output.argmax(dim=1) == sender_input.argmax(dim=1)).detach().float()
    loss = F.cross_entropy(receiver_output, sender_input.argmax(dim=1), reduction="none")
    return loss, {'acc': acc}

def loss_message_imitation(message,prob_reconstruction,message_lengths):

    # 1. len_mask selects only the symbols before EOS-token
    if message_lengths is not None:
        to_onehot=torch.eye(message.size(1)).to(message.device)
        to_onehot=torch.cat((to_onehot,torch.zeros((1,message.size(1))).to(message.device)),0)
        len_mask=[]
        for i in range(message_lengths.size(0)):
          len_mask.append(to_onehot[message_lengths[i]])
        len_mask=torch.stack(len_mask,dim=0)

        len_mask=torch.cumsum(len_mask,dim=1)
        len_mask=torch.ones(len_mask.size()).to(message.device).add_(-len_mask)

    # Reconstruction task
    batch_size=message.size(0)
    prob_reconstruction = prob_reconstruction.transpose(1,2)
    prob_reconstruction = prob_reconstruction.reshape((prob_reconstruction.size(0)*prob_reconstruction.size(1),prob_reconstruction.size(2)))
    message = message.reshape((message.size(0)*message.size(1)))

    acc_imitation = (prob_reconstruction.argmax(dim=1) == message).detach().float()
    loss_imitation = F.cross_entropy(torch.log(prob_reconstruction), message, reduction="none")

    loss_imitation = loss_imitation.reshape((batch_size,loss_imitation.size(0)//batch_size))
    acc_imitation = acc_imitation.reshape((batch_size,acc_imitation.size(0)//batch_size))

    if message_lengths is not None:

      loss_imitation = (loss_imitation*len_mask)/(len_mask.sum(1).unsqueeze(1)) # Add EOS mask
      acc_imitation = (acc_imitation*len_mask)/(len_mask.sum(1).unsqueeze(1))

      loss_imitation=loss_imitation.sum(1)
      acc_imitation=acc_imitation.sum(1)
    else:
      loss_imitation = loss_imitation.mean(dim=1)
      acc_imitation = acc_imitation.mean(dim=1)

    return loss_imitation, {'acc_imitation':acc_imitation}


def loss_model_2(sender_input, _message, message_length, _receiver_input, receiver_output, output_lm, _labels):

    """
    Compute the loss function for the Impatient Listener.
    It is equal to the average cross entropy of all the intermediate predictions

    Params:
    - sender_input: ground truth 1-hot vector | size=(batch_size,n_features)
    - receiver_output: receiver predictions | size=(batch_size,max_len,n_features)
    - message_lengh: message length | size=(batch_size)

    Returns:
    - loss: |  size= ????
    - {acc:acc}: mean accuracy | size=(batch_size)
    - crible_acc: accuracy by position | size=(batch_size,max_len)
    """

    # 1. len_mask selects only the symbols before EOS-token
    #to_onehot=torch.eye(_message.size(1)).to("cuda")
    #to_onehot=torch.cat((to_onehot,torch.zeros((1,_message.size(1))).to("cuda")),0)
    #len_mask=[]
    #for i in range(message_length.size(0)):
    #  len_mask.append(to_onehot[message_length[i]])
    #len_mask=torch.stack(len_mask,dim=0)

    #len_mask=torch.cumsum(len_mask,dim=1)
    #len_mask=torch.ones(len_mask.size()).to("cuda").add_(-len_mask)

    # 2. coef applies weights on each position. By default it is equal
    #coef=(1/message_length.to(float)).repeat(_message.size(1),1).transpose(1,0) # useless ?
    #len_mask.mul_((coef))
    #len_mask.mul_((1/len_mask.sum(1)).repeat((_message.size(1),1)).transpose(1,0))

    # Test: change positional wieghts
    #coef2=coef*torch.arange(_message.size(1),0,-1).repeat(_message.size(0),1).to("cuda")

    # 3. Loss candidate
    acc = (receiver_output[:,-1,:].argmax(dim=1) == sender_input.argmax(dim=1)).detach().float()
    loss = F.cross_entropy(receiver_output[:,-1,:], sender_input.argmax(dim=1), reduction="none")

    # 4. Loss language model

    output_lm = output_lm[:,:-1,:]
    target_lm = _message[:,1:message_length.max()]

    output_lm=output_lm.reshape((output_lm.size(0)*output_lm.size(1),output_lm.size(2)))
    target_lm=target_lm.reshape(target_lm.size(0)*target_lm.size(1))

    acc_lm = (output_lm.argmax(dim=1) == target_lm).detach().float()
    loss_lm = F.cross_entropy(output_lm, target_lm, reduction="none")

    loss_lm = loss_lm.reshape((loss.size(0),loss_lm.size(0)//loss.size(0)))
    acc_lm = acc_lm.reshape((acc.size(0),acc_lm.size(0)//acc.size(0)))

    loss_lm = loss_lm.mean(dim=1)
    acc_lm = acc_lm.mean(dim=1)

    return loss, loss_lm, {'acc': acc, "acc_lm": acc_lm}


def loss_model_3(sender_input, message, receiver_input, receiver_output,message_reconstruction,prob_reconstruction, labels,message_length=None):

    """
    Compute the loss function for the Impatient Listener.
    It is equal to the average cross entropy of all the intermediate predictions

    Params:
    - sender_input: ground truth 1-hot vector | size=(batch_size,n_features)
    - receiver_output: receiver predictions | size=(batch_size,max_len,n_features)
    - message_lengh: message length | size=(batch_size)

    Returns:
    - loss: |  size= ????
    - {acc:acc}: mean accuracy | size=(batch_size)
    - crible_acc: accuracy by position | size=(batch_size,max_len)
    """

    # 1. len_mask selects only the symbols before EOS-token
    if message_length is not None:
        to_onehot=torch.eye(message.size(1)).to("cuda")
        to_onehot=torch.cat((to_onehot,torch.zeros((1,message.size(1))).to("cuda")),0)
        len_mask=[]
        for i in range(message_length.size(0)):
          len_mask.append(to_onehot[message_length[i]])
        len_mask=torch.stack(len_mask,dim=0)

        len_mask=torch.cumsum(len_mask,dim=1)
        len_mask=torch.ones(len_mask.size()).to("cuda").add_(-len_mask)

    # Communication task

    acc = (receiver_output.argmax(dim=1) == sender_input.argmax(dim=1)).detach().float()
    loss = F.cross_entropy(receiver_output, sender_input.argmax(dim=1), reduction="none")

    # Reconstruction task
    prob_reconstruction = prob_reconstruction.transpose(1,2)
    prob_reconstruction = prob_reconstruction.reshape((prob_reconstruction.size(0)*prob_reconstruction.size(1),prob_reconstruction.size(2)))
    message = message.reshape((message.size(0)*message.size(1)))

    acc_imitation = (prob_reconstruction.argmax(dim=1) == message).detach().float()
    loss_imitation = F.cross_entropy(torch.log(prob_reconstruction), message, reduction="none")

    loss_imitation = loss_imitation.reshape((loss.size(0),loss_imitation.size(0)//loss.size(0)))
    acc_imitation = acc_imitation.reshape((acc.size(0),acc_imitation.size(0)//acc.size(0)))

    if message_length is not None:

      loss_imitation = (loss_imitation*len_mask)/(len_mask.sum(1).unsqueeze(1)) # Add EOS mask
      acc_imitation = (acc_imitation*len_mask)/(len_mask.sum(1).unsqueeze(1))

      loss_imitation=loss_imitation.sum(1)
      acc_imitation=acc_imitation.sum(1)
    else:
      loss_imitation = loss_imitation.mean(dim=1)
      acc_imitation = acc_imitation.mean(dim=1)

    return loss,loss_imitation, {'acc': acc}

def loss_pretraining(sender_input, message, pretrained_messages, receiver_input, receiver_output,message_reconstruction,prob_reconstruction, labels):

    """
    Compute the loss function for the Impatient Listener.
    It is equal to the average cross entropy of all the intermediate predictions

    Params:
    - sender_input: ground truth 1-hot vector | size=(batch_size,n_features)
    - receiver_output: receiver predictions | size=(batch_size,max_len,n_features)
    - message_lengh: message length | size=(batch_size)

    Returns:
    - loss: |  size= ????
    - {acc:acc}: mean accuracy | size=(batch_size)
    - crible_acc: accuracy by position | size=(batch_size,max_len)
    """
    # Communication task

    acc = (receiver_output.argmax(dim=1) == sender_input.argmax(dim=1)).detach().float()
    loss = F.cross_entropy(receiver_output, sender_input.argmax(dim=1), reduction="none")

    if pretrained_messages is not None:
        # Reconstruction task
        prob_reconstruction = prob_reconstruction.transpose(1,2)
        prob_reconstruction = prob_reconstruction.reshape((prob_reconstruction.size(0)*prob_reconstruction.size(1),prob_reconstruction.size(2)))
        pretrained_messages = pretrained_messages.reshape((pretrained_messages.size(0)*pretrained_messages.size(1)))

        acc_imitation = (prob_reconstruction.argmax(dim=1) == pretrained_messages).detach().float()
        loss_imitation = F.cross_entropy(torch.log(prob_reconstruction), pretrained_messages, reduction="none")

        loss_imitation = loss_imitation.mean(dim=0) # Add EOS mask
        acc_imitation = acc_imitation.mean(dim=0)
    else:
        loss_imitation=0.
        acc_imitation=0.

    return loss,loss_imitation, {'acc': acc}

def dump(game, n_features, device, gs_mode, epoch):
    # tiny "dataset"
    dataset = [[torch.eye(n_features).to(device), None]]

    sender_inputs, messages, receiver_inputs, receiver_outputs, _ = \
        core.dump_sender_receiver(game, dataset, gs=gs_mode, device=device, variable_length=True)

    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs, messages, receiver_outputs):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec[int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%50==0:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features

    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    return acc_vec, messages

def dump_dialog(game, n_features, device, gs_mode, epoch,past_messages_1=None,past_messages_2=None):
    # tiny "dataset"
    dataset = [[torch.eye(n_features).to(device), None]]

    sender_inputs_1, messages_1, receiver_inputs_1, receiver_outputs_1, \
    sender_inputs_2, messages_2, receiver_inputs_2, receiver_outputs_2, _ = \
        dump_sender_receiver_dialog(game, dataset, gs=gs_mode, device=device, variable_length=True)


    print("Language 1 (Agent 1 -> Agent 2)")

    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec_1=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs_1, messages_1, receiver_outputs_1):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec_1[int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%100==0:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features

    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    print("Language 2 (Agent 2 -> Agent 1)")

    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec_2=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs_2, messages_2, receiver_outputs_2):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec_2[int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%100==0:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features

    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    print("Similarity between language = {}".format(np.mean([levenshtein(messages_1[i],messages_2[i]) for i in range(len(messages_1))])),flush=True)

    if past_messages_1 is not None:
        print("Similarity evo language 1 = {}".format(np.mean([levenshtein(messages_1[i],past_messages_1[i]) for i in range(len(messages_1))])),flush=True)
    if past_messages_2 is not None:
        print("Similarity evo language 2 = {}".format(np.mean([levenshtein(messages_2[i],past_messages_2[i]) for i in range(len(messages_2))])),flush=True)


    return acc_vec_1, messages_1, acc_vec_2, messages_2

def dump_dialog_model_1(game, n_features, device, gs_mode, epoch,past_messages_1=None,past_messages_2=None):
    # tiny "dataset"
    dataset = [[torch.eye(n_features).to(device), None]]

    sender_inputs_1, messages_1, receiver_inputs_1, receiver_outputs_11,receiver_outputs_12, \
    sender_inputs_2, messages_2, receiver_inputs_2, receiver_outputs_21,receiver_outputs_22, _ = \
        dump_sender_receiver_dialog_model_1(game, dataset, gs=gs_mode, device=device, variable_length=True)


    print("Language 1 (Agent 1 -> Agent 2)")

    "1->2"
    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec_1=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs_1, messages_1, receiver_outputs_12):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec_1[int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%20==0:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features

    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    "1->1"
    print("internal listener")
    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec_11=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs_1, messages_1, receiver_outputs_11):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec_11  [int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%20==0:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features
    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    print("Language 2 (Agent 2 -> Agent 1)")

    "2->1"

    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec_2=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs_2, messages_2, receiver_outputs_21):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec_2[int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%20==0:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features

    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    print("internal listener")
    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec_22=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs_2, messages_2, receiver_outputs_22):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec_22[int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%20==0:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features
    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    #messages_1=[m[:np.min(np.where(m==0)[0])+1] if len(np.where(m==0)[0])>0 is not None else m for m in messages_1]
    #messages_2=[m[:np.min(np.where(m==0)[0])+1] if len(np.where(m==0)[0])>0 is not None else m for m in messages_2]


    print("Similarity between language = {}".format(np.mean([levenshtein(messages_1[i],messages_2[i]) for i in range(len(messages_1))])),flush=True)

    if past_messages_1 is not None:
        print("Similarity evo language 1 = {}".format(np.mean([levenshtein(messages_1[i],past_messages_1[i]) for i in range(len(messages_1))])),flush=True)
    if past_messages_2 is not None:
        print("Similarity evo language 2 = {}".format(np.mean([levenshtein(messages_2[i],past_messages_2[i]) for i in range(len(messages_2))])),flush=True)


    return messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22

def dump_dialog_model_2(game, n_features, device, gs_mode, epoch,past_messages_1=None,past_messages_2=None):
    # tiny "dataset"
    dataset = [[torch.eye(n_features).to(device), None]]

    sender_inputs_1, messages_1, receiver_inputs_1, receiver_outputs_1, \
    sender_inputs_2, messages_2, receiver_inputs_2, receiver_outputs_2, _ = \
        dump_sender_receiver_dialog_model_2(game, dataset, gs=gs_mode, device=device, variable_length=True)


    print("Language 1 (Agent 1 -> Agent 2)")

    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec_1=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs_1, messages_1, receiver_outputs_1):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec_1[int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%50==0:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features

    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    print("Language 2 (Agent 2 -> Agent 1)")

    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec_2=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs_2, messages_2, receiver_outputs_2):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec_2[int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%50==0:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features

    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    return acc_vec_1, messages_1, acc_vec_2, messages_2

def dump_dialog_model_6(game, n_features, device, gs_mode, epoch,past_messages_1=None,past_messages_2=None):
    # tiny "dataset"
    dataset = [[torch.eye(n_features).to(device), None]]

    sender_inputs_1, messages_1, receiver_inputs_1, receiver_outputs_11,receiver_outputs_12, \
    sender_inputs_2, messages_2, receiver_inputs_2, receiver_outputs_21,receiver_outputs_22, _ = \
        dump_sender_receiver_dialog_model_6(game, dataset, gs=gs_mode, device=device, variable_length=True)


    print("Language 1 (Agent 1 -> Agent 2)")

    "1->2"
    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec_1=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs_1, messages_1, receiver_outputs_12):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec_1[int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%100==0:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features

    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    "1->1"
    print("internal listener")
    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec_11=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs_1, messages_1, receiver_outputs_11):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec_11  [int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%100==0:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features
    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    print("Language 2 (Agent 2 -> Agent 1)")

    "2->1"

    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec_2=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs_2, messages_2, receiver_outputs_21):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec_2[int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%100==0:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features

    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    print("internal listener")
    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec_22=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs_2, messages_2, receiver_outputs_22):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec_22[int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%100==0:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features
    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    #messages_1=[m[:np.min(np.where(m==0)[0])+1] if len(np.where(m==0)[0])>0 is not None else m for m in messages_1]
    #messages_2=[m[:np.min(np.where(m==0)[0])+1] if len(np.where(m==0)[0])>0 is not None else m for m in messages_2]

    similarity_messages=np.mean([levenshtein(messages_1[i],messages_2[i])/np.max([len(messages_1[i]),len(messages_2[i])]) for i in range(len(messages_1))])

    print("Similarity between language = {}".format(similarity_messages),flush=True)

    if past_messages_1 is not None:
        print("Similarity evo language 1 = {}".format(np.mean([levenshtein(messages_1[i],past_messages_1[i]) for i in range(len(messages_1))])),flush=True)
    if past_messages_2 is not None:
        print("Similarity evo language 2 = {}".format(np.mean([levenshtein(messages_2[i],past_messages_2[i]) for i in range(len(messages_2))])),flush=True)


    return messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22, similarity_messages

def test_receiver_evolution(game,messages_test,device,gs_mode,past_preds_1,past_preds_2):
    # tiny "dataset"

    receiver_outputs_1,receiver_outputs_2 = \
        test_receiver_evolution_core(game,messages_test,gs=gs_mode, device=device, variable_length=True)

    preds_1=[]
    preds_2=[]

    for i in range(len(receiver_outputs_1)):
        preds_1.append(receiver_outputs_1[i].argmax().cpu().numpy())
        preds_2.append(receiver_outputs_2[i].argmax().cpu().numpy())

    sim_pred_1 = np.mean((np.array(preds_1)-np.array(past_preds_1))==0)
    sim_pred_2 = np.mean((np.array(preds_2)-np.array(past_preds_2))==0)

    print("Similarity predictions 1 = {}".format(sim_pred_1))
    print("Similarity predictions 2 = {}".format(sim_pred_2))

    return preds_1,preds_2,sim_pred_1,sim_pred_2

def dump_pretraining(game, n_features,pretrained_messages, device, gs_mode, epoch):
    # tiny "dataset"
    dataset = [[torch.eye(n_features).to(device), None]]

    sender_inputs_1, messages_1, receiver_inputs_1, receiver_outputs_1, _ = \
        dump_pretraining_u(game, dataset, gs=gs_mode, device=device, variable_length=True)


    print("Language 1 (Agent 1 -> Agent 1)")

    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec_1=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs_1, messages_1, receiver_outputs_1):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec_1[int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%10==0:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features

    print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    pretrained_messages=[m[:np.min(np.where(m==0)[0])+1] if len(np.where(m==0)[0])>0 is not None else m for m in pretrained_messages]

    print("Similarity between language = {}".format(np.mean([levenshtein(messages_1[i],pretrained_messages[i]) for i in range(len(messages_1))])),flush=True)


    return acc_vec_1, messages_1

def dump_impatient(game, n_features, device, gs_mode,epoch):
    # tiny "dataset"
    dataset = [[torch.eye(n_features).to(device), None]]

    sender_inputs, messages, receiver_inputs, receiver_outputs, _ = \
        dump_sender_receiver_impatient(game, dataset, gs=gs_mode, device=device, variable_length=True)

    unif_acc = 0.
    powerlaw_acc = 0.
    powerlaw_probs = 1 / np.arange(1, n_features+1, dtype=np.float32)
    powerlaw_probs /= powerlaw_probs.sum()

    acc_vec=np.zeros(n_features)

    for sender_input, message, receiver_output in zip(sender_inputs, messages, receiver_outputs):
        input_symbol = sender_input.argmax()
        output_symbol = receiver_output.argmax()
        acc = (input_symbol == output_symbol).float().item()

        acc_vec[int(input_symbol)]=acc

        unif_acc += acc
        powerlaw_acc += powerlaw_probs[input_symbol] * acc
        if epoch%100==0:
            print(f'input: {input_symbol.item()} -> message: {",".join([str(x.item()) for x in message])} -> output: {output_symbol.item()}', flush=True)

    unif_acc /= n_features

    #print(f'Mean accuracy wrt uniform distribution is {unif_acc}')
    #print(f'Mean accuracy wrt powerlaw distribution is {powerlaw_acc}')
    if epoch%25==0:
        print(json.dumps({'powerlaw': powerlaw_acc, 'unif': unif_acc}))

    return acc_vec, messages

def main(params):
    print(torch.cuda.is_available())
    opts = get_params(params)
    print(opts, flush=True)
    device = opts.device

    force_eos = opts.force_eos == 1

    if opts.probs == 'uniform':
        probs = np.ones(opts.n_features)
    elif opts.probs == 'powerlaw':
        probs = 1 / np.arange(1, opts.n_features+1, dtype=np.float32)
    else:
        probs = np.array([float(x) for x in opts.probs.split(',')], dtype=np.float32)

    probs /= probs.sum()

    print('the probs are: ', probs, flush=True)

    train_loader = OneHotLoader(n_features=opts.n_features, batch_size=opts.batch_size,
                                batches_per_epoch=opts.batches_per_epoch, probs=probs)

    # single batches with 1s on the diag
    test_loader = UniformLoader(opts.n_features)

    if not opts.dialog:
        if opts.sender_cell == 'transformer':
            sender = Sender(n_features=opts.n_features, n_hidden=opts.sender_embedding)
            sender = core.TransformerSenderReinforce(agent=sender, vocab_size=opts.vocab_size,
                                                     embed_dim=opts.sender_embedding, max_len=opts.max_len,
                                                     num_layers=opts.sender_num_layers, num_heads=opts.sender_num_heads,
                                                     hidden_size=opts.sender_hidden,
                                                     force_eos=opts.force_eos,
                                                     generate_style=opts.sender_generate_style,
                                                     causal=opts.causal_sender)
        else:
            sender = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)

            sender = core.RnnSenderReinforce(sender,
                                       opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                       cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                       force_eos=force_eos)
        if opts.receiver_cell == 'transformer':
            receiver = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_embedding)
            receiver = core.TransformerReceiverDeterministic(receiver, opts.vocab_size, opts.max_len,
                                                             opts.receiver_embedding, opts.receiver_num_heads, opts.receiver_hidden,
                                                             opts.receiver_num_layers, causal=opts.causal_receiver)
        else:

            if not opts.impatient:
              receiver = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)
              receiver = core.RnnReceiverDeterministic(receiver, opts.vocab_size, opts.receiver_embedding,
                                                     opts.receiver_hidden, cell=opts.receiver_cell,
                                                     num_layers=opts.receiver_num_layers)
            else:
              receiver = Receiver(n_features=opts.receiver_hidden, n_hidden=opts.vocab_size)
              # If impatient 1
              receiver = RnnReceiverImpatient(receiver, opts.vocab_size, opts.receiver_embedding,
                                                opts.receiver_hidden, cell=opts.receiver_cell,
                                                num_layers=opts.receiver_num_layers, max_len=opts.max_len, n_features=opts.n_features)

        if not opts.impatient:
            game = core.SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff,
                                                   receiver_entropy_coeff=opts.receiver_entropy_coeff,
                                                   length_cost=opts.length_cost,unigram_penalty=opts.unigram_pen,reg=opts.reg)
        else:
            game = SenderImpatientReceiverRnnReinforce(sender, receiver, loss_impatient, sender_entropy_coeff=opts.sender_entropy_coeff,
                                                       receiver_entropy_coeff=opts.receiver_entropy_coeff,
                                                       length_cost=opts.length_cost,unigram_penalty=opts.unigram_pen,reg=opts.reg)

        optimizer = core.build_optimizer(game.parameters())

        trainer = Trainer(game=game, optimizer_1=optimizer_1, optimizer_2=optimizer_2, train_data=train_loader,
                               validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

    if opts.dialog:

        if opts.model=="baseline":

            "Agent 1"

            sender_1 = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)
            sender_1 = core.RnnSenderReinforce(sender_1,
                                       opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                       cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                       force_eos=force_eos)

            receiver_1 = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)
            receiver_1 = core.RnnReceiverDeterministic(receiver_1, opts.vocab_size, opts.receiver_embedding,
                                                   opts.receiver_hidden, cell=opts.receiver_cell,
                                                   num_layers=opts.receiver_num_layers)

            agent_1=AgentBaseline(receiver = receiver_1, sender = sender_1)

            "Agent 2"

            sender_2 = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)
            sender_2 = core.RnnSenderReinforce(sender_2,
                                       opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                       cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                       force_eos=force_eos)

            receiver_2 = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)
            receiver_2 = core.RnnReceiverDeterministic(receiver_2, opts.vocab_size, opts.receiver_embedding,
                                                   opts.receiver_hidden, cell=opts.receiver_cell,
                                                   num_layers=opts.receiver_num_layers)

            agent_2=AgentBaseline(receiver = receiver_2, sender = sender_2)

            game = DialogReinforceBaseline(Agent_1=agent_1,
                                           Agent_2=agent_2,
                                           loss=loss,
                                           sender_entropy_coeff_1=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff_1=opts.receiver_entropy_coeff,
                                           sender_entropy_coeff_2=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff_2=opts.receiver_entropy_coeff,
                                           loss_weights=[0.5,0.5],
                                           length_cost=0.0,
                                           unigram_penalty=0.0,
                                           reg=False,
                                           device=device)

            optimizer_1 = core.build_optimizer(list(game.agent_1.sender.parameters())+list(game.agent_2.receiver.parameters()))
            optimizer_2 = core.build_optimizer(list(game.agent_2.sender.parameters())+list(game.agent_1.receiver.parameters()))


            trainer = TrainerDialogBaseline(game=game, optimizer_1=optimizer_1, optimizer_2=optimizer_2, train_data=train_loader,
                                            validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        elif opts.model=="model_1":

            "Agent 1"

            sender_1 = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)
            sender_1 = core.RnnSenderReinforce(sender_1,
                                       opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                       cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                       force_eos=force_eos)

            receiver_1 = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)
            receiver_1 = core.RnnReceiverDeterministic(receiver_1, opts.vocab_size, opts.receiver_embedding,
                                                   opts.receiver_hidden, cell=opts.receiver_cell,
                                                   num_layers=opts.receiver_num_layers)

            agent_1=AgentBaseline(receiver = receiver_1, sender = sender_1)

            "Agent 2"

            sender_2 = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)
            sender_2 = core.RnnSenderReinforce(sender_2,
                                       opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                       cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                       force_eos=force_eos)

            receiver_2 = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)
            receiver_2 = core.RnnReceiverDeterministic(receiver_2, opts.vocab_size, opts.receiver_embedding,
                                                   opts.receiver_hidden, cell=opts.receiver_cell,
                                                   num_layers=opts.receiver_num_layers)

            agent_2=AgentBaseline(receiver = receiver_2, sender = sender_2)

            game = DialogReinforceModel1(Agent_1=agent_1,
                                           Agent_2=agent_2,
                                           loss=loss,
                                           sender_entropy_coeff_1=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff_1=opts.receiver_entropy_coeff,
                                           sender_entropy_coeff_2=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff_2=opts.receiver_entropy_coeff,
                                           length_cost=0.0,
                                           unigram_penalty=0.0,
                                           reg=False,
                                           device=device)

            optimizer_sender_1 = core.build_optimizer(list(game.agent_1.sender.parameters()))
            optimizer_receiver_1 = core.build_optimizer(list(game.agent_1.receiver.parameters()))
            optimizer_sender_2 = core.build_optimizer(list(game.agent_2.sender.parameters()))
            optimizer_receiver_2 = core.build_optimizer(list(game.agent_2.receiver.parameters()))

            trainer = TrainerDialogModel1(game=game, optimizer_sender_1=optimizer_sender_1, optimizer_sender_2=optimizer_sender_2, \
                                          optimizer_receiver_1=optimizer_receiver_1, optimizer_receiver_2=optimizer_receiver_2, train_data=train_loader, \
                                          validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        elif opts.model=="model_2":

            "Agent 1"

            sender_1 = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)
            sender_1 = core.RnnSenderReinforce(sender_1,
                                               opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                               cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                               force_eos=force_eos)

            receiver_1 = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)
            receiver_1 = RnnReceiverWithHiddenStates(receiver_1, opts.vocab_size, opts.receiver_embedding,
                                                                   opts.receiver_hidden, cell=opts.receiver_cell,
                                                                   num_layers=opts.receiver_num_layers,max_len=opts.max_len,n_features=opts.n_features)

            agent_1=AgentModel2(receiver = receiver_1, sender = sender_1)

            "Agent 2"

            sender_2 = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)
            sender_2 = core.RnnSenderReinforce(sender_2,
                                               opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                               cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                               force_eos=force_eos)

            receiver_2 = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)
            receiver_2 = RnnReceiverWithHiddenStates(receiver_2, opts.vocab_size, opts.receiver_embedding,
                                                           opts.receiver_hidden, cell=opts.receiver_cell,
                                                           num_layers=opts.receiver_num_layers,max_len=opts.max_len,n_features=opts.n_features)

            agent_2=AgentModel2(receiver = receiver_2, sender = sender_2)

            "Game"

            game = DialogReinforceModel2(Agent_1=agent_1,
                                           Agent_2=agent_2,
                                           loss=loss_model_2,
                                           sender_entropy_coeff_1=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff_1=opts.receiver_entropy_coeff,
                                           sender_entropy_coeff_2=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff_2=opts.receiver_entropy_coeff,
                                           length_cost=0.0,
                                           unigram_penalty=0.0,
                                           reg=False,
                                           device=device)

            optimizer_1 = core.build_optimizer(list(game.agent_1.sender.parameters())+list(game.agent_2.parameters()))
            optimizer_2 = core.build_optimizer(list(game.agent_2.sender.parameters())+list(game.agent_1.parameters()))

            trainer = TrainerDialogModel2(game=game, optimizer_1=optimizer_1, optimizer_2=optimizer_2, \
                                          train_data=train_loader, \
                                          validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        elif opts.model=="model_3":

            "Agent 1"

            sender_1 = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)
            sender_1 = RnnSenderReinforceModel3(sender_1,
                                               opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                               cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                               force_eos=force_eos)

            receiver_1 = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)
            receiver_1 = core.RnnReceiverDeterministic(receiver_1, opts.vocab_size, opts.receiver_embedding,
                                                                   opts.receiver_hidden, cell=opts.receiver_cell,
                                                                   num_layers=opts.receiver_num_layers)

            agent_1=AgentModel3(receiver = receiver_1, sender = sender_1)

            "Agent 2"

            sender_2 = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)
            sender_2 = RnnSenderReinforceModel3(sender_2,
                                               opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                               cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                               force_eos=force_eos)

            receiver_2 = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)
            receiver_2 = core.RnnReceiverDeterministic(receiver_2, opts.vocab_size, opts.receiver_embedding,
                                                           opts.receiver_hidden, cell=opts.receiver_cell,
                                                           num_layers=opts.receiver_num_layers)

            agent_2=AgentModel3(receiver = receiver_2, sender = sender_2)

            "Game"

            game = DialogReinforceModel3(Agent_1=agent_1,
                                           Agent_2=agent_2,
                                           loss=loss_model_3,
                                           sender_entropy_coeff_1=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff_1=opts.receiver_entropy_coeff,
                                           sender_entropy_coeff_2=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff_2=opts.receiver_entropy_coeff,
                                           length_cost=0.0,
                                           unigram_penalty=0.0,
                                           reg=False,
                                           device=device)

            optimizer_1_comm = core.build_optimizer(list(game.agent_1.sender.parameters())+list(game.agent_2.receiver.parameters()))
            optimizer_1_imitation = core.build_optimizer(list(game.agent_2.sender.parameters()))
            optimizer_2_comm = core.build_optimizer(list(game.agent_2.sender.parameters())+list(game.agent_1.receiver.parameters()))
            optimizer_2_imitation = core.build_optimizer(list(game.agent_1.sender.parameters()))

            trainer = TrainerDialogModel3(game=game, optimizer_1_comm=optimizer_1_comm, optimizer_1_imitation=optimizer_1_imitation, \
                                          optimizer_2_comm=optimizer_2_comm, optimizer_2_imitation=optimizer_2_imitation, \
                                          train_data=train_loader, \
                                          validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        elif opts.model=="model_4":

            "Agent 1"

            sender_1 = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)
            sender_1 = RnnSenderReinforceModel3(sender_1,
                                       opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                       cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                       force_eos=force_eos)

            receiver_1 = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)
            receiver_1 = core.RnnReceiverDeterministic(receiver_1, opts.vocab_size, opts.receiver_embedding,
                                                   opts.receiver_hidden, cell=opts.receiver_cell,
                                                   num_layers=opts.receiver_num_layers)

            agent_1=AgentModel3(receiver = receiver_1, sender = sender_1)

            "Agent 2"

            sender_2 = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)
            sender_2 =RnnSenderReinforceModel3(sender_2,
                                       opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                       cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                       force_eos=force_eos)

            receiver_2 = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)
            receiver_2 = core.RnnReceiverDeterministic(receiver_2, opts.vocab_size, opts.receiver_embedding,
                                                   opts.receiver_hidden, cell=opts.receiver_cell,
                                                   num_layers=opts.receiver_num_layers)

            agent_2=AgentModel3(receiver = receiver_2, sender = sender_2)

            if opts.pretrain_agent_1:

                print("Pretrain Agent 1")

                pretrained_messages=None

                game = PretrainAgent(Agent_1=agent_1,
                                   loss=loss_pretraining,
                                   pretrained_messages=pretrained_messages,
                                   sender_entropy_coeff_1=opts.sender_entropy_coeff,
                                   receiver_entropy_coeff_1=opts.receiver_entropy_coeff,
                                   n_features=opts.n_features,
                                   length_cost=0.0,
                                   unigram_penalty=0.0,
                                   reg=False,
                                   device=device)

                optimizer_sender_1 = core.build_optimizer(list(game.agent_1.sender.parameters()))
                optimizer_receiver_1 = core.build_optimizer(list(game.agent_1.receiver.parameters()))

                trainer = TrainerPretraining(game=game, optimizer_sender_1=optimizer_sender_1,
                                              optimizer_receiver_1=optimizer_receiver_1, train_data=train_loader, \
                                              validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

                trainer.train(n_epochs=20)

            if opts.pretrain_agent_2:

                print("Pretrain Agent 2")

                pretrained_messages=None

                game = PretrainAgent(Agent_1=agent_2,
                                   loss=loss_pretraining,
                                   pretrained_messages=pretrained_messages,
                                   sender_entropy_coeff_1=opts.sender_entropy_coeff,
                                   receiver_entropy_coeff_1=opts.receiver_entropy_coeff,
                                   n_features=opts.n_features,
                                   length_cost=0.0,
                                   unigram_penalty=0.0,
                                   reg=False,
                                   device=device)

                optimizer_sender_1 = core.build_optimizer(list(game.agent_1.sender.parameters()))
                optimizer_receiver_1 = core.build_optimizer(list(game.agent_1.receiver.parameters()))

                trainer = TrainerPretraining(game=game, optimizer_sender_1=optimizer_sender_1,
                                              optimizer_receiver_1=optimizer_receiver_1, train_data=train_loader, \
                                              validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

                trainer.train(n_epochs=20)

            game = DialogReinforceModel4(Agent_1=agent_1,
                                           Agent_2=agent_2,
                                           loss=loss_model_3,
                                           sender_entropy_coeff_1=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff_1=opts.receiver_entropy_coeff,
                                           sender_entropy_coeff_2=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff_2=opts.receiver_entropy_coeff,
                                           length_cost=0.0,
                                           unigram_penalty=0.0,
                                           reg=False,
                                           device=device)

            optimizer_sender_1 = core.build_optimizer(list(game.agent_1.sender.parameters()))
            optimizer_receiver_1 = core.build_optimizer(list(game.agent_1.receiver.parameters()))
            optimizer_sender_2 = core.build_optimizer(list(game.agent_2.sender.parameters()))
            optimizer_receiver_2 = core.build_optimizer(list(game.agent_2.receiver.parameters()))

            trainer = TrainerDialogModel4(game=game, optimizer_sender_1=optimizer_sender_1, optimizer_sender_2=optimizer_sender_2, \
                                          optimizer_receiver_1=optimizer_receiver_1, optimizer_receiver_2=optimizer_receiver_2, train_data=train_loader, \
                                          validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        elif opts.model=="model_5":

            "Agent 1"

            sender_1 = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)

            receiver_1 = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)


            agent_1=AgentSharedEmbedding(receiver=receiver_1,
                                        sender=sender_1,
                                        vocab_size=opts.vocab_size,
                                        max_len=opts.max_len,
                                        sender_embedding=opts.sender_embedding,
                                        sender_hidden=opts.sender_hidden,
                                        sender_cell=opts.sender_cell,
                                        sender_num_layers=opts.sender_num_layers,
                                        force_eos=force_eos,
                                        receiver_embedding=opts.receiver_embedding,
                                        receiver_hidden=opts.receiver_hidden,
                                        receiver_cell=opts.receiver_cell,
                                        receiver_num_layers=opts.receiver_num_layers)

            "Agent 2"

            sender_2 = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)

            receiver_2 = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)

            agent_2=AgentSharedEmbedding(receiver=receiver_2,
                                        sender=sender_2,
                                        vocab_size=opts.vocab_size,
                                        max_len=opts.max_len,
                                        sender_embedding=opts.sender_embedding,
                                        sender_hidden=opts.sender_hidden,
                                        sender_cell=opts.sender_cell,
                                        sender_num_layers=opts.sender_num_layers,
                                        force_eos=force_eos,
                                        receiver_embedding=opts.receiver_embedding,
                                        receiver_hidden=opts.receiver_hidden,
                                        receiver_cell=opts.receiver_cell,
                                        receiver_num_layers=opts.receiver_num_layers)

            game = DialogReinforceModel4(Agent_1=agent_1,
                                           Agent_2=agent_2,
                                           loss=loss_model_3,
                                           sender_entropy_coeff_1=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff_1=opts.receiver_entropy_coeff,
                                           sender_entropy_coeff_2=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff_2=opts.receiver_entropy_coeff,
                                           length_cost=0.0,
                                           unigram_penalty=0.0,
                                           reg=False,
                                           device=device)

            optimizer_sender_1 = core.build_optimizer(list(game.agent_1.sender.parameters()))
            optimizer_receiver_1 = core.build_optimizer(list(game.agent_1.receiver.parameters()))
            optimizer_embedding_1 = core.build_optimizer(list(game.agent_1.embedding_layer.parameters()))
            optimizer_sender_2 = core.build_optimizer(list(game.agent_2.sender.parameters()))
            optimizer_receiver_2 = core.build_optimizer(list(game.agent_2.receiver.parameters()))
            optimizer_embedding_2 = core.build_optimizer(list(game.agent_2.embedding_layer.parameters()))

            trainer = TrainerDialogModel5(game=game, optimizer_sender_1=optimizer_sender_1, optimizer_sender_2=optimizer_sender_2, \
                                          optimizer_receiver_1=optimizer_receiver_1, optimizer_receiver_2=optimizer_receiver_2,\
                                          optimizer_embedding_1=optimizer_embedding_1,optimizer_embedding_2=optimizer_embedding_2, train_data=train_loader, \
                                          validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        elif opts.model=="shared_LSTM":

            "Agent 1"

            sender_1 = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)

            receiver_1 = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)


            agent_1=AgentSharedLSTM(receiver=receiver_1,
                                    sender=sender_1,
                                    vocab_size=opts.vocab_size,
                                    max_len=opts.max_len,
                                    embed_dim=opts.sender_embedding,
                                    hidden_size=opts.sender_hidden,
                                    cell=opts.sender_cell,
                                    num_layers=opts.sender_num_layers,
                                    force_eos=force_eos)

            "Agent 2"

            sender_2 = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)

            receiver_2 = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)

            agent_2=AgentSharedLSTM(receiver=receiver_2,
                                    sender=sender_2,
                                    vocab_size=opts.vocab_size,
                                    max_len=opts.max_len,
                                    embed_dim=opts.sender_embedding,
                                    hidden_size=opts.sender_hidden,
                                    cell=opts.sender_cell,
                                    num_layers=opts.sender_num_layers,
                                    force_eos=force_eos)
            if not opts.imitate:
                game = DialogReinforceModel6(Agent_1=agent_1,
                                               Agent_2=agent_2,
                                               loss=loss,
                                               sender_entropy_coeff_1=opts.sender_entropy_coeff,
                                               receiver_entropy_coeff_1=opts.receiver_entropy_coeff,
                                               sender_entropy_coeff_2=opts.sender_entropy_coeff,
                                               receiver_entropy_coeff_2=opts.receiver_entropy_coeff,
                                               imitate=opts.imitate,
                                               length_cost=0.0,
                                               unigram_penalty=0.0,
                                               reg=False,
                                               device=device)
            else:
                game = DialogReinforceModel6(Agent_1=agent_1,
                                               Agent_2=agent_2,
                                               loss=loss_model_3,
                                               sender_entropy_coeff_1=opts.sender_entropy_coeff,
                                               receiver_entropy_coeff_1=opts.receiver_entropy_coeff,
                                               sender_entropy_coeff_2=opts.sender_entropy_coeff,
                                               receiver_entropy_coeff_2=opts.receiver_entropy_coeff,
                                               imitate=opts.imitate,
                                               length_cost=0.0,
                                               unigram_penalty=0.0,
                                               reg=False,
                                               device=device)

            #optimizer_sender_1 = core.build_optimizer(list(game.agent_1.sender.parameters()))
            #optimizer_receiver_1 = core.build_optimizer(list(game.agent_1.receiver.parameters()))
            #optimizer_embedding_1 = core.build_optimizer(list(game.agent_1.embedding_layer.parameters()))
            #optimizer_sender_2 = core.build_optimizer(list(game.agent_2.sender.parameters()))
            #optimizer_receiver_2 = core.build_optimizer(list(game.agent_2.receiver.parameters()))
            #optimizer_embedding_2 = core.build_optimizer(list(game.agent_2.embedding_layer.parameters()))

            optimizer = core.build_optimizer(game.parameters())

            trainer = TrainerDialogModel6(game=game, optimizer=optimizer, train_data=train_loader, \
                                          validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        elif opts.model=="shared_embedding":

            "Agent 1"


            agent_1=AgentSharedEmbedding(n_features=opts.n_features,
                                        vocab_size=opts.vocab_size,
                                        max_len=opts.max_len,
                                        embed_dim=opts.sender_embedding,
                                        hidden_size=opts.sender_hidden,
                                        cell_sender=opts.sender_cell,
                                        cell_receiver=opts.receiver_cell,
                                        num_layers_sender=opts.sender_num_layers,
                                        num_layers_receiver=opts.receiver_num_layers,
                                        force_eos=force_eos)

            "Agent 2"

            agent_2=AgentSharedEmbedding(n_features=opts.n_features,
                                        vocab_size=opts.vocab_size,
                                        max_len=opts.max_len,
                                        embed_dim=opts.sender_embedding,
                                        hidden_size=opts.sender_hidden,
                                        cell_sender=opts.sender_cell,
                                        cell_receiver=opts.receiver_cell,
                                        num_layers_sender=opts.sender_num_layers,
                                        num_layers_receiver=opts.receiver_num_layers,
                                        force_eos=force_eos)
            if not opts.imitate:
                game = DialogReinforceModel6(Agent_1=agent_1,
                                               Agent_2=agent_2,
                                               loss=loss,
                                               sender_entropy_coeff_1=opts.sender_entropy_coeff,
                                               receiver_entropy_coeff_1=opts.receiver_entropy_coeff,
                                               sender_entropy_coeff_2=opts.sender_entropy_coeff,
                                               receiver_entropy_coeff_2=opts.receiver_entropy_coeff,
                                               imitate=opts.imitate,
                                               length_cost=0.0,
                                               unigram_penalty=0.0,
                                               reg=False,
                                               device=device)
            else:
                game = DialogReinforceModel6(Agent_1=agent_1,
                                               Agent_2=agent_2,
                                               loss=loss_model_3,
                                               sender_entropy_coeff_1=opts.sender_entropy_coeff,
                                               receiver_entropy_coeff_1=opts.receiver_entropy_coeff,
                                               sender_entropy_coeff_2=opts.sender_entropy_coeff,
                                               receiver_entropy_coeff_2=opts.receiver_entropy_coeff,
                                               imitate=opts.imitate,
                                               length_cost=0.0,
                                               unigram_penalty=0.0,
                                               reg=False,
                                               device=device)

            #optimizer_sender_1 = core.build_optimizer(list(game.agent_1.sender.parameters()))
            #optimizer_receiver_1 = core.build_optimizer(list(game.agent_1.receiver.parameters()))
            #optimizer_embedding_1 = core.build_optimizer(list(game.agent_1.embedding_layer.parameters()))
            #optimizer_sender_2 = core.build_optimizer(list(game.agent_2.sender.parameters()))
            #optimizer_receiver_2 = core.build_optimizer(list(game.agent_2.receiver.parameters()))
            #optimizer_embedding_2 = core.build_optimizer(list(game.agent_2.embedding_layer.parameters()))

            optimizer = core.build_optimizer(game.parameters())

            trainer = TrainerDialogModel6(game=game, optimizer=optimizer, train_data=train_loader, \
                                          validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        elif opts.model=="pretraining":

            "Agent 1"

            sender_1 = Sender(n_features=opts.n_features, n_hidden=opts.sender_hidden)
            sender_1 = RnnSenderReinforceModel3(sender_1,
                                       opts.vocab_size, opts.sender_embedding, opts.sender_hidden,
                                       cell=opts.sender_cell, max_len=opts.max_len, num_layers=opts.sender_num_layers,
                                       force_eos=force_eos)

            receiver_1 = Receiver(n_features=opts.n_features, n_hidden=opts.receiver_hidden)
            receiver_1 = core.RnnReceiverDeterministic(receiver_1, opts.vocab_size, opts.receiver_embedding,
                                                   opts.receiver_hidden, cell=opts.receiver_cell,
                                                   num_layers=opts.receiver_num_layers)

            agent_1=AgentModel3(receiver = receiver_1, sender = sender_1)

            "Pretrained_message"
            [[i]+[0]*(opts.max_len-1) for i in range(opts.n_features)]
            pretrained_messages=[]
            for i in range(1,opts.vocab_size):
                pretrained_messages.append([i]+[0]*(opts.max_len-1))
            for i in range(1,opts.vocab_size):
                pretrained_messages.append([1]+[i]+[0]*(opts.max_len-2))
            for i in range(1,opts.n_features-2*opts.vocab_size+3):
                pretrained_messages.append([2]+[i]+[0]*(opts.max_len-2))

            pretrained_messages=torch.tensor(pretrained_messages)

            game = PretrainAgent(Agent_1=agent_1,
                               loss=loss_pretraining,
                               pretrained_messages=pretrained_messages,
                               sender_entropy_coeff_1=opts.sender_entropy_coeff_1,
                               receiver_entropy_coeff_1=opts.receiver_entropy_coeff_1,
                               n_features=opts.n_features,
                               length_cost=0.0,
                               unigram_penalty=0.0,
                               reg=False,
                               device=device)

            optimizer_sender_1 = core.build_optimizer(list(game.agent_1.sender.parameters()))
            optimizer_receiver_1 = core.build_optimizer(list(game.agent_1.receiver.parameters()))

            trainer = TrainerPretraining(game=game, optimizer_sender_1=optimizer_sender_1,
                                          optimizer_receiver_1=optimizer_receiver_1, train_data=train_loader, \
                                          validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])




    if opts.model=="expe_1":

        "Define agents"


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


        "Define game"

        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_1":opts.receiver_entropy_coeff,
                      "sender_entropy_coeff_2":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_2":opts.receiver_entropy_coeff}

        if opts.optim_mode=="cross":
            loss_weights={"self":0.,"cross":1.,"imitation":0.}
        elif opts.optim_mode=="cross+self":
            loss_weights={"self":1.,"cross":1.,"imitation":0.}
        else:
            loss_weights={"self":1.,"cross":1.,"imitation":1.}
        #loss_weights={"self":opts.self_weight,"cross":opts.cross_weight,"imitation":opts.imitation_weight}

        game = DialogReinforce(Agent_1=agent_1,
                                Agent_2=agent_2,
                                loss_understanding=loss_understanding,
                                loss_imitation=loss_message_imitation,
                                optim_params=optim_params,
                                baseline_mode=opts.baseline_mode,
                                reward_mode=opts.reward_mode,
                                loss_weights=loss_weights,
                                device=device)

        "Create optimizers"
        #receiver_1_parameters = list(game.agent_1.agent_receiver.parameters()) + \
        #                      list(game.agent_1.receiver_cells.parameters()) + \
                              #list(game.agent_1.receiver_norm_h.parameters()) + \
                              #list(game.agent_1.receiver_norm_c.parameters()) + \
                              #list(game.agent_1.hidden_to_output.parameters()) + \
        #                      list(game.agent_1.receiver_embedding.parameters())

        #receiver_2_parameters = list(game.agent_2.agent_receiver.parameters()) + \
        #                      list(game.agent_2.receiver_cells.parameters()) + \
                              #list(game.agent_2.receiver_norm_h.parameters()) + \
                              #list(game.agent_2.receiver_norm_c.parameters()) + \
                              #list(game.agent_2.hidden_to_output.parameters()) + \
        #                      list(game.agent_2.receiver_embedding.parameters())

        #optimizer_agent_1 = core.build_optimizer(list(game.agent_1.parameters())+receiver_2_parameters)
        #optimizer_agent_2 = core.build_optimizer(list(game.agent_2.parameters())+receiver_1_parameters)
        #optimizer = core.build_optimizer(list(game.parameters()))
        optimizer = core.build_optimizer(list(game.parameters()))



        "Create trainer"
        #trainer = TrainerDialog(game=game, optimizer=optimizer, train_data=train_loader, \
        #                        validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])
        trainer = TrainerDialog(game=game, optimizer=optimizer, train_data=train_loader, \
                                validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        "Prepare training"

        # Create save dir
        if not path.exists(opts.dir_save):
            os.system("mkdir {}".format(opts.dir_save))
            os.system("mkdir -p {}/models {}/training_info {}/messages {}/accuracy {}/preds".format(opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save))

        # Main losses
        training_losses=[]
        eval_losses=[]
        training_entropy_1=[]
        training_entropy_2=[]
        training_loss_12=[]
        eval_loss_12=[]
        training_loss_21=[]
        eval_loss_21=[]

        # Specific losses
        training_loss_self_11=[]
        training_loss_cross_12=[]
        training_loss_imitation_12=[]
        training_loss_self_22=[]
        training_loss_cross_21=[]
        training_loss_imitation_21=[]
        eval_loss_self_11=[]
        eval_loss_cross_12=[]
        eval_loss_imitation_12=[]
        eval_loss_self_22=[]
        eval_loss_cross_21=[]
        eval_loss_imitation_21=[]

        # REINFORCE
        eval_reinforce_1=[]
        eval_reinforce_2=[]
        eval_baseline_1=[]
        eval_baseline_2=[]

        # Linguistic
        similarity_languages=[]

        "Prepare test"

        similarity_predictions_1=[]
        similarity_predictions_2=[]
        nb_messages_test=10000
        messages_test = torch.tensor(np.random.randint(opts.vocab_size,size=(nb_messages_test,opts.max_len)))

        "Train"

        for epoch in range(int(opts.n_epochs)):

            print("Epoch: {}".format(epoch))

            # Train
            list_train_loss,list_train_rest = trainer.train(n_epochs=1)

            #print(list_train_rest[-1],flush=True)

            # Eval
            eval_loss,eval_rest = trainer.eval()

            # Store results
            training_losses.append(list_train_loss[-1])
            eval_losses.append(eval_loss)
            training_entropy_1.append(list_train_rest[-1]["sender_entropy_1"])
            training_entropy_2.append(list_train_rest[-1]["sender_entropy_2"])
            training_loss_12.append(list_train_rest[-1]["loss_1"])
            eval_loss_12.append(eval_rest["loss_1"])
            training_loss_21.append(list_train_rest[-1]["loss_2"])
            eval_loss_21.append(eval_rest["loss_2"])
            training_loss_self_11.append(list_train_rest[-1]["loss_self_11"])
            training_loss_cross_12.append(list_train_rest[-1]["loss_cross_12"])
            training_loss_imitation_12.append(list_train_rest[-1]["loss_imitation_12"])
            training_loss_self_22.append(list_train_rest[-1]["loss_self_22"])
            training_loss_cross_21.append(list_train_rest[-1]["loss_cross_21"])
            training_loss_imitation_21.append(list_train_rest[-1]["loss_imitation_21"])
            eval_loss_self_11.append(eval_rest["loss_self_11"])
            eval_loss_cross_12.append(eval_rest["loss_cross_12"])
            eval_loss_imitation_12.append(eval_rest["loss_imitation_12"])
            eval_loss_self_22.append(eval_rest["loss_self_22"])
            eval_loss_cross_21.append(eval_rest["loss_cross_21"])
            eval_loss_imitation_21.append(eval_rest["loss_imitation_21"])
            eval_reinforce_1.append(eval_rest["reinforce_term_1"])
            eval_reinforce_2.append(eval_rest["reinforce_term_2"])
            eval_baseline_1.append(eval_rest["baseline_term_1"])
            eval_baseline_2.append(eval_rest["baseline_term_2"])

            if epoch==0:
                messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
            messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22, similarity_messages = dump_dialog_model_6(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
            np_messages_1 = convert_messages_to_numpy(messages_1)
            np_messages_2 = convert_messages_to_numpy(messages_2)
            similarity_languages.append(similarity_messages)

            if epoch==0:
                preds_1=preds_2=np.zeros((np.shape(messages_test)[0]))
            preds_1,preds_2,sim_pred_1,sim_pred_2 = test_receiver_evolution(trainer.game, messages_test, device,False,past_preds_1=preds_1,past_preds_2=preds_2)
            similarity_predictions_1.append(sim_pred_1)
            similarity_predictions_2.append(sim_pred_2)

            # Save models
            if epoch%20==0:
                torch.save(agent_1.state_dict(), f"{opts.dir_save}/models/agent_1_weights_{epoch}.pth")
                torch.save(agent_2.state_dict(), f"{opts.dir_save}/models/agent_2_weights_{epoch}.pth")

            # Save training info
            if epoch%10==0:
                np.save(opts.dir_save+'/training_info/training_loss_{}.npy'.format(epoch), training_losses)
                np.save(opts.dir_save+'/training_info/eval_loss_{}.npy'.format(epoch), eval_losses)
                np.save(opts.dir_save+'/training_info/training_entropy_1_{}.npy'.format(epoch), training_entropy_1)
                np.save(opts.dir_save+'/training_info/training_entropy_2_{}.npy'.format(epoch), training_entropy_2)
                np.save(opts.dir_save+'/training_info/training_loss_12_{}.npy'.format(epoch), training_loss_12)
                np.save(opts.dir_save+'/training_info/eval_loss_12_{}.npy'.format(epoch), eval_loss_12)
                np.save(opts.dir_save+'/training_info/training_loss_21_{}.npy'.format(epoch), training_loss_21)
                np.save(opts.dir_save+'/training_info/eval_loss_21_{}.npy'.format(epoch), eval_loss_21)
                np.save(opts.dir_save+'/training_info/training_loss_self_11_{}.npy'.format(epoch), training_loss_self_11)
                np.save(opts.dir_save+'/training_info/training_loss_cross_12_{}.npy'.format(epoch), training_loss_cross_12)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_12_{}.npy'.format(epoch), training_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/training_loss_self_22_{}.npy'.format(epoch), training_loss_self_22)
                np.save(opts.dir_save+'/training_info/training_loss_cross_21_{}.npy'.format(epoch), training_loss_cross_21)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_21_{}.npy'.format(epoch), training_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/eval_loss_self_11_{}.npy'.format(epoch), eval_loss_self_11)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_12_{}.npy'.format(epoch), eval_loss_cross_12)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_12_{}.npy'.format(epoch), eval_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/eval_loss_self_22_{}.npy'.format(epoch), eval_loss_self_22)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_21_{}.npy'.format(epoch), eval_loss_cross_21)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_21_{}.npy'.format(epoch), eval_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/similarity_languages_{}.npy'.format(epoch), similarity_languages)
                #Reinforce
                np.save(opts.dir_save+'/training_info/reinforce_term_1_{}.npy'.format(epoch), eval_reinforce_1)
                np.save(opts.dir_save+'/training_info/reinforce_term_2_{}.npy'.format(epoch), eval_reinforce_2)
                np.save(opts.dir_save+'/training_info/baseline_term_1_{}.npy'.format(epoch), eval_baseline_1)
                np.save(opts.dir_save+'/training_info/baseline_term_2_{}.npy'.format(epoch), eval_baseline_2)
                # Policy
                np.save(opts.dir_save+'/training_info/policy_1_{}.npy'.format(epoch),eval_rest["policy_1"].cpu().numpy())
                np.save(opts.dir_save+'/training_info/policy_2_{}.npy'.format(epoch),eval_rest["policy_2"].cpu().numpy())

            # Save accuracy/message results
            np.save(opts.dir_save+'/messages/agent_1_messages_{}.npy'.format(epoch), np_messages_1)
            np.save(opts.dir_save+'/messages/agent_2_messages_{}.npy'.format(epoch), np_messages_2)
            np.save(opts.dir_save+'/accuracy/12_accuracy_{}.npy'.format(epoch), acc_vec_1)
            np.save(opts.dir_save+'/accuracy/21_accuracy_{}.npy'.format(epoch), acc_vec_2)
            np.save(opts.dir_save+'/accuracy/11_accuracy_{}.npy'.format(epoch), acc_vec_11)
            np.save(opts.dir_save+'/accuracy/22_accuracy_{}.npy'.format(epoch), acc_vec_22)
            np.save(opts.dir_save+'/preds/preds_1_{}.npy'.format(epoch), preds_1)
            np.save(opts.dir_save+'/preds/preds_2_{}.npy'.format(epoch), preds_2)
            np.save(opts.dir_save+'/preds/sim_pred_1_{}.npy'.format(epoch), similarity_predictions_1)
            np.save(opts.dir_save+'/preds/sim_pred_2_{}.npy'.format(epoch), similarity_predictions_2)

    elif opts.model=="expe_single_listener":

        "Define agents"


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


        "Define game"

        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_1":opts.receiver_entropy_coeff,
                      "sender_entropy_coeff_2":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_2":opts.receiver_entropy_coeff}

        if opts.optim_mode=="cross":
            loss_weights={"self":0.,"cross":1.,"imitation":0.}
        elif opts.optim_mode=="cross+self":
            loss_weights={"self":1.,"cross":1.,"imitation":0.}
        else:
            loss_weights={"self":1.,"cross":1.,"imitation":1.}
        #loss_weights={"self":opts.self_weight,"cross":opts.cross_weight,"imitation":opts.imitation_weight}

        game = DialogReinforceSingleListener(Agent_1=agent_1,
                                            Agent_2=agent_2,
                                            loss_understanding=loss_understanding,
                                            loss_imitation=loss_message_imitation,
                                            optim_params=optim_params,
                                            baseline_mode=opts.baseline_mode,
                                            reward_mode=opts.reward_mode,
                                            loss_weights=loss_weights,
                                            device=device)

        "Create optimizers"
        speaker_parameters = list(game.agent_1.agent_sender.parameters()) + \
                               list(game.agent_1.sender_norm_h.parameters()) + \
                               list(game.agent_1.sender_norm_c.parameters()) + \
                               list(game.agent_1.hidden_to_output.parameters()) + \
                               list(game.agent_1.sender_embedding.parameters()) + \
                               list(game.agent_1.sender_cells.parameters()) + \
                               list(game.agent_2.agent_sender.parameters()) + \
                               list(game.agent_2.sender_norm_h.parameters()) + \
                               list(game.agent_2.sender_norm_c.parameters()) + \
                               list(game.agent_2.hidden_to_output.parameters()) + \
                               list(game.agent_2.sender_embedding.parameters()) + \
                               list(game.agent_2.sender_cells.parameters())

        listener_parameters = list(game.agent_1.agent_receiver.parameters()) + \
                              list(game.agent_1.receiver_cell.parameters()) + \
                              list(game.agent_1.receiver_embedding.parameters())

        # ADAM
        optimizer_speaker = core.build_optimizer(list(speaker_parameters),lr=opts.sender_lr)
        optimizer_listener = core.build_optimizer(list(listener_parameters),lr=opts.receiver_lr)

        # SGD
        #optimizer_speaker=torch.optim.SGD(speaker_parameters, lr=opts.sender_lr, momentum=0.9,nesterov=False)
        #optimizer_listener=torch.optim.SGD(listener_parameters, lr=opts.receiver_lr, momentum=0.9,nesterov=False)


        "Create trainer"
        #trainer = TrainerDialog(game=game, optimizer=optimizer, train_data=train_loader, \
        #                        validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])
        #trainer = TrainerDialogAsymLR(game=game, optimizer_speaker=optimizer_speaker,optimizer_listener=optimizer_listener, train_data=train_loader, \
        #                              validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])
        trainer = TrainerDialogAsymStep(game=game, optimizer_speaker=optimizer_speaker,optimizer_listener=optimizer_listener,\
                                        N_speaker=opts.N_speaker,N_listener=opts.N_listener,train_data=train_loader, \
                                        validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        "Prepare training"

        # Create save dir
        if not path.exists(opts.dir_save):
            os.system("mkdir {}".format(opts.dir_save))
            os.system("mkdir -p {}/models {}/training_info {}/messages {}/accuracy {}/preds".format(opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save))

        # Main losses
        training_losses=[]
        eval_losses=[]
        training_entropy_1=[]
        training_entropy_2=[]
        training_loss_12=[]
        eval_loss_12=[]
        training_loss_21=[]
        eval_loss_21=[]

        # Specific losses
        training_loss_self_11=[]
        training_loss_cross_12=[]
        training_loss_imitation_12=[]
        training_loss_self_22=[]
        training_loss_cross_21=[]
        training_loss_imitation_21=[]
        eval_loss_self_11=[]
        eval_loss_cross_12=[]
        eval_loss_imitation_12=[]
        eval_loss_self_22=[]
        eval_loss_cross_21=[]
        eval_loss_imitation_21=[]

        # REINFORCE
        eval_reinforce_1=[]
        eval_reinforce_2=[]
        eval_baseline_1=[]
        eval_baseline_2=[]

        # Linguistic
        similarity_languages=[]

        "Prepare test"

        similarity_predictions_1=[]
        similarity_predictions_2=[]
        nb_messages_test=10000
        messages_test = torch.tensor(np.random.randint(opts.vocab_size,size=(nb_messages_test,opts.max_len)))

        "Train"

        for epoch in range(int(opts.n_epochs)):

            print("Epoch: {}".format(epoch))

            # Train
            list_train_loss,list_train_rest = trainer.train(n_epochs=1)

            #print(list_train_rest[-1],flush=True)

            # Eval
            eval_loss,eval_rest = trainer.eval()

            # Store results
            training_losses.append(list_train_loss[-1])
            eval_losses.append(eval_loss)
            training_entropy_1.append(list_train_rest[-1]["sender_entropy_1"])
            training_entropy_2.append(list_train_rest[-1]["sender_entropy_2"])
            training_loss_12.append(list_train_rest[-1]["loss_1"])
            eval_loss_12.append(eval_rest["loss_1"])
            training_loss_21.append(list_train_rest[-1]["loss_2"])
            eval_loss_21.append(eval_rest["loss_2"])
            training_loss_self_11.append(list_train_rest[-1]["loss_self_11"])
            training_loss_cross_12.append(list_train_rest[-1]["loss_cross_12"])
            training_loss_imitation_12.append(list_train_rest[-1]["loss_imitation_12"])
            training_loss_self_22.append(list_train_rest[-1]["loss_self_22"])
            training_loss_cross_21.append(list_train_rest[-1]["loss_cross_21"])
            training_loss_imitation_21.append(list_train_rest[-1]["loss_imitation_21"])
            eval_loss_self_11.append(eval_rest["loss_self_11"])
            eval_loss_cross_12.append(eval_rest["loss_cross_12"])
            eval_loss_imitation_12.append(eval_rest["loss_imitation_12"])
            eval_loss_self_22.append(eval_rest["loss_self_22"])
            eval_loss_cross_21.append(eval_rest["loss_cross_21"])
            eval_loss_imitation_21.append(eval_rest["loss_imitation_21"])
            eval_reinforce_1.append(eval_rest["reinforce_term_1"])
            eval_reinforce_2.append(eval_rest["reinforce_term_2"])
            eval_baseline_1.append(eval_rest["baseline_term_1"])
            eval_baseline_2.append(eval_rest["baseline_term_2"])

            if epoch==0:
                messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
            messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22, similarity_messages = dump_dialog_model_6(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
            np_messages_1 = convert_messages_to_numpy(messages_1)
            np_messages_2 = convert_messages_to_numpy(messages_2)
            similarity_languages.append(similarity_messages)

            if epoch==0:
                preds_1=preds_2=np.zeros((np.shape(messages_test)[0]))
            preds_1,preds_2,sim_pred_1,sim_pred_2 = test_receiver_evolution(trainer.game, messages_test, device,False,past_preds_1=preds_1,past_preds_2=preds_2)
            similarity_predictions_1.append(sim_pred_1)
            similarity_predictions_2.append(sim_pred_2)

            # Save models
            if epoch%20==0:
                torch.save(agent_1.state_dict(), f"{opts.dir_save}/models/agent_1_weights_{epoch}.pth")
                torch.save(agent_2.state_dict(), f"{opts.dir_save}/models/agent_2_weights_{epoch}.pth")

            # Save training info
            if epoch%10==0:
                np.save(opts.dir_save+'/training_info/training_loss_{}.npy'.format(epoch), training_losses)
                np.save(opts.dir_save+'/training_info/eval_loss_{}.npy'.format(epoch), eval_losses)
                np.save(opts.dir_save+'/training_info/training_entropy_1_{}.npy'.format(epoch), training_entropy_1)
                np.save(opts.dir_save+'/training_info/training_entropy_2_{}.npy'.format(epoch), training_entropy_2)
                np.save(opts.dir_save+'/training_info/training_loss_12_{}.npy'.format(epoch), training_loss_12)
                np.save(opts.dir_save+'/training_info/eval_loss_12_{}.npy'.format(epoch), eval_loss_12)
                np.save(opts.dir_save+'/training_info/training_loss_21_{}.npy'.format(epoch), training_loss_21)
                np.save(opts.dir_save+'/training_info/eval_loss_21_{}.npy'.format(epoch), eval_loss_21)
                np.save(opts.dir_save+'/training_info/training_loss_self_11_{}.npy'.format(epoch), training_loss_self_11)
                np.save(opts.dir_save+'/training_info/training_loss_cross_12_{}.npy'.format(epoch), training_loss_cross_12)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_12_{}.npy'.format(epoch), training_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/training_loss_self_22_{}.npy'.format(epoch), training_loss_self_22)
                np.save(opts.dir_save+'/training_info/training_loss_cross_21_{}.npy'.format(epoch), training_loss_cross_21)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_21_{}.npy'.format(epoch), training_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/eval_loss_self_11_{}.npy'.format(epoch), eval_loss_self_11)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_12_{}.npy'.format(epoch), eval_loss_cross_12)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_12_{}.npy'.format(epoch), eval_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/eval_loss_self_22_{}.npy'.format(epoch), eval_loss_self_22)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_21_{}.npy'.format(epoch), eval_loss_cross_21)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_21_{}.npy'.format(epoch), eval_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/similarity_languages_{}.npy'.format(epoch), similarity_languages)
                #Reinforce
                np.save(opts.dir_save+'/training_info/reinforce_term_1_{}.npy'.format(epoch), eval_reinforce_1)
                np.save(opts.dir_save+'/training_info/reinforce_term_2_{}.npy'.format(epoch), eval_reinforce_2)
                np.save(opts.dir_save+'/training_info/baseline_term_1_{}.npy'.format(epoch), eval_baseline_1)
                np.save(opts.dir_save+'/training_info/baseline_term_2_{}.npy'.format(epoch), eval_baseline_2)
                # Policy
                np.save(opts.dir_save+'/training_info/policy_1_{}.npy'.format(epoch),eval_rest["policy_1"].cpu().numpy())
                np.save(opts.dir_save+'/training_info/policy_2_{}.npy'.format(epoch),eval_rest["policy_2"].cpu().numpy())

            # Save accuracy/message results
            np.save(opts.dir_save+'/messages/agent_1_messages_{}.npy'.format(epoch), np_messages_1)
            np.save(opts.dir_save+'/messages/agent_2_messages_{}.npy'.format(epoch), np_messages_2)
            np.save(opts.dir_save+'/accuracy/12_accuracy_{}.npy'.format(epoch), acc_vec_1)
            np.save(opts.dir_save+'/accuracy/21_accuracy_{}.npy'.format(epoch), acc_vec_2)
            np.save(opts.dir_save+'/accuracy/11_accuracy_{}.npy'.format(epoch), acc_vec_11)
            np.save(opts.dir_save+'/accuracy/22_accuracy_{}.npy'.format(epoch), acc_vec_22)
            np.save(opts.dir_save+'/preds/preds_1_{}.npy'.format(epoch), preds_1)
            np.save(opts.dir_save+'/preds/preds_2_{}.npy'.format(epoch), preds_2)
            np.save(opts.dir_save+'/preds/sim_pred_1_{}.npy'.format(epoch), similarity_predictions_1)
            np.save(opts.dir_save+'/preds/sim_pred_2_{}.npy'.format(epoch), similarity_predictions_2)

    elif opts.model=="expe_pretraining":

        "Define agents"


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

        "Prepare training"

        # Create save dir
        if not path.exists(opts.dir_save):
            os.system("mkdir {}".format(opts.dir_save))
            os.system("mkdir -p {}/models {}/training_info {}/messages {}/accuracy {}/preds".format(opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save))

        # Main losses
        training_losses=[]
        eval_losses=[]
        training_entropy_1=[]
        training_entropy_2=[]
        training_loss_12=[]
        eval_loss_12=[]
        training_loss_21=[]
        eval_loss_21=[]

        # Specific losses
        training_loss_self_11=[]
        training_loss_cross_12=[]
        training_loss_imitation_12=[]
        training_loss_self_22=[]
        training_loss_cross_21=[]
        training_loss_imitation_21=[]
        eval_loss_self_11=[]
        eval_loss_cross_12=[]
        eval_loss_imitation_12=[]
        eval_loss_self_22=[]
        eval_loss_cross_21=[]
        eval_loss_imitation_21=[]

        # REINFORCE
        eval_reinforce_1=[]
        eval_reinforce_2=[]
        eval_baseline_1=[]
        eval_baseline_2=[]

        # Linguistic
        similarity_languages=[]

        "Prepare test"

        similarity_predictions_1=[]
        similarity_predictions_2=[]
        nb_messages_test=10000
        messages_test = torch.tensor(np.random.randint(opts.vocab_size,size=(nb_messages_test,opts.max_len)))

        "Pretraining "

        print("Step 1 = Pretraining",flush=True)

        "Define game"

        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_1":opts.receiver_entropy_coeff,
                      "sender_entropy_coeff_2":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_2":opts.receiver_entropy_coeff}

        loss_weights={"self":1.,"cross":0.,"imitation":0.}
        #loss_weights={"self":opts.self_weight,"cross":opts.cross_weight,"imitation":opts.imitation_weight}

        game = DialogReinforce(Agent_1=agent_1,
                                Agent_2=agent_2,
                                loss_understanding=loss_understanding,
                                loss_imitation=loss_message_imitation,
                                optim_params=optim_params,
                                baseline_mode=opts.baseline_mode,
                                reward_mode=opts.reward_mode,
                                loss_weights=loss_weights,
                                device=device)

        "Create optimizers"
        optimizer = core.build_optimizer(list(game.parameters()))



        "Create trainer"
        trainer = TrainerDialog(game=game, optimizer=optimizer, train_data=train_loader, \
                                validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        for epoch in range(int(opts.n_epochs)):

            print("Epoch: {}".format(epoch))

            # Train
            list_train_loss,list_train_rest = trainer.train(n_epochs=1)

            #print(list_train_rest[-1],flush=True)

            # Eval
            eval_loss,eval_rest = trainer.eval()

            # Store results
            training_losses.append(list_train_loss[-1])
            eval_losses.append(eval_loss)
            training_entropy_1.append(list_train_rest[-1]["sender_entropy_1"])
            training_entropy_2.append(list_train_rest[-1]["sender_entropy_2"])
            training_loss_12.append(list_train_rest[-1]["loss_1"])
            eval_loss_12.append(eval_rest["loss_1"])
            training_loss_21.append(list_train_rest[-1]["loss_2"])
            eval_loss_21.append(eval_rest["loss_2"])
            training_loss_self_11.append(list_train_rest[-1]["loss_self_11"])
            training_loss_cross_12.append(list_train_rest[-1]["loss_cross_12"])
            training_loss_imitation_12.append(list_train_rest[-1]["loss_imitation_12"])
            training_loss_self_22.append(list_train_rest[-1]["loss_self_22"])
            training_loss_cross_21.append(list_train_rest[-1]["loss_cross_21"])
            training_loss_imitation_21.append(list_train_rest[-1]["loss_imitation_21"])
            eval_loss_self_11.append(eval_rest["loss_self_11"])
            eval_loss_cross_12.append(eval_rest["loss_cross_12"])
            eval_loss_imitation_12.append(eval_rest["loss_imitation_12"])
            eval_loss_self_22.append(eval_rest["loss_self_22"])
            eval_loss_cross_21.append(eval_rest["loss_cross_21"])
            eval_loss_imitation_21.append(eval_rest["loss_imitation_21"])
            eval_reinforce_1.append(eval_rest["reinforce_term_1"])
            eval_reinforce_2.append(eval_rest["reinforce_term_2"])
            eval_baseline_1.append(eval_rest["baseline_term_1"])
            eval_baseline_2.append(eval_rest["baseline_term_2"])

            if epoch==0:
                messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
            messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22, similarity_messages = dump_dialog_model_6(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
            np_messages_1 = convert_messages_to_numpy(messages_1)
            np_messages_2 = convert_messages_to_numpy(messages_2)
            similarity_languages.append(similarity_messages)

            if epoch==0:
                preds_1=preds_2=np.zeros((np.shape(messages_test)[0]))
            preds_1,preds_2,sim_pred_1,sim_pred_2 = test_receiver_evolution(trainer.game, messages_test, device,False,past_preds_1=preds_1,past_preds_2=preds_2)
            similarity_predictions_1.append(sim_pred_1)
            similarity_predictions_2.append(sim_pred_2)

            #game.optim_params["sender_entropy_coeff_1"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_11)
            #game.optim_params["sender_entropy_coeff_2"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_22)


            # Save models
            if epoch%20==0:
                torch.save(agent_1.state_dict(), f"{opts.dir_save}/models/agent_1_weights_{epoch}.pth")
                torch.save(agent_2.state_dict(), f"{opts.dir_save}/models/agent_2_weights_{epoch}.pth")

            # Save training info
            if epoch%10==0:
                np.save(opts.dir_save+'/training_info/training_loss_{}.npy'.format(epoch), training_losses)
                np.save(opts.dir_save+'/training_info/eval_loss_{}.npy'.format(epoch), eval_losses)
                np.save(opts.dir_save+'/training_info/training_entropy_1_{}.npy'.format(epoch), training_entropy_1)
                np.save(opts.dir_save+'/training_info/training_entropy_2_{}.npy'.format(epoch), training_entropy_2)
                np.save(opts.dir_save+'/training_info/training_loss_12_{}.npy'.format(epoch), training_loss_12)
                np.save(opts.dir_save+'/training_info/eval_loss_12_{}.npy'.format(epoch), eval_loss_12)
                np.save(opts.dir_save+'/training_info/training_loss_21_{}.npy'.format(epoch), training_loss_21)
                np.save(opts.dir_save+'/training_info/eval_loss_21_{}.npy'.format(epoch), eval_loss_21)
                np.save(opts.dir_save+'/training_info/training_loss_self_11_{}.npy'.format(epoch), training_loss_self_11)
                np.save(opts.dir_save+'/training_info/training_loss_cross_12_{}.npy'.format(epoch), training_loss_cross_12)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_12_{}.npy'.format(epoch), training_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/training_loss_self_22_{}.npy'.format(epoch), training_loss_self_22)
                np.save(opts.dir_save+'/training_info/training_loss_cross_21_{}.npy'.format(epoch), training_loss_cross_21)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_21_{}.npy'.format(epoch), training_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/eval_loss_self_11_{}.npy'.format(epoch), eval_loss_self_11)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_12_{}.npy'.format(epoch), eval_loss_cross_12)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_12_{}.npy'.format(epoch), eval_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/eval_loss_self_22_{}.npy'.format(epoch), eval_loss_self_22)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_21_{}.npy'.format(epoch), eval_loss_cross_21)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_21_{}.npy'.format(epoch), eval_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/similarity_languages_{}.npy'.format(epoch), similarity_languages)
                #Reinforce
                np.save(opts.dir_save+'/training_info/reinforce_term_1_{}.npy'.format(epoch), eval_reinforce_1)
                np.save(opts.dir_save+'/training_info/reinforce_term_2_{}.npy'.format(epoch), eval_reinforce_2)
                np.save(opts.dir_save+'/training_info/baseline_term_1_{}.npy'.format(epoch), eval_baseline_1)
                np.save(opts.dir_save+'/training_info/baseline_term_2_{}.npy'.format(epoch), eval_baseline_2)
                # Policy
                np.save(opts.dir_save+'/training_info/policy_1_{}.npy'.format(epoch),eval_rest["policy_1"].cpu().numpy())
                np.save(opts.dir_save+'/training_info/policy_2_{}.npy'.format(epoch),eval_rest["policy_2"].cpu().numpy())

            # Save accuracy/message results
            np.save(opts.dir_save+'/messages/agent_1_messages_{}.npy'.format(epoch), np_messages_1)
            np.save(opts.dir_save+'/messages/agent_2_messages_{}.npy'.format(epoch), np_messages_2)
            np.save(opts.dir_save+'/accuracy/12_accuracy_{}.npy'.format(epoch), acc_vec_1)
            np.save(opts.dir_save+'/accuracy/21_accuracy_{}.npy'.format(epoch), acc_vec_2)
            np.save(opts.dir_save+'/accuracy/11_accuracy_{}.npy'.format(epoch), acc_vec_11)
            np.save(opts.dir_save+'/accuracy/22_accuracy_{}.npy'.format(epoch), acc_vec_22)
            np.save(opts.dir_save+'/preds/preds_1_{}.npy'.format(epoch), preds_1)
            np.save(opts.dir_save+'/preds/preds_2_{}.npy'.format(epoch), preds_2)
            np.save(opts.dir_save+'/preds/sim_pred_1_{}.npy'.format(epoch), similarity_predictions_1)
            np.save(opts.dir_save+'/preds/sim_pred_2_{}.npy'.format(epoch), similarity_predictions_2)


        print("Step 2 = Interaction",flush=True)

        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_1":opts.receiver_entropy_coeff,
                      "sender_entropy_coeff_2":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_2":opts.receiver_entropy_coeff}

        loss_weights={"self":1.,"cross":1.,"imitation":0.}
        #loss_weights={"self":opts.self_weight,"cross":opts.cross_weight,"imitation":opts.imitation_weight}

        game = DialogReinforce(Agent_1=agent_1,
                                Agent_2=agent_2,
                                loss_understanding=loss_understanding,
                                loss_imitation=loss_message_imitation,
                                optim_params=optim_params,
                                baseline_mode=opts.baseline_mode,
                                reward_mode=opts.reward_mode,
                                loss_weights=loss_weights,
                                device=device)

        speaker_parameters = list(game.agent_1.agent_sender.parameters()) + \
                               list(game.agent_1.sender_norm_h.parameters()) + \
                               list(game.agent_1.sender_norm_c.parameters()) + \
                               list(game.agent_1.hidden_to_output.parameters()) + \
                               list(game.agent_1.sender_embedding.parameters()) + \
                               list(game.agent_1.sender_cells.parameters()) + \
                               list(game.agent_2.agent_sender.parameters()) + \
                               list(game.agent_2.sender_norm_h.parameters()) + \
                               list(game.agent_2.sender_norm_c.parameters()) + \
                               list(game.agent_2.hidden_to_output.parameters()) + \
                               list(game.agent_2.sender_embedding.parameters()) + \
                               list(game.agent_2.sender_cells.parameters())

        listener_parameters = list(game.agent_1.agent_receiver.parameters()) + \
                              list(game.agent_1.receiver_cell.parameters()) + \
                              list(game.agent_1.receiver_embedding.parameters()) + \
                              list(game.agent_2.agent_receiver.parameters()) + \
                              list(game.agent_2.receiver_cell.parameters()) + \
                              list(game.agent_2.receiver_embedding.parameters())

        "Create optimizers"
        optimizer_speaker = core.build_optimizer(list(speaker_parameters),lr=0.)
        optimizer_listener = core.build_optimizer(list(listener_parameters),lr=opts.receiver_lr)



        "Create trainer"
        trainer = TrainerDialogAsymLR(game=game, optimizer_speaker=optimizer_speaker,optimizer_listener=optimizer_listener, train_data=train_loader, \
                                      validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        for epoch in range(int(opts.n_epochs),2*int(opts.n_epochs)):

            print("Epoch: {}".format(epoch))

            # Train
            list_train_loss,list_train_rest = trainer.train(n_epochs=1)

            #print(list_train_rest[-1],flush=True)

            # Eval
            eval_loss,eval_rest = trainer.eval()

            # Store results
            training_losses.append(list_train_loss[-1])
            eval_losses.append(eval_loss)
            training_entropy_1.append(list_train_rest[-1]["sender_entropy_1"])
            training_entropy_2.append(list_train_rest[-1]["sender_entropy_2"])
            training_loss_12.append(list_train_rest[-1]["loss_1"])
            eval_loss_12.append(eval_rest["loss_1"])
            training_loss_21.append(list_train_rest[-1]["loss_2"])
            eval_loss_21.append(eval_rest["loss_2"])
            training_loss_self_11.append(list_train_rest[-1]["loss_self_11"])
            training_loss_cross_12.append(list_train_rest[-1]["loss_cross_12"])
            training_loss_imitation_12.append(list_train_rest[-1]["loss_imitation_12"])
            training_loss_self_22.append(list_train_rest[-1]["loss_self_22"])
            training_loss_cross_21.append(list_train_rest[-1]["loss_cross_21"])
            training_loss_imitation_21.append(list_train_rest[-1]["loss_imitation_21"])
            eval_loss_self_11.append(eval_rest["loss_self_11"])
            eval_loss_cross_12.append(eval_rest["loss_cross_12"])
            eval_loss_imitation_12.append(eval_rest["loss_imitation_12"])
            eval_loss_self_22.append(eval_rest["loss_self_22"])
            eval_loss_cross_21.append(eval_rest["loss_cross_21"])
            eval_loss_imitation_21.append(eval_rest["loss_imitation_21"])
            eval_reinforce_1.append(eval_rest["reinforce_term_1"])
            eval_reinforce_2.append(eval_rest["reinforce_term_2"])
            eval_baseline_1.append(eval_rest["baseline_term_1"])
            eval_baseline_2.append(eval_rest["baseline_term_2"])

            if epoch==0:
                messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
            messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22, similarity_messages = dump_dialog_model_6(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
            np_messages_1 = convert_messages_to_numpy(messages_1)
            np_messages_2 = convert_messages_to_numpy(messages_2)
            similarity_languages.append(similarity_messages)

            if epoch==0:
                preds_1=preds_2=np.zeros((np.shape(messages_test)[0]))
            preds_1,preds_2,sim_pred_1,sim_pred_2 = test_receiver_evolution(trainer.game, messages_test, device,False,past_preds_1=preds_1,past_preds_2=preds_2)
            similarity_predictions_1.append(sim_pred_1)
            similarity_predictions_2.append(sim_pred_2)

            #game.optim_params["sender_entropy_coeff_1"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_11)
            #game.optim_params["sender_entropy_coeff_2"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_22)


            # Save models
            if epoch%20==0:
                torch.save(agent_1.state_dict(), f"{opts.dir_save}/models/agent_1_weights_{epoch}.pth")
                torch.save(agent_2.state_dict(), f"{opts.dir_save}/models/agent_2_weights_{epoch}.pth")

            # Save training info
            if epoch%10==0:
                np.save(opts.dir_save+'/training_info/training_loss_{}.npy'.format(epoch), training_losses)
                np.save(opts.dir_save+'/training_info/eval_loss_{}.npy'.format(epoch), eval_losses)
                np.save(opts.dir_save+'/training_info/training_entropy_1_{}.npy'.format(epoch), training_entropy_1)
                np.save(opts.dir_save+'/training_info/training_entropy_2_{}.npy'.format(epoch), training_entropy_2)
                np.save(opts.dir_save+'/training_info/training_loss_12_{}.npy'.format(epoch), training_loss_12)
                np.save(opts.dir_save+'/training_info/eval_loss_12_{}.npy'.format(epoch), eval_loss_12)
                np.save(opts.dir_save+'/training_info/training_loss_21_{}.npy'.format(epoch), training_loss_21)
                np.save(opts.dir_save+'/training_info/eval_loss_21_{}.npy'.format(epoch), eval_loss_21)
                np.save(opts.dir_save+'/training_info/training_loss_self_11_{}.npy'.format(epoch), training_loss_self_11)
                np.save(opts.dir_save+'/training_info/training_loss_cross_12_{}.npy'.format(epoch), training_loss_cross_12)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_12_{}.npy'.format(epoch), training_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/training_loss_self_22_{}.npy'.format(epoch), training_loss_self_22)
                np.save(opts.dir_save+'/training_info/training_loss_cross_21_{}.npy'.format(epoch), training_loss_cross_21)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_21_{}.npy'.format(epoch), training_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/eval_loss_self_11_{}.npy'.format(epoch), eval_loss_self_11)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_12_{}.npy'.format(epoch), eval_loss_cross_12)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_12_{}.npy'.format(epoch), eval_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/eval_loss_self_22_{}.npy'.format(epoch), eval_loss_self_22)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_21_{}.npy'.format(epoch), eval_loss_cross_21)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_21_{}.npy'.format(epoch), eval_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/similarity_languages_{}.npy'.format(epoch), similarity_languages)
                #Reinforce
                np.save(opts.dir_save+'/training_info/reinforce_term_1_{}.npy'.format(epoch), eval_reinforce_1)
                np.save(opts.dir_save+'/training_info/reinforce_term_2_{}.npy'.format(epoch), eval_reinforce_2)
                np.save(opts.dir_save+'/training_info/baseline_term_1_{}.npy'.format(epoch), eval_baseline_1)
                np.save(opts.dir_save+'/training_info/baseline_term_2_{}.npy'.format(epoch), eval_baseline_2)
                # Policy
                np.save(opts.dir_save+'/training_info/policy_1_{}.npy'.format(epoch),eval_rest["policy_1"].cpu().numpy())
                np.save(opts.dir_save+'/training_info/policy_2_{}.npy'.format(epoch),eval_rest["policy_2"].cpu().numpy())

            # Save accuracy/message results
            np.save(opts.dir_save+'/messages/agent_1_messages_{}.npy'.format(epoch), np_messages_1)
            np.save(opts.dir_save+'/messages/agent_2_messages_{}.npy'.format(epoch), np_messages_2)
            np.save(opts.dir_save+'/accuracy/12_accuracy_{}.npy'.format(epoch), acc_vec_1)
            np.save(opts.dir_save+'/accuracy/21_accuracy_{}.npy'.format(epoch), acc_vec_2)
            np.save(opts.dir_save+'/accuracy/11_accuracy_{}.npy'.format(epoch), acc_vec_11)
            np.save(opts.dir_save+'/accuracy/22_accuracy_{}.npy'.format(epoch), acc_vec_22)
            np.save(opts.dir_save+'/preds/preds_1_{}.npy'.format(epoch), preds_1)
            np.save(opts.dir_save+'/preds/preds_2_{}.npy'.format(epoch), preds_2)
            np.save(opts.dir_save+'/preds/sim_pred_1_{}.npy'.format(epoch), similarity_predictions_1)
            np.save(opts.dir_save+'/preds/sim_pred_2_{}.npy'.format(epoch), similarity_predictions_2)


    elif opts.model=="expe_characteristic_time":

        "Define agents"


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

        "Prepare training"

        # Create save dir
        if not path.exists(opts.dir_save):
            os.system("mkdir {}".format(opts.dir_save))
            os.system("mkdir -p {}/models {}/training_info {}/messages {}/accuracy {}/preds".format(opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save))

        # Main losses
        training_losses=[]
        eval_losses=[]
        training_entropy_1=[]
        training_entropy_2=[]
        training_loss_12=[]
        eval_loss_12=[]
        training_loss_21=[]
        eval_loss_21=[]

        # Specific losses
        training_loss_self_11=[]
        training_loss_cross_12=[]
        training_loss_imitation_12=[]
        training_loss_self_22=[]
        training_loss_cross_21=[]
        training_loss_imitation_21=[]
        eval_loss_self_11=[]
        eval_loss_cross_12=[]
        eval_loss_imitation_12=[]
        eval_loss_self_22=[]
        eval_loss_cross_21=[]
        eval_loss_imitation_21=[]

        # REINFORCE
        eval_reinforce_1=[]
        eval_reinforce_2=[]
        eval_baseline_1=[]
        eval_baseline_2=[]

        # Linguistic
        similarity_languages=[]

        "Prepare test"

        similarity_predictions_1=[]
        similarity_predictions_2=[]
        nb_messages_test=10000
        messages_test = torch.tensor(np.random.randint(opts.vocab_size,size=(nb_messages_test,opts.max_len)))

        "Pretraining "

        print("Step 1 = Pretraining",flush=True)

        "Define game"

        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_1":opts.receiver_entropy_coeff,
                      "sender_entropy_coeff_2":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_2":opts.receiver_entropy_coeff}

        loss_weights={"self":1.,"cross":0.,"imitation":0.}
        #loss_weights={"self":opts.self_weight,"cross":opts.cross_weight,"imitation":opts.imitation_weight}

        game = DialogReinforce(Agent_1=agent_1,
                                Agent_2=agent_2,
                                loss_understanding=loss_understanding,
                                loss_imitation=loss_message_imitation,
                                optim_params=optim_params,
                                baseline_mode=opts.baseline_mode,
                                reward_mode=opts.reward_mode,
                                loss_weights=loss_weights,
                                device=device)

        "Create optimizers"

        speaker_1_parameters = list(game.agent_1.agent_sender.parameters()) + \
                               list(game.agent_1.sender_norm_h.parameters()) + \
                               list(game.agent_1.sender_norm_c.parameters()) + \
                               list(game.agent_1.hidden_to_output.parameters()) + \
                               list(game.agent_1.sender_embedding.parameters()) + \
                               list(game.agent_1.sender_cells.parameters())

        listener_1_parameters = list(game.agent_1.agent_receiver.parameters()) + \
                              list(game.agent_1.receiver_cell.parameters()) + \
                              list(game.agent_1.receiver_embedding.parameters())

        speaker_2_parameters = list(game.agent_2.agent_sender.parameters()) + \
                               list(game.agent_2.sender_norm_h.parameters()) + \
                               list(game.agent_2.sender_norm_c.parameters()) + \
                               list(game.agent_2.hidden_to_output.parameters()) + \
                               list(game.agent_2.sender_embedding.parameters()) + \
                               list(game.agent_2.sender_cells.parameters())

        listener_2_parameters = list(game.agent_2.agent_receiver.parameters()) + \
                              list(game.agent_2.receiver_cell.parameters()) + \
                              list(game.agent_2.receiver_embedding.parameters())

        optimizer_speaker_1 = core.build_optimizer(speaker_1_parameters,lr=opts.sender_lr)
        optimizer_listener_1 = core.build_optimizer(listener_1_parameters,lr=opts.receiver_lr)
        optimizer_speaker_2 = core.build_optimizer(speaker_2_parameters,lr=0.)
        optimizer_listener_2 = core.build_optimizer(listener_2_parameters,lr=0.)



        "Create trainer"
        trainer = TrainerDialog4Optim(game=game,
                                      optimizer_speaker_1=optimizer_speaker_1,
                                      optimizer_listener_1=optimizer_listener_1,
                                      optimizer_speaker_2=optimizer_speaker_2,
                                      optimizer_listener_2=optimizer_listener_2,
                                      train_data=train_loader,
                                      validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        for epoch in range(int(opts.n_epochs)):

            print("Epoch: {}".format(epoch))

            # Train
            list_train_loss,list_train_rest = trainer.train(n_epochs=1)

            #print(list_train_rest[-1],flush=True)

            # Eval
            eval_loss,eval_rest = trainer.eval()

            # Store results
            training_losses.append(list_train_loss[-1])
            eval_losses.append(eval_loss)
            training_entropy_1.append(list_train_rest[-1]["sender_entropy_1"])
            training_entropy_2.append(list_train_rest[-1]["sender_entropy_2"])
            training_loss_12.append(list_train_rest[-1]["loss_1"])
            eval_loss_12.append(eval_rest["loss_1"])
            training_loss_21.append(list_train_rest[-1]["loss_2"])
            eval_loss_21.append(eval_rest["loss_2"])
            training_loss_self_11.append(list_train_rest[-1]["loss_self_11"])
            training_loss_cross_12.append(list_train_rest[-1]["loss_cross_12"])
            training_loss_imitation_12.append(list_train_rest[-1]["loss_imitation_12"])
            training_loss_self_22.append(list_train_rest[-1]["loss_self_22"])
            training_loss_cross_21.append(list_train_rest[-1]["loss_cross_21"])
            training_loss_imitation_21.append(list_train_rest[-1]["loss_imitation_21"])
            eval_loss_self_11.append(eval_rest["loss_self_11"])
            eval_loss_cross_12.append(eval_rest["loss_cross_12"])
            eval_loss_imitation_12.append(eval_rest["loss_imitation_12"])
            eval_loss_self_22.append(eval_rest["loss_self_22"])
            eval_loss_cross_21.append(eval_rest["loss_cross_21"])
            eval_loss_imitation_21.append(eval_rest["loss_imitation_21"])
            eval_reinforce_1.append(eval_rest["reinforce_term_1"])
            eval_reinforce_2.append(eval_rest["reinforce_term_2"])
            eval_baseline_1.append(eval_rest["baseline_term_1"])
            eval_baseline_2.append(eval_rest["baseline_term_2"])

            if epoch==0:
                messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
            messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22, similarity_messages = dump_dialog_model_6(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
            np_messages_1 = convert_messages_to_numpy(messages_1)
            np_messages_2 = convert_messages_to_numpy(messages_2)
            similarity_languages.append(similarity_messages)

            if epoch==0:
                preds_1=preds_2=np.zeros((np.shape(messages_test)[0]))
            preds_1,preds_2,sim_pred_1,sim_pred_2 = test_receiver_evolution(trainer.game, messages_test, device,False,past_preds_1=preds_1,past_preds_2=preds_2)
            similarity_predictions_1.append(sim_pred_1)
            similarity_predictions_2.append(sim_pred_2)

            #game.optim_params["sender_entropy_coeff_1"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_11)
            #game.optim_params["sender_entropy_coeff_2"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_22)


            # Save models
            if epoch%20==0:
                torch.save(agent_1.state_dict(), f"{opts.dir_save}/models/agent_1_weights_{epoch}.pth")
                torch.save(agent_2.state_dict(), f"{opts.dir_save}/models/agent_2_weights_{epoch}.pth")

            # Save training info
            if epoch%10==0:
                np.save(opts.dir_save+'/training_info/training_loss_{}.npy'.format(epoch), training_losses)
                np.save(opts.dir_save+'/training_info/eval_loss_{}.npy'.format(epoch), eval_losses)
                np.save(opts.dir_save+'/training_info/training_entropy_1_{}.npy'.format(epoch), training_entropy_1)
                np.save(opts.dir_save+'/training_info/training_entropy_2_{}.npy'.format(epoch), training_entropy_2)
                np.save(opts.dir_save+'/training_info/training_loss_12_{}.npy'.format(epoch), training_loss_12)
                np.save(opts.dir_save+'/training_info/eval_loss_12_{}.npy'.format(epoch), eval_loss_12)
                np.save(opts.dir_save+'/training_info/training_loss_21_{}.npy'.format(epoch), training_loss_21)
                np.save(opts.dir_save+'/training_info/eval_loss_21_{}.npy'.format(epoch), eval_loss_21)
                np.save(opts.dir_save+'/training_info/training_loss_self_11_{}.npy'.format(epoch), training_loss_self_11)
                np.save(opts.dir_save+'/training_info/training_loss_cross_12_{}.npy'.format(epoch), training_loss_cross_12)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_12_{}.npy'.format(epoch), training_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/training_loss_self_22_{}.npy'.format(epoch), training_loss_self_22)
                np.save(opts.dir_save+'/training_info/training_loss_cross_21_{}.npy'.format(epoch), training_loss_cross_21)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_21_{}.npy'.format(epoch), training_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/eval_loss_self_11_{}.npy'.format(epoch), eval_loss_self_11)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_12_{}.npy'.format(epoch), eval_loss_cross_12)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_12_{}.npy'.format(epoch), eval_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/eval_loss_self_22_{}.npy'.format(epoch), eval_loss_self_22)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_21_{}.npy'.format(epoch), eval_loss_cross_21)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_21_{}.npy'.format(epoch), eval_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/similarity_languages_{}.npy'.format(epoch), similarity_languages)
                #Reinforce
                np.save(opts.dir_save+'/training_info/reinforce_term_1_{}.npy'.format(epoch), eval_reinforce_1)
                np.save(opts.dir_save+'/training_info/reinforce_term_2_{}.npy'.format(epoch), eval_reinforce_2)
                np.save(opts.dir_save+'/training_info/baseline_term_1_{}.npy'.format(epoch), eval_baseline_1)
                np.save(opts.dir_save+'/training_info/baseline_term_2_{}.npy'.format(epoch), eval_baseline_2)
                # Policy
                np.save(opts.dir_save+'/training_info/policy_1_{}.npy'.format(epoch),eval_rest["policy_1"].cpu().numpy())
                np.save(opts.dir_save+'/training_info/policy_2_{}.npy'.format(epoch),eval_rest["policy_2"].cpu().numpy())

            # Save accuracy/message results
            np.save(opts.dir_save+'/messages/agent_1_messages_{}.npy'.format(epoch), np_messages_1)
            np.save(opts.dir_save+'/messages/agent_2_messages_{}.npy'.format(epoch), np_messages_2)
            np.save(opts.dir_save+'/accuracy/12_accuracy_{}.npy'.format(epoch), acc_vec_1)
            np.save(opts.dir_save+'/accuracy/21_accuracy_{}.npy'.format(epoch), acc_vec_2)
            np.save(opts.dir_save+'/accuracy/11_accuracy_{}.npy'.format(epoch), acc_vec_11)
            np.save(opts.dir_save+'/accuracy/22_accuracy_{}.npy'.format(epoch), acc_vec_22)
            np.save(opts.dir_save+'/preds/preds_1_{}.npy'.format(epoch), preds_1)
            np.save(opts.dir_save+'/preds/preds_2_{}.npy'.format(epoch), preds_2)
            np.save(opts.dir_save+'/preds/sim_pred_1_{}.npy'.format(epoch), similarity_predictions_1)
            np.save(opts.dir_save+'/preds/sim_pred_2_{}.npy'.format(epoch), similarity_predictions_2)


        print("Step 2 = Interaction",flush=True)

        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_1":opts.receiver_entropy_coeff,
                      "sender_entropy_coeff_2":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_2":opts.receiver_entropy_coeff}

        loss_weights={"self":0.,"cross":1.,"imitation":0.}
        #loss_weights={"self":opts.self_weight,"cross":opts.cross_weight,"imitation":opts.imitation_weight}

        game = DialogReinforce(Agent_1=agent_1,
                                Agent_2=agent_2,
                                loss_understanding=loss_understanding,
                                loss_imitation=loss_message_imitation,
                                optim_params=optim_params,
                                baseline_mode=opts.baseline_mode,
                                reward_mode=opts.reward_mode,
                                loss_weights=loss_weights,
                                device=device)

        optimizer_speaker_1 = core.build_optimizer(speaker_1_parameters,lr=0.)
        optimizer_listener_1 = core.build_optimizer(listener_1_parameters,lr=0.)
        optimizer_speaker_2 = core.build_optimizer(speaker_2_parameters,lr=opts.sender_lr)
        optimizer_listener_2 = core.build_optimizer(listener_2_parameters,lr=opts.receiver_lr)



        "Create trainer"
        trainer = TrainerDialog4Optim(game=game,
                                      optimizer_speaker_1=optimizer_speaker_1,
                                      optimizer_listener_1=optimizer_listener_1,
                                      optimizer_speaker_2=optimizer_speaker_2,
                                      optimizer_listener_2=optimizer_listener_2,
                                      train_data=train_loader,
                                      validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        for epoch in range(int(opts.n_epochs),2*int(opts.n_epochs)):

            print("Epoch: {}".format(epoch))

            # Train
            list_train_loss,list_train_rest = trainer.train(n_epochs=1)

            #print(list_train_rest[-1],flush=True)

            # Eval
            eval_loss,eval_rest = trainer.eval()

            # Store results
            training_losses.append(list_train_loss[-1])
            eval_losses.append(eval_loss)
            training_entropy_1.append(list_train_rest[-1]["sender_entropy_1"])
            training_entropy_2.append(list_train_rest[-1]["sender_entropy_2"])
            training_loss_12.append(list_train_rest[-1]["loss_1"])
            eval_loss_12.append(eval_rest["loss_1"])
            training_loss_21.append(list_train_rest[-1]["loss_2"])
            eval_loss_21.append(eval_rest["loss_2"])
            training_loss_self_11.append(list_train_rest[-1]["loss_self_11"])
            training_loss_cross_12.append(list_train_rest[-1]["loss_cross_12"])
            training_loss_imitation_12.append(list_train_rest[-1]["loss_imitation_12"])
            training_loss_self_22.append(list_train_rest[-1]["loss_self_22"])
            training_loss_cross_21.append(list_train_rest[-1]["loss_cross_21"])
            training_loss_imitation_21.append(list_train_rest[-1]["loss_imitation_21"])
            eval_loss_self_11.append(eval_rest["loss_self_11"])
            eval_loss_cross_12.append(eval_rest["loss_cross_12"])
            eval_loss_imitation_12.append(eval_rest["loss_imitation_12"])
            eval_loss_self_22.append(eval_rest["loss_self_22"])
            eval_loss_cross_21.append(eval_rest["loss_cross_21"])
            eval_loss_imitation_21.append(eval_rest["loss_imitation_21"])
            eval_reinforce_1.append(eval_rest["reinforce_term_1"])
            eval_reinforce_2.append(eval_rest["reinforce_term_2"])
            eval_baseline_1.append(eval_rest["baseline_term_1"])
            eval_baseline_2.append(eval_rest["baseline_term_2"])

            if epoch==0:
                messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
            messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22, similarity_messages = dump_dialog_model_6(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
            np_messages_1 = convert_messages_to_numpy(messages_1)
            np_messages_2 = convert_messages_to_numpy(messages_2)
            similarity_languages.append(similarity_messages)

            if epoch==0:
                preds_1=preds_2=np.zeros((np.shape(messages_test)[0]))
            preds_1,preds_2,sim_pred_1,sim_pred_2 = test_receiver_evolution(trainer.game, messages_test, device,False,past_preds_1=preds_1,past_preds_2=preds_2)
            similarity_predictions_1.append(sim_pred_1)
            similarity_predictions_2.append(sim_pred_2)

            #game.optim_params["sender_entropy_coeff_1"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_11)
            #game.optim_params["sender_entropy_coeff_2"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_22)


            # Save models
            if epoch%20==0:
                torch.save(agent_1.state_dict(), f"{opts.dir_save}/models/agent_1_weights_{epoch}.pth")
                torch.save(agent_2.state_dict(), f"{opts.dir_save}/models/agent_2_weights_{epoch}.pth")

            # Save training info
            if epoch%10==0:
                np.save(opts.dir_save+'/training_info/training_loss_{}.npy'.format(epoch), training_losses)
                np.save(opts.dir_save+'/training_info/eval_loss_{}.npy'.format(epoch), eval_losses)
                np.save(opts.dir_save+'/training_info/training_entropy_1_{}.npy'.format(epoch), training_entropy_1)
                np.save(opts.dir_save+'/training_info/training_entropy_2_{}.npy'.format(epoch), training_entropy_2)
                np.save(opts.dir_save+'/training_info/training_loss_12_{}.npy'.format(epoch), training_loss_12)
                np.save(opts.dir_save+'/training_info/eval_loss_12_{}.npy'.format(epoch), eval_loss_12)
                np.save(opts.dir_save+'/training_info/training_loss_21_{}.npy'.format(epoch), training_loss_21)
                np.save(opts.dir_save+'/training_info/eval_loss_21_{}.npy'.format(epoch), eval_loss_21)
                np.save(opts.dir_save+'/training_info/training_loss_self_11_{}.npy'.format(epoch), training_loss_self_11)
                np.save(opts.dir_save+'/training_info/training_loss_cross_12_{}.npy'.format(epoch), training_loss_cross_12)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_12_{}.npy'.format(epoch), training_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/training_loss_self_22_{}.npy'.format(epoch), training_loss_self_22)
                np.save(opts.dir_save+'/training_info/training_loss_cross_21_{}.npy'.format(epoch), training_loss_cross_21)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_21_{}.npy'.format(epoch), training_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/eval_loss_self_11_{}.npy'.format(epoch), eval_loss_self_11)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_12_{}.npy'.format(epoch), eval_loss_cross_12)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_12_{}.npy'.format(epoch), eval_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/eval_loss_self_22_{}.npy'.format(epoch), eval_loss_self_22)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_21_{}.npy'.format(epoch), eval_loss_cross_21)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_21_{}.npy'.format(epoch), eval_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/similarity_languages_{}.npy'.format(epoch), similarity_languages)
                #Reinforce
                np.save(opts.dir_save+'/training_info/reinforce_term_1_{}.npy'.format(epoch), eval_reinforce_1)
                np.save(opts.dir_save+'/training_info/reinforce_term_2_{}.npy'.format(epoch), eval_reinforce_2)
                np.save(opts.dir_save+'/training_info/baseline_term_1_{}.npy'.format(epoch), eval_baseline_1)
                np.save(opts.dir_save+'/training_info/baseline_term_2_{}.npy'.format(epoch), eval_baseline_2)
                # Policy
                np.save(opts.dir_save+'/training_info/policy_1_{}.npy'.format(epoch),eval_rest["policy_1"].cpu().numpy())
                np.save(opts.dir_save+'/training_info/policy_2_{}.npy'.format(epoch),eval_rest["policy_2"].cpu().numpy())

            # Save accuracy/message results
            np.save(opts.dir_save+'/messages/agent_1_messages_{}.npy'.format(epoch), np_messages_1)
            np.save(opts.dir_save+'/messages/agent_2_messages_{}.npy'.format(epoch), np_messages_2)
            np.save(opts.dir_save+'/accuracy/12_accuracy_{}.npy'.format(epoch), acc_vec_1)
            np.save(opts.dir_save+'/accuracy/21_accuracy_{}.npy'.format(epoch), acc_vec_2)
            np.save(opts.dir_save+'/accuracy/11_accuracy_{}.npy'.format(epoch), acc_vec_11)
            np.save(opts.dir_save+'/accuracy/22_accuracy_{}.npy'.format(epoch), acc_vec_22)
            np.save(opts.dir_save+'/preds/preds_1_{}.npy'.format(epoch), preds_1)
            np.save(opts.dir_save+'/preds/preds_2_{}.npy'.format(epoch), preds_2)
            np.save(opts.dir_save+'/preds/sim_pred_1_{}.npy'.format(epoch), similarity_predictions_1)
            np.save(opts.dir_save+'/preds/sim_pred_2_{}.npy'.format(epoch), similarity_predictions_2)


    elif opts.model=="expe_characteristic_time":

        "Define agents"


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

        "Prepare training"

        # Create save dir
        if not path.exists(opts.dir_save):
            os.system("mkdir {}".format(opts.dir_save))
            os.system("mkdir -p {}/models {}/training_info {}/messages {}/accuracy {}/preds".format(opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save))

        # Main losses
        training_losses=[]
        eval_losses=[]
        training_entropy_1=[]
        training_entropy_2=[]
        training_loss_12=[]
        eval_loss_12=[]
        training_loss_21=[]
        eval_loss_21=[]

        # Specific losses
        training_loss_self_11=[]
        training_loss_cross_12=[]
        training_loss_imitation_12=[]
        training_loss_self_22=[]
        training_loss_cross_21=[]
        training_loss_imitation_21=[]
        eval_loss_self_11=[]
        eval_loss_cross_12=[]
        eval_loss_imitation_12=[]
        eval_loss_self_22=[]
        eval_loss_cross_21=[]
        eval_loss_imitation_21=[]

        # REINFORCE
        eval_reinforce_1=[]
        eval_reinforce_2=[]
        eval_baseline_1=[]
        eval_baseline_2=[]

        # Linguistic
        similarity_languages=[]

        "Prepare test"

        similarity_predictions_1=[]
        similarity_predictions_2=[]
        nb_messages_test=10000
        messages_test = torch.tensor(np.random.randint(opts.vocab_size,size=(nb_messages_test,opts.max_len)))

        "Pretraining "

        print("Step 1 = Pretraining",flush=True)

        "Define game"

        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_1":opts.receiver_entropy_coeff,
                      "sender_entropy_coeff_2":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_2":opts.receiver_entropy_coeff}

        loss_weights={"self":1.,"cross":0.,"imitation":0.}
        #loss_weights={"self":opts.self_weight,"cross":opts.cross_weight,"imitation":opts.imitation_weight}

        game = DialogReinforce(Agent_1=agent_1,
                                Agent_2=agent_2,
                                loss_understanding=loss_understanding,
                                loss_imitation=loss_message_imitation,
                                optim_params=optim_params,
                                baseline_mode=opts.baseline_mode,
                                reward_mode=opts.reward_mode,
                                loss_weights=loss_weights,
                                device=device)

        "Create optimizers"

        speaker_1_parameters = list(game.agent_1.agent_sender.parameters()) + \
                               list(game.agent_1.sender_norm_h.parameters()) + \
                               list(game.agent_1.sender_norm_c.parameters()) + \
                               list(game.agent_1.hidden_to_output.parameters()) + \
                               list(game.agent_1.sender_embedding.parameters()) + \
                               list(game.agent_1.sender_cells.parameters())

        listener_1_parameters = list(game.agent_1.agent_receiver.parameters()) + \
                              list(game.agent_1.receiver_cell.parameters()) + \
                              list(game.agent_1.receiver_embedding.parameters())

        speaker_2_parameters = list(game.agent_2.agent_sender.parameters()) + \
                               list(game.agent_2.sender_norm_h.parameters()) + \
                               list(game.agent_2.sender_norm_c.parameters()) + \
                               list(game.agent_2.hidden_to_output.parameters()) + \
                               list(game.agent_2.sender_embedding.parameters()) + \
                               list(game.agent_2.sender_cells.parameters())

        listener_2_parameters = list(game.agent_2.agent_receiver.parameters()) + \
                              list(game.agent_2.receiver_cell.parameters()) + \
                              list(game.agent_2.receiver_embedding.parameters())

        optimizer_speaker_1 = core.build_optimizer(speaker_1_parameters,lr=opts.sender_lr)
        optimizer_listener_1 = core.build_optimizer(listener_1_parameters,lr=opts.receiver_lr)
        optimizer_speaker_2 = core.build_optimizer(speaker_2_parameters,lr=0.)
        optimizer_listener_2 = core.build_optimizer(listener_2_parameters,lr=0.)



        "Create trainer"
        trainer = TrainerDialog4Optim(game=game,
                                      optimizer_speaker_1=optimizer_speaker_1,
                                      optimizer_listener_1=optimizer_listener_1,
                                      optimizer_speaker_2=optimizer_speaker_2,
                                      optimizer_listener_2=optimizer_listener_2,
                                      train_data=train_loader,
                                      validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        for epoch in range(int(opts.n_epochs)):

            print("Epoch: {}".format(epoch))

            # Train
            list_train_loss,list_train_rest = trainer.train(n_epochs=1)

            #print(list_train_rest[-1],flush=True)

            # Eval
            eval_loss,eval_rest = trainer.eval()

            # Store results
            training_losses.append(list_train_loss[-1])
            eval_losses.append(eval_loss)
            training_entropy_1.append(list_train_rest[-1]["sender_entropy_1"])
            training_entropy_2.append(list_train_rest[-1]["sender_entropy_2"])
            training_loss_12.append(list_train_rest[-1]["loss_1"])
            eval_loss_12.append(eval_rest["loss_1"])
            training_loss_21.append(list_train_rest[-1]["loss_2"])
            eval_loss_21.append(eval_rest["loss_2"])
            training_loss_self_11.append(list_train_rest[-1]["loss_self_11"])
            training_loss_cross_12.append(list_train_rest[-1]["loss_cross_12"])
            training_loss_imitation_12.append(list_train_rest[-1]["loss_imitation_12"])
            training_loss_self_22.append(list_train_rest[-1]["loss_self_22"])
            training_loss_cross_21.append(list_train_rest[-1]["loss_cross_21"])
            training_loss_imitation_21.append(list_train_rest[-1]["loss_imitation_21"])
            eval_loss_self_11.append(eval_rest["loss_self_11"])
            eval_loss_cross_12.append(eval_rest["loss_cross_12"])
            eval_loss_imitation_12.append(eval_rest["loss_imitation_12"])
            eval_loss_self_22.append(eval_rest["loss_self_22"])
            eval_loss_cross_21.append(eval_rest["loss_cross_21"])
            eval_loss_imitation_21.append(eval_rest["loss_imitation_21"])
            eval_reinforce_1.append(eval_rest["reinforce_term_1"])
            eval_reinforce_2.append(eval_rest["reinforce_term_2"])
            eval_baseline_1.append(eval_rest["baseline_term_1"])
            eval_baseline_2.append(eval_rest["baseline_term_2"])

            if epoch==0:
                messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
            messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22, similarity_messages = dump_dialog_model_6(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
            np_messages_1 = convert_messages_to_numpy(messages_1)
            np_messages_2 = convert_messages_to_numpy(messages_2)
            similarity_languages.append(similarity_messages)

            if epoch==0:
                preds_1=preds_2=np.zeros((np.shape(messages_test)[0]))
            preds_1,preds_2,sim_pred_1,sim_pred_2 = test_receiver_evolution(trainer.game, messages_test, device,False,past_preds_1=preds_1,past_preds_2=preds_2)
            similarity_predictions_1.append(sim_pred_1)
            similarity_predictions_2.append(sim_pred_2)

            #game.optim_params["sender_entropy_coeff_1"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_11)
            #game.optim_params["sender_entropy_coeff_2"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_22)


            # Save models
            if epoch%20==0:
                torch.save(agent_1.state_dict(), f"{opts.dir_save}/models/agent_1_weights_{epoch}.pth")
                torch.save(agent_2.state_dict(), f"{opts.dir_save}/models/agent_2_weights_{epoch}.pth")

            # Save training info
            if epoch%10==0:
                np.save(opts.dir_save+'/training_info/training_loss_{}.npy'.format(epoch), training_losses)
                np.save(opts.dir_save+'/training_info/eval_loss_{}.npy'.format(epoch), eval_losses)
                np.save(opts.dir_save+'/training_info/training_entropy_1_{}.npy'.format(epoch), training_entropy_1)
                np.save(opts.dir_save+'/training_info/training_entropy_2_{}.npy'.format(epoch), training_entropy_2)
                np.save(opts.dir_save+'/training_info/training_loss_12_{}.npy'.format(epoch), training_loss_12)
                np.save(opts.dir_save+'/training_info/eval_loss_12_{}.npy'.format(epoch), eval_loss_12)
                np.save(opts.dir_save+'/training_info/training_loss_21_{}.npy'.format(epoch), training_loss_21)
                np.save(opts.dir_save+'/training_info/eval_loss_21_{}.npy'.format(epoch), eval_loss_21)
                np.save(opts.dir_save+'/training_info/training_loss_self_11_{}.npy'.format(epoch), training_loss_self_11)
                np.save(opts.dir_save+'/training_info/training_loss_cross_12_{}.npy'.format(epoch), training_loss_cross_12)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_12_{}.npy'.format(epoch), training_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/training_loss_self_22_{}.npy'.format(epoch), training_loss_self_22)
                np.save(opts.dir_save+'/training_info/training_loss_cross_21_{}.npy'.format(epoch), training_loss_cross_21)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_21_{}.npy'.format(epoch), training_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/eval_loss_self_11_{}.npy'.format(epoch), eval_loss_self_11)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_12_{}.npy'.format(epoch), eval_loss_cross_12)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_12_{}.npy'.format(epoch), eval_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/eval_loss_self_22_{}.npy'.format(epoch), eval_loss_self_22)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_21_{}.npy'.format(epoch), eval_loss_cross_21)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_21_{}.npy'.format(epoch), eval_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/similarity_languages_{}.npy'.format(epoch), similarity_languages)
                #Reinforce
                np.save(opts.dir_save+'/training_info/reinforce_term_1_{}.npy'.format(epoch), eval_reinforce_1)
                np.save(opts.dir_save+'/training_info/reinforce_term_2_{}.npy'.format(epoch), eval_reinforce_2)
                np.save(opts.dir_save+'/training_info/baseline_term_1_{}.npy'.format(epoch), eval_baseline_1)
                np.save(opts.dir_save+'/training_info/baseline_term_2_{}.npy'.format(epoch), eval_baseline_2)
                # Policy
                np.save(opts.dir_save+'/training_info/policy_1_{}.npy'.format(epoch),eval_rest["policy_1"].cpu().numpy())
                np.save(opts.dir_save+'/training_info/policy_2_{}.npy'.format(epoch),eval_rest["policy_2"].cpu().numpy())

            # Save accuracy/message results
            np.save(opts.dir_save+'/messages/agent_1_messages_{}.npy'.format(epoch), np_messages_1)
            np.save(opts.dir_save+'/messages/agent_2_messages_{}.npy'.format(epoch), np_messages_2)
            np.save(opts.dir_save+'/accuracy/12_accuracy_{}.npy'.format(epoch), acc_vec_1)
            np.save(opts.dir_save+'/accuracy/21_accuracy_{}.npy'.format(epoch), acc_vec_2)
            np.save(opts.dir_save+'/accuracy/11_accuracy_{}.npy'.format(epoch), acc_vec_11)
            np.save(opts.dir_save+'/accuracy/22_accuracy_{}.npy'.format(epoch), acc_vec_22)
            np.save(opts.dir_save+'/preds/preds_1_{}.npy'.format(epoch), preds_1)
            np.save(opts.dir_save+'/preds/preds_2_{}.npy'.format(epoch), preds_2)
            np.save(opts.dir_save+'/preds/sim_pred_1_{}.npy'.format(epoch), similarity_predictions_1)
            np.save(opts.dir_save+'/preds/sim_pred_2_{}.npy'.format(epoch), similarity_predictions_2)


        print("Step 2 = Interaction",flush=True)

        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_1":opts.receiver_entropy_coeff,
                      "sender_entropy_coeff_2":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_2":opts.receiver_entropy_coeff}

        loss_weights={"self":0.,"cross":1.,"imitation":0.}
        #loss_weights={"self":opts.self_weight,"cross":opts.cross_weight,"imitation":opts.imitation_weight}

        game = DialogReinforce(Agent_1=agent_1,
                                Agent_2=agent_2,
                                loss_understanding=loss_understanding,
                                loss_imitation=loss_message_imitation,
                                optim_params=optim_params,
                                baseline_mode=opts.baseline_mode,
                                reward_mode=opts.reward_mode,
                                loss_weights=loss_weights,
                                device=device)

        optimizer_speaker_1 = core.build_optimizer(speaker_1_parameters,lr=0.)
        optimizer_listener_1 = core.build_optimizer(listener_1_parameters,lr=0.)
        optimizer_speaker_2 = core.build_optimizer(speaker_2_parameters,lr=opts.sender_lr)
        optimizer_listener_2 = core.build_optimizer(listener_2_parameters,lr=opts.receiver_lr)



        "Create trainer"
        trainer = TrainerDialog4Optim(game=game,
                                      optimizer_speaker_1=optimizer_speaker_1,
                                      optimizer_listener_1=optimizer_listener_1,
                                      optimizer_speaker_2=optimizer_speaker_2,
                                      optimizer_listener_2=optimizer_listener_2,
                                      train_data=train_loader,
                                      validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        for epoch in range(int(opts.n_epochs),2*int(opts.n_epochs)):

            print("Epoch: {}".format(epoch))

            # Train
            list_train_loss,list_train_rest = trainer.train(n_epochs=1)

            #print(list_train_rest[-1],flush=True)

            # Eval
            eval_loss,eval_rest = trainer.eval()

            # Store results
            training_losses.append(list_train_loss[-1])
            eval_losses.append(eval_loss)
            training_entropy_1.append(list_train_rest[-1]["sender_entropy_1"])
            training_entropy_2.append(list_train_rest[-1]["sender_entropy_2"])
            training_loss_12.append(list_train_rest[-1]["loss_1"])
            eval_loss_12.append(eval_rest["loss_1"])
            training_loss_21.append(list_train_rest[-1]["loss_2"])
            eval_loss_21.append(eval_rest["loss_2"])
            training_loss_self_11.append(list_train_rest[-1]["loss_self_11"])
            training_loss_cross_12.append(list_train_rest[-1]["loss_cross_12"])
            training_loss_imitation_12.append(list_train_rest[-1]["loss_imitation_12"])
            training_loss_self_22.append(list_train_rest[-1]["loss_self_22"])
            training_loss_cross_21.append(list_train_rest[-1]["loss_cross_21"])
            training_loss_imitation_21.append(list_train_rest[-1]["loss_imitation_21"])
            eval_loss_self_11.append(eval_rest["loss_self_11"])
            eval_loss_cross_12.append(eval_rest["loss_cross_12"])
            eval_loss_imitation_12.append(eval_rest["loss_imitation_12"])
            eval_loss_self_22.append(eval_rest["loss_self_22"])
            eval_loss_cross_21.append(eval_rest["loss_cross_21"])
            eval_loss_imitation_21.append(eval_rest["loss_imitation_21"])
            eval_reinforce_1.append(eval_rest["reinforce_term_1"])
            eval_reinforce_2.append(eval_rest["reinforce_term_2"])
            eval_baseline_1.append(eval_rest["baseline_term_1"])
            eval_baseline_2.append(eval_rest["baseline_term_2"])

            if epoch==0:
                messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
            messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22, similarity_messages = dump_dialog_model_6(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
            np_messages_1 = convert_messages_to_numpy(messages_1)
            np_messages_2 = convert_messages_to_numpy(messages_2)
            similarity_languages.append(similarity_messages)

            if epoch==0:
                preds_1=preds_2=np.zeros((np.shape(messages_test)[0]))
            preds_1,preds_2,sim_pred_1,sim_pred_2 = test_receiver_evolution(trainer.game, messages_test, device,False,past_preds_1=preds_1,past_preds_2=preds_2)
            similarity_predictions_1.append(sim_pred_1)
            similarity_predictions_2.append(sim_pred_2)

            #game.optim_params["sender_entropy_coeff_1"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_11)
            #game.optim_params["sender_entropy_coeff_2"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_22)


            # Save models
            if epoch%20==0:
                torch.save(agent_1.state_dict(), f"{opts.dir_save}/models/agent_1_weights_{epoch}.pth")
                torch.save(agent_2.state_dict(), f"{opts.dir_save}/models/agent_2_weights_{epoch}.pth")

            # Save training info
            if epoch%10==0:
                np.save(opts.dir_save+'/training_info/training_loss_{}.npy'.format(epoch), training_losses)
                np.save(opts.dir_save+'/training_info/eval_loss_{}.npy'.format(epoch), eval_losses)
                np.save(opts.dir_save+'/training_info/training_entropy_1_{}.npy'.format(epoch), training_entropy_1)
                np.save(opts.dir_save+'/training_info/training_entropy_2_{}.npy'.format(epoch), training_entropy_2)
                np.save(opts.dir_save+'/training_info/training_loss_12_{}.npy'.format(epoch), training_loss_12)
                np.save(opts.dir_save+'/training_info/eval_loss_12_{}.npy'.format(epoch), eval_loss_12)
                np.save(opts.dir_save+'/training_info/training_loss_21_{}.npy'.format(epoch), training_loss_21)
                np.save(opts.dir_save+'/training_info/eval_loss_21_{}.npy'.format(epoch), eval_loss_21)
                np.save(opts.dir_save+'/training_info/training_loss_self_11_{}.npy'.format(epoch), training_loss_self_11)
                np.save(opts.dir_save+'/training_info/training_loss_cross_12_{}.npy'.format(epoch), training_loss_cross_12)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_12_{}.npy'.format(epoch), training_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/training_loss_self_22_{}.npy'.format(epoch), training_loss_self_22)
                np.save(opts.dir_save+'/training_info/training_loss_cross_21_{}.npy'.format(epoch), training_loss_cross_21)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_21_{}.npy'.format(epoch), training_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/eval_loss_self_11_{}.npy'.format(epoch), eval_loss_self_11)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_12_{}.npy'.format(epoch), eval_loss_cross_12)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_12_{}.npy'.format(epoch), eval_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/eval_loss_self_22_{}.npy'.format(epoch), eval_loss_self_22)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_21_{}.npy'.format(epoch), eval_loss_cross_21)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_21_{}.npy'.format(epoch), eval_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/similarity_languages_{}.npy'.format(epoch), similarity_languages)
                #Reinforce
                np.save(opts.dir_save+'/training_info/reinforce_term_1_{}.npy'.format(epoch), eval_reinforce_1)
                np.save(opts.dir_save+'/training_info/reinforce_term_2_{}.npy'.format(epoch), eval_reinforce_2)
                np.save(opts.dir_save+'/training_info/baseline_term_1_{}.npy'.format(epoch), eval_baseline_1)
                np.save(opts.dir_save+'/training_info/baseline_term_2_{}.npy'.format(epoch), eval_baseline_2)
                # Policy
                np.save(opts.dir_save+'/training_info/policy_1_{}.npy'.format(epoch),eval_rest["policy_1"].cpu().numpy())
                np.save(opts.dir_save+'/training_info/policy_2_{}.npy'.format(epoch),eval_rest["policy_2"].cpu().numpy())

            # Save accuracy/message results
            np.save(opts.dir_save+'/messages/agent_1_messages_{}.npy'.format(epoch), np_messages_1)
            np.save(opts.dir_save+'/messages/agent_2_messages_{}.npy'.format(epoch), np_messages_2)
            np.save(opts.dir_save+'/accuracy/12_accuracy_{}.npy'.format(epoch), acc_vec_1)
            np.save(opts.dir_save+'/accuracy/21_accuracy_{}.npy'.format(epoch), acc_vec_2)
            np.save(opts.dir_save+'/accuracy/11_accuracy_{}.npy'.format(epoch), acc_vec_11)
            np.save(opts.dir_save+'/accuracy/22_accuracy_{}.npy'.format(epoch), acc_vec_22)
            np.save(opts.dir_save+'/preds/preds_1_{}.npy'.format(epoch), preds_1)
            np.save(opts.dir_save+'/preds/preds_2_{}.npy'.format(epoch), preds_2)
            np.save(opts.dir_save+'/preds/sim_pred_1_{}.npy'.format(epoch), similarity_predictions_1)
            np.save(opts.dir_save+'/preds/sim_pred_2_{}.npy'.format(epoch), similarity_predictions_2)

    if opts.model=="expe_lr":

        "Define agents"


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


        "Define game"

        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_1":opts.receiver_entropy_coeff,
                      "sender_entropy_coeff_2":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_2":opts.receiver_entropy_coeff}

        if opts.optim_mode=="cross":
            loss_weights={"self":0.,"cross":1.,"imitation":0.}
        elif opts.optim_mode=="cross+self":
            loss_weights={"self":1.,"cross":1.,"imitation":0.}
        else:
            loss_weights={"self":1.,"cross":1.,"imitation":1.}
        #loss_weights={"self":opts.self_weight,"cross":opts.cross_weight,"imitation":opts.imitation_weight}

        game = DialogReinforce(Agent_1=agent_1,
                                Agent_2=agent_2,
                                loss_understanding=loss_understanding,
                                loss_imitation=loss_message_imitation,
                                optim_params=optim_params,
                                baseline_mode=opts.baseline_mode,
                                reward_mode=opts.reward_mode,
                                loss_weights=loss_weights,
                                device=device)

        speaker_parameters = list(game.agent_1.agent_sender.parameters()) + \
                               list(game.agent_1.sender_norm_h.parameters()) + \
                               list(game.agent_1.sender_norm_c.parameters()) + \
                               list(game.agent_1.hidden_to_output.parameters()) + \
                               list(game.agent_1.sender_embedding.parameters()) + \
                               list(game.agent_1.sender_cells.parameters()) + \
                               list(game.agent_2.agent_sender.parameters()) + \
                               list(game.agent_2.sender_norm_h.parameters()) + \
                               list(game.agent_2.sender_norm_c.parameters()) + \
                               list(game.agent_2.hidden_to_output.parameters()) + \
                               list(game.agent_2.sender_embedding.parameters()) + \
                               list(game.agent_2.sender_cells.parameters())

        listener_parameters = list(game.agent_1.agent_receiver.parameters()) + \
                              list(game.agent_1.receiver_cell.parameters()) + \
                              list(game.agent_1.receiver_embedding.parameters()) + \
                              list(game.agent_2.agent_receiver.parameters()) + \
                              list(game.agent_2.receiver_cell.parameters()) + \
                              list(game.agent_2.receiver_embedding.parameters())

        "Create optimizers"
        # ADAM
        #optimizer_speaker = core.build_optimizer(list(speaker_parameters),lr=opts.sender_lr)
        #optimizer_listener = core.build_optimizer(list(listener_parameters),lr=opts.receiver_lr)

        # SGD
        optimizer_speaker=torch.optim.SGD(speaker_parameters, lr=opts.sender_lr, momentum=0.9,nesterov=False)
        optimizer_listener=torch.optim.SGD(listener_parameters, lr=opts.receiver_lr, momentum=0.9,nesterov=False)


        "Create trainer"
        #trainer = TrainerDialog(game=game, optimizer=optimizer, train_data=train_loader, \
        #                        validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])
        trainer = TrainerDialogAsymLR(game=game, optimizer_speaker=optimizer_speaker,optimizer_listener=optimizer_listener, train_data=train_loader, \
                                      validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        "Prepare training"

        # Create save dir
        if not path.exists(opts.dir_save):
            os.system("mkdir {}".format(opts.dir_save))
            os.system("mkdir -p {}/models {}/training_info {}/messages {}/accuracy {}/preds".format(opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save))

        # Main losses
        training_losses=[]
        eval_losses=[]
        training_entropy_1=[]
        training_entropy_2=[]
        training_loss_12=[]
        eval_loss_12=[]
        training_loss_21=[]
        eval_loss_21=[]

        # Specific losses
        training_loss_self_11=[]
        training_loss_cross_12=[]
        training_loss_imitation_12=[]
        training_loss_self_22=[]
        training_loss_cross_21=[]
        training_loss_imitation_21=[]
        eval_loss_self_11=[]
        eval_loss_cross_12=[]
        eval_loss_imitation_12=[]
        eval_loss_self_22=[]
        eval_loss_cross_21=[]
        eval_loss_imitation_21=[]

        # REINFORCE
        eval_reinforce_1=[]
        eval_reinforce_2=[]
        eval_baseline_1=[]
        eval_baseline_2=[]

        # Linguistic
        similarity_languages=[]

        "Prepare test"

        similarity_predictions_1=[]
        similarity_predictions_2=[]
        nb_messages_test=10000
        messages_test = torch.tensor(np.random.randint(opts.vocab_size,size=(nb_messages_test,opts.max_len)))

        gradient_speaker=[]
        gradient_listener=[]

        "Train"

        for epoch in range(int(opts.n_epochs)):

            print("Epoch: {}".format(epoch))

            # Train
            list_train_loss,list_train_rest = trainer.train(n_epochs=1)

            #print(list_train_rest[-1],flush=True)

            # Eval
            eval_loss,eval_rest = trainer.eval()

            # Store results
            training_losses.append(list_train_loss[-1])
            eval_losses.append(eval_loss)
            training_entropy_1.append(list_train_rest[-1]["sender_entropy_1"])
            training_entropy_2.append(list_train_rest[-1]["sender_entropy_2"])
            training_loss_12.append(list_train_rest[-1]["loss_1"])
            eval_loss_12.append(eval_rest["loss_1"])
            training_loss_21.append(list_train_rest[-1]["loss_2"])
            eval_loss_21.append(eval_rest["loss_2"])
            training_loss_self_11.append(list_train_rest[-1]["loss_self_11"])
            training_loss_cross_12.append(list_train_rest[-1]["loss_cross_12"])
            training_loss_imitation_12.append(list_train_rest[-1]["loss_imitation_12"])
            training_loss_self_22.append(list_train_rest[-1]["loss_self_22"])
            training_loss_cross_21.append(list_train_rest[-1]["loss_cross_21"])
            training_loss_imitation_21.append(list_train_rest[-1]["loss_imitation_21"])
            eval_loss_self_11.append(eval_rest["loss_self_11"])
            eval_loss_cross_12.append(eval_rest["loss_cross_12"])
            eval_loss_imitation_12.append(eval_rest["loss_imitation_12"])
            eval_loss_self_22.append(eval_rest["loss_self_22"])
            eval_loss_cross_21.append(eval_rest["loss_cross_21"])
            eval_loss_imitation_21.append(eval_rest["loss_imitation_21"])
            eval_reinforce_1.append(eval_rest["reinforce_term_1"])
            eval_reinforce_2.append(eval_rest["reinforce_term_2"])
            eval_baseline_1.append(eval_rest["baseline_term_1"])
            eval_baseline_2.append(eval_rest["baseline_term_2"])

            if epoch==0:
                messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
            messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22, similarity_messages = dump_dialog_model_6(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
            np_messages_1 = convert_messages_to_numpy(messages_1)
            np_messages_2 = convert_messages_to_numpy(messages_2)
            similarity_languages.append(similarity_messages)

            if epoch==0:
                preds_1=preds_2=np.zeros((np.shape(messages_test)[0]))
            preds_1,preds_2,sim_pred_1,sim_pred_2 = test_receiver_evolution(trainer.game, messages_test, device,False,past_preds_1=preds_1,past_preds_2=preds_2)
            similarity_predictions_1.append(sim_pred_1)
            similarity_predictions_2.append(sim_pred_2)

            #game.optim_params["sender_entropy_coeff_1"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_11)
            #game.optim_params["sender_entropy_coeff_2"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_22)


            # Save models
            if epoch%20==0:
                torch.save(agent_1.state_dict(), f"{opts.dir_save}/models/agent_1_weights_{epoch}.pth")
                torch.save(agent_2.state_dict(), f"{opts.dir_save}/models/agent_2_weights_{epoch}.pth")

            # Save training info
            if epoch%10==0:
                np.save(opts.dir_save+'/training_info/training_loss_{}.npy'.format(epoch), training_losses)
                np.save(opts.dir_save+'/training_info/eval_loss_{}.npy'.format(epoch), eval_losses)
                np.save(opts.dir_save+'/training_info/training_entropy_1_{}.npy'.format(epoch), training_entropy_1)
                np.save(opts.dir_save+'/training_info/training_entropy_2_{}.npy'.format(epoch), training_entropy_2)
                np.save(opts.dir_save+'/training_info/training_loss_12_{}.npy'.format(epoch), training_loss_12)
                np.save(opts.dir_save+'/training_info/eval_loss_12_{}.npy'.format(epoch), eval_loss_12)
                np.save(opts.dir_save+'/training_info/training_loss_21_{}.npy'.format(epoch), training_loss_21)
                np.save(opts.dir_save+'/training_info/eval_loss_21_{}.npy'.format(epoch), eval_loss_21)
                np.save(opts.dir_save+'/training_info/training_loss_self_11_{}.npy'.format(epoch), training_loss_self_11)
                np.save(opts.dir_save+'/training_info/training_loss_cross_12_{}.npy'.format(epoch), training_loss_cross_12)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_12_{}.npy'.format(epoch), training_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/training_loss_self_22_{}.npy'.format(epoch), training_loss_self_22)
                np.save(opts.dir_save+'/training_info/training_loss_cross_21_{}.npy'.format(epoch), training_loss_cross_21)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_21_{}.npy'.format(epoch), training_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/eval_loss_self_11_{}.npy'.format(epoch), eval_loss_self_11)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_12_{}.npy'.format(epoch), eval_loss_cross_12)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_12_{}.npy'.format(epoch), eval_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/eval_loss_self_22_{}.npy'.format(epoch), eval_loss_self_22)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_21_{}.npy'.format(epoch), eval_loss_cross_21)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_21_{}.npy'.format(epoch), eval_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/similarity_languages_{}.npy'.format(epoch), similarity_languages)
                #Reinforce
                np.save(opts.dir_save+'/training_info/reinforce_term_1_{}.npy'.format(epoch), eval_reinforce_1)
                np.save(opts.dir_save+'/training_info/reinforce_term_2_{}.npy'.format(epoch), eval_reinforce_2)
                np.save(opts.dir_save+'/training_info/baseline_term_1_{}.npy'.format(epoch), eval_baseline_1)
                np.save(opts.dir_save+'/training_info/baseline_term_2_{}.npy'.format(epoch), eval_baseline_2)
                # Policy
                np.save(opts.dir_save+'/training_info/policy_1_{}.npy'.format(epoch),eval_rest["policy_1"].cpu().numpy())
                np.save(opts.dir_save+'/training_info/policy_2_{}.npy'.format(epoch),eval_rest["policy_2"].cpu().numpy())

            # Save accuracy/message results
            np.save(opts.dir_save+'/messages/agent_1_messages_{}.npy'.format(epoch), np_messages_1)
            np.save(opts.dir_save+'/messages/agent_2_messages_{}.npy'.format(epoch), np_messages_2)
            np.save(opts.dir_save+'/accuracy/12_accuracy_{}.npy'.format(epoch), acc_vec_1)
            np.save(opts.dir_save+'/accuracy/21_accuracy_{}.npy'.format(epoch), acc_vec_2)
            np.save(opts.dir_save+'/accuracy/11_accuracy_{}.npy'.format(epoch), acc_vec_11)
            np.save(opts.dir_save+'/accuracy/22_accuracy_{}.npy'.format(epoch), acc_vec_22)
            np.save(opts.dir_save+'/preds/preds_1_{}.npy'.format(epoch), preds_1)
            np.save(opts.dir_save+'/preds/preds_2_{}.npy'.format(epoch), preds_2)
            np.save(opts.dir_save+'/preds/sim_pred_1_{}.npy'.format(epoch), similarity_predictions_1)
            np.save(opts.dir_save+'/preds/sim_pred_2_{}.npy'.format(epoch), similarity_predictions_2)

            grads_listener=[]
            for p in listener_parameters:
              grads_listener.append(p.detach().cpu().numpy())


            grads_speaker=[]
            for p in speaker_parameters:
              grads_speaker.append(p.detach().cpu().numpy())

            if epoch>0:
              diff_grad_sp=[]
              diff_grad_list=[]
              for i in range(len(grads_speaker)):
                diff_grad_sp.append(np.linalg.norm(grads_speaker[i]-ex_grads_speaker[i],2))

              for i in range(len(grads_listener)):
                diff_grad_list.append(np.linalg.norm(grads_listener[i]-ex_grads_listener[i],2))

              print("Grads listener = {}".format(np.mean(diff_grad_list)))
              gradient_listener.append(np.mean(diff_grad_list))
              np.save(opts.dir_save+'/training_info/grads_listener_{}.npy'.format(epoch),gradient_listener)
              print("Grads speaker = {}".format(np.mean(diff_grad_sp)))
              gradient_speaker.append(np.mean(diff_grad_sp))
              np.save(opts.dir_save+'/training_info/grads_speaker_{}.npy'.format(epoch),gradient_speaker)

            ex_grads_speaker=grads_speaker
            ex_grads_listener=grads_listener


    elif opts.model=="expe_step":

        "Define agents"


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


        "Define game"

        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_1":opts.receiver_entropy_coeff,
                      "sender_entropy_coeff_2":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_2":opts.receiver_entropy_coeff}

        if opts.optim_mode=="cross":
            loss_weights={"self":0.,"cross":1.,"imitation":0.}
        elif opts.optim_mode=="cross+self":
            loss_weights={"self":1.,"cross":1.,"imitation":0.}
        else:
            loss_weights={"self":1.,"cross":1.,"imitation":1.}
        #loss_weights={"self":opts.self_weight,"cross":opts.cross_weight,"imitation":opts.imitation_weight}

        game = DialogReinforce(Agent_1=agent_1,
                                Agent_2=agent_2,
                                loss_understanding=loss_understanding,
                                loss_imitation=loss_message_imitation,
                                optim_params=optim_params,
                                baseline_mode=opts.baseline_mode,
                                reward_mode=opts.reward_mode,
                                loss_weights=loss_weights,
                                device=device)

        speaker_parameters = list(game.agent_1.agent_sender.parameters()) + \
                               list(game.agent_1.sender_norm_h.parameters()) + \
                               list(game.agent_1.sender_norm_c.parameters()) + \
                               list(game.agent_1.hidden_to_output.parameters()) + \
                               list(game.agent_1.sender_embedding.parameters()) + \
                               list(game.agent_1.sender_cells.parameters()) + \
                               list(game.agent_2.agent_sender.parameters()) + \
                               list(game.agent_2.sender_norm_h.parameters()) + \
                               list(game.agent_2.sender_norm_c.parameters()) + \
                               list(game.agent_2.hidden_to_output.parameters()) + \
                               list(game.agent_2.sender_embedding.parameters()) + \
                               list(game.agent_2.sender_cells.parameters())

        listener_parameters = list(game.agent_1.agent_receiver.parameters()) + \
                              list(game.agent_1.receiver_cell.parameters()) + \
                              list(game.agent_1.receiver_embedding.parameters()) + \
                              list(game.agent_2.agent_receiver.parameters()) + \
                              list(game.agent_2.receiver_cell.parameters()) + \
                              list(game.agent_2.receiver_embedding.parameters())

        "Create optimizers"
        # ADAM
        optimizer_speaker = core.build_optimizer(list(speaker_parameters),lr=opts.sender_lr)
        optimizer_listener = core.build_optimizer(list(listener_parameters),lr=opts.receiver_lr)

        # SGD
        #optimizer_speaker=torch.optim.SGD(speaker_parameters, lr=opts.sender_lr, momentum=0.9,nesterov=False)
        #optimizer_listener=torch.optim.SGD(listener_parameters, lr=opts.receiver_lr, momentum=0.9,nesterov=False)


        "Create trainer"
        #trainer = TrainerDialog(game=game, optimizer=optimizer, train_data=train_loader, \
        #                        validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])
        #trainer = TrainerDialogAsymLR(game=game, optimizer_speaker=optimizer_speaker,optimizer_listener=optimizer_listener, train_data=train_loader, \
        #                              validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])
        trainer = TrainerDialogAsymStep(game=game, optimizer_speaker=optimizer_speaker,optimizer_listener=optimizer_listener,\
                                        N_speaker=opts.N_speaker,N_listener=opts.N_listener,train_data=train_loader, \
                                        validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])


        "Prepare training"

        # Create save dir
        if not path.exists(opts.dir_save):
            os.system("mkdir {}".format(opts.dir_save))
            os.system("mkdir -p {}/models {}/training_info {}/messages {}/accuracy {}/preds".format(opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save))

        # Main losses
        training_losses=[]
        eval_losses=[]
        training_entropy_1=[]
        training_entropy_2=[]
        training_loss_12=[]
        eval_loss_12=[]
        training_loss_21=[]
        eval_loss_21=[]

        # Specific losses
        training_loss_self_11=[]
        training_loss_cross_12=[]
        training_loss_imitation_12=[]
        training_loss_self_22=[]
        training_loss_cross_21=[]
        training_loss_imitation_21=[]
        eval_loss_self_11=[]
        eval_loss_cross_12=[]
        eval_loss_imitation_12=[]
        eval_loss_self_22=[]
        eval_loss_cross_21=[]
        eval_loss_imitation_21=[]

        # REINFORCE
        eval_reinforce_1=[]
        eval_reinforce_2=[]
        eval_baseline_1=[]
        eval_baseline_2=[]

        # Linguistic
        similarity_languages=[]

        "Prepare test"

        similarity_predictions_1=[]
        similarity_predictions_2=[]
        nb_messages_test=10000
        messages_test = torch.tensor(np.random.randint(opts.vocab_size,size=(nb_messages_test,opts.max_len)))

        gradient_speaker=[]
        gradient_listener=[]

        "Train"

        for epoch in range(int(opts.n_epochs)):

            print("Epoch: {}".format(epoch))

            # Train
            list_train_loss,list_train_rest = trainer.train(n_epochs=1)

            #print(list_train_rest[-1],flush=True)

            # Eval
            eval_loss,eval_rest = trainer.eval()

            # Store results
            training_losses.append(list_train_loss[-1])
            eval_losses.append(eval_loss)
            training_entropy_1.append(list_train_rest[-1]["sender_entropy_1"])
            training_entropy_2.append(list_train_rest[-1]["sender_entropy_2"])
            training_loss_12.append(list_train_rest[-1]["loss_1"])
            eval_loss_12.append(eval_rest["loss_1"])
            training_loss_21.append(list_train_rest[-1]["loss_2"])
            eval_loss_21.append(eval_rest["loss_2"])
            training_loss_self_11.append(list_train_rest[-1]["loss_self_11"])
            training_loss_cross_12.append(list_train_rest[-1]["loss_cross_12"])
            training_loss_imitation_12.append(list_train_rest[-1]["loss_imitation_12"])
            training_loss_self_22.append(list_train_rest[-1]["loss_self_22"])
            training_loss_cross_21.append(list_train_rest[-1]["loss_cross_21"])
            training_loss_imitation_21.append(list_train_rest[-1]["loss_imitation_21"])
            eval_loss_self_11.append(eval_rest["loss_self_11"])
            eval_loss_cross_12.append(eval_rest["loss_cross_12"])
            eval_loss_imitation_12.append(eval_rest["loss_imitation_12"])
            eval_loss_self_22.append(eval_rest["loss_self_22"])
            eval_loss_cross_21.append(eval_rest["loss_cross_21"])
            eval_loss_imitation_21.append(eval_rest["loss_imitation_21"])
            eval_reinforce_1.append(eval_rest["reinforce_term_1"])
            eval_reinforce_2.append(eval_rest["reinforce_term_2"])
            eval_baseline_1.append(eval_rest["baseline_term_1"])
            eval_baseline_2.append(eval_rest["baseline_term_2"])

            if epoch==0:
                messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
            messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22, similarity_messages = dump_dialog_model_6(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
            np_messages_1 = convert_messages_to_numpy(messages_1)
            np_messages_2 = convert_messages_to_numpy(messages_2)
            similarity_languages.append(similarity_messages)

            if epoch==0:
                preds_1=preds_2=np.zeros((np.shape(messages_test)[0]))
            preds_1,preds_2,sim_pred_1,sim_pred_2 = test_receiver_evolution(trainer.game, messages_test, device,False,past_preds_1=preds_1,past_preds_2=preds_2)
            similarity_predictions_1.append(sim_pred_1)
            similarity_predictions_2.append(sim_pred_2)

            #game.optim_params["sender_entropy_coeff_1"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_11)
            #game.optim_params["sender_entropy_coeff_2"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_22)


            # Save models
            if epoch%20==0:
                torch.save(agent_1.state_dict(), f"{opts.dir_save}/models/agent_1_weights_{epoch}.pth")
                torch.save(agent_2.state_dict(), f"{opts.dir_save}/models/agent_2_weights_{epoch}.pth")

            # Save training info
            if epoch%10==0:
                np.save(opts.dir_save+'/training_info/training_loss_{}.npy'.format(epoch), training_losses)
                np.save(opts.dir_save+'/training_info/eval_loss_{}.npy'.format(epoch), eval_losses)
                np.save(opts.dir_save+'/training_info/training_entropy_1_{}.npy'.format(epoch), training_entropy_1)
                np.save(opts.dir_save+'/training_info/training_entropy_2_{}.npy'.format(epoch), training_entropy_2)
                np.save(opts.dir_save+'/training_info/training_loss_12_{}.npy'.format(epoch), training_loss_12)
                np.save(opts.dir_save+'/training_info/eval_loss_12_{}.npy'.format(epoch), eval_loss_12)
                np.save(opts.dir_save+'/training_info/training_loss_21_{}.npy'.format(epoch), training_loss_21)
                np.save(opts.dir_save+'/training_info/eval_loss_21_{}.npy'.format(epoch), eval_loss_21)
                np.save(opts.dir_save+'/training_info/training_loss_self_11_{}.npy'.format(epoch), training_loss_self_11)
                np.save(opts.dir_save+'/training_info/training_loss_cross_12_{}.npy'.format(epoch), training_loss_cross_12)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_12_{}.npy'.format(epoch), training_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/training_loss_self_22_{}.npy'.format(epoch), training_loss_self_22)
                np.save(opts.dir_save+'/training_info/training_loss_cross_21_{}.npy'.format(epoch), training_loss_cross_21)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_21_{}.npy'.format(epoch), training_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/eval_loss_self_11_{}.npy'.format(epoch), eval_loss_self_11)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_12_{}.npy'.format(epoch), eval_loss_cross_12)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_12_{}.npy'.format(epoch), eval_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/eval_loss_self_22_{}.npy'.format(epoch), eval_loss_self_22)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_21_{}.npy'.format(epoch), eval_loss_cross_21)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_21_{}.npy'.format(epoch), eval_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/similarity_languages_{}.npy'.format(epoch), similarity_languages)
                #Reinforce
                np.save(opts.dir_save+'/training_info/reinforce_term_1_{}.npy'.format(epoch), eval_reinforce_1)
                np.save(opts.dir_save+'/training_info/reinforce_term_2_{}.npy'.format(epoch), eval_reinforce_2)
                np.save(opts.dir_save+'/training_info/baseline_term_1_{}.npy'.format(epoch), eval_baseline_1)
                np.save(opts.dir_save+'/training_info/baseline_term_2_{}.npy'.format(epoch), eval_baseline_2)
                # Policy
                np.save(opts.dir_save+'/training_info/policy_1_{}.npy'.format(epoch),eval_rest["policy_1"].cpu().numpy())
                np.save(opts.dir_save+'/training_info/policy_2_{}.npy'.format(epoch),eval_rest["policy_2"].cpu().numpy())

            # Save accuracy/message results
            np.save(opts.dir_save+'/messages/agent_1_messages_{}.npy'.format(epoch), np_messages_1)
            np.save(opts.dir_save+'/messages/agent_2_messages_{}.npy'.format(epoch), np_messages_2)
            np.save(opts.dir_save+'/accuracy/12_accuracy_{}.npy'.format(epoch), acc_vec_1)
            np.save(opts.dir_save+'/accuracy/21_accuracy_{}.npy'.format(epoch), acc_vec_2)
            np.save(opts.dir_save+'/accuracy/11_accuracy_{}.npy'.format(epoch), acc_vec_11)
            np.save(opts.dir_save+'/accuracy/22_accuracy_{}.npy'.format(epoch), acc_vec_22)
            np.save(opts.dir_save+'/preds/preds_1_{}.npy'.format(epoch), preds_1)
            np.save(opts.dir_save+'/preds/preds_2_{}.npy'.format(epoch), preds_2)
            np.save(opts.dir_save+'/preds/sim_pred_1_{}.npy'.format(epoch), similarity_predictions_1)
            np.save(opts.dir_save+'/preds/sim_pred_2_{}.npy'.format(epoch), similarity_predictions_2)

            grads_listener=[]
            for p in listener_parameters:
              grads_listener.append(p.detach().cpu().numpy())


            grads_speaker=[]
            for p in speaker_parameters:
              grads_speaker.append(p.detach().cpu().numpy())

            if epoch>0:
              diff_grad_sp=[]
              diff_grad_list=[]
              for i in range(len(grads_speaker)):
                diff_grad_sp.append(np.linalg.norm(grads_speaker[i]-ex_grads_speaker[i],2))

              for i in range(len(grads_listener)):
                diff_grad_list.append(np.linalg.norm(grads_listener[i]-ex_grads_listener[i],2))

              print("Grads listener = {}".format(np.mean(diff_grad_list)))
              gradient_listener.append(np.mean(diff_grad_list))
              np.save(opts.dir_save+'/training_info/grads_listener_{}.npy'.format(epoch),gradient_listener)
              print("Grads speaker = {}".format(np.mean(diff_grad_sp)))
              gradient_speaker.append(np.mean(diff_grad_sp))
              np.save(opts.dir_save+'/training_info/grads_speaker_{}.npy'.format(epoch),gradient_speaker)

            ex_grads_speaker=grads_speaker
            ex_grads_listener=grads_listener

    elif opts.model=="expe_KL":

        "Define agents"


        agent_1=AgentBaselineKL(vocab_size=opts.vocab_size,
                        n_features=opts.n_features,
                        max_len=opts.max_len,
                        embed_dim=opts.sender_embedding,
                        hidden_size=opts.sender_hidden,
                        sender_cell=opts.sender_cell,
                        receiver_cell=opts.receiver_cell,
                        sender_num_layers=opts.sender_num_layers,
                        receiver_num_layers=opts.receiver_num_layers,
                        force_eos=force_eos)

        agent_2=AgentBaselineKL(vocab_size=opts.vocab_size,
                        n_features=opts.n_features,
                        max_len=opts.max_len,
                        embed_dim=opts.sender_embedding,
                        hidden_size=opts.sender_hidden,
                        sender_cell=opts.sender_cell,
                        receiver_cell=opts.receiver_cell,
                        sender_num_layers=opts.sender_num_layers,
                        receiver_num_layers=opts.receiver_num_layers,
                        force_eos=force_eos)


        "Define game"

        optim_params={"length_cost":0.,
                      "sender_entropy_coeff_1":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_1":opts.receiver_entropy_coeff,
                      "sender_entropy_coeff_2":opts.sender_entropy_coeff,
                      "receiver_entropy_coeff_2":opts.receiver_entropy_coeff}

        if opts.optim_mode=="cross":
            loss_weights={"self":0.,"cross":1.,"imitation":0.}
        elif opts.optim_mode=="cross+self":
            loss_weights={"self":1.,"cross":1.,"imitation":0.}
        else:
            loss_weights={"self":1.,"cross":1.,"imitation":1.}
        #loss_weights={"self":opts.self_weight,"cross":opts.cross_weight,"imitation":opts.imitation_weight}

        game = DialogReinforceKL(Agent_1=agent_1,
                                Agent_2=agent_2,
                                loss_understanding=loss_understanding,
                                loss_imitation=loss_message_imitation,
                                optim_params=optim_params,
                                loss_weights=loss_weights,
                                device=device)

        "Create optimizers"
        receiver_1_parameters = list(game.agent_1.agent_receiver.parameters()) + \
                              list(game.agent_1.receiver_cells.parameters()) + \
                              list(game.agent_1.receiver_norm_h.parameters()) + \
                              list(game.agent_1.receiver_norm_c.parameters()) + \
                              list(game.agent_1.hidden_to_output.parameters()) + \
                              list(game.agent_1.receiver_embedding.parameters())

        receiver_2_parameters = list(game.agent_2.agent_receiver.parameters()) + \
                              list(game.agent_2.receiver_cells.parameters()) + \
                              list(game.agent_2.receiver_norm_h.parameters()) + \
                              list(game.agent_2.receiver_norm_c.parameters()) + \
                              list(game.agent_2.hidden_to_output.parameters()) + \
                              list(game.agent_2.receiver_embedding.parameters())

        optimizer_agent_1 = core.build_optimizer(list(game.agent_1.parameters())+receiver_2_parameters)
        optimizer_agent_2 = core.build_optimizer(list(game.agent_2.parameters())+receiver_1_parameters)
        #optimizer = core.build_optimizer(list(game.parameters()))



        "Create trainer"
        #trainer = TrainerDialog(game=game, optimizer=optimizer, train_data=train_loader, \
        #                        validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])
        trainer = TrainerDialog(game=game, optimizer_agent_1=optimizer_agent_1,optimizer_agent_2=optimizer_agent_2, train_data=train_loader, \
                                validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

        "Prepare training"

        # Create save dir
        if not path.exists(opts.dir_save):
            os.system("mkdir {}".format(opts.dir_save))
            os.system("mkdir -p {}/models {}/training_info {}/messages {}/accuracy".format(opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save))

        # Main losses
        training_losses=[]
        eval_losses=[]
        training_entropy_1=[]
        training_entropy_2=[]
        training_loss_12=[]
        eval_loss_12=[]
        training_loss_21=[]
        eval_loss_21=[]

        # Specific losses
        training_loss_self_11=[]
        training_loss_cross_12=[]
        training_loss_imitation_12=[]
        training_loss_self_22=[]
        training_loss_cross_21=[]
        training_loss_imitation_21=[]
        eval_loss_self_11=[]
        eval_loss_cross_12=[]
        eval_loss_imitation_12=[]
        eval_loss_self_22=[]
        eval_loss_cross_21=[]
        eval_loss_imitation_21=[]

        # Linguistic
        similarity_languages=[]

        "Train"

        for epoch in range(int(opts.n_epochs)):

            print("Epoch: {}".format(epoch))

            # Train
            list_train_loss,list_train_rest = trainer.train(n_epochs=1)

            print(list_train_rest[-1],flush=True)

            # Eval
            eval_loss,eval_rest = trainer.eval()

            # Store results
            training_losses.append(list_train_loss[-1])
            eval_losses.append(eval_loss)
            training_entropy_1.append(list_train_rest[-1]["sender_entropy_1"])
            training_entropy_2.append(list_train_rest[-1]["sender_entropy_2"])
            training_loss_12.append(list_train_rest[-1]["loss_1"])
            eval_loss_12.append(eval_rest["loss_1"])
            training_loss_21.append(list_train_rest[-1]["loss_2"])
            eval_loss_21.append(eval_rest["loss_2"])
            training_loss_self_11.append(list_train_rest[-1]["loss_self_11"])
            training_loss_cross_12.append(list_train_rest[-1]["loss_cross_12"])
            training_loss_imitation_12.append(list_train_rest[-1]["loss_imitation_12"])
            training_loss_self_22.append(list_train_rest[-1]["loss_self_22"])
            training_loss_cross_21.append(list_train_rest[-1]["loss_cross_21"])
            training_loss_imitation_21.append(list_train_rest[-1]["loss_imitation_21"])
            eval_loss_self_11.append(eval_rest["loss_self_11"])
            eval_loss_cross_12.append(eval_rest["loss_cross_12"])
            eval_loss_imitation_12.append(eval_rest["loss_imitation_12"])
            eval_loss_self_22.append(eval_rest["loss_self_22"])
            eval_loss_cross_21.append(eval_rest["loss_cross_21"])
            eval_loss_imitation_21.append(eval_rest["loss_imitation_21"])

            if epoch==0:
                messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
            messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22, similarity_messages = dump_dialog_model_6(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
            np_messages_1 = convert_messages_to_numpy(messages_1)
            np_messages_2 = convert_messages_to_numpy(messages_2)
            similarity_languages.append(similarity_messages)

            #game.optim_params["sender_entropy_coeff_1"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_11)
            #game.optim_params["sender_entropy_coeff_2"]=opts.sender_entropy_coeff-(opts.sender_entropy_coeff+0.05)*np.mean(acc_vec_22)


            # Save models
            if epoch%20==0:
                torch.save(agent_1.state_dict(), f"{opts.dir_save}/models/agent_1_weights_{epoch}.pth")
                torch.save(agent_2.state_dict(), f"{opts.dir_save}/models/agent_2_weights_{epoch}.pth")

            # Save training info
            if epoch%10==0:
                np.save(opts.dir_save+'/training_info/training_loss_{}.npy'.format(epoch), training_losses)
                np.save(opts.dir_save+'/training_info/eval_loss_{}.npy'.format(epoch), eval_losses)
                np.save(opts.dir_save+'/training_info/training_entropy_1_{}.npy'.format(epoch), training_entropy_1)
                np.save(opts.dir_save+'/training_info/training_entropy_2_{}.npy'.format(epoch), training_entropy_2)
                np.save(opts.dir_save+'/training_info/training_loss_12_{}.npy'.format(epoch), training_loss_12)
                np.save(opts.dir_save+'/training_info/eval_loss_12_{}.npy'.format(epoch), eval_loss_12)
                np.save(opts.dir_save+'/training_info/training_loss_21_{}.npy'.format(epoch), training_loss_21)
                np.save(opts.dir_save+'/training_info/eval_loss_21_{}.npy'.format(epoch), eval_loss_21)
                np.save(opts.dir_save+'/training_info/training_loss_self_11_{}.npy'.format(epoch), training_loss_self_11)
                np.save(opts.dir_save+'/training_info/training_loss_cross_12_{}.npy'.format(epoch), training_loss_cross_12)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_12_{}.npy'.format(epoch), training_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/training_loss_self_22_{}.npy'.format(epoch), training_loss_self_22)
                np.save(opts.dir_save+'/training_info/training_loss_cross_21_{}.npy'.format(epoch), training_loss_cross_21)
                np.save(opts.dir_save+'/training_info/training_loss_imitation_21_{}.npy'.format(epoch), training_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/eval_loss_self_11_{}.npy'.format(epoch), eval_loss_self_11)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_12_{}.npy'.format(epoch), eval_loss_cross_12)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_12_{}.npy'.format(epoch), eval_loss_imitation_12)
                np.save(opts.dir_save+'/training_info/eval_loss_self_22_{}.npy'.format(epoch), eval_loss_self_22)
                np.save(opts.dir_save+'/training_info/eval_loss_cross_21_{}.npy'.format(epoch), eval_loss_cross_21)
                np.save(opts.dir_save+'/training_info/eval_loss_imitation_21_{}.npy'.format(epoch), eval_loss_imitation_21)
                np.save(opts.dir_save+'/training_info/similarity_languages_{}.npy'.format(epoch), similarity_languages)

            # Save accuracy/message results
            np.save(opts.dir_save+'/messages/agent_1_messages_{}.npy'.format(epoch), np_messages_1)
            np.save(opts.dir_save+'/messages/agent_2_messages_{}.npy'.format(epoch), np_messages_2)
            np.save(opts.dir_save+'/accuracy/12_accuracy_{}.npy'.format(epoch), acc_vec_1)
            np.save(opts.dir_save+'/accuracy/21_accuracy_{}.npy'.format(epoch), acc_vec_2)
            np.save(opts.dir_save+'/accuracy/11_accuracy_{}.npy'.format(epoch), acc_vec_11)
            np.save(opts.dir_save+'/accuracy/22_accuracy_{}.npy'.format(epoch), acc_vec_22)


    else:

        for epoch in range(int(opts.n_epochs)):
            trainer.train(n_epochs=1)

            if opts.checkpoint_dir:
                trainer.save_checkpoint(name=f'{opts.name}_vocab{opts.vocab_size}_rs{opts.random_seed}_lr{opts.lr}_shid{opts.sender_hidden}_rhid{opts.receiver_hidden}_sentr{opts.sender_entropy_coeff}_reg{opts.length_cost}_max_len{opts.max_len}')

            if not opts.dialog:
                if not opts.impatient:
                    acc_vec_1,messages_1=dump(trainer.game, opts.n_features, device, False,epoch)
                else:
                    acc_vec_1,messages_1=dump_impatient(trainer.game, opts.n_features, device, False,epoch)

            else:
                if opts.model=="baseline":
                    if epoch==0:
                        messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
                    messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22 = dump_dialog_model_1(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
                elif opts.model=="model_1":
                    if epoch==0:
                        messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
                    messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22 = dump_dialog_model_1(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
                elif opts.model=="model_2":
                    if epoch==0:
                        messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
                    acc_vec_1, messages_1, acc_vec_2, messages_2 = dump_dialog_model_2(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
                elif opts.model=="model_3":
                    if epoch==0:
                        messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
                    acc_vec_1, messages_1, acc_vec_2, messages_2 = dump_dialog(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
                elif opts.model=="model_4":
                    if epoch==0:
                        messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
                    messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22 = dump_dialog_model_1(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
                elif opts.model=="model_5":
                    if epoch==0:
                        messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
                    messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22 = dump_dialog_model_1(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
                elif opts.model=="shared_LSTM" or opts.model=="shared_embedding":
                    if epoch==0:
                        messages_1=messages_2=np.zeros((opts.n_features,opts.max_len))
                    messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22 = dump_dialog_model_6(trainer.game, opts.n_features, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
                elif opts.model=="pretraining":
                    acc_vec_1, messages_1 = dump_pretraining(trainer.game, opts.n_features,pretrained_messages, device, False,epoch)


            if opts.dialog and opts.model!="pretraining":

                if opts.entropy_scheduling:
                    if opts.model!="baseline":
                        game.sender_entropy_coeff_1=0.5*(1-np.mean(acc_vec_11)**10)
                        game.sender_entropy_coeff_2=0.5*(1-np.mean(acc_vec_22)**10)
                    else:
                        game.sender_entropy_coeff_1=0.5*(1-np.mean(acc_vec_1)**10)
                        game.sender_entropy_coeff_2=0.5*(1-np.mean(acc_vec_2)**10)

                # Convert to numpy to save messages
                all_messages_1=[]
                for x in messages_1:
                    x = x.cpu().numpy()
                    all_messages_1.append(x)
                all_messages_1 = np.asarray(all_messages_1)

                all_messages_2=[]
                for x in messages_2:
                    x = x.cpu().numpy()
                    all_messages_2.append(x)
                all_messages_2 = np.asarray(all_messages_2)

                if epoch%50==0:
                    if opts.model!="model_6" and opts.model!="shared_embedding":
                        torch.save(agent_1.sender.state_dict(), f"{opts.dir_save}/sender/agent_1_sender_weights_{epoch}.pth")
                        torch.save(agent_1.receiver.state_dict(), f"{opts.dir_save}/receiver/agent_1_receiver_weights_{epoch}.pth")
                        torch.save(agent_2.sender.state_dict(), f"{opts.dir_save}/sender/agent_2_sender_weights_{epoch}.pth")
                        torch.save(agent_2.receiver.state_dict(), f"{opts.dir_save}/receiver/agent_2_receiver_weights_{epoch}.pth")
                    else:
                        torch.save(agent_1.state_dict(), f"{opts.dir_save}/sender/agent_1_weights_{epoch}.pth")
                        torch.save(agent_2.state_dict(), f"{opts.dir_save}/sender/agent_2_weights_{epoch}.pth")

                np.save(opts.dir_save+'/messages/agent_1_messages_{}.npy'.format(epoch), all_messages_1)
                np.save(opts.dir_save+'/accuracy/agent_1_accuracy_{}.npy'.format(epoch), acc_vec_1)
                np.save(opts.dir_save+'/messages/agent_2_messages_{}.npy'.format(epoch), all_messages_2)
                np.save(opts.dir_save+'/accuracy/agent_2_accuracy_{}.npy'.format(epoch), acc_vec_2)
                np.save(opts.dir_save+'/accuracy/agent_1_1_accuracy_{}.npy'.format(epoch), acc_vec_11)
                np.save(opts.dir_save+'/accuracy/agent_2_2_accuracy_{}.npy'.format(epoch), acc_vec_22)

            else:

                if opts.entropy_scheduling:
                    game.sender_entropy_coeff_1=0.5*(1-np.mean(acc_vec_1))

                # Convert to numpy to save messages
                all_messages_1=[]
                for x in messages_1:
                    x = x.cpu().numpy()
                    all_messages_1.append(x)
                all_messages_1 = np.asarray(all_messages_1)

                if epoch%10==0:
                    torch.save(agent_1.sender.state_dict(), f"{opts.dir_save}/sender/agent_1_sender_weights_{epoch}.pth")
                    torch.save(agent_1.receiver.state_dict(), f"{opts.dir_save}/receiver/agent_1_receiver_weights_{epoch}.pth")

                np.save(opts.dir_save+'/messages/agent_1_messages_{}.npy'.format(epoch), all_messages_1)
                np.save(opts.dir_save+'/accuracy/agent_1_accuracy_{}.npy'.format(epoch), acc_vec_1)


    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
