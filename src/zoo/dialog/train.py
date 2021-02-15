# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from src.core.util import dump_sender_receiver_impatient,levenshtein
#Dialog
from src.core.reinforce_wrappers import RnnReceiverWithHiddenStates,RnnSenderReinforceModel3
from src.core.reinforce_wrappers import  AgentBaseline,AgentModel2,AgentModel3,AgentSharedEmbedding
from src.core.reinforce_wrappers import DialogReinforceBaseline,DialogReinforceModel1,DialogReinforceModel2, DialogReinforceModel3,DialogReinforceModel4,PretrainAgent
from src.core.util import dump_sender_receiver_dialog,dump_sender_receiver_dialog_model_1,dump_sender_receiver_dialog_model_2,dump_pretraining
from src.core.trainers import TrainerDialog, TrainerDialogModel1, TrainerDialogModel2, TrainerDialogModel3,TrainerDialogModel4,TrainerDialogModel5,TrainerPretraining


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


def loss_model_3(sender_input, message, receiver_input, receiver_output,message_reconstruction,prob_reconstruction, labels):

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

    # Reconstruction task
    prob_reconstruction = prob_reconstruction.transpose(1,2)
    prob_reconstruction = prob_reconstruction.reshape((prob_reconstruction.size(0)*prob_reconstruction.size(1),prob_reconstruction.size(2)))
    message = message.reshape((message.size(0)*message.size(1)))

    acc_imitation = (prob_reconstruction.argmax(dim=1) == message).detach().float()
    loss_imitation = F.cross_entropy(torch.log(prob_reconstruction), message, reduction="none")

    loss_imitation = loss_imitation.reshape((loss.size(0),loss_imitation.size(0)//loss.size(0)))
    acc_imitation = acc_imitation.reshape((acc.size(0),acc_imitation.size(0)//acc.size(0)))

    loss_imitation = loss_imitation.mean(dim=1) # Add EOS mask
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

    # Reconstruction task
    prob_reconstruction = prob_reconstruction.transpose(1,2)
    prob_reconstruction = prob_reconstruction.reshape((prob_reconstruction.size(0)*prob_reconstruction.size(1),prob_reconstruction.size(2)))
    pretrained_messages = pretrained_messages.reshape((pretrained_messages.size(0)*pretrained_messages.size(1)))

    acc_imitation = (prob_reconstruction.argmax(dim=1) == pretrained_messages).detach().float()
    loss_imitation = F.cross_entropy(torch.log(prob_reconstruction), pretrained_messages, reduction="none")

    loss_imitation = loss_imitation.reshape((loss.size(0),loss_imitation.size(0)//loss.size(0)))
    acc_imitation = acc_imitation.reshape((acc.size(0),acc_imitation.size(0)//acc.size(0)))

    loss_imitation = loss_imitation.mean(dim=1) # Add EOS mask
    acc_imitation = acc_imitation.mean(dim=1)

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

def dump_dialog(game, n_features, device, gs_mode, epoch):
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

def dump_dialog_model_1(game, n_features, device, gs_mode, epoch):
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

    print("Similarity between language = {}".format(np.mean([levenshtein(messages_1[i],messages_2[i]) for i in range(len(messages_1))])),flush=True)

    return acc_vec_1, messages_1, acc_vec_2, messages_2

def dump_dialog_model_2(game, n_features, device, gs_mode, epoch):
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

def dump_pretraining(game, n_features, device, gs_mode, epoch):
    # tiny "dataset"
    dataset = [[torch.eye(n_features).to(device), None]]

    sender_inputs_1, messages_1, receiver_inputs_1, receiver_outputs_1, _ = \
        dump_pretraining(game, dataset, gs=gs_mode, device=device, variable_length=True)


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
                                           sender_entropy_coeff=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff=opts.receiver_entropy_coeff,
                                           loss_weights=[0.5,0.5],
                                           length_cost=0.0,
                                           unigram_penalty=0.0,
                                           reg=False,
                                           device=device)

            optimizer_1 = core.build_optimizer(list(game.agent_1.sender.parameters())+list(game.agent_2.receiver.parameters()))
            optimizer_2 = core.build_optimizer(list(game.agent_2.sender.parameters())+list(game.agent_1.receiver.parameters()))


            trainer = TrainerDialog(game=game, optimizer_1=optimizer_1, optimizer_2=optimizer_2, train_data=train_loader,
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
                                           sender_entropy_coeff=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff=opts.receiver_entropy_coeff,
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
                                           sender_entropy_coeff=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff=opts.receiver_entropy_coeff,
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
                                           sender_entropy_coeff=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff=opts.receiver_entropy_coeff,
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

            game = DialogReinforceModel4(Agent_1=agent_1,
                                           Agent_2=agent_2,
                                           loss=loss_model_3,
                                           sender_entropy_coeff=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff=opts.receiver_entropy_coeff,
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
                                           sender_entropy_coeff=opts.sender_entropy_coeff,
                                           receiver_entropy_coeff=opts.receiver_entropy_coeff,
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
            pretrained_messsages=[[i,0] for i in range(opts.n_features)]

            game = PretrainAgent(Agent_1=agent_1,
                               loss=loss_pretraining,
                               pretrained_messages=pretrained_messages,
                               sender_entropy_coeff=opts.sender_entropy_coeff,
                               receiver_entropy_coeff=opts.receiver_entropy_coeff,
                               length_cost=0.0,
                               unigram_penalty=0.0,
                               reg=False,
                               device=device)

            optimizer_sender_1 = core.build_optimizer(list(game.agent_1.sender.parameters()))
            optimizer_receiver_1 = core.build_optimizer(list(game.agent_1.receiver.parameters()))

            trainer = TrainerPretraining(game=game, optimizer_sender_1=optimizer_sender_1,
                                          optimizer_receiver_1=optimizer_receiver_1, train_data=train_loader, \
                                          validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])





    for epoch in range(int(opts.n_epochs)):

        print("Epoch: {}".format(epoch))

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
                acc_vec_1, messages_1, acc_vec_2, messages_2 = dump_dialog(trainer.game, opts.n_features, device, False,epoch)
            elif opts.model=="model_1":
                acc_vec_1, messages_1, acc_vec_2, messages_2 = dump_dialog_model_1(trainer.game, opts.n_features, device, False,epoch)
            elif opts.model=="model_2":
                acc_vec_1, messages_1, acc_vec_2, messages_2 = dump_dialog_model_2(trainer.game, opts.n_features, device, False,epoch)
            elif opts.model=="model_3":
                acc_vec_1, messages_1, acc_vec_2, messages_2 = dump_dialog(trainer.game, opts.n_features, device, False,epoch)
            elif opts.model=="model_4":
                acc_vec_1, messages_1, acc_vec_2, messages_2 = dump_dialog_model_1(trainer.game, opts.n_features, device, False,epoch)
            elif opts.model=="model_5":
                acc_vec_1, messages_1, acc_vec_2, messages_2 = dump_dialog_model_1(trainer.game, opts.n_features, device, False,epoch)
            elif opts.model=="pretraining":
                acc_vec_1, messages_1 = dump_pretraining(trainer.game, opts.n_features, device, False,epoch)


        if opts.dialog and opts.model!="pretraining":
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
                torch.save(agent_1.sender.state_dict(), f"{opts.dir_save}/sender/agent_1_sender_weights_{epoch}.pth")
                torch.save(agent_1.receiver.state_dict(), f"{opts.dir_save}/receiver/agent_1_receiver_weights_{epoch}.pth")
                torch.save(agent_2.sender.state_dict(), f"{opts.dir_save}/sender/agent_2_sender_weights_{epoch}.pth")
                torch.save(agent_2.receiver.state_dict(), f"{opts.dir_save}/receiver/agent_2_receiver_weights_{epoch}.pth")

            np.save(opts.dir_save+'/messages/agent_1_messages_{}.npy'.format(epoch), all_messages_1)
            np.save(opts.dir_save+'/accuracy/agent_1_accuracy_{}.npy'.format(epoch), acc_vec_1)
            np.save(opts.dir_save+'/messages/agent_2_messages_{}.npy'.format(epoch), all_messages_2)
            np.save(opts.dir_save+'/accuracy/agent_2_accuracy_{}.npy'.format(epoch), acc_vec_2)

        else:
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
