# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import numpy as np
import os
from os import path
import itertools
import pickle
import torch.utils.data
import torch.nn.functional as F
from scipy.stats import entropy
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
from src.core.reinforce_wrappers import DialogReinforceCompositionality, AgentBaselineCompositionality, DialogReinforceCompositionalitySingleListener
from src.core.trainers import CompoTrainer,TrainerDialogCompositionality,TrainerDialogAsymLR,TrainerDialogAsymStep

# MultiAgents
from src.core.reinforce_wrappers import DialogReinforceCompositionalityMultiAgent
from src.core.trainers import TrainerDialogMultiAgent
from src.core.util import dump_multiagent_compositionality,sample_messages


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
    parser.add_argument('--model', type=str, default="expe_1",help='Choice of expe')

    # Split
    parser.add_argument('--split_proportion', type=float, default=0.8,help='Train/test split prop')

    # Compo
    parser.add_argument('--sender_lr', type=float, default=0.001,help='Lr for senders (for asymmetric expe)')
    parser.add_argument('--receiver_lr', type=float, default=0.01,help='Lr for receivers (for asymmetric expe)')

    # Asym learning
    parser.add_argument('--step_ratio', type=float, default=1,help='N_step_speaker/N_step_listener')

    # Estimation
    parser.add_argument('--agents_weights', type=str,help='Path to agent weights')
    parser.add_argument('--dataset_split', type=str,help='Path to dataset split')
    parser.add_argument('--n_sampling', type=int,help='Number of sampling iteration for estimation')
    parser.add_argument('--by_position', type=bool,help='Measure entropy by position')

    args = core.init(parser, params)

    return args

def estimate_policy(agents,
                   compo_dataset,
                   split,
                   n_sampling,
                   vocab_size,
                   max_len,
                   n_attributes,
                   n_values,
                   device,
                   by_position=False):

    """
    Estimate agent (speaker module) policies based on message samples
    """

    dataset=[]
    combination=[]

    for i in range(len(compo_dataset)):
        if i in split:
          dataset.append(torch.from_numpy(compo_dataset[i]).float())
          combination.append(np.reshape(compo_dataset[i],(n_attributes,n_values)).argmax(1))

    dataset = [[torch.stack(dataset).to(device), None]]

    policies={}

    if by_position:

        for agent in agents:
            policies[agent]=np.zeros((len(split),max_len,vocab_size))
        policies["mean_policy"]=np.zeros((len(split),max_len,vocab_size))

        for _ in range(n_sampling):
            idx=np.random.choice(len(agents))
            agent=agents["agent_{}".format(idx)]
            messages = sample_messages(agent,dataset,device)

            for i,message in enumerate(messages):
                for j,symbol in enumerate(message):
                    policies["agent_{}".format(idx)][i,j,symbol]+=1/n_sampling
                    policies["mean_policy"][i,j,symbol]+=1/n_sampling


    else:

        for agent in agents:
            policies[agent]=[{} for i in range(len(split))]
        policies["mean_policy"]=[{} for i in range(len(split))]

        for _ in range(n_sampling):
            idx=np.random.choice(len(agents))
            agent=agents["agent_{}".format(idx)]
            messages = sample_messages(agent,dataset,device)

            for i,message in enumerate(messages):
                message="".join([str(s.cpu().numpy()) for s in message])
                # Individual policy
                if message in policies["agent_{}".format(idx)][i]:
                    policies["agent_{}".format(idx)][i][message]+=1/n_sampling
                else:
                    policies["agent_{}".format(idx)][i][message]=1/n_sampling

                # Mean policy
                if message in policies["mean_policy"][i]:
                    policies["mean_policy"][i][message]+=1/n_sampling
                else:
                    policies["mean_policy"][i][message]=1/n_sampling

    return policies

def fill_to_max_len(messages,max_len):
    lengths=[len(m) for m in messages]
    new_messages=[np.concatenate((m,[-1]*(max_len-lengths[i]))) for i,m in enumerate(messages)]
    return new_messages

def loss_understanding_compositionality(sender_input, receiver_output,n_attributes,n_values):

    loss=0.

    sender_input=sender_input.reshape(sender_input.size(0),n_attributes,n_values)

    crible_acc=(receiver_output.argmax(dim=2)==sender_input.argmax(2)).detach().float()

    for j in range(receiver_output.size(1)):
        loss+=F.cross_entropy(receiver_output[:,j,:], sender_input[:,j,:].argmax(dim=1), reduction="none")

    return loss, {'acc': crible_acc}

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

def dump_compositionality_multiagent(game,compo_dataset,split,list_speakers,list_listeners,n_attributes,n_values,device, epoch,past_messages=None,compute_similarity=False):

    dataset=[]
    combination=[]

    for i in range(len(compo_dataset)):
        if i in split:
          dataset.append(torch.from_numpy(compo_dataset[i]).float())
          combination.append(np.reshape(compo_dataset[i],(n_attributes,n_values)).argmax(1))

    dataset = [[torch.stack(dataset).to(device), None]]

    sender_inputs, messages, _ , receiver_outputs,labels = \
        dump_multiagent_compositionality(game, dataset,list_speakers,list_listeners, device=device, variable_length=True)
    # Rq. sender_inputs = list, messages = dict , receiver_outputs = dict

    n_messages = len(dataset[0][0])
    n_agents = len(messages)

    "1. Accuracy"

    accuracy_vectors={}
    accs_tot=[]

    for agent_speaker in messages:

        accuracy_vectors[agent_speaker]={}

        for agent_listener in receiver_outputs[agent_speaker]:

            unif_acc = 0.
            unif_acc_general=0.
            acc_vec=np.zeros(((n_messages), n_attributes))

            for i in range(len(receiver_outputs[agent_speaker][agent_listener])):
              message=messages[agent_speaker][i]
              correct=True
              if i<n_messages:
                  for j in range(len(list(combination[i]))):
                    if receiver_outputs[agent_speaker][agent_listener][i][j]==list(combination[i])[j]:
                      unif_acc+=1
                      acc_vec[i,j]=1
                    else:
                      correct=False
                  if correct:
                    unif_acc_general+=1.

            accuracy_vectors[agent_speaker][agent_listener] = acc_vec
            unif_acc /= (n_messages) * n_attributes
            unif_acc_general/=n_messages
            accs_tot.append(unif_acc)

        #print(agent)
    print(json.dumps({'unif accuracy': np.mean(accs_tot)}))

    if compute_similarity:
        "2. Similarity messages"
        similarity_messages = np.zeros((n_agents,n_agents))
        mean_similarity=[]
        for j,agent_1 in enumerate(messages):
            for k,agent_2 in enumerate(messages):
                similarity_val=np.mean([levenshtein(messages[agent_1][i],messages[agent_2][i])/np.max([len(messages[agent_1][i]),len(messages[agent_2][i])]) for i in range(len(messages[agent_1]))])
                similarity_messages[j,k]=similarity_val
                if j!=k:
                  mean_similarity.append(similarity_val)


        print("Similarity between language = {}".format(np.mean(mean_similarity)),flush=True)
    else:
        similarity_messages = []

    #if past_messages is not None:
    #    print("Similarity evo language 1 = {}".format(np.mean([levenshtein(messages_1[i],past_messages_1[i]) for i in range(len(messages_1))])),flush=True)

    return messages,accuracy_vectors, similarity_messages

def main(params):
    print(torch.cuda.is_available())
    opts = get_params(params)
    print(opts, flush=True)
    device = opts.device

    force_eos = opts.force_eos == 1


    compo_dataset = build_compo_dataset(opts.n_values, opts.n_attributes)

    split = np.sort(np.load(opts.dataset_split))

    with open(opts.agents_weights, "rb") as fp:
        agents_weights = pickle.load(fp)


    agents={}

    for i in range(len(agents_weights)):


        agent=AgentBaselineCompositionality(vocab_size=opts.vocab_size,
                                                n_attributes=opts.n_attributes,
                                                n_values=opts.n_values,
                                                max_len=opts.max_len,
                                                embed_dim=opts.sender_embedding,
                                                sender_hidden_size=opts.sender_hidden,
                                                receiver_hidden_size=opts.receiver_hidden,
                                                sender_cell=opts.sender_cell,
                                                receiver_cell=opts.receiver_cell,
                                                sender_num_layers=opts.sender_num_layers,
                                                receiver_num_layers=opts.receiver_num_layers,
                                                force_eos=force_eos)

        agent.load_state_dict(torch.load(agents_weights[i],map_location=torch.device('cpu')))
        agent.to(device)
        agents["agent_{}".format(i)] = agent

        #(agent,compo_dataset,split,n_sampling,vocab_size,max_len,device)

    if opts.by_position:

        policies = estimate_policy(agents=agents,
                                   compo_dataset=compo_dataset,
                                   split=split,
                                   n_sampling=opts.n_sampling,
                                   vocab_size=opts.vocab_size,
                                   max_len=opts.max_len,
                                   n_attributes=opts.n_attributes,
                                   n_values=opts.n_values,
                                   device=device,
                                   by_position=True)

        for agent in policies:
            mean_entropy=0.
            for i in range(np.shape(policies[agent])[0]):
                for j in range(np.shape(policies[agent])[1]):
                  probs=[policies[agent][i,j,k] for k in range(np.shape(policies[agent])[2])]
                  mean_entropy+=entropy(probs,base=10)
            mean_entropy/=(np.shape(policies[agent])[0]*np.shape(policies[agent])[1])

            np.save(opts.dir_save+'/training_info/entropy_by_pos_{}.npy'.format(agent),np.array(mean_entropy))

    else:
        policies = estimate_policy(agents=agents,
                                   compo_dataset=compo_dataset,
                                   split=split,
                                   n_sampling=opts.n_sampling,
                                   vocab_size=opts.vocab_size,
                                   max_len=opts.max_len,
                                   n_attributes=opts.n_attributes,
                                   n_values=opts.n_values,
                                   device=device)

        for agent in policies:
            mean_entropy=0.
            for i in range(len(policies[agent])):
              probs=[policies[agent][i][m] for m in policies[agent][i]]
              mean_entropy+=entropy(probs,base=10)
            mean_entropy/=len(policies[agent])

            np.save(opts.dir_save+'/training_info/entropy_{}.npy'.format(agent),np.array(mean_entropy))

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
