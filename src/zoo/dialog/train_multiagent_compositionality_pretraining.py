# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import numpy as np
import os
from os import path
import itertools
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
from src.core.reinforce_wrappers import DialogReinforceCompositionality, AgentBaselineCompositionality, DialogReinforceCompositionalitySingleListener
from src.core.trainers import CompoTrainer,TrainerDialogCompositionality,TrainerDialogAsymLR,TrainerDialogAsymStep,TrainerDialogMultiAgentPretraining

# MultiAgents
from src.core.reinforce_wrappers import DialogReinforceCompositionalityMultiAgent
from src.core.trainers import TrainerDialogMultiAgent
from src.core.util import dump_multiagent_compositionality


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

    # MultiAgent
    parser.add_argument('--N_speakers', type=int, default=1,help='Number agents')
    parser.add_argument('--N_listeners', type=int, default=1,help='Number agents')
    parser.add_argument('--compute_similarity', type=bool, default=False,help='Compute similarity')
    parser.add_argument('--N_listener_sampled',type=int,default=1, help='Numbr of Listeners sampled at each step')
    parser.add_argument('--save_probs',type=str,default=None, help='Save probs during inference')

    args = core.init(parser, params)

    return args

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

    # Distribution of the inputs
    if opts.probs=="uniform":
        probs=[]
        probs_by_att = np.ones(opts.n_values)
        probs_by_att /= probs_by_att.sum()
        for i in range(opts.n_attributes):
            probs.append(probs_by_att)

    if opts.probs=="entropy_test":
        probs=[]
        for i in range(opts.n_attributes):
            probs_by_att = np.ones(opts.n_values)
            probs_by_att[0]=1+(1*i)
            probs_by_att /= probs_by_att.sum()
            probs.append(probs_by_att)

    if opts.probs_attributes=="uniform":
        probs_attributes=[1]*opts.n_attributes

    if opts.probs_attributes=="uniform_indep":
        probs_attributes=[]
        probs_attributes=[0.2]*opts.n_attributes

    if opts.probs_attributes=="echelon":
        probs_attributes=[]
        for i in range(opts.n_attributes):
            #probs_attributes.append(1.-(0.2)*i)
            #probs_attributes.append(0.7+0.3/(i+1))
            probs_attributes=[1.,0.95,0.9,0.85]

    print("Probability by attribute is:",probs_attributes)


    compo_dataset = build_compo_dataset(opts.n_values, opts.n_attributes)

    if opts.split_proportion<1.:
        train_split = np.random.RandomState(opts.random_seed).choice(opts.n_values**opts.n_attributes,size=(int(opts.split_proportion*(opts.n_values**opts.n_attributes))),replace=False)
        test_split=[]

        for j in range(opts.n_values**opts.n_attributes):
          if j not in train_split:
            test_split.append(j)
        test_split = np.array(test_split)

    else:
        train_split=test_split=np.arange(opts.n_values**opts.n_attributes)

    train_loader = OneHotLoaderCompositionality(dataset=compo_dataset,split=train_split,n_values=opts.n_values, n_attributes=opts.n_attributes, batch_size=opts.batch_size,
                                                batches_per_epoch=opts.batches_per_epoch, probs=probs, probs_attributes=probs_attributes)

    # single batches with 1s on the diag
    #test_loader = TestLoaderCompositionality(dataset=compo_dataset,n_values=opts.n_values,n_attributes=opts.n_attributes)
    test_loader = TestLoaderCompositionality(dataset=compo_dataset,split=test_split,n_values=opts.n_values, n_attributes=opts.n_attributes, batch_size=opts.batch_size,
                                            batches_per_epoch=opts.batches_per_epoch, probs=probs, probs_attributes=probs_attributes)

    agents={}
    optim_params={}
    loss_weights={}
    speaker_parameters={}
    listener_parameters={}

    for i in range(max(opts.N_speakers,opts.N_listeners)):

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

        agents["agent_{}".format(i)] = agent

        optim_params["agent_{}".format(i)] = {"length_cost":0.,
                                              "sender_entropy_coeff":opts.sender_entropy_coeff,
                                              "receiver_entropy_coeff":opts.receiver_entropy_coeff}

        loss_weights["agent_{}".format(i)]= {"self":0.,"cross":1.,"imitation":0.}

        if i<opts.N_speakers:
            speaker_parameters["agent_{}".format(i)]=list(agent.agent_sender.parameters()) + \
                                                       list(agent.sender_norm_h.parameters()) + \
                                                       list(agent.sender_norm_c.parameters()) + \
                                                       list(agent.hidden_to_output.parameters()) + \
                                                       list(agent.sender_embedding.parameters()) + \
                                                       list(agent.sender_cells.parameters())


        if i<opts.N_listeners:
            listener_parameters["agent_{}".format(i)]=list(agent.agent_receiver.parameters()) + \
                                  list(agent.receiver_cell.parameters()) + \
                                  list(agent.receiver_embedding.parameters())


    game = DialogReinforceCompositionalityMultiAgent(Agents=agents,
                                                    n_attributes=opts.n_attributes,
                                                    n_values=opts.n_values,
                                                    loss_understanding=loss_understanding_compositionality,
                                                    optim_params=optim_params,
                                                    baseline_mode=opts.baseline_mode,
                                                    reward_mode=opts.reward_mode,
                                                    loss_weights=loss_weights,
                                                    device=device)

    # Optimizers
    optimizer_speaker={}
    optimizer_listener={}

    for i in range(max(opts.N_speakers,opts.N_listeners)):
        if i<opts.N_speakers:
            optimizer_speaker["agent_{}".format(i)] = core.build_optimizer(list(speaker_parameters["agent_{}".format(i)]),lr=opts.sender_lr)
        if i<opts.N_listeners:
            optimizer_listener["agent_{}".format(i)] = core.build_optimizer(list(listener_parameters["agent_{}".format(i)]),lr=opts.receiver_lr)


    "Create trainer"
    list_speakers=[i for i in range(opts.N_speakers)]
    list_listeners=[i for i in range(opts.N_listeners)]
    trainer = TrainerDialogMultiAgentPretraining(game=game, optimizer_speaker=optimizer_speaker,optimizer_listener=optimizer_listener,\
                                                list_speakers=list_speakers,list_listeners=list_listeners,save_probs_eval=opts.save_probs,\
                                                N_listener_sampled = opts.N_listener_sampled,step_ratio=opts.step_ratio,train_data=train_loader, \
                                                validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])


    # Create save dir
    if not path.exists(opts.dir_save):
        os.system("mkdir {}".format(opts.dir_save))
        os.system("mkdir -p {}/models {}/training_info {}/messages {}/accuracy {}/test".format(opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save,opts.dir_save))

    # Save train split
    np.save(opts.dir_save+'/training_info/train_split.npy', train_split)
    np.save(opts.dir_save+'/training_info/test_split.npy', test_split)


    # Main losses
    training_losses=[]
    eval_losses=[]
    training_entropy=[]
    training_loss_cross=[]
    eval_loss_cross=[]

    # Pretraining

    print("Pretraining")

    for epoch in range(10):
        # Train
        list_train_loss,list_train_rest = trainer.train(n_epochs=1,pretraining=True)

        # Eval
        eval_loss,eval_rest = trainer.eval()

        if epoch==0:
            messages=[np.zeros((opts.n_values**opts.n_attributes,opts.max_len)) for _ in range(max(opts.N_speakers,opts.N_listeners))]
        messages,accuracy_vectors, similarity_messages = dump_compositionality_multiagent(trainer.game,compo_dataset,train_split,list_speakers,list_listeners, opts.n_attributes, opts.n_values, device,epoch,past_messages=messages,compute_similarity=compute_similarity)
        np_messages = {agent:convert_messages_to_numpy(messages[agent]) for agent in messages}

    print("Interactions")

    for epoch in range(int(opts.n_epochs)):

        print("Epoch: "+str(epoch))
        if epoch%10==0:
            if opts.N_speakers<4:
                compute_similarity=True
            else:
                compute_similarity=opts.compute_similarity
        else:
            compute_similarity=opts.compute_similarity

        # Train
        list_train_loss,list_train_rest = trainer.train(n_epochs=1,pretraining=False)

        # Eval
        eval_loss,eval_rest = trainer.eval()

        # Store results
        training_losses.append(list_train_loss[-1])
        eval_losses.append(eval_loss)

        training_entropy=[-1]*max(opts.N_speakers,opts.N_listeners)
        training_loss_cross=[-1]*max(opts.N_speakers,opts.N_listeners)
        eval_loss_cross=[-1]*max(opts.N_speakers,opts.N_listeners)

        for i in range(max(opts.N_speakers,opts.N_listeners)):
            if "sender_entropy_{}".format(i) in list_train_rest[-1]:
                training_entropy[i]=list_train_rest[-1]["sender_entropy_{}".format(i)]
            if "loss_{}".format(i) in list_train_rest[-1]:
                training_loss_cross[i]=list_train_rest[-1]["loss_{}".format(i)]
            if "loss_{}".format(i) in eval_rest:
                eval_loss_cross[i] = eval_rest["loss_{}".format(i)]

        print("Train")
        if epoch==0:
            messages=[np.zeros((opts.n_values**opts.n_attributes,opts.max_len)) for _ in range(max(opts.N_speakers,opts.N_listeners))]
        messages,accuracy_vectors, similarity_messages = dump_compositionality_multiagent(trainer.game,compo_dataset,train_split,list_speakers,list_listeners, opts.n_attributes, opts.n_values, device,epoch,past_messages=messages,compute_similarity=compute_similarity)
        np_messages = {agent:convert_messages_to_numpy(messages[agent]) for agent in messages}

        print("Test")
        if epoch==0:
            messages_test=[np.zeros((opts.n_values**opts.n_attributes,opts.max_len)) for _ in range(max(opts.N_speakers,opts.N_listeners))]
        messages_test,accuracy_vectors_test, similarity_messages_test = dump_compositionality_multiagent(trainer.game,compo_dataset,test_split,list_speakers,list_listeners, opts.n_attributes, opts.n_values, device,epoch,past_messages=messages_test,compute_similarity=compute_similarity)
        np_messages_test = {agent:convert_messages_to_numpy(messages_test[agent]) for agent in messages_test}

        # Save models
        if epoch%20==0:
            for agent in agents:
                torch.save(agents[agent].state_dict(), f"{opts.dir_save}/models/{agent}_weights_{epoch}.pth")

        # Save training info
        if epoch%10==0:
            np.save(opts.dir_save+'/training_info/training_loss_{}.npy'.format(epoch), training_losses)
            np.save(opts.dir_save+'/training_info/eval_loss_{}.npy'.format(epoch), eval_losses)
            np.save(opts.dir_save+'/training_info/training_entropy_{}.npy'.format(epoch), training_entropy)
            np.save(opts.dir_save+'/training_info/training_loss_cross_{}.npy'.format(epoch), training_loss_cross)
            np.save(opts.dir_save+'/training_info/eval_loss_cross_{}.npy'.format(epoch), eval_loss_cross)
            np.save(opts.dir_save+'/training_info/similarity_languages_{}.npy'.format(epoch), similarity_messages)
            np.save(opts.dir_save+'/training_info/similarity_languages_test_{}.npy'.format(epoch), similarity_messages_test)

        # Save accuracy/message results
        messages_to_be_saved = np.stack([fill_to_max_len(np_messages[agent],opts.max_len) for agent in np_messages])
        accuracy_vectors_to_be_saved = np.zeros((len(list_speakers),len(list_listeners),len(train_split),opts.n_attributes))
        for i,agent_speaker in enumerate(accuracy_vectors):
            for j,agent_listener in enumerate(accuracy_vectors[agent_speaker]):
                accuracy_vectors_to_be_saved[i,j,:,:] = accuracy_vectors[agent_speaker][agent_listener]


        np.save(opts.dir_save+'/messages/messages_{}.npy'.format(epoch), messages_to_be_saved)
        np.save(opts.dir_save+'/accuracy/accuracy_{}.npy'.format(epoch), accuracy_vectors_to_be_saved)

        # Test set
        messages_test_to_be_saved = np.stack([fill_to_max_len(np_messages_test[agent],opts.max_len) for agent in np_messages_test])
        accuracy_vectors_test_to_be_saved = np.zeros((len(list_speakers),len(list_listeners),len(test_split),opts.n_attributes))
        for i,agent_speaker in enumerate(accuracy_vectors_test):
            for j,agent_listener in enumerate(accuracy_vectors_test[agent_speaker]):
                accuracy_vectors_test_to_be_saved[i,j,:,:] = accuracy_vectors_test[agent_speaker][agent_listener]

        np.save(opts.dir_save+'/test/messages_test_{}.npy'.format(epoch), messages_test_to_be_saved)
        np.save(opts.dir_save+'/test/accuracy_test_{}.npy'.format(epoch), accuracy_vectors_test_to_be_saved)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
