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
from src.core.reinforce_wrappers import DialogReinforceCompositionality, AgentBaselineCompositionality
from src.core.trainers import CompoTrainer,TrainerDialogCompositionality,TrainerDialogAsymLR,TrainerDialogAsymStep


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
    parser.add_argument('--N_speaker', type=float, default=10,help='Number of speaker training step')
    parser.add_argument('--N_listener', type=float, default=10,help='Number of listener training step')

    args = core.init(parser, params)

    return args


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

def dump_compositionality(game,compo_dataset,split,n_attributes,n_values,device, gs_mode, epoch,past_messages_1=None,past_messages_2=None):

    dataset=[]
    combination=[]

    for i in range(len(compo_dataset)):
        if i in split:
          dataset.append(torch.from_numpy(compo_dataset[i]).float())
          combination.append(np.reshape(compo_dataset[i],(n_attributes,n_values)).argmax(1))

    dataset = [[torch.stack(dataset).to(device), None]]

    sender_inputs_1, messages_1, receiver_inputs_1, receiver_outputs_11,receiver_outputs_12, \
    sender_inputs_2, messages_2, receiver_inputs_2, receiver_outputs_21,receiver_outputs_22, _ = \
        dump_dialog_compositionality(game, dataset, gs=gs_mode, device=device, variable_length=True)

    n_messages = len(dataset[0][0])

    print("Language 1 (Agent 1 -> Agent 2)")

    "1->2"
    unif_acc = 0.
    unif_acc_general=0.
    acc_vec_1=np.zeros(((n_messages), n_attributes))

    for i in range(len(receiver_outputs_12)):
      message=messages_1[i]
      correct=True
      if i<n_messages:
          for j in range(len(list(combination[i]))):
            if receiver_outputs_12[i][j]==list(combination[i])[j]:
              unif_acc+=1
              acc_vec_1[i,j]=1
            else:
              correct=False
          if correct:
            unif_acc_general+=1.

      if epoch%100==99:
          print(f'input: {",".join([str(x) for x in combination[i]])} -> message: {",".join([str(x.item()) for x in message])} -> output: {",".join([str(x) for x in receiver_outputs_12[i]])}', flush=True)

    unif_acc /= (n_messages) * n_attributes
    unif_acc_general/=n_messages

    print(json.dumps({'unif': unif_acc,'unif_general':unif_acc_general}))

    print(np.mean(acc_vec_1,axis=0))

    "1->1"
    print("internal listener")
    unif_acc = 0.
    unif_acc_general=0.
    acc_vec_11=np.zeros(((n_messages), n_attributes))

    for i in range(len(receiver_outputs_11)):
      message=messages_1[i]
      correct=True
      if i<n_messages:
          for j in range(len(list(combination[i]))):
            if receiver_outputs_11[i][j]==list(combination[i])[j]:
              unif_acc+=1
              acc_vec_11[i,j]=1
            else:
              correct=False

          if correct:
            unif_acc_general+=1.

      if epoch%100==99:
          print(f'input: {",".join([str(x) for x in combination[i]])} -> message: {",".join([str(x.item()) for x in message])} -> output: {",".join([str(x) for x in receiver_outputs_11[i]])}', flush=True)

    unif_acc /= (n_messages) * n_attributes
    unif_acc_general/=n_messages

    print(json.dumps({'unif': unif_acc,'unif_general':unif_acc_general}))

    print(np.mean(acc_vec_11,axis=0))

    print("Language 2 (Agent 2 -> Agent 1)")

    "2->1"

    unif_acc = 0.
    unif_acc_general=0.
    acc_vec_2=np.zeros(((n_messages), n_attributes))

    for i in range(len(receiver_outputs_21)):
      message=messages_2[i]
      correct=True
      if i<n_messages:
          for j in range(len(list(combination[i]))):
            if receiver_outputs_21[i][j]==list(combination[i])[j]:
              unif_acc+=1
              acc_vec_2[i,j]=1
            else:
              correct=False

          if correct:
            unif_acc_general+=1.

      if epoch%100==99:
          print(f'input: {",".join([str(x) for x in combination[i]])} -> message: {",".join([str(x.item()) for x in message])} -> output: {",".join([str(x) for x in receiver_outputs_21[i]])}', flush=True)

    unif_acc /= (n_messages) * n_attributes
    unif_acc_general/=n_messages

    print(json.dumps({'unif': unif_acc,'unif_general':unif_acc_general}))
    print(np.mean(acc_vec_2,axis=0))

    print("internal listener")

    unif_acc = 0.
    unif_acc_general=0.
    acc_vec_22=np.zeros(((n_messages), n_attributes))

    for i in range(len(receiver_outputs_22)):
      message=messages_2[i]
      correct=True
      if i<n_messages:
          for j in range(len(list(combination[i]))):
            if receiver_outputs_22[i][j]==list(combination[i])[j]:
              unif_acc+=1
              acc_vec_22[i,j]=1
            else:
              correct=False

          if correct:
            unif_acc_general+=1.

      if epoch%100==99:
          print(f'input: {",".join([str(x) for x in combination[i]])} -> message: {",".join([str(x.item()) for x in message])} -> output: {",".join([str(x) for x in receiver_outputs_22[i]])}', flush=True)

    unif_acc /= (n_messages) * n_attributes
    unif_acc_general/=n_messages

    print(json.dumps({'unif': unif_acc,'unif_general':unif_acc_general}))
    print(np.mean(acc_vec_22,axis=0))

    similarity_messages=np.mean([levenshtein(messages_1[i],messages_2[i])/np.max([len(messages_1[i]),len(messages_2[i])]) for i in range(len(messages_1))])

    print("Similarity between language = {}".format(similarity_messages),flush=True)

    if past_messages_1 is not None:
        print("Similarity evo language 1 = {}".format(np.mean([levenshtein(messages_1[i],past_messages_1[i]) for i in range(len(messages_1))])),flush=True)
    if past_messages_2 is not None:
        print("Similarity evo language 2 = {}".format(np.mean([levenshtein(messages_2[i],past_messages_2[i]) for i in range(len(messages_2))])),flush=True)


    return messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22, similarity_messages

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

    train_split = np.random.RandomState(opts.random_seed).choice(opts.n_values**opts.n_attributes,size=(int(opts.split_proportion*(opts.n_values**opts.n_attributes))),replace=False)
    test_split=[]

    for j in range(opts.n_values**opts.n_attributes):
      if j not in train_split:
        test_split.append(j)
    test_split = np.array(test_split)

    train_loader = OneHotLoaderCompositionality(dataset=compo_dataset,split=train_split,n_values=opts.n_values, n_attributes=opts.n_attributes, batch_size=opts.batch_size,
                                                batches_per_epoch=opts.batches_per_epoch, probs=probs, probs_attributes=probs_attributes)

    # single batches with 1s on the diag
    #test_loader = TestLoaderCompositionality(dataset=compo_dataset,n_values=opts.n_values,n_attributes=opts.n_attributes)
    test_loader = TestLoaderCompositionality(dataset=compo_dataset,split=test_split,n_values=opts.n_values, n_attributes=opts.n_attributes, batch_size=opts.batch_size,
                                            batches_per_epoch=opts.batches_per_epoch, probs=probs, probs_attributes=probs_attributes)

    agent_1=AgentBaselineCompositionality(vocab_size=opts.vocab_size,
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

    agent_2=AgentBaselineCompositionality(vocab_size=opts.vocab_size,
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

    game = DialogReinforceCompositionality(Agent_1=agent_1,
                                            Agent_2=agent_2,
                                            n_attributes=opts.n_attributes,
                                            n_values=opts.n_values,
                                            loss_understanding=loss_understanding_compositionality,
                                            optim_params=optim_params,
                                            baseline_mode=opts.baseline_mode,
                                            reward_mode=opts.reward_mode,
                                            loss_weights=loss_weights,
                                            device=device)

    "Create optimizers"
    if opts.model=="expe_1":
        optimizer = core.build_optimizer(list(game.parameters()))

        trainer = TrainerDialogCompositionality(n_attributes=opts.n_attributes,n_values=opts.n_values,game=game, optimizer=optimizer, train_data=train_loader,
                                                validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

    elif opts.model=="expe_lr":

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

        # SGD
        #optimizer_speaker=torch.optim.SGD(speaker_parameters, lr=opts.sender_lr, momentum=0.9,nesterov=False)
        #optimizer_listener=torch.optim.SGD(listener_parameters, lr=opts.receiver_lr, momentum=0.9,nesterov=False)
        optimizer_speaker = core.build_optimizer(list(speaker_parameters),lr=opts.sender_lr)
        optimizer_listener = core.build_optimizer(list(listener_parameters),lr=opts.receiver_lr)


        "Create trainer"
        trainer = TrainerDialog(game=game, optimizer=optimizer, train_data=train_loader, \
                                validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

    elif opts.model=="expe_step":

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

        # SGD
        #optimizer_speaker=torch.optim.SGD(speaker_parameters, lr=opts.sender_lr, momentum=0.9,nesterov=False)
        #optimizer_listener=torch.optim.SGD(listener_parameters, lr=opts.receiver_lr, momentum=0.9,nesterov=False)
        optimizer_speaker = core.build_optimizer(list(speaker_parameters),lr=opts.sender_lr)
        optimizer_listener = core.build_optimizer(list(listener_parameters),lr=opts.receiver_lr)


        "Create trainer"
        #trainer = TrainerDialog(game=game, optimizer=optimizer, train_data=train_loader, \
        #                        validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])
        trainer = TrainerDialogAsymStep(game=game, optimizer_speaker=optimizer_speaker,optimizer_listener=optimizer_listener,\
                                        N_speaker=opts.N_speaker,N_listener=opts.N_listener,train_data=train_loader, \
                                        validation_data=test_loader, callbacks=[EarlyStopperAccuracy(opts.early_stopping_thr)])

    else:
        raise("Model not indicated")

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
    similarity_languages_test=[]


    for epoch in range(int(opts.n_epochs)):

        print("Epoch: "+str(epoch))

        # Train
        list_train_loss,list_train_rest = trainer.train(n_epochs=1)

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
        #training_loss_imitation_12.append(list_train_rest[-1]["loss_imitation_12"])
        training_loss_self_22.append(list_train_rest[-1]["loss_self_22"])
        training_loss_cross_21.append(list_train_rest[-1]["loss_cross_21"])
        #training_loss_imitation_21.append(list_train_rest[-1]["loss_imitation_21"])
        eval_loss_self_11.append(eval_rest["loss_self_11"])
        eval_loss_cross_12.append(eval_rest["loss_cross_12"])
        #eval_loss_imitation_12.append(eval_rest["loss_imitation_12"])
        eval_loss_self_22.append(eval_rest["loss_self_22"])
        eval_loss_cross_21.append(eval_rest["loss_cross_21"])
        #eval_loss_imitation_21.append(eval_rest["loss_imitation_21"])

        print("Train")
        if epoch==0:
            messages_1=messages_2=np.zeros((opts.n_values**opts.n_attributes,opts.max_len))
        messages_1, messages_2,acc_vec_1, acc_vec_2, acc_vec_11, acc_vec_22, similarity_messages = dump_compositionality(trainer.game,compo_dataset,train_split, opts.n_attributes, opts.n_values, device, False,epoch,past_messages_1=messages_1,past_messages_2=messages_2)
        np_messages_1 = convert_messages_to_numpy(messages_1)
        np_messages_2 = convert_messages_to_numpy(messages_2)
        similarity_languages.append(similarity_messages)

        print("Test")
        if epoch==0:
            messages_1_test=messages_2_test=np.zeros((opts.n_values**opts.n_attributes,opts.max_len))
        messages_1_test, messages_2_test,acc_vec_1_test, acc_vec_2_test, acc_vec_11_test, acc_vec_22_test, similarity_messages_test = dump_compositionality(trainer.game,compo_dataset,test_split, opts.n_attributes, opts.n_values, device, False,epoch,past_messages_1=messages_1_test,past_messages_2=messages_2_test)
        np_messages_1_test = convert_messages_to_numpy(messages_1_test)
        np_messages_2_test = convert_messages_to_numpy(messages_2_test)
        similarity_languages_test.append(similarity_messages_test)

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
            np.save(opts.dir_save+'/training_info/similarity_languages_test_{}.npy'.format(epoch), similarity_languages_test)

        # Save accuracy/message results
        np.save(opts.dir_save+'/messages/agent_1_messages_{}.npy'.format(epoch), np_messages_1)
        np.save(opts.dir_save+'/messages/agent_2_messages_{}.npy'.format(epoch), np_messages_2)
        np.save(opts.dir_save+'/accuracy/12_accuracy_{}.npy'.format(epoch), acc_vec_1)
        np.save(opts.dir_save+'/accuracy/21_accuracy_{}.npy'.format(epoch), acc_vec_2)
        np.save(opts.dir_save+'/accuracy/11_accuracy_{}.npy'.format(epoch), acc_vec_11)
        np.save(opts.dir_save+'/accuracy/22_accuracy_{}.npy'.format(epoch), acc_vec_22)

        # Test set
        np.save(opts.dir_save+'/test/agent_1_messages_test_{}.npy'.format(epoch), np_messages_1_test)
        np.save(opts.dir_save+'/test/agent_2_messages_test_{}.npy'.format(epoch), np_messages_2_test)
        np.save(opts.dir_save+'/test/12_accuracy_test_{}.npy'.format(epoch), acc_vec_1_test)
        np.save(opts.dir_save+'/test/21_accuracy_test_{}.npy'.format(epoch), acc_vec_2_test)
        np.save(opts.dir_save+'/test/11_accuracy_test_{}.npy'.format(epoch), acc_vec_11_test)
        np.save(opts.dir_save+'/test/22_accuracy_test_{}.npy'.format(epoch), acc_vec_22_test)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
