import time
import math
import numpy as np
import os
import random
from functools import reduce
from scipy.stats import entropy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.query_network import QueryNetworkDQN
from models.AptamerLSTM import AptamerLSTM
from utils.final_utils import get_logfile
from utils.progressbar import progress_bar

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10


def create_models():
    """Creates the 3 Networks needed for Training

    :return: sequence network, query network, target network (same construction as query network)
    """

    # Sequence network

    net = AptamerLSTM().cuda()
    print("Model has " + str(count_parameters(net)))

    # Query network (and target network for DQN)
    state_dataset_size = 30  # This depends on size of dataset V
    action_state_length = 3

    policy_net = QueryNetworkDQN(
        model_state_length=state_dataset_size,
        action_state_length=action_state_length,
        bias_average=1,  # TODO Figure out the query network bias trick from 2018 paper.
    ).cuda()
    target_net = QueryNetworkDQN(
        model_state_length=state_dataset_size,
        action_state_length=action_state_length,
        bias_average=1,  # TODO Figure out the query network bias trick from 2018 paper.
    ).cuda()
    print("Policy network has " + str(count_parameters(policy_net)))

    print("Models created!")
    return net, policy_net, target_net


def count_parameters(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def load_models(
    net,
    load_weights,
    exp_name_toload,
    snapshot,
    exp_name,
    ckpt_path,
    checkpointer,
    exp_name_toload_rl="",
    policy_net=None,
    target_net=None,
    test=False,
    dataset="cityscapes",
    al_algorithm="random",
):
    """Load model weights.
    :param net: Segmentation network
    :param load_weights: (bool) True if segmentation network is loaded from pretrained weights in 'exp_name_toload'
    :param exp_name_toload: (str) Folder where to find pre-trained segmentation network's weights.
    :param snapshot: (str) Name of the checkpoint.
    :param exp_name: (str) Experiment name.
    :param ckpt_path: (str) Checkpoint name.
    :param checkpointer: (bool) If True, load weights from the same folder.
    :param exp_name_toload_rl: (str) Folder where to find trained weights for the query network (DQN). Used to test
    query network.
    :param policy_net: Policy network.
    :param target_net: Target network.
    :param test: (bool) If True and al_algorithm='ralis' and there exists a checkpoint in 'exp_name_toload_rl',
    we will load checkpoint for trained query network (DQN).
    :param dataset: (str) Which dataset.
    :param al_algorithm: (str) Which active learning algorithm.
    """

    policy_path = os.path.join(ckpt_path, exp_name_toload_rl, "policy_" + snapshot)
    net_path = os.path.join(ckpt_path, exp_name_toload, "best_jaccard_val.pth")

    policy_checkpoint_path = os.path.join(ckpt_path, exp_name, "policy_" + snapshot)
    net_checkpoint_path = os.path.join(ckpt_path, exp_name, "last_jaccard_val.pth")

    ####------ Load policy (RL) from one folder and network from another folder ------####
    if al_algorithm == "ralis" and test and os.path.isfile(policy_path):
        print("(RL and TEST) Testing policy from another experiment folder!")
        policy_net.load_state_dict(torch.load(policy_path))
        # Load pre-trained segmentation network from another folder (best checkpoint)
        if load_weights and len(exp_name_toload) > 0:
            print("Loading pre-trained segmentation network (best checkpoint).")
            net_dict = torch.load(net_path)
            if len([key for key, value in net_dict.items() if "module" in key.lower()]) > 0:
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in net_dict.items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                net_dict = new_state_dict
            net.load_state_dict(net_dict)
            net.cuda()

        if (
            checkpointer
        ):  # In case the experiment is interrupted, load most recent segmentation network
            if os.path.isfile(net_checkpoint_path):
                print("(RL and TEST) Loading checkpoint for segmentation network!")
                net_dict = torch.load(net_checkpoint_path)
                if len([key for key, value in net_dict.items() if "module" in key.lower()]) > 0:
                    from collections import OrderedDict

                    new_state_dict = OrderedDict()
                    for k, v in net_dict.items():
                        name = k[7:]  # remove module.
                        new_state_dict[name] = v
                    net_dict = new_state_dict
                net.load_state_dict(net_dict)
    else:
        ####------ Load experiment from another folder ------####
        if load_weights and len(exp_name_toload) > 0:
            print("(From another exp) training resumes from best_jaccard_val.pth")
            net_dict = torch.load(net_path)
            if len([key for key, value in net_dict.items() if "module" in key.lower()]) > 0:
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in net_dict.items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                net_dict = new_state_dict
            net.load_state_dict(net_dict)

        ####------ Resume experiment ------####
        if checkpointer:
            ##-- Check if weights exist --##
            if os.path.isfile(net_checkpoint_path):
                print("(Checkpointer) training resumes from last_jaccard_val.pth")
                net_dict = torch.load(net_checkpoint_path)
                if len([key for key, value in net_dict.items() if "module" in key.lower()]) > 0:
                    from collections import OrderedDict

                    new_state_dict = OrderedDict()
                    for k, v in net_dict.items():
                        name = k[7:]  # remove module.
                        new_state_dict[name] = v
                    net_dict = new_state_dict
                net.load_state_dict(net_dict)
                if (
                    al_algorithm == "ralis"
                    and os.path.isfile(policy_checkpoint_path)
                    and policy_net is not None
                    and target_net is not None
                ):
                    print("(Checkpointer RL) training resumes from " + snapshot)
                    policy_net.load_state_dict(torch.load(policy_checkpoint_path))
                    policy_net.cuda()
                    target_net.load_state_dict(torch.load(policy_checkpoint_path))
                    target_net.cuda()
            else:
                print("(Checkpointer) Training starts from scratch")

    ####------ Get log file ------####
    if al_algorithm == "ralis":
        logger = None
        best_record = None
        curr_epoch = None
    else:
        num_classes = 11 if "camvid" in dataset else 19
        logger, best_record, curr_epoch = get_logfile(
            ckpt_path, exp_name, checkpointer, snapshot, num_classes=num_classes
        )

    return logger, curr_epoch, best_record


def compute_state(net, stateLoader, unlabelledLoader, labelled_dataset, unlabelled_dataset):
    ## Adapted from LAL-RL, https://github.com/ksenia-konyushkova/LAL-RL
    """Function for computing the state depending on the sequence model and next available actions.

    This function computes 1) model_state that characterises
    the current state of the model and it is computed as
    a function of predictions on the hold-out dataset
    2) next_action_state that characterises all possible actions
    (unlabelled datapoints) that can be taken at the next step.

    :param: net:
    :param: state_dataset:
    :param: unlabelled_dataset:
    :param: labelled_dataset:

    :return: model_state: a numpy.ndarray of size of number of datapoints in dataset.state_data characterizing the current model and, thus, the state of the environment
    :return: next_action_state: a numpy.ndarray of size #features characterising actions (currently, 3) x #unlabelled datapoints where each column corresponds to the vector characterizing each possible action.
    """
    # COMPUTE MODEL_STATE
    net.eval()
    # TODO Fix Jankiness here.
    predictions = [net(torch.tensor(datapoint).to("cuda:0")) for datapoint in stateLoader]
    predictions = torch.cat([*predictions])[:, 0]
    idx = torch.argsort(predictions)
    # the state representation is the *sorted* list of scores
    model_state = predictions[idx]

    # COMPUTE ACTION_STATE
    # prediction (score) of model on each unlabelled sample
    a1 = [net(datapoint) for datapoint in unlabelledLoader]
    a1 = torch.cat([*a1])[:, 0]
    a2 = []
    a3 = []
    for idx, xi in enumerate(unlabelled_dataset):
        # average distance to every unlabelled datapoint
        a2 += [
            torch.mean(
                torch.tensor(
                    [torch.sum(xi != datapoint) for datapoint in unlabelled_dataset],
                    dtype=torch.float32,
                    device=torch.device("cuda:0"),
                )
            )
        ]
        # average distance to every labelled datapoint
        a3 += [
            torch.mean(
                torch.tensor(
                    [torch.sum(xi != datapoint[0]) for datapoint in labelled_dataset],
                    dtype=torch.float32,
                    device=torch.device("cuda:0"),
                )
            )
        ]
    a2 = torch.stack(a2)
    a3 = torch.stack(a3)
    action_state = torch.stack([a1, a2, a3])
    action_state = torch.transpose(action_state, 0, 1)
    return model_state, action_state


def select_action(args, policy_net, model_state, action_state, steps_done, test=False):
    """We select the action: index of the image to label.
    :(argparse.ArgumentParser) args: The parser with all the defined arguments.
    :param policy_net: policy network.
    :param model_state: (torch.Variable) Torch tensor containing the model state representation.
    :param action_state: (torch.Variable) Torch tensor containing the action state representations.
    :param steps_done: (int) Number of images labeled so far.
    :param test: (bool) Whether we are testing the DQN or training it.
    :return: Action (indexes of the regions to label)
    """

    policy_net.eval()
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    q_val_ = []
    if sample > eps_threshold or test:
        print("Action selected with DQN!")
        with torch.no_grad():
            # Get Q-values for every action
            q_val_ = [policy_net(model_state, action_i_state) for action_i_state in action_state]

            action = torch.argmax(torch.stack(q_val_))
            del q_val_
    else:
        action = Variable(
            torch.Tensor(
                [np.random.choice(range(args.rl_pool), action_state.size()[0], replace=True)]
            )
            .type(torch.LongTensor)
            .view(-1)
        )  # .cuda()

    return action

def optimize_q_network(
    args,
    memory,
    Transition,
    policy_net,
    target_net,
    optimizerP,
    BATCH_SIZE=32,
    GAMMA=0.999,
    dqn_epochs=1,
):
    """This function optimizes the policy network

    :(ReplayMemory) memory: Experience replay buffer
    :param Transition: definition of the experience replay tuple
    :param policy_net: Policy network
    :param target_net: Target network
    :param optimizerP: Optimizer of the policy network
    :param BATCH_SIZE: (int) Batch size to sample from the experience replay
    :param GAMMA: (float) Discount factor
    :param dqn_epochs: (int) Number of epochs to train the DQN
    """
    # Code adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    if len(memory) < BATCH_SIZE:
        return
    print("Optimize model...")
    print(len(memory))
    policy_net.train()
    loss_item = 0
    for ep in range(dqn_epochs):
        optimizerP.zero_grad()
        transitions = memory.sample(BATCH_SIZE)
        expected_q_values = []
        for transition in transitions:
            # Predict q-value function value for all available actions in transition
            action_i = select_action(
                args, policy_net, transition.model_state, transition.action_state, test=False
            )
            with torch.autograd.set_detect_anomaly(True):
                policy_net.train()
                q_policy = policy_net(
                    transition.model_state.detach(), transition.action_state[action_i].detach()
                )
                q_target = target_net(
                    transition.model_state.detach(), transition.action_state[action_i].detach()
                )

                # Compute the expected Q values
                expected_q_values = (q_target * GAMMA) + transition.reward

                # Compute MSE loss
                loss = F.mse_loss(q_policy, expected_q_values)
                # loss_item += loss.item()
                # progress_bar(ep, dqn_epochs, "[DQN loss %.5f]" % (loss_item / (ep + 1)))
                loss.backward()
        optimizerP.step()

        del loss
        del transitions

    lab_set = open(os.path.join(args.ckpt_path, args.exp_name, "q_loss.txt"), "a")
    lab_set.write("%f" % (loss_item))
    lab_set.write("\n")
    lab_set.close()
