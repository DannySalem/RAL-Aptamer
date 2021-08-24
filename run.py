from data import dataset
import os
import sys
import shutil
import random
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR

from models.model_utils import (
    create_models,
    load_models,
    compute_state,
    select_action,
    optimize_q_network,
)
from utils.final_utils import (
    check_mkdir,
    create_and_load_optimizers,
    get_logfile,
    get_training_stage,
    set_training_stage,
)

from data.data_utils import get_exp_data, get_iter_data, add_labeled_datapoint
from utils.replay_buffer import ReplayMemory
from utils.replay_new import ReplayBuffer
import utils.parser as parser
from utils.final_utils import train, validate, final_test

cudnn.benchmark = False
cudnn.deterministic = True


def main(args):
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ####------ Create experiment folder  ------####
    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))

    ####------ Print and save arguments in experiment folder  ------####
    parser.save_arguments(args)
    ####------ Copy current config file to ckpt folder ------####
    fn = sys.argv[0].rsplit("/", 1)[-1]
    # shutil.copy(sys.argv[0], os.path.join(args.ckpt_path, args.exp_name, fn))

    ####------ Create segmentation, query and target networks ------####

    net, policy_net, target_net = create_models()

    # TODO remove unnecessary things from load_models
    ####------ Load weights if necessary and create log file ------####
    # kwargs_load = {
    #    "net": net,
    #    "load_weights": args.load_weights,
    #    "exp_name_toload": args.exp_name_toload,
    #    "snapshot": args.snapshot,
    #    "exp_name": args.exp_name,
    #    "ckpt_path": args.ckpt_path,
    #    "checkpointer": args.checkpointer,
    #    "exp_name_toload_rl": args.exp_name_toload_rl,
    #    "policy_net": policy_net,
    #    "target_net": target_net,
    #    "test": args.test,
    #    "dataset": args.dataset,
    #    "al_algorithm": args.al_algorithm,
    # }
    # _ = load_models(**kwargs_load)

    ####------ Load training and validation data ------####
    # kwargs_data = {
    #    "data_path": args.data_path,
    #    "tr_bs": args.train_batch_size,
    #    "vl_bs": args.val_batch_size,
    #    "n_workers": 4,
    #    "input_size": args.input_size,
    #    "num_each_iter": args.num_each_iter,
    #    "dataset": args.dataset,
    #    "test": args.test,
    # }

    state_dataset, labelled_dataset, val_dataset = get_exp_data()
    labelledLoader = DataLoader(
        labelled_dataset, batch_size=12, shuffle=True, num_workers=0, pin_memory=False
    )
    valLoader = DataLoader(
        val_dataset, batch_size=12, shuffle=True, num_workers=0, pin_memory=False
    )
    stateLoader = DataLoader(
        state_dataset, batch_size=15, shuffle=True, num_workers=0, pin_memory=False
    )

    ####------ Create loss ------####
    criterion = nn.MSELoss().cuda()

    ####------ Create optimizers (and load them if necessary) ------####
    kwargs_load_opt = {
        "net": net,
        "opt_choice": args.optimizer,
        "lr": args.lr,
        "wd": args.weight_decay,
        "momentum": args.momentum,
        "ckpt_path": args.ckpt_path,
        "exp_name_toload": args.exp_name_toload,
        "exp_name": args.exp_name,
        "snapshot": args.snapshot,
        "checkpointer": args.checkpointer,
        "load_opt": args.load_opt,
        "policy_net": policy_net,
        "lr_dqn": args.lr_dqn,
    }

    optimizer, optimizerP = create_and_load_optimizers(**kwargs_load_opt)

    #####################################################################
    ####################### TRAIN ######################
    #####################################################################
    if args.train:
        print("Starting training...")

        # Create schedulers
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
        schedulerP = ExponentialLR(optimizerP, gamma=args.gamma_scheduler_dqn)

        # Set Train Mode on Sequence Model
        net.train()

        num_episodes = args.rl_episodes
        epoch_num = args.epoch_num
        patience = args.patience
        best_val_episode = 0
        steps_done = 0

        ## Deep Q-Network variables ##
        Transition = namedtuple(
            "Transition",
            (
                "model_state",
                "action_state",
                "next_model_state",
                "next_action_state",
                "reward",
                "terminal",
            ),
        )

        memory = ReplayBuffer(buffer_size=args.rl_buffer)
        memory2 = ReplayMemory(capacity=args.rl_buffer)
        # Load Target Network, set to eval Mode, Set how often to update the weights
        TARGET_UPDATE = 5
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        for n_ep in range(num_episodes):
            ## [ --- Start of episode ---]  ##
            print("---------------Current episode: " + str(n_ep) + "/" + str(num_episodes))

            # TODO Update logging so that it works with our data instead of images/patches. Need to change the Logger class and the get_logfile method
            logger, best_record, curr_epoch = get_logfile(
                args.ckpt_path,
                args.exp_name,
                args.checkpointer,
                args.snapshot,
                log_name="ep" + str(n_ep) + "_log.txt",
            )

            # Initialize budget and counter to update target_network
            terminal = False
            counter_iter = 0

            # Sample Unlabelled datapoints
            unlabelled_dataset = get_iter_data()
            unlabelledLoader = DataLoader(
                unlabelled_dataset, batch_size=15, shuffle=True, num_workers=0, pin_memory=False
            )
            # Compute state. Shape:[group_size, num regions, dim, w,h]
            model_state, action_state = compute_state(
                net, stateLoader, unlabelledLoader, labelled_dataset, unlabelled_dataset
            )

            args.epoch_num = epoch_num
            args.patience = patience
            args.rl_pool = 100
            # Take images while the budget is not met
            while not terminal:
                # Choose actions. The actions are the regions to label at a given step
                action = select_action(args, policy_net, model_state, action_state, steps_done)
                steps_done += 1

                # TODO Update to match our data
                # list_existing_images = add_labeled_datapoint(
                #    args,
                #    list_existing_images=list_existing_images,
                #    region_candidates=region_candidates,
                #    train_set=train_set,
                #    action_list=action,
                #    budget=args.budget_labels,
                #    n_ep=n_ep,
                # )
                add_labeled_datapoint(labelled_dataset, unlabelled_dataset[action])
                # Train segmentation network with selected regions:
                print("Train network with selected images...")
                train_loss, val_loss, val_rmse = train_seq_model(
                    args,
                    0,
                    labelledLoader,
                    net,
                    criterion,
                    optimizer,
                    valLoader,
                    best_record,
                    logger,
                    scheduler,
                    schedulerP,
                )

                reward = -1

                # Sample Unlabelled datapoint candidates for next state
                unlabelled_dataset = get_iter_data()
                unlabelledLoader = DataLoader(
                    unlabelled_dataset, batch_size=15, shuffle=True, num_workers=0, pin_memory=False
                )

                # Compute next state
                if not terminal:
                    next_model_state, next_action_state = compute_state(
                        net, stateLoader, unlabelledLoader, labelled_dataset, unlabelled_dataset
                    )
                else:
                    next_model_state = None
                    next_action_state = None

                # Store the transition in experience replay. Next state is None if the budget has been reached (final
                # state)

                memory2.push(
                    model_state,
                    action_state,
                    next_model_state,
                    next_action_state,
                    reward,
                    terminal,
                )

                # Move to the next state
                del model_state, action_state
                model_state = next_model_state
                action_state = next_action_state
                del next_model_state, next_action_state

                # Perform optimization on the target network
                optimize_q_network(
                    args,
                    memory2,
                    Transition,
                    policy_net,
                    target_net,
                    optimizerP,
                    GAMMA=args.dqn_gamma,
                    BATCH_SIZE=4,  # args.dqn_bs,
                )

                # Save weights of policy_net and target_net
                torch.save(
                    policy_net.cpu().state_dict(),
                    os.path.join(args.ckpt_path, args.exp_name, "policy_last_jaccard_val.pth"),
                )
                policy_net.cuda()
                torch.save(
                    optimizerP.state_dict(),
                    os.path.join(args.ckpt_path, args.exp_name, "opt_policy_last_jaccard_val.pth"),
                )

                # Update target network every TARGET_UPDATE iterations
                if counter_iter % TARGET_UPDATE == 0:
                    print("Update target network")
                    target_net.load_state_dict(policy_net.state_dict())
                    torch.save(
                        target_net.cpu().state_dict(),
                        os.path.join(args.ckpt_path, args.exp_name, "target_last_jaccard_val.pth"),
                    )
                    target_net.cuda()
                counter_iter += 1

            # Training to convergence not relevant for the training of the DQN.
            # We get no rewards, states or actions once the budget is met
            print("Training with all images, training with patience 0")
            args.patience = 0
            # Train until convergence
            _, val_acc_episode, _ = train_seq_model(
                args,
                curr_epoch,
                train_loader,
                net,
                criterion,
                optimizer,
                val_loader,
                best_record,
                logger,
                scheduler,
                schedulerP,
                final_train=True,
            )

            # End of budget
            logger.close()
            train_set.reset()
            list_existing_images = []

            del net
            del optimizer
            del train_loader
            del train_set
            del candidate_set

            print("Resetting the networks, optimizers and data!")
            # Create the networks from scratch, except the policy and target networks.
            ####------ Create segmentation, query and target network ------####
            kwargs_models = {
                "dataset": args.dataset,
                "al_algorithm": args.al_algorithm,
                "region_size": args.region_size,
            }
            net, _, _ = create_models(**kwargs_models)

            ####------ Load weights if necessary and create log file ------####
            kwargs_load = {
                "net": net,
                "load_weights": args.load_weights,
                "exp_name_toload": args.exp_name_toload,
                "snapshot": args.snapshot,
                "exp_name": args.exp_name,
                "ckpt_path": args.ckpt_path,
                "checkpointer": args.checkpointer,
                "exp_name_toload_rl": args.exp_name_toload_rl,
                "policy_net": None,
                "target_net": None,
                "test": args.test,
                "dataset": args.dataset,
                "al_algorithm": args.al_algorithm,
            }
            _ = load_models(**kwargs_load)

            ####------ Load training and validation data ------####
            kwargs_data = {
                "data_path": args.data_path,
                "tr_bs": args.train_batch_size,
                "vl_bs": args.val_batch_size,
                "n_workers": 4,
                "scale_size": args.scale_size,
                "input_size": args.input_size,
                "num_each_iter": args.num_each_iter,
                "only_last_labeled": args.only_last_labeled,
                "dataset": args.dataset,
                "test": args.test,
                "al_algorithm": args.al_algorithm,
                "full_res": args.full_res,
                "region_size": args.region_size,
            }

            train_loader, train_set, val_loader, candidate_set = get_data(**kwargs_data)

            ####------ Create loss ------####
            criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.ignore_label).cuda()

            ####------ Create optimizers (and load them if necessary) ------####
            kwargs_load_opt = {
                "net": net,
                "opt_choice": args.optimizer,
                "lr": args.lr,
                "wd": args.weight_decay,
                "momentum": args.momentum,
                "ckpt_path": args.ckpt_path,
                "exp_name_toload": args.exp_name_toload,
                "exp_name": args.exp_name,
                "snapshot": args.snapshot,
                "checkpointer": args.checkpointer,
                "load_opt": args.load_opt,
                "policy_net": None,
                "lr_dqn": args.lr_dqn,
                "al_algorithm": args.al_algorithm,
            }
            optimizer, _ = create_and_load_optimizers(**kwargs_load_opt)
            scheduler = ExponentialLR(optimizer, gamma=args.gamma)

            net.train()

            # Save final policy network with the best accuracy on the validation set
            if val_acc_episode > best_val_episode:
                best_val_episode = val_acc_episode
                torch.save(
                    policy_net.cpu().state_dict(),
                    os.path.join(args.ckpt_path, args.exp_name, "policy_best_jaccard_val.pth"),
                )
                policy_net.cuda()
        ## [ --- End of episode iteration --- ] ##
    # TODO I haven't looked at TEST yet. Should be pretty similar to Train though.
    #####################################################################
    ################################ TEST ########################
    #####################################################################
    if args.test:
        print("Starting test...")
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
        schedulerP = None

        # We are TESTING the DQN, but we still train the segmentation network
        net.train()

        # Load regions already labeled so far
        list_existing_images = []
        if os.path.isfile(os.path.join(args.ckpt_path, args.exp_name, "labeled_set_0.txt")):
            file = open(os.path.join(args.ckpt_path, args.exp_name, "labeled_set_0.txt"), "r")
            lab_set = file.read()
            lab_set = lab_set.split("\n")
            for elem in lab_set:
                if not elem == "":
                    paths = elem.split(",")
                    list_existing_images.append((int(paths[0]), int(paths[1]), int(paths[2])))
                    train_set.add_index(int(paths[0]), (int(paths[1]), int(paths[2])))

        print("-----Evaluating policy network -------")
        # Get log file
        logger, best_record, curr_epoch = get_logfile(
            args.ckpt_path,
            args.exp_name,
            args.checkpointer,
            args.snapshot,
            log_name="log.txt",
            num_classes=train_set.num_classes,
        )
        ## Initialize budget
        budget_reached = False

        if get_training_stage(args) is None:
            set_training_stage(args, "")

        # Choose candidate pool
        num_regions = args.num_each_iter * args.rl_pool
        num_groups = args.num_each_iter

        if get_training_stage(args) == "":
            candidates = train_set.get_candidates(num_regions_unlab=num_regions)
            candidate_set.reset()
            # Test adding a list of candidate images
            candidate_set.add_index(list(candidates))
            # Choose candidate pool
            region_candidates = get_region_candidates(
                candidates, train_set, num_regions=num_regions
            )
            # Compute state. Shape: [num_regions, dimensions state, w, h] Wanted: [group_size, num regions, dim, w,h]
            current_state, region_candidates = compute_state(
                args,
                net,
                region_candidates,
                candidate_set,
                train_set,
                num_groups=num_groups,
                reg_sz=args.region_size,
            )

        sel_act = False

        if (
            train_set.get_num_labeled_regions() >= args.budget_labels
            and (get_training_stage(args) == "trained")
            or "final_train" in get_training_stage(args)
        ):
            budget_reached = True

        while not budget_reached:
            # Select and perform an action. The action is the index of the 'candidates' image to label
            if (
                get_training_stage(args) == ""
                or (get_training_stage(args) == "trained" and sel_act)
                or (
                    get_training_stage(args).split("-")[0] == "computed"
                    if get_training_stage(args) != None
                    else True
                )
            ):
                action, steps_done, chosen_stats = select_action(
                    args, policy_net, current_state, 0, test=True
                )
                list_existing_images = add_labeled_datapoint(
                    args,
                    list_existing_images=list_existing_images,
                    region_candidates=region_candidates,
                    train_set=train_set,
                    action_list=action,
                    budget=args.budget_labels,
                    n_ep=0,
                )
                set_training_stage(args, "added")

            # Train network for regression with selected images:
            print("Train network with selected images...")
            if get_training_stage(args) == "added":
                tr_iu, vl_iu, _ = train_seq_model(
                    args,
                    0,
                    train_loader,
                    net,
                    criterion,
                    optimizer,
                    val_loader,
                    best_record,
                    logger,
                    scheduler,
                    schedulerP,
                )
                set_training_stage(args, "trained")

            if get_training_stage(args) == "trained":
                if train_set.get_num_labeled_regions() < args.budget_labels:
                    candidates = train_set.get_candidates(num_regions_unlab=num_regions)
                    candidate_set.reset()
                    # Test adding a list of candidate images
                    candidate_set.add_index(list(candidates))
                    # Choose candidate pool
                    region_candidates = get_region_candidates(
                        candidates, train_set, num_regions=num_regions
                    )
                    # Compute state. Shape: [num_regions, dimensions state, w, h] Wanted: [group_size, num regions,
                    # dim, w,h]
                    next_state, region_candidates = compute_state(
                        args,
                        net,
                        region_candidates,
                        candidate_set,
                        train_set,
                        num_groups=num_groups,
                        reg_sz=args.region_size,
                    )
                    # Move to the next state
                    current_state = next_state
                    del next_state
                    sel_act = True
                else:
                    next_state = None
                    budget_reached = True

        if terminal:
            if args.only_last_labeled:
                train_set.end_al = True
            print("Training with all regions.")
            args.epoch_num = 1000

            # Train until convergence
            _, val_acc_episode, _ = train_seq_model(
                args,
                curr_epoch,
                train_loader,
                net,
                criterion,
                optimizer,
                val_loader,
                best_record,
                logger,
                scheduler,
                schedulerP,
                final_train=True,
            )

        # End of budget
        logger.close()

        ## Test with test set. Getting final performance number on the test set. ##
        if args.final_test:
            final_test(args, net, criterion)


# TODO Update sequence model training
def train_seq_model(
    args,
    curr_epoch,
    train_loader,
    net,
    criterion,
    optimizer,
    val_loader,
    best_record,
    logger,
    scheduler,
    schedulerP,
    final_train=False,
):

    for epoch in range(curr_epoch, args.epoch_num):
        print("Epoch %i /%i" % (epoch, args.epoch_num + 1))
        train_loss = train(train_loader, net, criterion, optimizer)

        val_loss, rmse, best_record = validate(
            val_loader, net, criterion, optimizer, epoch, best_record, args
        )

        if final_train:
            scheduler.step()
            scheduler.step()
            if schedulerP is not None:
                schedulerP.step()

            ## Append info to logger

            info = [
                epoch,
                optimizer.param_groups[0]["lr"],
                train_loss,
                val_loss,
            ]

            logger.append(info)

    return train_loss, val_loss, rmse


if __name__ == "__main__":
    ####------ Parse arguments from console  ------####
    args = parser.get_arguments()
    main(args)
