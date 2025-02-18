import argparse
import os, sys
import numpy as np
import copy
import time
import random

from models.config import config
from models.DPKNet import DPKNet as Model
from datasets_setting.h36m import Datasets as H36MDataset
from datasets_setting.h36m_eval import Datasets as H36MEval

from test_H36M import results_keys
from test_H36M import test

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default=None, help='=exp name')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--iter', type=int, default=40000, help='=iter number')
parser.add_argument('--similar_weight', type=float, default=0., help='=the weight of similar calculate')
parser.add_argument('--distill_layer_num', type=int, default=24, help='=num of blocks to distill posterior knowledge')
args = parser.parse_args()

config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.num_layers = args.num
config.cos_lr_total_iters=args.iter
config.node_num=22

# setting the weight of similar calculate
config.similar_weight=args.similar_weight

# setting ernum of blocks to distill posterior knowledge
config.motion_mlp.distill_layer_num=args.distill_layer_num

current_train_id = time.strftime("%Y_%m_%d_%HH_%MM_%SS", time.localtime())
script_name = os.path.basename(sys.argv[0])[:-3]
exp_name="{}_{}.txt".format(script_name,current_train_id)
log_name = '{}_bs{}_num{}_similar_weight{}_distill_layer_num{}_{}'.format(script_name,config.batch_size,config.motion_mlp.num_layers,config.similar_weight,args.distill_layer_num,current_train_id)
ckpt = os.path.join(config.checkpoint, log_name)
if not os.path.isdir(ckpt):
    os.makedirs(ckpt)

log_dir=os.path.join(ckpt,exp_name)
acc_log = open(log_dir, 'a')
writer = SummaryWriter()

for key in config :
    print(key,":",config[key])
    acc_log.write(''.join(str(key)+' : '+ str(config[key]) + '\n'))

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

dct_m,idct_m = get_dct_matrix(config.motion.input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer) :
    if nb_iter > 30000:
        current_lr = 1e-5
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm

def train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr) :

    # input
    h36m_motion_input_ = h36m_motion_input.clone()
    h36m_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.input_length], h36m_motion_input_.cuda())

    b, n, c = h36m_motion_target.shape
    h36m_motion_target = h36m_motion_target.cuda().reshape(b, n, 22, 3).reshape(-1, 3)

    # straining step 1
    # normal output
    motion_pred,Posterior_Knowledge = model(h36m_motion_input_.cuda())
    motion_pred = torch.matmul(idct_m[:, :config.motion.input_length, :], motion_pred)
    # prepare final prediction
    offset = h36m_motion_input[:, -1:].cuda()
    motion_pred = motion_pred[:, :config.motion.target_length] + offset
    loss = torch.mean(torch.norm(motion_pred.reshape(b, n, 22, 3).reshape(-1, 3) - h36m_motion_target, 2, 1))

    motion_pred = motion_pred.reshape(b, n, 22, 3)
    dmotion_pred = gen_velocity(motion_pred)
    motion_gt = h36m_motion_target.reshape(b, n, 22, 3)
    dmotion_gt = gen_velocity(motion_gt)
    dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1, 3), 2, 1))
    loss = loss + dloss

    # #similarity
    Posterior_Knowledge=torch.matmul(idct_m[:, :config.motion.input_length, :], Posterior_Knowledge)
    Posterior_Knowledge = Posterior_Knowledge[:, :config.motion.target_length] + offset
    similarity = torch.mean(torch.norm(motion_pred.reshape(b, n, 22, 3).reshape(-1, 3) - h36m_motion_target - Posterior_Knowledge.reshape(b, n, 22, 3).reshape(-1, 3), 2, 1))
    loss = loss + config.similar_weight*similarity

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # straining step 2
    # normal output
    enhanced_motion_pred, Posterior_Knowledge = model(h36m_motion_input_.cuda())
    enhanced_motion_pred = enhanced_motion_pred+Posterior_Knowledge
    enhanced_motion_pred = torch.matmul(idct_m[:, :config.motion.input_length, :], enhanced_motion_pred)
    # prepare final prediction
    enhanced_motion_pred = enhanced_motion_pred[:, :config.motion.target_length] + offset
    loss = torch.mean(torch.norm(enhanced_motion_pred.reshape(b, n, 22, 3).reshape(-1, 3) - h36m_motion_target, 2, 1))

    enhanced_motion_pred = enhanced_motion_pred.reshape(b, n, 22, 3)
    denhanced_motion_pred = gen_velocity(enhanced_motion_pred)
    dloss = torch.mean(torch.norm((denhanced_motion_pred - dmotion_gt).reshape(-1, 3), 2, 1))
    loss = loss + dloss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)


    return loss.item(), optimizer, current_lr

model = Model(config)
model.train()
model.cuda()

config.motion.target_length = config.motion.target_length_train
dataset = H36MDataset(config,split=0, data_aug=config.data_aug)

shuffle = True
sampler = None
dataloader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, sampler=sampler, shuffle=shuffle, pin_memory=True)


actions = ["walking", "eating", "smoking", "discussion", "directions",
           "greeting", "phoning", "posing", "purchases", "sitting",
           "sittingdown", "takingphoto", "waiting", "walkingdog",
           "walkingtogether"]
eval_config = copy.deepcopy(config)
eval_config.motion.target_length = eval_config.motion.target_length_eval
shuffle = False

test_data = dict()
for act in actions:
    test_dataset = H36MEval(eval_config,actions=[act],split=2, data_aug=False)
    test_data[act] = DataLoader(test_dataset, batch_size=128,num_workers=0, drop_last=False,sampler=sampler, shuffle=shuffle, pin_memory=True)

# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),  lr=config.cos_lr_max,  weight_decay=config.weight_decay)

##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.

while (nb_iter + 1) < config.cos_lr_total_iters:


    for (h36m_motion_input, h36m_motion_target) in dataloader:

        loss, optimizer, current_lr = train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
        avg_loss += loss
        avg_lr += current_lr
        # print('training.............')
        if (nb_iter + 1) % config.print_every ==  0 :
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every
            print('iter:{}   loss:{}'.format(nb_iter + 1, avg_loss))
            acc_log.write('iter:' + ''.join(str(nb_iter + 1) + '    loss:' + str(avg_loss) + '\n'))


        if (((nb_iter + 1)<25000) &((nb_iter + 1) % (config.save_every*5) == 0))|(((nb_iter + 1)>25000) &((nb_iter + 1) % (config.save_every*5) == 0)):
            print('iter:', nb_iter+1)
            model.eval()
            errs = np.zeros([len(actions) + 1, len(results_keys)])
            for i, act in enumerate(actions):
                errs[i] = test(eval_config, model, test_data[act])
                # print('act:',act)
                # print(errs[i])
            errs[-1] = np.mean(errs[:-1], axis=0)
            print("Mean Error:",errs[-1])
            acc_log.write(''.join(str(nb_iter + 1) + '\n'))

            line = ''
            err_name=''
            for ii in errs[-1]:
                line += str('{:.3f}'.format(ii)) + ' '
                err_name += str('{:.3f}'.format(ii)) + '_'
            line += '\n'
            err_name = err_name[:-1]
            acc_log.write(''.join(line))
            torch.save(model.state_dict(), config.checkpoint + '/' + log_name+ '/' + 'iter-' + str(nb_iter + 1) +'-AverageEror:'+err_name+ '.pth')
            model.train()

        if (nb_iter + 1) == config.cos_lr_total_iters :
            break
        nb_iter += 1

writer.close()
