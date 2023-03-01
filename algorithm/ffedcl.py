from utils import fmodule
from .fedbase import BasicServer, BasicClient
import numpy as np
import numpy as np
from utils import fmodule
import copy
from multiprocessing import Pool as ThreadPool
from main import logger
import os
import utils.fflow as flw
import utils.network_simulator as ns
import torch
from torch import nn
import torch.nn.functional as F
import utils.fmodule as fmod
import math
from scipy.spatial.distance import cdist
import heapq

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.con_lr = option['con_lr']
        self.con_n = option['con_n']
        self.con_topnum = option['con_topnum']
        self.con_th = option['con_th']
        self.con_th_num = 0
        self.con_th_state = False
        self.isfedservercon7 = True

    def iterate(self, t):
        # sample clients
        selected_clients = self.sample()
        # training
        ws, losses = self.communicate(selected_clients)
        
        # aggregate
        # self.model = self.aggregate(ws, p= 1.0 / len(selected_clients))
        self.model = self.aggregate(ws, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
       
        ws_loss_dict = {}
        for i in range(self.clients_per_round):
            ws_loss_dict[ws[i]]=losses[i]
        sorted_ws = sorted(ws_loss_dict.items(), key=lambda x: x[1])

        loss_avg = np.mean(losses)
        posi_w, nega_w = [],[]

        for i in range(int(self.clients_per_round * 1)):
            if sorted_ws[i][1] > loss_avg:
                posi_w.append(list(sorted_ws[i])[0])
            elif sorted_ws[i][1] <= loss_avg:
                nega_w.append(list(sorted_ws[i])[0])

        if(self.isfedservercon7==True):
            
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.con_lr)
            alpha = torch.tensor(0.0,requires_grad=False).float().to(device)
            top_num = self.con_topnum

            final_loss = 0
    

            for j in range(len(posi_w)):
                list2 = list(posi_w[j].parameters())
                list2[-2] = list2[-2].flatten()
                max_number = heapq.nlargest(top_num, list(map(abs, list2[-2]))) 
            
                for index in range(len(list2[-2])):
                    if abs(list2[-2][index]) < max_number[top_num-1]:
                        with torch.no_grad():
                            list2[-2][index] = 0

            for j in range(len(nega_w)):
                list2 = list(nega_w[j].parameters())
                list2[-2] = list2[-2].flatten()
                max_number = heapq.nlargest(top_num, list(map(abs, list2[-2]))) 
            
                for index in range(len(list2[-2])):
                    if abs(list2[-2][index]) < max_number[top_num-1]:
                        with torch.no_grad():
                            list2[-2][index] = 0

            for i in range(self.con_n):
                posid2, negad2 = torch.tensor(0.0,requires_grad=True).float().to(device), torch.tensor(0.0,requires_grad=True).float().to(device)
            
                list1 = list(self.model.parameters())
                list1[-2] = list1[-2].flatten()
        
                max_number = heapq.nlargest(top_num, list(map(abs, list1[-2]))) 
            
                for index in range(len(list1[-2])):
                    if abs(list1[-2][index]) < max_number[top_num-1]:
                        with torch.no_grad():
                            list1[-2][index] = 0

                for j in range(len(posi_w)):
                    list2 = list(posi_w[j].parameters())
                    list2[-2] = list2[-2].flatten()
                    cos=torch.nn.CosineSimilarity(dim=-1)
                    cos1 = cos(list1[-2], list2[-2])

                    print(cos1)

                    posid2 = posid2 + (1 - cos1)
        
                posid2 = posid2 / len(posi_w)

                print('----------------')

                for j in range(len(nega_w)):
                    list2 = list(nega_w[j].parameters())
                    list2[-2] = list2[-2].flatten()
                    cos=torch.nn.CosineSimilarity(dim=-1)
                    cos1 = cos(list1[-2], list2[-2])
                    print(cos1)
                    negad2 = negad2 + (1 - cos1)
        
                negad2 = negad2 / len(nega_w)

                print('-------------')
                loss = max(posid2-negad2+alpha, torch.tensor(0.0,requires_grad=True).to(device)).float().to(device)
                final_loss = loss
                print(posid2)
                print(negad2)
                print(loss)
                print('=================')
                loss.requires_grad_(True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if self.con_th_state == True:
                if final_loss < self.con_th and final_loss >0:
                    self.con_th_num = self.con_th_num + 1
                    if self.con_th_num == 3:
                        self.isfedservercon7 = False
                else:
                    self.con_th_state = False
                    self.con_th_num == 0
            
            else:
                if final_loss < self.con_th and final_loss >0:
                    self.con_th_num = self.con_th_num + 1
                    self.con_th_state = True

        return selected_clients

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)



    