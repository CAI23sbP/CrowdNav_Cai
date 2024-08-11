
import random,torch, os
from collections import namedtuple, deque

class DQNBuffer():
    def __init__(self, policy_config ,train_config, device, predict_net, target_net, opt_lossf):
        self.memory = deque([], maxlen= train_config.train.capacity)
        tuples = {"Transition":("state","action","next_state","reward","done")}
        key = list(tuples.keys())[0]
        self.Transition = namedtuple(key,
                        tuples[f"{key}"])
        self.device = device
        self.setting_param(train_config)
        self.init_batch(policy_config)
        self.predict_net = predict_net
        self.target_net = target_net 
        self.optimizer, self.criterion = opt_lossf
        self.loss_fun = None

    def setting_param(self,train_config):
        self.batch_size = train_config.train.batch_size
        self.epoch = train_config.train.epoch
        self.gamma = train_config.train.gamma
        self.tau = train_config.train.tau
        self.target_policy_step = train_config.train.target_policy_step

    def add(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)

    def init_batch(self,policy_config):
        self.state_batch = torch.zeros(self.batch_size, policy_config.dqn.input_dim)
        self.action_batch = torch.zeros(self.batch_size, 1, dtype=torch.int64)
        self.next_state_batch = torch.zeros(self.batch_size, policy_config.dqn.input_dim)
        self.reward_batch = torch.zeros(self.batch_size)
        self.done_batch = torch.zeros(self.batch_size,dtype = torch.bool)
        self.policy_config = policy_config
        self.update_step = 0

    def extractor_loss(self,loss_fun):
        self.loss_fun = loss_fun

    def update(self,*args):
        self.add(*args)
        
        if len(self.memory) < self.batch_size:
            return
        
        for n in range(self.epoch):
            dataset = self.sample()
            for i, data in enumerate([*dataset]):
                self.state_batch[i] = torch.FloatTensor(data[0])
                self.action_batch [i] = torch.FloatTensor([data[1]])
                self.next_state_batch[i] = torch.FloatTensor(data[2])
                self.reward_batch [i] = torch.FloatTensor([data[3]])
                self.done_batch [i] = torch.BoolTensor([data[4]])

            predict_q_value = self.predict_net(self.state_batch).gather(1, self.action_batch)
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            with torch.no_grad():
                next_states = torch.cat([next_state for done, next_state in zip(self.done_batch, self.next_state_batch) if not done]).reshape(-1,self.policy_config.dqn.input_dim)
                next_state_values[~self.done_batch] = self.target_net(next_states).max(1)[0]
            
            expected_func = (next_state_values * self.gamma) + self.reward_batch
            loss = self.criterion(expected_func,predict_q_value.view(-1))
            

            self.loss_fun

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.predict_net.parameters(), 100)
            self.optimizer.step()

        self.update_step += 1
        if self.update_step % self.target_policy_step ==0:
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.predict_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key]*(1- self.tau)
            self.target_net.load_state_dict(target_net_state_dict)

    def save(self,path):
        path = os.path.join(path)
        torch.save(self.target_net.state_dict(), path)
    
    def load(self,path):
        path = os.path.join(path)
        self.predict_net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        self.target_net.load_state_dict(self.predict_net.state_dict())

class DDQNBuffer(DQNBuffer):
    def __init__(self,configs, predict_net, target_net, opt_lossf):
        super().__init__(configs, predict_net, target_net, opt_lossf)
    
    def init_batch(self):
        super().init_batch()

    def add(self,*args):
        super().add(*args)

    def sample(self):
        return super().sample()

    def __len__(self):
        return super().__len__()

    def update(self,*args):
        self.add(*args)

        if len(self.memory) < self.batch_size:
            return
        
        for n in range(self.epoch):
            dataset = self.sample()
            for i, data in enumerate([*dataset]):
                self.state_batch[i] = torch.FloatTensor(data[0])
                self.action_batch [i] = torch.FloatTensor([data[1]])
                self.next_state_batch[i] = torch.FloatTensor(data[2])
                self.reward_batch [i] = torch.FloatTensor([data[3]])
                self.done_batch [i] = torch.BoolTensor([data[4]])


            predict_q_value = self.predict_net(self.state_batch).gather(1, self.action_batch)
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            with torch.no_grad():
                next_states = torch.cat([next_state for done, next_state in zip(self.done_batch, self.next_state_batch) if not done]).reshape(-1,self.policy_config.dqn.input_dim)
                arg_max_a = self.predict_net(next_states).max(1)[1]
                next_state_values[~self.done_batch] = self.target_net(next_states).max(1)[0]
            
            expected_func = (next_state_values * arg_max_a).sum(1) *self.gamma + self.reward_batch
            loss = self.criterion(expected_func,predict_q_value.view(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.predict_net.parameters(), 100)
            self.optimizer.step()
    
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.predict_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key]*(1- self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def load(self):
        super().save()

    def load(self):
        super().save()
