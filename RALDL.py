import copy
import logging
import numpy as np
import os
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import math
from inc_net import SimpleVitNet
from utils.toolkit import target2onehot, tensor2numpy, accuracy

num_workers = 8

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments=[]
        self._network = None

        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        acc_total,grouped = self._evaluate(y_pred, y_true)
        return acc_total,grouped,y_pred[:,0],y_true

    def _eval_cnn(self, loader):
        import time
        self._network.eval()
        y_pred, y_true = [], []
        total_time = 0
        total_images = 0

        #time inference
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(tqdm(loader, desc="Inference timing")):
                inputs = inputs.to(self._device)
                batch_size = inputs.size(0)
                total_images += batch_size

                if total_images == 0 and torch.cuda.is_available():
                    _ = self._network(inputs)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()

                    outputs = self._network(inputs)["logits"]

                    end_event.record()
                    torch.cuda.synchronize()
                    batch_time = start_event.elapsed_time(end_event)  
                else:
                    start_time = time.time()
                    outputs = self._network(inputs)["logits"]
                    batch_time = (time.time() - start_time) * 1000 

                total_time += batch_time

                predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.cpu().numpy())

        avg_time = total_time / total_images
        print(f"✅ Average inference time per image: {avg_time:.2f} ms ({avg_time/1000:.4f} s)")

        return np.concatenate(y_pred), np.concatenate(y_true)

    
    def _evaluate(self, y_pred, y_true):
        ret = {}
        acc_total,grouped = accuracy(y_pred.T[0], y_true, self._known_classes,self.class_increments)
        return acc_total,grouped 
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if args["model_name"]=='adapter' and '_adapter' not in args["convnet_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')
        
        self._network = SimpleVitNet(args, True)
        self._batch_size= args["batch_size"]
        
        self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8

        self.args=args

    def after_task(self):
        self._known_classes = self._classes_seen_so_far
    
    def replace_fc(self,trainloader):
        self._network = self._network.eval()

        if self.args['use_RP']:
            self._network.fc.use_RP=True
            if self.args['M']>0:
                self._network.fc.W_rand=self.W_rand
                self._network.fc.down_proj=self.down_proj
                self._network.fc.up_proj = self.up_proj
                self._network.fc.act = self.act
            else:
                self._network.fc.W_rand=None

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.cuda()
                label=label.cuda()
                embedding = self._network.convnet(data)
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        Y=target2onehot(label_list,self.total_classnum)
        if self.args['use_RP']:
            if self.args['M']>0:
                Features_h=torch.nn.functional.relu(Features_f@ self._network.fc.W_rand.cpu())
                #lrp
                Features_f = Features_f.to(self._device)
                Features_h = Features_h.to(self._device)
            
                Features_lrp = torch.nn.functional.relu(self._network.fc.up_proj(self._network.fc.act(self._network.fc.down_proj(Features_f))))
                Features_h = Features_h + Features_lrp
            
            else:
                Features_h=Features_f
                
            Features_h = Features_h.cpu()
            
            self.Q=self.Q+Features_h.T @ Y 
            self.G=self.G+Features_h.T @ Features_h
            ridge=self.optimise_ridge_parameter(Features_h,Y)
            Wo=torch.linalg.solve(self.G+ridge*torch.eye(self.G.size(dim=0)),self.Q).T 
            self._network.fc.weight.data=Wo[0:self._network.fc.weight.shape[0],:].to(device='cuda')
        else:
            for class_index in np.unique(self.train_dataset.labels):
                data_index=(label_list==class_index).nonzero().squeeze(-1)
                if self.is_dil:
                    class_prototype=Features_f[data_index].sum(0)
                    self._network.fc.weight.data[class_index]+=class_prototype.to(device='cuda') 
                else:
                    class_prototype=Features_f[data_index].mean(0)
                    self._network.fc.weight.data[class_index]=class_prototype 

    def optimise_ridge_parameter(self,Features,Y):
        ridges=10.0**np.arange(-8,9)
        num_val_samples=int(Features.shape[0]*0.8)
        losses=[]
        Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
        G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T 
            Y_train_pred=Features[num_val_samples::,:]@Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]).detach().cpu().numpy())
        ridge=ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: "+str(ridge))
        return ridge
    
    def incremental_train(self, data_manager):
        with torch.cuda.device(2):
            self.total_classnum = data_manager.get_total_classnum()
            self._cur_task += 1
            self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(self._cur_task)
            if self.args['use_RP']:
                del self._network.fc
                self._network.fc=None
            self._network.update_fc(self._classes_seen_so_far) 
            if self.is_dil == False:
                logging.info("Starting CIL Task {}".format(self._cur_task+1))
            logging.info("Learning on classes {}-{}".format(self._known_classes, self._classes_seen_so_far-1))
            self.class_increments.append([self._known_classes, self._classes_seen_so_far-1])
            self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train", mode="train", )
            self.train_loader = DataLoader(self.train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=num_workers)
            train_dataset_for_CPs = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train", mode="test", )
            self.train_loader_for_CPs = DataLoader(train_dataset_for_CPs, batch_size=self._batch_size, shuffle=True, num_workers=num_workers)
            test_dataset = data_manager.get_dataset(np.arange(0, self._classes_seen_so_far), source="test", mode="test" )
            self.test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=num_workers)
            if len(self._multiple_gpus) > 1:
                print('Multiple GPUs')
                self._network = nn.DataParallel(self._network, self._multiple_gpus)
            self._train(self.train_loader, self.test_loader, self.train_loader_for_CPs)
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module

    def freeze_backbone(self,is_first_session=False):
        # Freeze the parameters for ViT.
        if 'vit' in self.args['convnet_type']:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "head." not in name and "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False
        else:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False

    def show_num_params(self,verbose=False):
        # show total parameters and trainable parameters
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} training parameters.')
        if total_params != total_trainable_params and verbose:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())

    def _train(self, train_loader, test_loader, train_loader_for_CPs):
        with torch.cuda.device(2):
            self._network.to(self._device)
            self._network.fc._cur_task = self._cur_task
            
            if self._cur_task == 0 and self.dil_init==False:
                self.show_num_params()
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
                scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
                #train the PETL method for the first task:
                logging.info("Starting PETL training on first task using "+self.args["model_name"]+" method")
                self._init_train(train_loader, test_loader, optimizer, scheduler)
                self.freeze_backbone()
                
                torch.save(self._network.state_dict(), "model_task0.pth")
                logging.info("Model saved after task 0.")
            
                if self.args['use_RP'] and self.dil_init==False:
                    self.setup_RP()
                
                lrp_params = (
                    list(self._network.fc.down_proj.parameters()) +
                    list(self._network.fc.up_proj.parameters()) +
                    (list(self._network.fc.act.parameters()) if isinstance(self._network.fc.act, nn.Module) else []) 
                )
                optimizer = optim.SGD(lrp_params, momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
                scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=0.01)
                self.show_num_params()
                self._init_train(train_loader, test_loader, optimizer, scheduler)
                
                
            if self.is_dil and self.dil_init==False:
                self.dil_init=True
                self._network.fc.weight.data.fill_(0.0)
            self.replace_fc(train_loader_for_CPs)
            self.show_num_params()
        
    
    def setup_RP(self):
        self.initiated_G=False
        self._network.fc.use_RP=True
        if self.args['M']>0:
            M=self.args['M']
            self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(device='cuda')) 
            self._network.fc.reset_parameters()
            self._network.fc.W_rand=torch.randn(self._network.fc.in_features,M).to(device='cuda')
            
            self.W_rand=copy.deepcopy(self._network.fc.W_rand) 
            
            # LRP 
            r = 64  
            dim = self._network.fc.in_features
            self._network.fc.down_proj = nn.Linear(dim, r).to(device='cuda')
            self._network.fc.up_proj = nn.Linear(r, M).to(device='cuda')
            self._network.fc.act = nn.GELU().to(device='cuda')
            self.down_proj=copy.deepcopy(self._network.fc.down_proj)
            self.up_proj=copy.deepcopy(self._network.fc.up_proj)
            self.act=copy.deepcopy(self._network.fc.act)
            
        else:
            M=self._network.fc.in_features 
        self.Q=torch.zeros(M,self.total_classnum)
        self.G=torch.zeros(M,M)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)
        
    

   