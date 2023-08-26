import math
import torch


import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from pdb import set_trace as stop

from monitor_model import CLASSIFIER
from .utils import Backbone,positionalencoding2d,SelfAttnLayer,weights_init,custom_replace
from .evaluate import *

from typing import Type, Any, Callable, Union, List, Optional

@CLASSIFIER.register_module('CTran')
class CTran(nn.Module):
    def __init__(self,
                 num_classes,
                 use_lmt,
                 pos_emb=False,
                 layers=3,
                 heads=4,
                 dropout=0.1,
                 int_loss=0,
                 no_x_features=False,
                **kwargs: Any):
        super(CTran, self).__init__()
        self.use_lmt = use_lmt
        self.num_classes = num_classes
        self.no_x_features = no_x_features # (for no image features)

        # ResNet backbone
        self.backbone = Backbone()
        hidden = 2048 # this should match the backbone output feature size

        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(hidden,hidden,(1,1))
        
        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_classes)).view(1,-1).long()
        self.label_lt = torch.nn.Embedding(num_classes, hidden, padding_idx=None)

        # State Embeddings
        self.known_label_lt = torch.nn.Embedding(3, hidden, padding_idx=0)

        # Position Embeddings (for image features)
        self.use_pos_enc = pos_emb
        if self.use_pos_enc:
            # self.position_encoding = PositionEmbeddingSine(int(hidden/2), normalize=True)
            self.position_encoding = positionalencoding2d(hidden, 18, 18).unsqueeze(0)

        # Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden,heads,dropout) for _ in range(layers)])

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        self.output_linear = torch.nn.Linear(hidden,num_classes)

        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_lt.apply(weights_init)
        self.known_label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)
        self.step = 0
    #     self.args = get_cls_args()

    # def get_cls_args(self):
    #     return 
        
    def forward(self, images, mask):
        const_label_input = self.label_input.repeat(images.size(0),1).cuda()
        init_label_embeddings = self.label_lt(const_label_input)

        features = self.backbone(images)
        
        if self.downsample:
            features = self.conv_downsample(features)
        if self.use_pos_enc:
            pos_encoding = self.position_encoding(features,torch.zeros(features.size(0),18,18, dtype=torch.bool).cuda())
            features = features + pos_encoding

        features = features.view(features.size(0),features.size(1),-1).permute(0,2,1) 

        if self.use_lmt:
            # Convert mask values to positive integers for nn.Embedding
            label_feat_vec = custom_replace(mask,0,1,2).long()

            # Get state embeddings
            state_embeddings = self.known_label_lt(label_feat_vec)

            # Add state embeddings to label embeddings
            init_label_embeddings += state_embeddings
        
        if self.no_x_features:
            embeddings = init_label_embeddings 
        else:
            # Concat image and label embeddings
            embeddings = torch.cat((features,init_label_embeddings),1)

        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)        
        attns = []
        for layer in self.self_attn_layers:
            embeddings,attn = layer(embeddings,mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:,-init_label_embeddings.size(1):,:]
        output = self.output_linear(label_embeddings) 
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0),1,1).cuda()
        output = (output*diag_mask).sum(-1)

        return output,None,attns

    def _train(self, 
               data, 
               args, 
               optimizer, 
               writer,
               scheduler = None
               ):
        
        self.train()
        optimizer.zero_grad()
        # pre-allocate full prediction and target tensors
        # all_predictions = torch.zeros(len(data.dataset), self.num_classes).cpu()
        # all_targets = torch.zeros(len(data.dataset), self.num_classes).cpu()
        # all_masks = torch.zeros(len(data.dataset), self.num_classes).cpu()
        # # all_image_ids = []

        # batch_idx = 0
        # loss_total = 0
        # unk_loss_total = 0

        for batch in tqdm(data, mininterval=0.5, leave=False, ncols=50):

            labels = batch['labels'].float()
            images = batch['image'].float()
            mask = batch['mask'].float()
            unk_mask = custom_replace(mask,1,0,0)
            # all_image_ids += batch['imageIDs']

            mask_in = mask.clone()
            pred,int_pred,attns = self(images.cuda(),mask_in.cuda())

            loss =  F.binary_cross_entropy_with_logits(pred.view(labels.size(0),-1),labels.cuda(),reduction='none')

            if self.use_lmt: 
                # only use unknown labels for loss
                loss_out = (unk_mask.cuda()*loss).sum()
            else: 
                # use all labels for loss
                loss_out = loss.sum() 

                # loss_out = loss_out/unk_mask.cuda().sum()
            loss_out.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            
            
            if (self.step+1) % args.train_param.log_interval_step == 0:
                # self.loggerInfo(loss)
                writer.add_scalar(f'train_loss', loss, self.step)


            ## Updates ##
            # loss_total += loss_out.item()
            # unk_loss_total += loss_out.item()
            # start_idx,end_idx=(batch_idx*data.batch_size),((batch_idx+1)*data.batch_size)

            # if pred.size(0) != all_predictions[start_idx:end_idx].size(0):
            #     pred = pred.view(labels.size(0),-1)

            # all_predictions[start_idx:end_idx] = pred.data.cpu()
            # all_targets[start_idx:end_idx] = labels.data.cpu()
            # all_masks[start_idx:end_idx] = mask.data.cpu()
            # batch_idx +=1

        # loss_total = loss_total/float(all_predictions.size(0))
        # unk_loss_total = unk_loss_total/float(all_predictions.size(0))

        # train_metrics = self.evaluate(all_predictions, all_targets, all_masks,loss_total,unk_loss_total)
    def _eval(self,
            data, 
            args, 
            ):
        self.eval()
        for batch in tqdm(data, mininterval=0.5, leave=False, ncols=50):

            labels = batch['labels'].float()
            images = batch['image'].float()
            mask = batch['mask'].float()
            unk_mask = custom_replace(mask,1,0,0)
            # all_image_ids += batch['imageIDs']

            mask_in = mask.clone()
            pred,int_pred,attns = self(images.cuda(),mask_in.cuda())

            #TODO 待补充


    def evaluate(args,
                 all_predictions,
                 all_targets,
                 all_masks,
                 loss,
                 loss_unk,
                 elapsed,
                 known_labels=0,
                 all_metrics=False,
                 verbose=True):
        all_predictions = F.sigmoid(all_predictions)

        if args.dataset =='cub':
            all_preds_concepts = all_predictions[:,0:112].clone()
            all_targets_concepts = all_targets[:,0:112].clone()
            all_preds_concepts[all_preds_concepts >= 0.5] = 1
            all_preds_concepts[all_preds_concepts < 0.5] = 0

            concept_accs = []
            for i in range(all_preds_concepts.size(1)): 
                concept_accs.append(metrics.accuracy_score(all_targets_concepts[:,i],all_preds_concepts[:,i]))
            concept_acc = np.array(concept_accs).mean()

            all_preds_classes = all_predictions[:,112:].clone()
            all_targets_classes = all_targets[:,112:].clone()
            pred_max_val,pred_max_idx = torch.max(all_preds_classes,1)
            _,target_max_idx = torch.max(all_targets_classes,1)

            class_acc = (pred_max_idx==target_max_idx).sum().item()/pred_max_idx.size(0)

        else:
            concept_acc = 0
            class_acc = 0
        

        unknown_label_mask = custom_replace(all_masks,1,0,0)


        if known_labels > 0:
            meanAP = custom_mean_avg_precision(all_targets,all_predictions,unknown_label_mask)
        else:
            meanAP = metrics.average_precision_score(all_targets,all_predictions, average='macro', pos_label=1)

        optimal_threshold = 0.5 

        all_targets = all_targets.numpy()
        all_predictions = all_predictions.numpy()

        top_3rd = np.sort(all_predictions)[:,-3].reshape(-1,1)
        all_predictions_top3 = all_predictions.copy()
        all_predictions_top3[all_predictions_top3<top_3rd] = 0
        all_predictions_top3[all_predictions_top3<optimal_threshold] = 0
        all_predictions_top3[all_predictions_top3>=optimal_threshold] = 1

        CP_top3 = metrics.precision_score(all_targets, all_predictions_top3, average='macro')
        CR_top3 = metrics.recall_score(all_targets, all_predictions_top3, average='macro')
        CF1_top3 = (2*CP_top3*CR_top3)/(CP_top3+CR_top3)
        OP_top3 = metrics.precision_score(all_targets, all_predictions_top3, average='micro')
        OR_top3 = metrics.recall_score(all_targets, all_predictions_top3, average='micro')
        OF1_top3 = (2*OP_top3*OR_top3)/(OP_top3+OR_top3)


        all_predictions_thresh = all_predictions.copy()
        all_predictions_thresh[all_predictions_thresh < optimal_threshold] = 0
        all_predictions_thresh[all_predictions_thresh >= optimal_threshold] = 1
        CP = metrics.precision_score(all_targets, all_predictions_thresh, average='macro')
        CR = metrics.recall_score(all_targets, all_predictions_thresh, average='macro')
        CF1 = (2*CP*CR)/(CP+CR)
        OP = metrics.precision_score(all_targets, all_predictions_thresh, average='micro')
        OR = metrics.recall_score(all_targets, all_predictions_thresh, average='micro')
        OF1 = (2*OP*OR)/(OP+OR)  

        acc_ = list(subset_accuracy(all_targets, all_predictions_thresh, axis=1, per_sample=True))
        hl_ = list(hamming_loss(all_targets, all_predictions_thresh, axis=1, per_sample=True))
        exf1_ = list(example_f1_score(all_targets, all_predictions_thresh, axis=1, per_sample=True))        
        acc = np.mean(acc_)
        hl = np.mean(hl_)
        exf1 = np.mean(exf1_)

        eval_ret = OrderedDict([('Subset accuracy', acc),
                            ('Hamming accuracy', 1 - hl),
                            ('Example-based F1', exf1),
                            ('Label-based Micro F1', OF1),
                            ('Label-based Macro F1', CF1)])


        ACC = eval_ret['Subset accuracy']
        HA = eval_ret['Hamming accuracy']
        ebF1 = eval_ret['Example-based F1']
        OF1 = eval_ret['Label-based Micro F1']
        CF1 = eval_ret['Label-based Macro F1']

        if verbose:
            print('loss:  {:0.3f}'.format(loss))
            print('lossu: {:0.3f}'.format(loss_unk))
            print('----')
            print('mAP:   {:0.1f}'.format(meanAP*100))
            print('----')
            print('CP:    {:0.1f}'.format(CP*100))
            print('CR:    {:0.1f}'.format(CR*100))
            print('CF1:   {:0.1f}'.format(CF1*100))
            print('OP:    {:0.1f}'.format(OP*100))
            print('OR:    {:0.1f}'.format(OR*100))
            print('OF1:   {:0.1f}'.format(OF1*100))
            if args.dataset in ['coco','vg']:
                print('----')
                print('CP_t3: {:0.1f}'.format(CP_top3*100))
                print('CR_t3: {:0.1f}'.format(CR_top3*100))
                print('CF1_t3:{:0.1f}'.format(CF1_top3*100))
                print('OP_t3: {:0.1f}'.format(OP_top3*100))
                print('OR_t3: {:0.1f}'.format(OR_top3*100))
                print('OF1_t3:{:0.1f}'.format(OF1_top3*100)) 

        metrics_dict = {}
        metrics_dict['mAP'] = meanAP
        metrics_dict['ACC'] = ACC
        metrics_dict['HA'] = HA
        metrics_dict['ebF1'] = ebF1
        metrics_dict['OF1'] = OF1
        metrics_dict['CF1'] = CF1
        metrics_dict['loss'] = loss
        metrics_dict['time'] = elapsed

        if args.dataset =='cub':
            print('Concept Acc:    {:0.3f}'.format(concept_acc))
            print('Class Acc:    {:0.3f}'.format(class_acc))
            metrics_dict['concept_acc'] = concept_acc
            metrics_dict['class_acc'] = class_acc

        print('')

        return metrics_dict