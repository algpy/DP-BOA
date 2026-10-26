import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class DINOHead_(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, cls="random"):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # random
        print(cls)
        if cls == "random":
            self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        elif cls == "etf":
            self.last_layer = ETF_Classifier(in_dim, out_dim)
        else:
            raise NotImplementedError
        # # etf
        # self.last_layer = ETF_Classifier(in_dim, out_dim)
        # print("ETF classifier")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        return x_proj, logits
    

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm_last_layer=True):
        super().__init__()
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = nn.functional.normalize(x, dim=-1, p=2)
        logits = self.last_layer(nn.functional.normalize(x, dim=-1, p=2))
        return x, logits


class DINOHead_v2(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, cls="random", is_feature=True):
        super().__init__()
        # random
        print(cls)
        if cls == "random":
            self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        elif cls == "etf":
            self.last_layer = ETF_Classifier(in_dim, out_dim)
        elif cls == "learnable":
            self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
        else:
            raise NotImplementedError
        self.is_feature = is_feature
        # # etf
        # self.last_layer = ETF_Classifier(in_dim, out_dim)
        # print("ETF classifier")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        logits = self.last_layer(nn.functional.normalize(x, dim=-1, p=2))
        if self.is_feature:
            return x, logits
        else:
            return logits
    

class TeacherHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm_last_layer=True, middle_dim=768, nlayer=1):
        super().__init__()
        if nlayer == 0:
            self.transform = nn.Sequential(nn.ReLU())
        else:
            self.transform = nn.Sequential(nn.Linear(in_dim, middle_dim, bias=False), nn.ReLU())
            for i in range(1, nlayer):
                self.transform.add_module(str(i * 2), nn.Linear(middle_dim, middle_dim, bias=False))
                self.transform.add_module(str(i * 2 + 1), nn.ReLU())
        #self.transform = nn.Sequential(nn.ReLU(), )
        self.last_layer = nn.utils.weight_norm(nn.Linear(middle_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.transform(x)
        logits = self.last_layer(nn.functional.normalize(x, dim=-1, p=2))
        return x, logits
    
class StudentHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm_last_layer=True, teacher_dim=768, student_dim=768, nlayer=1):
        super().__init__()
        middle_dim = student_dim
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        if nlayer == 0:
            self.transform = nn.Sequential(nn.ReLU())
        else:
            self.transform = nn.Sequential(nn.Linear(in_dim, middle_dim, bias=False), nn.ReLU())
            for i in range(1, nlayer):
                self.transform.add_module(str(i * 2), nn.Linear(middle_dim, middle_dim, bias=False))
                self.transform.add_module(str(i * 2 + 1), nn.ReLU())
        #self.transform = nn.Sequential(nn.ReLU(), )
        self.last_layer = nn.utils.weight_norm(nn.Linear(middle_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.transform(x)
        x_norm = torch.zeros(x.shape).cuda()
        x_norm[:, :self.teacher_dim] = self.teacher_dim ** 0.5 * nn.functional.normalize(x[:, :self.teacher_dim], dim=-1, p=2).clone()
        x_norm[:, self.teacher_dim:] = (self.student_dim - self.teacher_dim) ** 0.5 * nn.functional.normalize(x[:, self.teacher_dim:], dim=-1, p=2).clone()
        logits = self.last_layer(nn.functional.normalize(x_norm, dim=-1, p=2))
        return x, logits
    
class ClassifierHead(nn.Module):
    def __init__(self, out_dim, norm_last_layer=True, middle_dim=768, nlayer=1):
        super().__init__()
        self.last_layer = nn.utils.weight_norm(nn.Linear(middle_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        logits = self.last_layer(nn.functional.normalize(x, dim=-1, p=2))
        return logits
    
class TeacherHead_vis(nn.Module):
    def __init__(self, in_dim, out_dim, norm_last_layer=True, middle_dim=768, nlayer=1):
        super().__init__()
        if nlayer == 0:
            self.transform = nn.Sequential(nn.ReLU())
        else:
            self.transform = nn.Sequential(nn.Linear(in_dim, middle_dim, bias=False), nn.ReLU())
            for i in range(1, nlayer):
                self.transform.add_module(str(i * 2), nn.Linear(middle_dim, middle_dim, bias=False))
                self.transform.add_module(str(i * 2 + 1), nn.ReLU())
        #self.transform = nn.Sequential(nn.ReLU(), )
        self.last_layer = nn.utils.weight_norm(nn.Linear(middle_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.transform(x)
        logits = self.last_layer(nn.functional.normalize(x, dim=-1, p=2))
        return x
    
class StudentHead_vis(nn.Module):
    def __init__(self, in_dim, out_dim, norm_last_layer=True, teacher_dim=768, student_dim=768, nlayer=1):
        super().__init__()
        middle_dim = student_dim
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        if nlayer == 0:
            self.transform = nn.Sequential(nn.ReLU())
        else:
            self.transform = nn.Sequential(nn.Linear(in_dim, middle_dim, bias=False), nn.ReLU())
            for i in range(1, nlayer):
                self.transform.add_module(str(i * 2), nn.Linear(middle_dim, middle_dim, bias=False))
                self.transform.add_module(str(i * 2 + 1), nn.ReLU())
        #self.transform = nn.Sequential(nn.ReLU(), )
        self.last_layer = nn.utils.weight_norm(nn.Linear(middle_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, inp):
        x = self.transform(inp)
        x_norm = torch.zeros(x.shape).cuda()
        x_norm[:, :self.teacher_dim] = self.teacher_dim ** 0.5 * nn.functional.normalize(x[:, :self.teacher_dim], dim=-1, p=2).clone()
        x_norm[:, self.teacher_dim:] = (self.student_dim - self.teacher_dim) ** 0.5 * nn.functional.normalize(x[:, self.teacher_dim:], dim=-1, p=2).clone()
        logits = self.last_layer(nn.functional.normalize(x_norm, dim=-1, p=2))
        return inp
    
class concept(nn.Module):
    def __init__(self, in_dim, out_dim, norm_last_layer=True):
        super().__init__()
        self.concepts = nn.Parameter(torch.zeros((768, 3, 224, 224)))

    def _init_weights(self, m):
        nn.init.kaiming_normal_(self.concepts)

    def forward(self, x):
        return self.concepts
        
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def info_nce_logits_v2(features, n_views=2, temperature=1.0, device='cuda', memory_feature=None, memory_mask_lab=None):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    if memory_feature is not None:
        memory_feature = F.normalize(memory_feature, dim=1)
        negatives = torch.matmul(features, memory_feature.T)
        
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1, num_labeled_classes=100, start_hard_epoch=10):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp = teacher_temp
        self.threshold = 0.8
        self.nlc = num_labeled_classes
        self.start_hard_epoch = start_hard_epoch

    def forward(self, student_output, teacher_output, epoch, prob=None):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        # temp = self.teacher_temp
        teacher_out = F.softmax(teacher_output / temp, dim=-1)

        # hard threshold
        if epoch >= self.start_hard_epoch and prob is not None:
            prob = prob.squeeze()
            confident_novel = (prob <= (1 - self.threshold)) & (teacher_out.argmax(dim=1) >= self.nlc)
            confident_known = (prob >= self.threshold) & (teacher_out.argmax(dim=1) < self.nlc)
            if confident_novel.sum() > 0:
                teacher_out[confident_novel] = (teacher_out[confident_novel] == teacher_out[confident_novel].max(dim=1, keepdim=True)[0]).float()
            if confident_known.sum() > 0:
                teacher_out[confident_known] = (teacher_out[confident_known] == teacher_out[confident_known].max(dim=1, keepdim=True)[0]).float()
        # end

        teacher_out = teacher_out.detach().chunk(self.ncrops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


class ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=False, LWS=False, reg_ETF=False):
        super(ETF_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))
        self.ori_M = M.cuda()
        self.LWS = LWS
        self.reg_ETF = reg_ETF

        if fix_bn:
            self.BN_H.weight.requires_grad = False
            self.BN_H.bias.requires_grad = False

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        # feat in has to be larger than num classes.
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-06), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def forward(self, x):
        # x = self.BN_H(x)
        # x = x / torch.clamp(
        #     torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        output = torch.matmul(x, self.ori_M)


        return output
    

# class Discriminator(nn.Module):
#     def __init__(self, input_dim=1024):
#         super().__init__()
#         self.mlp = nn.Sequential(nn.Linear(input_dim, 1024),
#                                  nn.ReLU(inplace=True),
#                                  nn.Linear(1024, 1024),
#                                  nn.ReLU(inplace=True),
#                                  nn.Linear(1024, 1))

#         self._reinit_all_layers()

#     def forward(self, x):
#         return self.mlp(x)

#     @torch.no_grad()
#     def _reinit_all_layers(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)


class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            layers = [nn.Linear(in_dim, bottleneck_dim)]
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(bottleneck_dim, out_dim, bias=False))

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # # etf
        # self.last_layer = ETF_Classifier(in_dim, out_dim)
        # print("ETF classifier")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)

                
def model_statistics(model):
    total_params = sum(param.numel() for param in model.parameters()) / 1000000.0
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0
    print('    Total params: %.2fM, trainable params: %.2fM' % (total_params, trainable_params))


class EMA():

    def __init__(self, ema_model, decay):

        self.decay = decay
        self.shadow = ema_model  # create a copy of the model
        self.shadow.eval()  # set the shadow model to evaluation mode

    def update(self, model):

        with torch.no_grad():

            # update the shadow parameters
            for shadow_param, param in zip(self.shadow.parameters(), model.parameters()):
                shadow_param.copy_(self.decay*shadow_param + (1.0-self.decay)*param)

    def set(self, model):

        with torch.no_grad():

            # copy the shadow parameters to the model
            for model_param, shadow_param in zip(model.parameters(), self.shadow.parameters()):
                model_param.copy_(shadow_param)

    def __call__(self, x):
        with torch.no_grad():
            return self.shadow(x)