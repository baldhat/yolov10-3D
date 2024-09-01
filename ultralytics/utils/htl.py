import torch

class Hierarchical_Task_Learning:
    def __init__(self, stat_epoch_nums=5, max_epochs=200):
        self.stat_epoch_nums = stat_epoch_nums
        self.max_epochs = max_epochs
        self.past_losses = []
        self.loss_graph = {'bbox_om': [],
                           'cls_om': [],
                           'dep_om': [0, 4],  # bbox_om, s3d_om
                           'o3d_om': [0], # bbox_om
                           's3d_om': [0], # bbox_om
                           'hd_om': [0], # bbox_om

                           'bbox_oo': [],
                           'cls_oo': [],
                           'dep_oo': [6, 10],  # 'bbox_oo', 's3d_oo'
                           'o3d_oo': [6], # 'bbox_oo'
                           's3d_oo': [6], # 'bbox_oo'
                           'hd_oo': [6],  # 'bbox_oo'
                           }

    def compute_weight(self, current_loss, epoch):
        T = self.max_epochs
        # compute initial weights
        loss_weights = torch.zeros((len(self.loss_graph.keys())), device=current_loss.device)
        eval_loss_input = current_loss
        for i, term in enumerate(self.loss_graph):
            if len(self.loss_graph[term]) == 0:
                loss_weights[i] = torch.tensor(1.0).to(current_loss[i].device)
            else:
                loss_weights[i] = torch.tensor(0.0).to(current_loss[i].device)
                # update losses list
        if len(self.past_losses) == self.stat_epoch_nums:
            past_loss = torch.stack(self.past_losses).squeeze(1)
            mean_diff = (past_loss[:-2] - past_loss[2:]).mean(0)
            if not hasattr(self, 'init_diff'):
                self.init_diff = mean_diff
            c_weights = 1 - (mean_diff / self.init_diff).relu().unsqueeze(0)

            time_value = min(((epoch - 5) / (T - 5)), 1.0)
            for i, current_topic in enumerate(self.loss_graph.keys()):
                if len(self.loss_graph[current_topic]) != 0:
                    control_weight = 1.0
                    for pre_topic_idx in self.loss_graph[current_topic]:
                        control_weight *= c_weights[0][pre_topic_idx]
                    loss_weights[i] = time_value ** (1 - control_weight)
                    if loss_weights[i] != loss_weights[i]:
                        for pre_topic in self.loss_graph[i]:
                            print('NAN===============', time_value, control_weight,
                                  c_weights[0][pre_topic], pre_topic, pre_topic)
            # pop first list
            self.past_losses.pop(0)
        self.past_losses.append(eval_loss_input.unsqueeze(0))
        print("Loss Weights:", "\t".join([f"{list(self.loss_graph.keys())[i]}: {x.item()}" for i, x in enumerate(loss_weights)]))
        return ((loss_weights / loss_weights.sum()) * 6.0).detach()

    def update_e0(self, eval_loss):
        self.epoch0_loss = eval_loss #torch.cat([_.unsqueeze(0) for _ in eval_loss.values()]).unsqueeze(0)
