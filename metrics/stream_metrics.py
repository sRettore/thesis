import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        pass

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def synch(self, device):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes, opts, classNames = None):
        super().__init__()
        self.where_to_sim = opts.where_to_sim
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.total_samples = 0
        self.classNames = classNames
        self.stringLenght = max(len(s) for s in classNames)
        
    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        self.total_samples += len(label_trues)

    def to_str(self, results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU" and k!="Class Prc" and k!="Class Acc" and k!="Confusion Matrix":
                string += "%s: %f\n"%(k, v)
        
        string+='Class IoU:\n'
        if self.classNames is None:
            for k, v in results['Class IoU'].items():
                string += "\tclass %d: %s\n"%(k, str(v))
                
            string+='Class Acc:\n'
            for k, v in results['Class Acc'].items():
                string += "\tclass %d: %s\n"%(k, str(v))
                
            string+='Class Prc:\n'
            for k, v in results['Class Prc'].items():
                string += "\tclass %d: %s\n"%(k, str(v))
        else:
            for k, v in results['Class IoU'].items():
                string += f"\tclass {self.classNames[k]:<{self.stringLenght}} : {str(v)}\n"
                
            string+='Class Acc:\n'
            for k, v in results['Class Acc'].items():
                string += f"\tclass {self.classNames[k]:<{self.stringLenght}} : {str(v)}\n"
                
            string+='Class Prc:\n'
            for k, v in results['Class Prc'].items():
                string += f"\tclass {self.classNames[k]:<{self.stringLenght}} : {str(v)}\n"
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy (aka pixel accuracy)
            - mean accuracy (aka class accuracy)
            - mean IoU
            - fwavacc
        """
        EPS = 1e-6
        hist = self.confusion_matrix

        gt_sum = hist.sum(axis=1)
        mask = (gt_sum != 0)
        diag = np.diag(hist)
        # acc = TP / TP+FN
        acc = diag.sum() / hist.sum()
        acc_cls_c = diag / (gt_sum + EPS)
        acc_cls = np.mean(acc_cls_c[mask])
        
        # prc = TP / TP+FP
        prc_cls_c = diag / (hist.sum(axis=0) + EPS)
        prc_cls = np.mean(prc_cls_c[mask])
        iu = diag / (gt_sum + hist.sum(axis=0) - diag + EPS)
        mean_iu = np.mean(iu[mask])
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), [iu[i] if m else "X" for i, m in enumerate(mask)]))
        cls_acc = dict(zip(range(self.n_classes), [acc_cls_c[i] if m else "X" for i, m in enumerate(mask)]))
        cls_prc = dict(zip(range(self.n_classes), [prc_cls_c[i] if m else "X" for i, m in enumerate(mask)]))

        return {
                "Total samples":  self.total_samples,
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "Mean Prc": prc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
                "Class Acc": cls_acc,
                "Class Prc": cls_prc,
                "Confusion Matrix": self.confusion_matrix_to_fig()
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.total_samples = 0

    def synch(self, device):
        # collect from multi-processes
        confusion_matrix = torch.tensor(self.confusion_matrix).to(device)
        samples = torch.tensor(self.total_samples).to(device)

        if self.where_to_sim == 'GPU_server':
            torch.distributed.reduce(confusion_matrix, dst=0)
            torch.distributed.reduce(samples, dst=0)

            if torch.distributed.get_rank() == 0:
                self.confusion_matrix = confusion_matrix.cpu().numpy()
                self.total_samples = samples.cpu().numpy()
        else:
            self.confusion_matrix = confusion_matrix.cpu().numpy()
            self.total_samples = samples.cpu().numpy()

    def confusion_matrix_to_fig(self):
        cm = self.confusion_matrix.astype('float') / (self.confusion_matrix.sum(axis=1)+0.000001)[:, np.newaxis]
        
        if plt.gcf().number == 19:
            plt.close('all')
            
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(title=f'Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')

        fig.tight_layout()
        return fig


class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]




