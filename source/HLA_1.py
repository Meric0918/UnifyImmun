import time
from models.HLA import *
import random
import warnings
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import roc_auc_score, auc, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
import os
import torch
import torch.nn as nn
import torch.optim as optim
import swanlab

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

warnings.filterwarnings("ignore")
seed = 66
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
pep_max_len = 15
hla_max_len = 34
tcr_max_len = 34
tgt_len = pep_max_len + hla_max_len
vocab_size = len(vocab)
n_heads = 1
d_model = 64
d_ff = 512
d_k = d_v = 64
n_layers = 1
threshold = 0.5
use_cuda = torch.cuda.is_available()
model = Mymodel_HLA().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def performance(y_true, y_pred, y_pred_transfer):
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred_transfer)
    tn, fp, fn, tp = (
        confusion_matrix(y_true, y_pred_transfer, labels=[0, 1]).ravel().tolist()
    )
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true=y_true, y_pred=y_pred_transfer, zero_division=0)
    recall = recall_score(y_true=y_true, y_pred=y_pred_transfer, zero_division=0)
    f1 = f1_score(y_true=y_true, y_pred=y_pred_transfer, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred)
    prec, reca, _ = precision_recall_curve(y_true, y_pred)
    aupr = auc(reca, prec)
    mcc = matthews_corrcoef(y_true, y_pred_transfer)
    print("tn = {}, fp = {}, fn = {}, tp = {}".format(tn, fp, fn, tp))
    print(
        "y_pred: 0 = {} | 1 = {}".format(
            Counter(y_pred_transfer)[0], Counter(y_pred_transfer)[1]
        )
    )
    print("y_true: 0 = {} | 1 = {}".format(Counter(y_true)[0], Counter(y_true)[1]))
    print(
        "auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}".format(
            roc_auc, sensitivity, specificity, accuracy, mcc
        )
    )
    print(
        "precision={:.4f}|recall={:.4f}|f1={:.4f}|aupr={:.4f}".format(
            precision, recall, f1, aupr
        )
    )
    return (
        roc_auc,
        accuracy,
        mcc,
        f1,
        aupr,
        sensitivity,
        specificity,
        precision,
        recall,
    )


f_mean = lambda l: sum(l) / len(l)


def performances_to_pd(performances_list):
    metrics_name = [
        "roc_auc",
        "accuracy",
        "mcc",
        "f1",
        "aupr",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
    ]
    performances_pd = pd.DataFrame(performances_list, columns=metrics_name)
    performances_pd.loc["mean"] = performances_pd.mean(axis=0)
    performances_pd.loc["std"] = performances_pd.std(axis=0)
    return performances_pd


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup1 = {}
        self.backup2 = {}

    def attack(self, epsilon=1.0, emb_name="emb"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if emb_name == "encoder_H.src_emb":
                    self.backup1[name] = param.data.clone()
                if emb_name == "encoder_P.src_emb":
                    self.backup2[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="emb"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if emb_name == "encoder_H.src_emb":
                    assert name in self.backup1
                    param.data = self.backup1[name]
                if emb_name == "encoder_P.src_emb":
                    assert name in self.backup2
                    param.data = self.backup2[name]
        if emb_name == "encoder_H.src_emb":
            self.backup1 = {}
        if emb_name == "encoder_P.src_emb":
            self.backup2 = {}


def train_HLA(model, train_loader, fold, epoch, epochs):
    train_time = 0
    model.train()
    y_true_list, y_pred_list, attention_list = [], [], []
    loss_list = []
    fgm = FGM(model)
    for train_anti_inputs, train_hla_inputs, train_labels in tqdm(
        train_loader, colour="yellow"
    ):
        train_anti_inputs, train_hla_inputs, train_labels = (
            train_anti_inputs.to(device),
            train_hla_inputs.to(device),
            train_labels.to(device),
        )
        start = time.time()
        train_outputs, cross_attention = model(train_anti_inputs, train_hla_inputs)
        train_loss = criterion(train_outputs, train_labels)
        train_time += time.time() - start
        train_loss.backward()
        fgm.attack(emb_name="encoder_H.src_emb")
        fgm.attack(emb_name="encoder_P.src_emb")
        train_outputs2, train_dec_self_attns2 = model(
            train_anti_inputs, train_hla_inputs
        )
        loss_sum = criterion(train_outputs2, train_labels)
        loss_sum.backward()
        fgm.restore(emb_name="encoder_H.src_emb")
        fgm.restore(emb_name="encoder_P.src_emb")
        optimizer.step()
        optimizer.zero_grad()
        y_true = train_labels.cpu().numpy()
        y_pred = nn.Softmax(dim=1)(train_outputs)[:, 1].cpu().detach().numpy()
        y_true_list.extend(y_true)
        y_pred_list.extend(y_pred)
        loss_list.append(train_loss)
        attention_list.append(cross_attention)
    y_pred_transfer_list = transfer(y_pred_list, threshold)
    result_train = (y_true_list, y_pred_list, y_pred_transfer_list)
    print(
        "Fold-{} Train: Epoch:{}/{} Loss = {:.4f} Time = {:.4f} seconds".format(
            fold, epoch, epochs, f_mean(loss_list), train_time
        )
    )
    performance_train = performance(y_true_list, y_pred_list, y_pred_transfer_list)
    return result_train, performance_train, train_time, attention_list


def valid_HLA(model, val_loader, fold, epoch, epochs):
    model.eval()
    torch.manual_seed(66)
    torch.cuda.manual_seed(66)
    with torch.no_grad():
        y_true_val_list, y_pred_val_list, attention_val_list = [], [], []
        loss_val_list = []
        for val_anti_inputs, val_hla_inputs, val_labels in tqdm(
            val_loader, colour="blue"
        ):
            val_anti_inputs, val_hla_inputs, val_labels = (
                val_anti_inputs.to(device),
                val_hla_inputs.to(device),
                val_labels.to(device),
            )
            val_outputs, cross_attention_val = model(val_anti_inputs, val_hla_inputs)
            val_loss = criterion(val_outputs, val_labels)
            y_true_val = val_labels.cpu().numpy()
            y_pred_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
            y_true_val_list.extend(y_true_val)
            y_pred_val_list.extend(y_pred_val)
            loss_val_list.append(val_loss)
        y_pred_transfer_val_list = transfer(y_pred_val_list, threshold)
        result_val = (y_true_val_list, y_pred_val_list, y_pred_transfer_val_list)
        print(
            "Fold-{} Valid: Epoch:{}/{} Loss = {:.4f}".format(
                fold, epoch, epochs, f_mean(loss_val_list)
            )
        )
        performance_val = performance(
            y_true_val_list, y_pred_val_list, y_pred_transfer_val_list
        )
    return result_val, performance_val, cross_attention_val


independent_loader = data_load_HLA(
    type_="independent", fold=None, batch_size=batch_size
)
external_loader = data_load_HLA(type_="external", fold=None, batch_size=batch_size)

(
    train_fold_performance_list,
    val_fold_performance_list,
    independent_fold_performance_list,
    external_fold_performance_list,
) = [], [], [], []
(
    attention_train_dict,
    attention_val_dict,
    attention_independent_dict,
    attention_external_dict,
) = {}, {}, {}, {}

swanlab.init(
    project="unifyimmun",
    experiment_name="HLA_1",
    config={
        "model": "HLA_1",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 1e-3,
        "threshold": threshold,
        "seed": seed,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
    },
)

for fold in range(1, 6):
    print("Fold-{}:".format(fold))
    print("Load HLA Data:")
    train_loader = data_load_HLA(type_="train", fold=fold, batch_size=batch_size)
    val_loader = data_load_HLA(type_="val", fold=fold, batch_size=batch_size)
    train_data = pd.read_csv(
        os.path.join(
            _project_root, "data", "data_HLA", "train_fold_{}.csv".format(fold)
        )
    )
    val_data = pd.read_csv(
        os.path.join(
            _project_root, "data", "data_HLA", "train_fold_{}.csv".format(fold)
        )
    )
    print(
        "Fold-{} Label: Train = {} | Val = {}".format(
            fold, Counter(train_data.label), Counter(val_data.label)
        )
    )
    print("HLA Train:")
    path_all = os.path.join(_project_root, "trained_model", "HLA_1")
    save_path = os.path.join(
        _project_root, "trained_model", "HLA_1", "model_HLA_fold{}.pkl".format(fold)
    )
    encoder_path = os.path.join(
        _project_root, "trained_model", "HLA_1", "encoder_P_{}.pth".format(fold)
    )
    print("save path: ", save_path)
    performance_best, epoch_best = 0, -1
    time_train = 0
    for epoch in range(1, epochs + 1):
        result_train, performance_train, train_time, attention_score = train_HLA(
            model, train_loader, fold, epoch, epochs
        )
        result_val, performance_val, attention_score_val = valid_HLA(
            model, val_loader, fold, epoch, epochs
        )
        performance_avg = sum(performance_val[:5]) / 5

        swanlab.log(
            {
                "fold": fold,
                "epoch": epoch,
                "train_loss": float(f_mean(result_train[0])),
                "train_roc_auc": performance_train[0],
                "train_accuracy": performance_train[1],
                "train_mcc": performance_train[2],
                "train_f1": performance_train[3],
                "train_aupr": performance_train[4],
                "train_sensitivity": performance_train[5],
                "train_specificity": performance_train[6],
                "val_loss": float(f_mean(result_val[0])),
                "val_roc_auc": performance_val[0],
                "val_accuracy": performance_val[1],
                "val_mcc": performance_val[2],
                "val_f1": performance_val[3],
                "val_aupr": performance_val[4],
                "val_sensitivity": performance_val[5],
                "val_specificity": performance_val[6],
                "val_performance_avg": performance_avg,
            }
        )

        if performance_avg > performance_best:
            performance_best, epoch_best = performance_avg, epoch
            if not os.path.exists(path_all):
                os.makedirs(path_all)
            print(
                "Save model: Best epoch = {} | Performance_avg = {:.4f}".format(
                    epoch_best, performance_best
                )
            )
            print("Save Path: ", save_path)
            torch.save(model.eval().state_dict(), save_path)
            torch.save(model.eval().encoder_P.state_dict(), encoder_path)
            swanlab.log(
                {"best_epoch": epoch_best, "best_performance_avg": performance_best}
            )
        time_train += train_time
    print("HLA Training Finished")
    print("-----Evaluate Results-----")
    if epoch_best >= 0:
        print("*****Path saver: ", save_path)
        model.load_state_dict(torch.load(save_path))
        model_eval = model.eval()
        valid_result, valid_performance, valid_attention = valid_HLA(
            model_eval, val_loader, fold, epoch_best, epochs
        )
        independent_result, independent_performance, independent_attention = valid_HLA(
            model_eval, independent_loader, fold, epoch_best, epochs
        )
        external_result, external_performance, external_attention = valid_HLA(
            model_eval, external_loader, fold, epoch_best, epochs
        )
        val_fold_performance_list.append(valid_performance)
        independent_fold_performance_list.append(independent_performance)
        external_fold_performance_list.append(external_performance)

        swanlab.log(
            {
                f"fold_{fold}_val_roc_auc": valid_performance[0],
                f"fold_{fold}_val_accuracy": valid_performance[1],
                f"fold_{fold}_independent_roc_auc": independent_performance[0],
                f"fold_{fold}_independent_accuracy": independent_performance[1],
                f"fold_{fold}_external_roc_auc": external_performance[0],
                f"fold_{fold}_external_accuracy": external_performance[1],
            }
        )
    print("Total training time: {:6.2f} sec".format(time_train))

print("****Independent set:")
print(performances_to_pd(independent_fold_performance_list).to_string())
print("****External set:")
print(performances_to_pd(external_fold_performance_list).to_string())
print("****Val set:")
print(performances_to_pd(val_fold_performance_list).to_string())

swanlab.finish()
