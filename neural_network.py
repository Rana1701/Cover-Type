import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from torch.utils.data import DataLoader, TensorDataset


class _MLPBackbone(nn.Module):
    def __init__(self, n_features, hidden_layer_sizes, n_classes, dropout, use_batch_norm):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(prev, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _init_linear(module: nn.Module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    MLP multi-classe (PyTorch) compatible scikit-learn : RandomizedSearchCV, cross_validate, etc.
    Attend des entrées numériques (ex. données normalisées comme les autres modèles).
    """

    def __init__(
        self,
        hidden_layer_sizes=(128, 64),
        lr=1e-3,
        batch_size=256,
        max_epochs=40,
        dropout=0.1,
        weight_decay=0.0,
        optimizer="adamw",
        val_fraction=0.15,
        early_stopping=True,
        patience=10,
        min_delta=0.0,
        scheduler_factor=0.5,
        scheduler_patience=3,
        use_batch_norm=True,
        random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.val_fraction = val_fraction
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.use_batch_norm = use_batch_norm
        self.random_state = random_state

    def _set_seed(self):
        s = self.random_state
        if s is not None:
            torch.manual_seed(int(s))
            np.random.seed(int(s))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(s))
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        X = np.asarray(X, dtype=np.float32)

        self._set_seed()

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        hidden = tuple(self.hidden_layer_sizes) if not isinstance(self.hidden_layer_sizes, tuple) else self.hidden_layer_sizes
        self.pytorch_model_ = _MLPBackbone(
            n_features,
            hidden,
            n_classes,
            self.dropout,
            bool(self.use_batch_norm),
        )
        _init_linear(self.pytorch_model_)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pytorch_model_.to(self.device_)

        use_val = bool(self.early_stopping) and float(self.val_fraction) > 0.0 and len(X) >= 50
        if use_val:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X,
                y_enc,
                test_size=float(self.val_fraction),
                stratify=y_enc,
                random_state=self.random_state,
            )
        else:
            X_tr, y_tr = X, y_enc
            X_va, y_va = None, None

        Xt = torch.from_numpy(np.ascontiguousarray(X_tr))
        yt = torch.from_numpy(y_tr.astype(np.int64))
        ds = TensorDataset(Xt, yt)
        bs = min(int(self.batch_size), len(ds))
        loader = DataLoader(ds, batch_size=max(bs, 1), shuffle=True)

        if use_val:
            Xv = torch.from_numpy(np.ascontiguousarray(X_va))
            yv = torch.from_numpy(y_va.astype(np.int64))
            val_ds = TensorDataset(Xv, yv)
            vbs = min(int(self.batch_size), len(val_ds))
            val_loader = DataLoader(val_ds, batch_size=max(vbs, 1), shuffle=False)
        else:
            val_loader = None

        opt_name = str(self.optimizer).lower()
        if opt_name == "adamw":
            opt = torch.optim.AdamW(
                self.pytorch_model_.parameters(),
                lr=float(self.lr),
                weight_decay=float(self.weight_decay),
            )
        elif opt_name == "adam":
            opt = torch.optim.Adam(
                self.pytorch_model_.parameters(),
                lr=float(self.lr),
                weight_decay=float(self.weight_decay),
            )
        else:
            raise ValueError("optimizer doit etre 'adam' ou 'adamw'")
        crit = nn.CrossEntropyLoss()
        scheduler = None
        if use_val:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=float(self.scheduler_factor),
                patience=int(self.scheduler_patience),
            )

        best_state = None
        best_val_loss = float("inf")
        no_improve = 0

        self.pytorch_model_.train()
        for _ in range(int(self.max_epochs)):
            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                opt.zero_grad()
                logits = self.pytorch_model_(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()

            if val_loader is not None:
                self.pytorch_model_.eval()
                val_loss_sum = 0.0
                val_batches = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device_)
                        yb = yb.to(self.device_)
                        logits = self.pytorch_model_(xb)
                        val_loss_sum += float(crit(logits, yb).item())
                        val_batches += 1
                mean_val = val_loss_sum / max(val_batches, 1)
                scheduler.step(mean_val)
                if mean_val + float(self.min_delta) < best_val_loss:
                    best_val_loss = mean_val
                    best_state = copy.deepcopy(self.pytorch_model_.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                self.pytorch_model_.train()
                if no_improve >= int(self.patience):
                    break

        if best_state is not None:
            self.pytorch_model_.load_state_dict(best_state)

        self.n_features_in_ = n_features
        return self

    def predict(self, X):
        check_is_fitted(self, ("pytorch_model_", "classes_"))
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        X = np.asarray(X, dtype=np.float32)

        self.pytorch_model_.eval()
        with torch.no_grad():
            t = torch.from_numpy(X).to(self.device_)
            logits = self.pytorch_model_(t)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

        return self.label_encoder_.inverse_transform(pred)
