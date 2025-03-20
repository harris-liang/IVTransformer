import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb
from sklearn.linear_model import Ridge

from base_models import NeuralNetwork, ParallelNetworks

import pdb


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif conf.family == "EncoderTF":
        # backward compatible
        if 'encoder_activation' not in conf.keys():
            conf.encoder_activation = "relu"
        if 'normalize_attn' not in conf.keys():
            conf.normalize_attn = True

        model = EncoderTransformer(
            n_dims=conf.n_dims + 1,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            activation=conf.encoder_activation,
            normalize_attn=conf.normalize_attn,
        )
    elif conf.family == 'gpt2-loop':
        model = TransformerModelLooped(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    else:
        raise NotImplementedError

    return model


RIDGE_LAMS = [0.2, 1.25, 5, 20]

def get_relevant_baselines(task_name, ridge_lams=RIDGE_LAMS):
    task_to_baselines = {
        "iv_regression":[
            (LeastSquaresModel_iv, {}), 
            (IVRegressionModel, {}),
        ],
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (LogisticModel, {}),
        ],
        "noisy_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(RidgeModel, {"lam": lam}) for lam in ridge_lams],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


def get_activation(activation="relu"):
    if activation == "relu":
        return F.relu
    elif activation == "softmax":
        return lambda x: F.softmax(x, dim=-1)
    else:
        raise NotImplementedError

class EncoderTransformer(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4,
                 activation="relu", normalize_attn=True, mlp=True, layernorm=True):
        super(EncoderTransformer, self).__init__()
        self.name = f"EncoderTF_embd={n_embd}_layer={n_layer}_head={n_head}"

        # configs
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.activation = get_activation(activation)
        self.normalize_attn = normalize_attn
        self.layernorm = layernorm
        self.mlp = mlp

        # layers
        self._read_in = nn.Linear(n_dims, n_embd)
        self._queries = nn.ModuleList()
        self._keys = nn.ModuleList()
        self._values = nn.ModuleList()
        self._mlps = nn.ModuleList()
        self._lns_1 = nn.ModuleList()
        self._lns_2 = nn.ModuleList()
        for i in range(n_layer):
            self._queries.append(nn.Linear(n_embd, n_embd, bias=False))
            self._keys.append(nn.Linear(n_embd, n_embd, bias=False))
            self._values.append(nn.Linear(n_embd, n_embd, bias=False))
            self._lns_1.append(nn.LayerNorm([self.n_embd]))
            self._mlps.append(
                nn.Sequential(
                    nn.Linear(n_embd, n_embd),
                    nn.ReLU(),
                    nn.Linear(n_embd, n_embd),
                )
            )
            self._lns_2.append(nn.LayerNorm([self.n_embd]))
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """
        Directly stack the x's and y's into the same location
        resulting sequence would be Bx(N+1)x(d+1), where (N+1)-th token is test
        """
        zs = torch.cat((xs_b, ys_b.unsqueeze(2)), dim=2)
        zs[:, -1, -1].zero_()
        return zs

    def forward(self, xs, ys, inds=None, return_hidden_states=False):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        hidden_states = []
        H = self._read_in(zs)
        hidden_states.append(H)
        for (q, k, v, ln1, mlp, ln2) in zip(
            self._queries, self._keys, self._values,
            self._lns_1, self._mlps, self._lns_2,
        ):
            query = q(H)
            key = k(H)
            value = v(H)

            attn_weights = self.activation(torch.einsum('bid,bjd->bij', query, key))
            if self.normalize_attn:
                attn_weights = attn_weights / ys.shape[1]
            H = H + torch.einsum('bij,bjd->bid', attn_weights, value)
            if self.layernorm:
                H = ln1(H)

            if self.mlp:
                H = H + mlp(H)
                if self.layernorm:
                    H = ln2(H)

            hidden_states.append(H)

        prediction = self._read_out(H)
        if return_hidden_states:
            return prediction[:, inds, 0], hidden_states
        return prediction[:, inds, 0]


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs
    

class TransformerModelLooped(TransformerModel):
    def __init__(
            self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):

        super(TransformerModelLooped, self).__init__(
            n_dims, n_positions, n_embd, n_layer, n_head)
        self.loop_func = 'z=f(x+z)'
        self._pred_type = 'regression'


    def forward(self, xs, ys, inds=None, n_loop_start=0, n_loops=10):
        """
        :param xs: [B, n, d]
        :param ys: [B, n]
        :param n_loop_start: int
        :param n_loops: int
        :return:
        """
        B, n, d_in = xs.shape
        zs = self._combine(xs, ys)  # [B, n, d_in], [B, n], [B, n] -> [B, 2n, d_in + 1]
        embeds = self._read_in(zs)  # [B, 2n, d_in + 1] -> [B, 2n, d]
        if self.loop_func in ['z=f(x+z)']:
            output = embeds
        elif self.loop_func in ['z=f(x*z)']:
            output = torch.ones_like(embeds)  # also of shape [B, 2n, d]
        else:
            raise NotImplementedError("Currently we only support loop function z=f(x+z) or z=f(x*z).")
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
   
        for idx in range(n_loops):
            if idx < n_loop_start:  # this will save memory when n_loops large.
                with torch.no_grad():
                    output = self.f(output, embeds)
            else:
                output = self._backbone(inputs_embeds=output).last_hidden_state
                prediction = self._read_out(output)  # [B, 2n, d] -> [B, 2n, 1]
        if self._pred_type == 'regression':
            y = prediction[:, ::2, 0][:, inds]
        elif self._pred_type == 'classification':
            y = prediction[:, ::2]
        else:
            raise NotImplementedError
        return y


class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)

class IVRegressionModel:
    def __init__(self):
        self.name = f"iv_regression"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xzs, ys, n_dims_truncated=None, inds=None, return_beta = False):

        xzs, ys = xzs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xzs and ys are not defined")

        preds = []
        p = int(1/3 * xzs.shape[2])
        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
         
            train_xzs, train_ys = xzs[:, :i], ys[:, :i]
            train_xs = train_xzs[:, :, :p]
            train_zs = train_xzs[:, :, p:]

            test_xz = xzs[:, i : i + 1]
            test_x = test_xz[:, :, :p]
            test_z = test_xz[:, :, p:]
            
            Theta = torch.linalg.lstsq(train_zs, train_xs).solution
            train_xs_hat = train_zs @ Theta
            beta = torch.linalg.lstsq(train_xs_hat, train_ys.unsqueeze(2)).solution
            
            pred_x_hat = test_z @ Theta
            pred = test_x @ beta
            
            preds.append(pred[:, 0, 0])
            
        if return_beta:
            return torch.stack(preds, dim=1), beta
        else:
            return torch.stack(preds, dim=1)
        
    def get_y_query(self, xzs, ys):
        p = int(xzs.shape[2] / 3)
        train_xzs, train_ys = xzs[:, :-1], ys[:, :-1]
        train_xs = train_xzs[:, :, :p]
        train_zs = train_xzs[:, :, p:]

        test_xz = xzs[:, -1:]
        test_x = test_xz[:, :, :p]
        test_z = test_xz[:, :, p:]

        Theta = torch.linalg.lstsq(train_zs, train_xs).solution
        train_xs_hat = train_zs @ Theta
        beta = torch.linalg.lstsq(train_xs_hat, train_ys.unsqueeze(2)).solution
        pred = test_x @ beta
        return pred

class RidgeIVRegressionModel:
    def __init__(self, lambda_theta=1.0, lambda_beta=1.0):
        self.name = f"Ridge iv"
        self.lambda_theta = lambda_theta
        self.lambda_beta = lambda_beta

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xzs, ys, n_dims_truncated=None, inds=None, return_beta=False):
    
        xzs, ys = xzs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xzs and ys are not defined")

        preds = []
        p = int(1 / 3 * xzs.shape[2])
        beta = None

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue

            train_xzs, train_ys = xzs[:, :i], ys[:, :i]
            train_xs = train_xzs[:, :, :p]
            train_zs = train_xzs[:, :, p:]

            test_xz = xzs[:, i : i + 1]
            test_x = test_xz[:, :, :p]
            test_z = test_xz[:, :, p:]

            # First stage: Ridge regression to estimate Theta
            XtZ = train_zs.transpose(1, 2) @ train_zs
            Theta = torch.linalg.solve(
                XtZ + self.lambda_theta * torch.eye(XtZ.shape[-1]),
                train_zs.transpose(1, 2) @ train_xs,
            )

            train_xs_hat = train_zs @ Theta

            # Second stage: Ridge regression to estimate beta
            XtX_hat = train_xs_hat.transpose(1, 2) @ train_xs_hat
            beta = torch.linalg.solve(
                XtX_hat + self.lambda_beta * torch.eye(XtX_hat.shape[-1]),
                train_xs_hat.transpose(1, 2) @ train_ys.unsqueeze(2),
            )

            # Predict for the current test point
            pred_x_hat = test_z @ Theta
            pred = test_x @ beta

            preds.append(pred[:, 0, 0])

        if return_beta:
            return torch.stack(preds, dim=1), beta
        else:
            return torch.stack(preds, dim=1)

        
# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )
            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)
    
    def get_y_query(self, xs, ys):
        # xs, ys = xs.cpu(), ys.cpu()
        train_xs, train_ys = xs[:, :-1], ys[:, :-1]
        test_x = xs[:, -1:]
        ws, _, _, _ = torch.linalg.lstsq(
            train_xs, train_ys.unsqueeze(2), driver=self.driver
        )
        pred = test_x @ ws
        return pred
    
class RidgeLeastSquaresModel:
    def __init__(self, lambda_ridge=1.0, driver=None):
        self.driver = driver
        self.lambda_ridge = lambda_ridge
        self.name = "Ridge OLS"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue

            # Split training and test sets
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            # Compute ridge regression weights
            XtX = train_xs.transpose(1, 2) @ train_xs
            Xty = train_xs.transpose(1, 2) @ train_ys.unsqueeze(2)

            # Add ridge regularization (lambda * I)
            reg_term = self.lambda_ridge * torch.eye(XtX.shape[-1], device=XtX.device)

            # Solve for weights
            ws = torch.linalg.solve(XtX + reg_term, Xty)

            # Predict
            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)

    
class LeastSquaresModel_iv:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS"

    def __call__(self, xzs, ys, inds=None):
        xzs, ys = xzs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []
        p = int(xzs.shape[2] / 3)

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xzs, train_ys = xzs[:, :i], ys[:, :i]
            train_xs = train_xzs[:, :, :p]
            test_xz = xzs[:, i : i + 1]
            test_x = test_xz[:, :, :p]

            ws = torch.linalg.lstsq(train_xs, train_ys.unsqueeze(2)).solution

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)

    def get_y_query(self, xzs, ys):
        p = int(xzs.shape[2] / 3)
        train_xzs, train_ys = xzs[:, :-1], ys[:, :-1]
        train_xs = train_xzs[:, :, :p]
        test_xz = xzs[:, -1:]
        test_x = test_xz[:, :, :p]

        ws = torch.linalg.lstsq(train_xs, train_ys.unsqueeze(2)).solution
        pred = test_x @ ws
        return pred
    