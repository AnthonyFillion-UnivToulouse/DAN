"""
This file contains the DAN and some functions to test it
"""
import torch
import numpy as np
import random as rnd
import torch.nn as nn
from torch.optim import SGD, Adam
from lorenz95 import M
from os import mkdir, path

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class DAN(nn.Module):
    """
    A Data Assimilation Network class
    """

    def __init__(self, **kwargs):

        super(DAN, self).__init__()

        # Inputs ##############################################################
        self.training_mode = kwargs.get("training_mode", "supervised")
        print(self.training_mode)
        #   w_dim is the reparameterization trick number of samples,
        #     it is useless when the training is supervised
        self.w_dim = kwargs.get("w_dim", 20)
        #   late instantiation of the net transformations from kwargs
        self.net = nn.ModuleDict({})  # transformations dict
        for key in ["analyzer", "propagater", "procoder"]:
            d = kwargs[key]
            self.net[key] = d["module"](**d["kwargs"])
        #######################################################################

    def forward(self, ha, y):
        """
        forward pass in the DAN

        Inputs:
            ha: batch of last cycle posterior memories
            y: batch of observations
        Outputs:
            ha: batch of posterior memories
            pdf_b: batch of prior pdfs
            pdf_a: batch of posterior pdfs
            pdf_r: batch of decoded pdfs, None when unsupervised training
        Comments:
            the pdfs are torch.distributions.multivariate_normal objects
        """

        hb = self.net["propagater"](ha)  # propagate past mem into prior mem
        pdf_b = self.net["procoder"](hb)  # procode prior mem into prior pdf
        ha = self.net["analyzer"](torch.cat((hb, y), 1))  # analyze prior mem
        #  and obs into post mem
        pdf_a = self.net["procoder"](ha)  # procode post mem into post pdf

        return ha, pdf_b, pdf_a

    def loss_selector(self, name, pdf_b, pdf_a, x, y):
        """
        implements some instantaneous loss functions for the
        DAN training or testing
        Inputs
          name: "rmse_b"   | "rmse_a"   | "rmse"   (supervised)
                "logpdf_b" | "logpdf_a" | "logpdf" (supervised)
                "reg"      | "rec"      | "elbo"   (unsupervised)
          pdf_b: batch of prior pdfs
          pdf_a: batch of posterior pdfs
          pdf_r: batch of decoded pdfs, None when unsupervised training
          x: a batch of states
          y: a batch of observations
        Outputs
          a scalar tensor
        """
        if name == "rmse_b":
            x_dim = x.size(1)
            return torch.mean(torch.norm(
                x-pdf_b.mean[:, :x_dim], dim=1)/(x_dim**0.5))
        elif name == "rmse_a":
            x_dim = x.size(1)
            return torch.mean(torch.norm(
                x-pdf_a.mean[:, :x_dim], dim=1)/(x_dim**0.5))
        elif name == "rmse":
            x_dim = x.size(1)
            rmse_b = torch.mean(
                torch.norm(x-pdf_b.mean[:, :x_dim], dim=1)/(x_dim**0.5))
            rmse_a = torch.mean(
                torch.norm(x-pdf_a.mean[:, :x_dim], dim=1)/(x_dim**0.5))
            return rmse_b + rmse_a
        elif name == "logpdf_b":
            return -torch.mean(pdf_b.log_prob(x))
        elif name == "logpdf_a":
            return -torch.mean(pdf_a.log_prob(x))
        elif name == "logpdf":
            logpdf_b = -torch.mean(pdf_b.log_prob(x))
            logpdf_a = -torch.mean(pdf_a.log_prob(x))
            return logpdf_b + logpdf_a
        else:
            print("ERROR: name loss_selector")
            return None


def train(**kwargs):  # TODO: change in exp
    """
    Train functions for the DAN
    """

    # Inputs ##################################################################
    net = kwargs["net"]  # the data assimilation network
    fprint = kwargs.get("fprint", 10**2)  # printing and saving outputs freq
    n_obs = kwargs.get("n_obs", 10**4)  # number of observations
    op_obs = kwargs.get("op_obs", lambda x: x)  # observation operator
    sigma_r = kwargs.get("sigma_r", 1.)  # obs error std
    op_prop = kwargs.get("op_prop", lambda x: M(x, 1))  # propagation operator
    sigma_q = kwargs.get("sigma_q", .1)  # prop error std
    x = kwargs.get("x_truth", M(3*torch.ones(1024, 40) +  # initial truth
                                torch.randn(1024, 40), 10**3))
    x_dim = x.size(1)
    batch_size = x.size(0)
    y_dim = kwargs.get("y_dim", x_dim)
    ha0m1 = kwargs.get("hidden_state",  # TODO: change in exp
                       torch.zeros(batch_size, 800))  # initial memory batch
    direxp = kwargs.get("direxp", "")  # experiment directory
    outputs = kwargs.get("outputs",  # dict of the outputs
                         {key: np.array([]) for key in
                          ["logpdf_b", "logpdf_a",
                           "rmse_b", "rmse_a",
                           "elbo", "rec", "reg"]})
    opt_loss = kwargs.get("opt_loss", "logpdf")  # loss function name
    S = kwargs.get("DAW_step", 1)  # step between outer QS optimizations
    Ls = kwargs.get("QS_DAW_lengths", [1])  # Quasi static DAW lengths
    Js = kwargs.get("QS_DAW_iterations", [1])  # Quasi static optimizations
    #                                            iteration number
    tol = kwargs.get("tolerance")  # optimization min grad norm
    optimizer = kwargs["optimizer"]
    # Outputs #################################################################
    # outputs: is updated with the computed loss and saved on disk
    # net: the last net weights are saved on disk
    # h: the last computed memory batch is saved on disk for another training
    # x: the last computed state batch is saved on disk for another training
    ###########################################################################

    """
    TBPTT-QS-DAW optimization controlled by
      n_obs: int; the total number of obs
      Ls: [int]; the QS window lengths, L is its last element
      Js: [int]; the iteration numbers of each inner QS optimization
      S: int; TBPTT lag
      tol: float; the gradient min norm

      --  ha-1  y0  y1  y2  y3  y4  y6  --  --
      |   S=2   |
                | L0=3  |                      : QS0
                |     L1=5      |              : QS1
                |         L=6       |          : QS2
                -- ha1  y2  y3  y4  y5  y6  y7 : TBPTT
                |  S=2  |
                        | L0=3  |              : QS0
                        |     L1=5      |      : QS1
                        |         L=6       |  : QS2
      ...
    """
    # obs and state first initialization
    # only from 0 to L-S, the rest is updated in the loop over observations
    xs = torch.empty(batch_size, x_dim, Ls[-1])
    ys = torch.empty(batch_size, y_dim, Ls[-1])
    for l in range(0, Ls[-1]-S):
        ys[:, :, l] = op_obs(x) + sigma_r*torch.randn(batch_size, y_dim)
        xs[:, :, l] = x
        x = op_prop(x) + sigma_q*torch.randn(batch_size, x_dim)

    # loop over observations
    c, t = 0, 0  # t counts the total iteration number
    while c < n_obs:
        # states and observations update,
        # the L-S firsts items are shifted on the left
        ys = torch.roll(ys, -S, 1)
        xs = torch.roll(xs, -S, 1)
        #   the S last items can now be updated
        for l in range(Ls[-1]-S, Ls[-1]):
            ys[:, :, l] = op_obs(x) +\
                sigma_r*torch.randn(batch_size, y_dim)
            xs[:, :, l] = x
            x = op_prop(x) + sigma_q*torch.randn(batch_size, x_dim)

        # loop over QS minimizations,
        # each has its own cost function length
        for i in range(len(Ls)):

            # iterated gradient descent
            j, crit = 0, 1.
            while (j < Js[i]) and ((j == 0) or (crit >= tol)):
                optimizer.zero_grad()

                # instantaneous cost functions sum over Ls[i] time steps
                h = ha0m1.clone()  # initial memory
                # (h analyzed minus 1 i.e. y_{-1} is the last analyzed obs)

                # loss is a dict containing the instantaneous
                # losses sum for each outputs
                loss = {name_loss: 0. for name_loss in outputs.keys()}
                for l in range(Ls[i]):

                    # apply the net on obs l, update the memory
                    h, pdf_b, pdf_a = net(h, ys[:, :, l])
                    if l == S-1:  # save h analyzed S minus 1 for
                        haSm1 = h  # the next TBPTT window

                    # update the losses
                    for key in loss.keys():
                        if key == opt_loss:
                            # if the key corresponds to the loss to optimize
                            # computes it with require_grad
                            loss[key] += net.loss_selector(
                                key, pdf_b, pdf_a,
                                xs[:, :, l], ys[:, :, l])
                        else:
                            # otherwize, computes it without require_grad
                            with torch.no_grad():
                                loss[key] += net.loss_selector(
                                    key, pdf_b, pdf_a,
                                    xs[:, :, l], ys[:, :, l])

                # The loss and its derivative is constructed,
                # perform an optimization step and update outputs
                for key, val in loss.items():
                    if key == opt_loss:
                        val.backward()  # TODO: maybe error val -> loss[key]
                        optimizer.step()
                    outputs[key] = np.append(
                        outputs[key], val.item()/Ls[i])

                # Checkpoint
                if t % fprint == 0:
                    print("Train, obs_nb: "+str(c) +
                          " QS_L: "+str(Ls[i]) +
                          " opt_ite: "+str(j) +
                          " tot_ite: "+str(t) +
                          " grad_norm: "+str("N/A"),
                          end=" ")
                    # Checkpoint, save outputs on disk
                    for key, val in outputs.items():
                        np.save(direxp+"_train_"+key+".npy", val)
                        print(key+": "+str(val[-1]), end=" ")
                    print("")
                j += 1
                t += 1

            # End gradient descent

        # End QS minimizations are over

        # Detach the grad to perform TBPTT in the next window
        ha0m1 = haSm1.detach()
        c += S

    # End loop over observations

    # Saves the net, the mem and the state at the end of the cycles
    print("Train, obs_nb: "+str(c) +
          " QS_L: "+str(Ls[i]) +
          " opt_ite: "+str(j) +
          " tot_ite: "+str(t) +
          " grad_norm: "+str("N/A"))
    for key, val in outputs.items():
        np.save(direxp+"_train_"+key+".npy", val)
        print(key+": "+str(val[-1]), end=" ")
    print("")
    torch.save(net.state_dict(), direxp+"_state_dict.pt")
    torch.save(optimizer.state_dict(), direxp+"_opt_state_dict.pt")
    torch.save(h, direxp+"_hidden_state_train.pt")
    torch.save(x, direxp+"_x_truth_train.pt")


@torch.no_grad()
def test(**kwargs):
    """
    Test functions for the DAN
    """

    # Inputs ##################################################################
    net = kwargs["net"]  # the data assimilation network
    fprint = kwargs.get("fprint", 10**2)  # printing and saving outputs freq
    n_obs = kwargs.get("n_obs", 10**4)  # number of observations
    op_obs = kwargs.get("op_obs", lambda x: x)  # observation operator
    sigma_r = kwargs.get("sigma_r", 1.)  # obs error std
    op_prop = kwargs.get("op_prop", lambda x: M(x, 1))  # propagation operator
    sigma_q = kwargs.get("sigma_q", .1)  # prop error std
    x = kwargs.get("x_truth", M(3*torch.ones(1024, 40) +  # initial truth
                                torch.randn(1024, 40), 10**3))
    x_dim = x.size(1)
    batch_size = x.size(0)
    y_dim = kwargs.get("y_dim", x_dim)
    h = kwargs.get("hidden_state",  # TODO: change in exp
                   torch.zeros(batch_size, 800))  # initial memory batch
    direxp = kwargs.get("direxp", "")  # experiment directory
    outputs = kwargs.get("outputs",  # dict of the outputs
                         {key: np.array([]) for key in
                          ["logpdf_b", "logpdf_a",
                           "rmse_b", "rmse_a",
                           "elbo", "rec", "reg"]})

    # Outputs #################################################################
    # outputs: is updated with the computed loss and saved on disk
    # h: the last computed memory batch is saved on disk for another test
    # x: the last computed state batch is saved on disk for another test
    ###########################################################################

    # Main loop over cycles
    for c in range(n_obs):

        # observation generation
        y = op_obs(x) + sigma_r*torch.randn(batch_size, y_dim)

        # Evaluates the loss
        h, pdf_b, pdf_a = net(h, y)
        for key in outputs.keys():
            loss = net.loss_selector(key, pdf_b, pdf_a, x, y)
            outputs[key] = np.append(outputs[key], loss.item())

        # Checkpoint
        if (c % fprint == 0) or (c+1 >= n_obs):
            # save outputs and print
            print("Test, ite: "+str(c), end=" ")
            for key, val in outputs.items():
                np.save(direxp+"_test_"+key+".npy", val)  # save it
                print(key+": "+str(val[-1]), end=" ")
            print("")

        # state propagation
        x = op_prop(x) + sigma_q*torch.randn(batch_size, x_dim)

    # save outputs and print
    print("Test, ite: "+str(c), end=" ")
    for key, val in outputs.items():
        np.save(direxp+"_test_"+key+".npy", val)  # save it
        print(key+": "+str(val[-1]), end=" ")
    print("")
    torch.save(h, direxp+"_hidden_state_test.pt")
    torch.save(x, direxp+"_x_truth_test.pt")


def repeat(**kwargs):  # TODO: change in exp
    """
    Repeats a DAN test and train
    """

    # Inputs ##################################################################
    x_dim = kwargs.get("x_dim", 40)  # state dim
    h_dim = kwargs.get("h_dim", 20*x_dim)  # mem dim
    batch_size = kwargs.get("batch_size", 1024)  # batch size
    nb_repeat = kwargs.get("nb_repeat", 10**10)  # number of test and train
    directory = kwargs["directory"]  # directory of the experiment
    exp = kwargs["exp"]  # name of the experiment
    append = kwargs.get("append", False)  # start from an existing dan state
    stateless =\
        kwargs.get("stateless",  # clear train or test initial state
                   {"test": True, "train": False})  # and mem each
    #                                               # repetition
    burn = kwargs.get("burn", 10**3)  # skip any transiant regime
    outputs =\
        kwargs.get("outputs",  # dict for the test/train outputted losses
                   {key: {key2: np.array([])
                          for key2 in
                          ["rmse", "rmse_b", "rmse_a",
                           "logpdf", "logpdf_b", "logpdf_a",
                           "elbo", "reg", "rec"]}
                    for key in
                    ["test", "train"]})
    ktt =\
        {"test": kwargs["ktest"],  # kwargs for test and train function
         "train": kwargs["ktrain"]}
    kDAN = kwargs["kDAN"]  # kwargs for DAN
    direxp = directory + "/" + exp

    # Outputs #################################################################
    # the outputs of test and train
    ###########################################################################

    # DAN initialization
    net = DAN(**kDAN)

    # Main loop over repetitions
    for i in range(nb_repeat):

        if not path.exists(directory):
            mkdir(directory)

        if append:
            # load previous net weights
            net.load_state_dict(torch.load(direxp+"_state_dict.pt"))

        for mode in outputs.keys():
            if append:
                # load previous outputs dict
                for key in outputs[mode].keys():
                    outputs[mode][key] =\
                        np.load(direxp+"_"+mode+"_"+key+".npy")

            if append and (not stateless[mode]):
                # load mem and truth
                ktt[mode]["hidden_state"] =\
                    torch.load(direxp+"_hidden_state_"+mode+".pt")
                ktt[mode]["x_truth"] =\
                    torch.load(direxp+"_x_truth_"+mode+".pt")

            else:
                # clear initialization of mem and truth
                ktt[mode]["hidden_state"] =\
                    torch.zeros(batch_size, h_dim)
                ktt[mode]["x_truth"] =\
                    M(3*torch.ones(batch_size, x_dim) +
                      torch.randn(batch_size, x_dim), burn)
        append = True  # to append in next repetition

        # test then train
        print("Launch test "+str(i))
        test(net=net,
             outputs=outputs["test"],
             **ktt["test"])
        print("Launch train "+str(i))
        train(net=net,
              outputs=outputs["train"],
              **ktt["train"])


def train_then_test(**kwargs):
    """
    a DAN test and train
    """

    # Inputs ##################################################################
    #  Files
    directory = kwargs["directory"]
    exp = kwargs["exp"]
    direxp = directory + "/" + exp
    #  Dimensions
    x_dim = kwargs.get("x_dim", 40)  # state dim
    h_dim = kwargs.get("h_dim", 20*x_dim)  # mem dim
    batch_sizes = kwargs.get("batch_size",
                             {"train": 1024, "test": 1})
    burn = kwargs.get("burn", 10**3)  # skip any transiant regime
    #  Controls
    modes = kwargs.get("modes", ["train", "test"])
    load_weights = kwargs.get("load_weights", False)
    append_outputs = kwargs.get("append_outputs",
                                {"train": False, "test": False})
    load_state = kwargs.get("load_state", {"train": [False],
                                           "test": [False]})
    seeds = kwargs.get("seeds", {"train": [0],
                                 "test": [1]})
    kwargs["ktest"]["batch_size"] = kwargs["batch_sizes"]["test"]
    kwargs["ktrain"]["batch_size"] = kwargs["batch_sizes"]["train"]

    # Outputs #################################################################
    # the outputs of test and train
    ###########################################################################

    # DAN initialization
    net = DAN(**kwargs["kDAN"])
    if load_weights:
        net.load_state_dict(torch.load(direxp+"_state_dict.pt"))

    # Optimizer initialization
    optidict = kwargs.get("optidict", {"optimizer": "Adam", "lr": 10**-4})
    if optidict["optimizer"] == "SGD":
        optimizer = SGD(net.parameters(),
                        lr=optidict["lr"],
                        momentum=optidict["momentum"],
                        nesterov=optidict["nesterov"])
    elif optidict["optimizer"] == "Adam":
        optimizer = Adam(net.parameters(),
                         lr=optidict["lr"])

    i = {"train": 0, "test": 0}
    for mode in modes:
        # set seeds
        seed = seeds[mode][i[mode]]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        rnd.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if append_outputs[mode]:
            # load previous outputs dict
            for key, val in kwargs["k"+mode]["outputs"].items():
                val = np.load(direxp+"_"+mode+"_"+key+".npy")
        else:
            append_outputs[mode] = True

        if load_state[mode][i[mode]]:
            # load mem and truth
            h =\
                torch.load(direxp+"_hidden_state_"+mode+".pt")
            x =\
                torch.load(direxp+"_x_truth_"+mode+".pt")
        else:
            # clear mem and truth
            h =\
                torch.zeros(batch_sizes[mode], h_dim)
            x =\
                M(3*torch.ones(batch_sizes[mode], x_dim) +
                  torch.randn(batch_sizes[mode], x_dim), burn)

        if mode == "train":
            print("Launch "+mode)
            if optidict.get("load", False) and (i[mode] != 0):
                optimizer.load_state_dict(
                    torch.load(direxp+"_opt_state_dict.pt"))
            train(net=net,
                  hidden_state=h,
                  x_truth=x,
                  optimizer=optimizer,
                  **kwargs["k"+mode])
        elif mode == "test":
            print("Launch "+mode)
            test(net=net,
                 hidden_state=h,
                 x_truth=x,
                 **kwargs["k"+mode])
        i[mode] += 1


# Classes to build analyzers, propagaters, procoders and decoders #############
class Coder(torch.nn.Module):
    """
    This module implements a function from an input into a pdf
    """

    def __init__(self, eta_func, eta_args, eta_dim, out_dim):
        """
        eta: a vectorized mean-cov sqrt function of input
        eta_dim: dimension of the results of eta
        out_dim: dimension of the pdf domain
        """
        super().__init__()
        self.eta_dim = eta_dim
        self.out_dim = out_dim
        self.eta = eta_func(**eta_args)

        """
        Initializes the triangular inf column and line indices of a matrix,
        they are ordered from the main diagonal, then the subdiagonal, etc
        """
        ldiag, d, c = out_dim, 0, 0  # diag length, diag index, column index
        self.inds = [[], []]  # list of line and column indexes
        for i in range(eta_dim - out_dim):  # loop over the non-mean coeff
            self.inds[0].append(c+d)  # line index
            self.inds[1].append(c)  # column index
            if c == ldiag-1:  # the current diag end is reached
                ldiag += -1  # the diag length is decremented
                c = 0  # the column index is reinitialized
                d += 1  # the diag index is incremented
            else:  # otherwize, only the column index is incremented
                c += 1

    def forward(self,  inp):
        eta = self.eta(inp)
        mu = eta[:, :self.out_dim]  # the first coeff are the mean
        lbda = torch.cat(  # the next out_dim ones are the log diag
            (torch.exp(eta[:, self.out_dim:2*self.out_dim]),
             eta[:, 2*self.out_dim:]), 1)
        Lambda = torch.zeros(eta.size(0), self.out_dim, self.out_dim)
        Lambda[:, self.inds[0], self.inds[1]] = lbda  # the lower tri cov sqrt
        return torch.distributions.multivariate_normal.\
            MultivariateNormal(mu, scale_tril=Lambda)  # makes the gaussian pdf


class FcZero(torch.nn.Module):
    """
    Fully connected neural network with ReZero trick
    """
    def __init__(self, layers):
        """
        layers: the list of the linear layers dimensions
        """
        super().__init__()
        n = len(layers)
        self.lins = nn.ModuleList(
            [nn.Linear(d0, d1) for d0, d1 in zip(layers[:-1], layers[1:])])
        self.acts = nn.ModuleList([nn.LeakyReLU()]*(n-1))
        self.alphas = torch.nn.Parameter(torch.zeros(n-1))

    def forward(self, h):
        for lin, act, alpha in zip(self.lins[:-1], self.acts, self.alphas):
            h = h + alpha*act(lin(h))
        return self.lins[-1](h)


class OpFixEta(torch.nn.Module):
    """
    Observation eta with unparameterized mean function and
    unparameterized constant covariance function
    """

    def __init__(self, op_obs, eta_dim, y_dim):
        """
        Inputs
          op_obs: the function H
          eta_dim: the flattened mean-cov sqrt dimension
          y_dim: the observation dimension
        """
        super().__init__()
        self.op_obs = op_obs
        self.lbda = torch.zeros(eta_dim - y_dim)

    def forward(self, x):
        """
        Inputs
          x: a batch of states
        Ouputs
           : a batch of eta vector
        """
        tmp = self.lbda.unsqueeze(0).repeat(x.size(0), 1)
        return torch.cat((self.op_obs(x), tmp), 1)


class OpCstEta(torch.nn.Module):
    """
    Observation eta with unparameterized mean function and
    parameterized constant covariance function
    """
    def __init__(self, op_obs, eta_dim, y_dim):
        """
        Inputs
          op_obs: the function H
          eta_dim: the flattened mean-cov sqrt dimension
          y_dim: the observation dimension
        """
        super().__init__()
        self.op_obs = op_obs
        self.lbda = torch.nn.Parameter(
            torch.zeros(eta_dim - y_dim))

    def forward(self, x):
        """
        Inputs
          x: a batch of states
        Ouputs
           : a batch of eta vector
        """
        tmp = self.lbda.unsqueeze(0).repeat(x.size(0), 1)
        return torch.cat((self.op_obs(x), tmp), 1)


def id_obs(x):
    """
    identity observation operator
    """
    return x


def half_obs(x):
    """
    one coordinate over 2 observation operator
    """
    return x[:, ::2]


class dumbpdf:

    def __init__(self, mu):
        self.mean = mu
