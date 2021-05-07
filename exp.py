"""
This file writes or loads a config dict and lauches the specified DAN
experiment. Else it prints a config dict, or
plot an experiment result.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sys import argv
from os import path, mkdir, system
import pickle
import nets
from copy import deepcopy

DIS_DIR = "afillion@osirim-slurm.irit.fr:" +\
    "/projets/aniti-daml/daml/afillion/DAN/" +\
    argv[0][:-3]  # Data directory on the distant computer
LOC_DIR = argv[0][:-3]  # Data directory on the local computer
OVERWRITE = True

if len(argv) == 1:
    """
    COMPUTE MODE:
    - The config file is saved if overwrite else loaded
    - The experiment is launched
    """

    if OVERWRITE:

        # CONFIG DICT #########################################################
        #   train_then_test parameters
        kttt = {}
        # Files
        kttt["exp"] = EXP =\
            "exp"  # Experiment name (no _)
        kttt["directory"] = DIRECTORY =\
            "/projets/aniti-daml/daml/afillion/DAN/" +\
            argv[0][:-3]  # Data directory
        kttt["comments"] = "nn 20 layers"
        # Dimensions
        kttt["x_dim"] = x_dim = 40  # State dimension
        kttt["h_dim"] = h_dim = 20*x_dim  # Memory dimension
        kttt["batch_sizes"] = batch_sizes = {"train": 2**10, "test": 1}
        kttt["burn"] = 10**3  # Burn in to skip system transient regime of
        # Controls
        rep = 6
        kttt["modes"] = rep*["train", "test"]
        kttt["load_weights"] = False
        kttt["append_outputs"] = {"train": False, "test": False}
        kttt["load_state"] = {"train": [False] + (rep-1)*[True, True],
                              "test": rep*[False]}
        kttt["seeds"] = {"train": rep*[0],
                         "test": [i for i in range(1, rep+1)]}
        # test parameters
        kttt["ktest"] = k = {}
        # k["net"] = None  # defined in train_then_test
        k["fprint"] = fprint = 10**3  # print frequency
        k["n_obs"] = n_obs = fprint*10**2  # nb of observations
        k["op_obs"] = op_obs = nets.id_obs  # true obs operator
        if op_obs == nets.id_obs:
            k["y_dim"] = y_dim = x_dim  # its out dim
        elif op_obs == nets.half_obs:
            k["y_dim"] = y_dim = int((x_dim+1)/2)  # its out dim
        else:
            print("UNKNOWN obs operator")
        k["sigma_r"] = 1.  # observation error std
        # k["op_prop"] = lambda x: M(x, 1)  # propagation operator
        k["sigma_q"] = .1  # propagation error std
        # k["x_truth"] = None  # defined in train_then_test
        # k["hidden_state"] = None  # defined in train_then_test
        k["direxp"] = DIRECTORY + "/" + EXP  # experiment directory
        k["outputs"] = {"rmse_b": np.array([]),
                        "rmse_a": np.array([]),
                        "logpdf": np.array([])}
        #   train parameters
        kttt["ktrain"] = k = deepcopy(kttt["ktest"])
        k["opt_loss"] = "logpdf"
        k["DAW_step"] = T = 1
        k["QS_DAW_lengths"] = [T]
        k["QS_DAW_iterations"] = [1]
        k["tolerance"] = 1.
        k["opti"] = {"optimizer": "Adam",
                     "lr": 10**(-4),
                     "momentum": 0.,
                     "nesterov": False}
        #   DAN parameters
        kttt["kDAN"] = k = {}
        if kttt["ktrain"]["opt_loss"] == "elbo":
            k["training_mode"] = "unsupervised"
        else:
            k["training_mode"] = "supervised"
        k["w_dim"] = w_dim = 20  # reparameterization trick sample nb
        #   Analyzer parameters
        nblayers = 20
        kttt["kDAN"]["analyzer"] = k = {}
        k["module"] = nets.FcZero  # its nn.module
        k["kwargs"] = {"layers": nblayers*[h_dim + y_dim] + [h_dim]}  # its args
        #   Propagater parameters
        kttt["kDAN"]["propagater"] = k = {}
        k["module"] = nets.FcZero  # its nn.module
        k["kwargs"] = {"layers": nblayers*[h_dim] + [h_dim]}  # its args
        #   Procoder parameters
        eta_dim = x_dim + int(x_dim*(x_dim+1)/2)
        kttt["kDAN"]["procoder"] = k = {}
        k["module"] = nets.Coder  # its nn.module
        k["kwargs"] = {
            "eta_func": nets.FcZero,
            "eta_args": {"layers": [h_dim] + [eta_dim]},
            "eta_dim": eta_dim,
            "out_dim": x_dim}

        # save the parameter dict
        if not path.exists(DIRECTORY):
            mkdir(DIRECTORY)
        with open(DIRECTORY+"/"+EXP+"_kttt.pkl", "wb") as f:
            pickle.dump(kttt, f, pickle.HIGHEST_PROTOCOL)

    else:
        # load the parameter dict
        with open(DIRECTORY+"/"+EXP+"_kttt.pkl", "rb") as f:
            CFG = pickle.load(f)

    # launch train_then_test
    print("--> "+EXP+" <--")
    nets.train_then_test(**kttt)

else:
    """
    POSTPROC MODE
    - argv[1]: print | plot
    - argv[2]: exp
    - argv[3]: dl | ndl
    """
    EXP = argv[2]
    if argv[3] == "dl":
        DOWNLOAD = True
    else:
        DOWNLOAD = False

    # download and load kttt dict
    if not path.exists(LOC_DIR+"/"+EXP+"_kttt.pkl") or OVERWRITE:
        if not path.exists(LOC_DIR):
            mkdir(LOC_DIR)
        if DOWNLOAD:
            system("scp "+DIS_DIR+"/"+EXP+"_kttt.pkl "+LOC_DIR)
    with open(LOC_DIR+"/"+EXP+"_kttt.pkl", "rb") as f:
        kttt = pickle.load(f)

    if argv[1] == "print":
        """
        The exp config dict is printed
        """

        def printrec(v, prefix):
            if isinstance(v, dict):
                print(prefix+"{")
                for k, val in v.items():
                    printrec(val, len(prefix)*" "+" "+k)
                print(prefix+"}")
            elif isinstance(v, tuple):
                print(prefix+"(")
                for val in v:
                    printrec(val, len(prefix)*" "+"  ")
                print(prefix+")")
            elif isinstance(v, list):
                print(prefix+"[")
                for val in v:
                    printrec(val, len(prefix)*" "+"  ")
                print(prefix+"]")
            else:
                print(prefix+": "+str(v))

        print("=== "+kttt["exp"]+" ===")
        printrec(kttt, "kttt")
        print("======================")

    elif argv[1] == "plot":
        """
        The exp outputs are plotted
        - argv[4] = tr | None
        """

        DIREXP = LOC_DIR+"/"+EXP

        # download distant data
        if not path.exists(LOC_DIR):
            mkdir(LOC_DIR)
        if DOWNLOAD:
            system("scp "+DIS_DIR+"/"+EXP+"_*.npy "+LOC_DIR)

        if len(argv) < 5:
            modes = ["train", "test"]
        elif argv[4] == "tr":
            modes = ["train"]
        else:
            modes = ["test"]

        fprint = kttt["ktest"]["fprint"]
        n_obs = kttt["ktest"]["n_obs"]
        nrep = 6
        nite = 10**5
        start = 0  # int(10**5/fprint)
        stop = -1  # int(0.5*10**6/fprint)
        step = 2500
        nplots = 2

        p = mpl.rcParams
        p["figure.figsize"] = 1.5*6.4, 4.8
        p["figure.subplot.left"] = 0.07
        p["figure.subplot.right"] = 0.99
        p["figure.subplot.bottom"] = 0.10
        p["figure.subplot.top"] = 0.83
        p["figure.subplot.wspace"] = 0.10
        p["figure.subplot.hspace"] = 0.2
        p["legend.columnspacing"] = 1.4

        fig, ax = plt.subplots(1, nplots)
        linestyles = {"train": "solid", "test": "solid"}
        linewidth = {"train": 3, "test": 2}
        opacity = {"train": 1., "test": 1}
        markers = {"train": "None", "test": "None"}
        idplot = {"rmse": 1, "rmse_b": 1, "rmse_a": 1,
                  "logpdf": 0, "logpdf_b": 0, "logpdf_a": 0,
                  "elbo": 2, "rec": 2, "reg": 2}
        colors = {"test": {"rmse": "C0", "rmse_b": "C1", "rmse_a": "C2",
                           "logpdf": "C0", "logpdf_b": "C1", "logpdf_a": "C2",
                           "elbo": "C0", "rec": "C1", "reg": "C2"},
                  "train": {"rmse": "C0", "rmse_b": "sienna", "rmse_a": "darkgreen",
                            "logpdf": "darkblue", "logpdf_b": "C1", "logpdf_a": "C2",
                            "elbo": "C0", "rec": "C1", "reg": "C2"}}
        zorder = {"test": 0, "train": 2}
        names = {"logpdf": r"$L_t$", "rmse_b": r"rmse$_t^{b}$", "rmse_a": r"rmse$_t^{a}$"}

        for mode in modes:
            for key in kttt["k"+mode]["outputs"].keys():
                tmp = np.load(DIREXP+"_"+mode+"_"+key+".npy")
                if mode == "test":
                    print(key + " mean= "+str(np.mean(tmp)))
                if np.any(np.isnan(tmp)):
                    print("WARNING NAN IN DATA")
                if mode == "train":
                    tmp2 = tmp[start:stop:step]
                    ax[idplot[key]].plot([start + i*step for i in range(len(tmp2))],
                                         tmp2,
                                         label=mode+" "+names[key],
                                         linestyle=linestyles[mode],
                                         linewidth=linewidth[mode],
                                         color=colors[mode][key],
                                         marker=markers[mode],
                                         alpha=opacity[mode],
                                         zorder=zorder[mode])
                elif mode == "test":
                    for r in range(1, nrep, 2):
                        start2 = start+r*nite
                        stop2 = start+(r+1)*nite
                        tmp2 = tmp[start2:stop2:step]
                        if r == 1:
                            ax[idplot[key]].plot([start2+i*step for i in range(len(tmp2))],
                                                 tmp2,
                                                 label=mode + " " + names[key],
                                                 linestyle=linestyles[mode],
                                                 linewidth=linewidth[mode],
                                                 color=colors[mode][key],
                                                 marker=markers[mode],
                                                 alpha=opacity[mode],
                                                 zorder=zorder[mode])
                        else:
                            ax[idplot[key]].plot([start2+i*step for i in range(len(tmp2))],
                                                 tmp2,
                                                 linestyle=linestyles[mode],
                                                 linewidth=linewidth[mode],
                                                 color=colors[mode][key],
                                                 marker=markers[mode],
                                                 alpha=opacity[mode],
                                                 zorder=zorder[mode])
                        # IEnKF-Q
                        if key == "rmse_a":
                            tmp3 = np.load("exp_IEnKFQ/IEnKFQ_test_rmse_a_"+str(int((r-1)/2))+".npy")
                            tmp3 = tmp3[0:-1:step]
                            if r == nrep-1:
                                ax[1].plot([start2 + i*step for i in range(len(tmp3))],
                                           tmp3,
                                           color='C3',
                                           label=r"IEnKF-Q rmse$^{a}_t$",
                                           alpha=opacity["test"],
                                           zorder=1)
                            else:
                                ax[1].plot([start2 + i*step for i in range(len(tmp3))],
                                           tmp3,
                                           color='C3',
                                           alpha=opacity["test"],
                                           zorder=1)
        for i in range(nplots):
            ax[i].set_xticks([i for i in range(0, 6*10**5, 10**5)])
            # ax[i].set_xticklabels([i*step for i in range(len(tmp))])
            # ax[i].set_title(["rmse", "logpdf", "ELBO"][i])
            ax[i].set_yscale("log")
            ax[i].set_xlabel("iterations")
            ax[i].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax[i].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                         ncol=3, mode=None, borderaxespad=0.)
            ax[i].xaxis.grid()
        ax[1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.1f}'))
        ax[1].yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
        # fig.tight_layout()
        plt.savefig(DIREXP+".pdf")
        plt.show()

#  LocalWords:  fprint projets afillion
