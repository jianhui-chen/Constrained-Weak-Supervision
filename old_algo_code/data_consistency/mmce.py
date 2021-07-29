""" Code adapted from https://www.cs.utexas.edu/~lqiang/codes/crowd_imcl14_demo.zip
    Original code in Matlab
"""
import os
import sys
import __main__ as main
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from time import time


class Dummy:
    pass


def initial_fields():
    Key = Dummy()
    Key.L = []
    Key.A = []
    Key.task_labels = []
    Key.worker_labels = []
    Key.label_domain = []
    Key.no_tasks = []
    Key.no_workers = []
    Key.class_counts = []
    Key.no_workers_per_task = []
    Key.no_tasks_per_worker = []
    Key.worker_indices_per_task = []
    Key.task_indices_per_worker = []
    Key.true_labels = []
    return Key


def crowd_model(L, **kwargs):
    #### generate a crowd_model structure based on the labeling matrix L ###
    A = (L != 0).astype(int)
    # parameters ...
    no_tasks = L.shape[0]  # Number of task
    no_workers = L.shape[1]  # Number of workers
    no_workers_per_task = np.sum(A, axis=1)
    no_tasks_per_worker = np.sum(A, axis=0)
    # neiborhoods
    worker_indices_per_task = [0 for i in range(no_tasks)]
    for i in range(no_tasks):
        worker_indices_per_task[i] = np.argwhere(A[i, :]).flatten()
    task_indices_per_worker = [0 for i in range(no_workers)]
    for j in range(no_workers):
        task_indices_per_worker[j] = np.argwhere(A[:, j]).flatten()
    Key = initial_fields()
    task_labels = [0 for i in range(no_tasks)]
    for i in range(no_tasks):
        task_labels[i] = L[i, worker_indices_per_task[i]]
    worker_labels = [0 for i in range(no_workers)]
    for j in range(no_workers):
        worker_labels[j] = L[task_indices_per_worker[j], j]
    Key.task_labels = task_labels
    Key.worker_labels = worker_labels
    label_domain = np.unique(L[L != 0])
    Key.label_domain = label_domain
    Key.class_counts = len(label_domain)
    Key.no_tasks = no_tasks
    Key.no_workers = no_workers
    Key.no_workers_per_task = no_workers_per_task
    Key.no_tasks_per_worker = no_tasks_per_worker
    Key.worker_indices_per_task = worker_indices_per_task
    Key.task_indices_per_worker = task_indices_per_worker
    Key.L = L
    Key.A = A
    for k, v in kwargs.items():
        setattr(Key, k, v)

    return Key


def MajorityVote_crowd_model(Model):
    np.random.seed(1)
    verbose = 1
    weights = np.ones(Model.no_workers)
    breakties = 'random'
    # majority voting
    L = Model.L
    no_tasks = Model.no_tasks
    worker_indices_per_task = Model.worker_indices_per_task
    ans_labels = np.zeros(no_tasks)
    domb = Model.label_domain
    Model.class_counts = len(Model.label_domain)
    mu = np.zeros((Model.class_counts, no_tasks))
    mu_count = np.zeros((Model.class_counts, no_tasks))
    # main algorithm
    for i in range(no_tasks):
        labvec = L[i, worker_indices_per_task[i]]
        wts = weights[worker_indices_per_task[i]]
        if len(labvec) == 0:
            ans_labels[i] = np.nan
            continue
        numb = np.zeros(len(domb))
        for ii in range(len(domb)):
            numb[ii] = np.sum(wts[labvec == domb[ii]])
        mu_count[:, i] = numb
        mu[:, i] = np.exp(numb-np.max(numb))
        mu[:, i] = mu[:, i]/np.sum(mu[:, i])
        dx = np.argwhere(np.max(numb) == numb)
        # break ties
        if dx.size > 1:
            if breakties == 'random':
                if verbose >= 1:
                    print(
                        'Marjority Voting: ties happens in %d-th task, randomly select one' % i)
                np.random.shuffle(dx)
                dx = dx[0]
            elif breakties == 'first':
                if verbose >= 1:
                    print(
                        'Marjority Voting: ties happens in %d-th task, select the first one' % i)
                dx = dx[0]
            else:
                raise Exception('wrong definition of breaking ties')
        ans_labels[i] = domb[dx]
    Key_mvote = Dummy()
    Key_mvote.ans_labels = ans_labels
    if (hasattr(Model, 'true_labels')) and (len(Model.true_labels) > 0):
        Key_mvote.MoreInfo = Dummy()
        Key_mvote.error_rate = np.mean(ans_labels != Model.true_labels)
    Key_mvote.counts = mu_count
    if verbose >= 1:
        if hasattr(Key_mvote, 'error_rate'):
            print(main.__file__, '\t-- error rate = %f' % Key_mvote.error_rate)
    return Key_mvote


def DawidSkene_crowd_model(Model):
    verbose = 1
    maxIter = 100
    TOL = 1e-3
    L = Model.L
    prior_workers = []
    prior_tasks = []
    partial_truth = [[], []]
    no_tasks = Model.no_tasks
    no_workers = Model.no_workers
    worker_indices_per_task = Model.worker_indices_per_task
    task_indices_per_worker = Model.task_indices_per_worker
    label_domain = Model.label_domain
    class_counts = len(label_domain)
    partial_dx = partial_truth[0]
    partial_array = np.ones(
        (Model.class_counts, len(partial_dx)))/Model.class_counts
    eps = np.spacing(1)
    for i in range(len(partial_dx)):
        partial_array[:, i] = eps
        partial_array[partial_truth[2][i], i] = 1-eps
        partial_array[:, i] = partial_array[:, i]/np.sum(partial_array[:, i])
    other_dx = np.setdiff1d(range(no_tasks), partial_dx)
    # set default prior parameters
    if len(prior_tasks) == 0:
        prior_tasks = np.ones((class_counts, no_tasks))/class_counts
    if len(prior_workers) == 0:
        prior_workers = np.ones((class_counts, class_counts))
    alpha = np.ones((class_counts, class_counts, no_workers))
    mu = np.zeros((class_counts, no_tasks))
    # initializing mu using frequency counts
    for i in range(no_tasks):
        neib = worker_indices_per_task[i]
        labs = L[i, neib]
        for k in range(len(label_domain)):
            mu[k, i] = prior_tasks[k, i] * \
                np.sum(labs == label_domain[k]) / len(labs)
        mu[:, i] = mu[:, i]/np.sum(mu[:, i])
    mu[:, partial_dx] = partial_array
    # main iteration
    err = np.nan
    for iter_ in range(maxIter):
        # M-Step: Updating workers' confusion matrix (alpha)
        for j in range(no_workers):
            neib = task_indices_per_worker[j]
            labs = L[neib, j]
            alpha[:, :, j] = prior_workers - 1 + eps
            for ell in range(class_counts):
                dx = neib[labs == label_domain[ell]]
                alpha[:, ell, j] = alpha[:, ell, j] + np.sum(mu[:, dx], axis=1)
        alpha = alpha / np.expand_dims(np.sum(alpha, axis=1), axis=1)
        # E-Step: Updating tasks' posterior probabilities (mu)
        old_mu = mu.copy()
        for i in other_dx:
            neib = worker_indices_per_task[i]
            labs = L[i, neib]
            tmp = 0
            for ell in range(class_counts):
                jdx = neib[labs == label_domain[ell]]
                tmp = tmp + np.sum(np.log(alpha[:, ell, jdx]), axis=-1)
            mu[:, i] = prior_tasks[:, i] * np.exp(tmp - np.max(tmp))
            mu[:, i] = mu[:, i]/np.sum(mu[:, i])
        err = np.max(np.abs(old_mu-mu)).astype('double')
        if verbose >= 2:
            print('%s: %d-th iteration, converge error=%d\n' %
                  (main.__file__, iter_, err))
        if err < TOL:
            break
    # decode the labels of tasks
    # add some random noise to break ties.
    mxdx = np.argmax(mu*(1+np.random.uniform(0, 1, size=mu.shape))*eps, axis=0)
    ans_labels = label_domain[mxdx]
    Key = Dummy()
    Key.ans_labels = ans_labels
    if hasattr(Model, 'true_labels') and (len(Model.true_labels) > 0):
        Key.MoreInfo = Dummy()
        Key.error_rate = np.mean(Key.ans_labels != Model.true_labels)
    Key.soft_labels = mu
    Key.parameter_worker = alpha
    Key.converge_error = err.astype('double')
    # Print out final information
    if verbose >= 1:
        printstr = '%s:\n\t-- break at %dth iteration, congerr=%f\n' % (
            main.__file__, iter_, err)
        if hasattr(Key, 'error_rate'):
            printstr = printstr + '\t-- error rate = %f' % Key.error_rate
    print(printstr)
    return Key


def obj_update_alpha_minimaxent_per_worker(alpha, wX_neibj, mu_neibj, labs, label_domain, lambda_=0, offdiag=False):
    obj = 0
    class_counts = len(label_domain)
    alpha = np.reshape(alpha, (class_counts, class_counts)).T
    Dobj_Dalpha = np.zeros(alpha.shape)
    for i in range(len(labs)):
        wxi = wX_neibj[:, :, i]
        MM = wxi + alpha
        maxtmp = MM.max(axis=-1, keepdims=True)
        expMM = np.exp(MM - maxtmp)
        sumexpMM = np.sum(expMM, axis=-1, keepdims=True)
        logz = np.log(sumexpMM) + maxtmp
        obj = obj + np.dot(MM[:, labs[i]-1] - logz.flatten(), mu_neibj[:, i])
        probmatrix = expMM / sumexpMM
        Dobj_Dalpha[:, labs[i]-1] = Dobj_Dalpha[:, labs[i]-1] + mu_neibj[:, i]
        Dobj_Dalpha = Dobj_Dalpha - mu_neibj[:, i:i+1] * probmatrix
    if offdiag:
        tmp = np.diag(np.diag(alpha))
        obj = -obj + 0.5 * lambda_ * \
            ((alpha**2).sum() - (np.diag(alpha)**2).sum())
        Dobj_Dalpha = -Dobj_Dalpha.T.flatten() + lambda_ * \
            (alpha.T.flatten() - tmp.T.flatten())
    else:
        obj = -obj + 0.5 * lambda_ * (alpha**2).sum()
        Dobj_Dalpha = -Dobj_Dalpha.T.flatten() + lambda_ * alpha.T.flatten()
    return obj, Dobj_Dalpha


def obj_update_wx_per_task(wxi, mui, alpha_neibi, labs, label_domain, lambda_=0):
    class_counts = len(label_domain)
    wxi = np.reshape(wxi, (class_counts, class_counts)).T
    obj = 0
    Dobj_Dw = 0*wxi
    for j in range(len(labs)):
        MM = wxi + alpha_neibi[:, :, j]
        maxtmp = MM.max(axis=-1, keepdims=True)
        expMM = np.exp(MM - maxtmp)
        sumexpMM = np.sum(expMM, axis=-1, keepdims=True)
        logz = np.log(sumexpMM) + maxtmp
        obj = obj + np.dot(MM[:, labs[j]-1] - logz.flatten(), mui)
        probmatrix = expMM / sumexpMM
        Dobj_Dw[:, labs[j]-1] = Dobj_Dw[:, labs[j]-1] + mui
        Dobj_Dw = Dobj_Dw - mui.reshape((-1, 1)) * probmatrix

    obj = -obj + 0.5 * lambda_ * np.sum(wxi**2)
    Dobj = -Dobj_Dw.T.flatten() + lambda_ * wxi.T.flatten()
    return obj, Dobj


def minFunc(funObj, x0, options):
    opt_res = minimize(funObj, x0, method=options.Method, jac=True,
                       tol=options.progTOL,
                       options={'maxiter': options.maxIter, 'disp': False})
    return opt_res.x


def logsumexp2_stable(tmp):
    maxtmp = np.max(tmp, axis=-1, keepdims=True)
    logz = np.log(np.sum(np.exp(tmp - maxtmp),
                  axis=-1, keepdims=True)) + maxtmp
    return logz


def MinimaxEntropy_Categorical_crowd_model(Model, opts):
    verbose, inner_maxIter, damping = opts['verbose'], 1, 0
    maxIter, TOL = opts['maxIter'], opts['TOL']
    prior_workers = np.array([])
    prior_tasks = np.array([])
    offdiag = False
    L = Model.L
    lambda_alpha_Vec, lambda_w_Vec = opts['lambda_worker'], opts['lambda_task']
    update_alpha, update_w = True, True
    no_tasks = Model.no_tasks
    no_workers = Model.no_workers
    worker_indices_per_task = Model.worker_indices_per_task
    task_indices_per_worker = Model.task_indices_per_worker
    label_domain = Model.label_domain
    class_counts = len(label_domain)
    if np.array(lambda_alpha_Vec).size == 1:
        lambda_alpha_Vec = lambda_alpha_Vec*np.ones(no_workers)
    if np.array(lambda_w_Vec).size == 1:
        lambda_w_Vec = lambda_w_Vec*np.ones(no_tasks)
    if prior_tasks.size == 0:
        prior_tasks = np.ones((class_counts, no_tasks))/class_counts
    if prior_workers.size == 0:
        prior_workers = np.ones((class_counts, class_counts))
    # set optimization parameters with LBFGS
    maxIter_Mstep, optTOL_Mstep, progTOL_Mstep = 25, 1e-3, 1e-3
    options = Dummy()
    options.bbType = 1
    options.Method = 'L-BFGS-B'
    options.maxIter = maxIter_Mstep
    options.optTol = optTOL_Mstep
    options.progTOL = progTOL_Mstep
    if verbose > 3:
        options.Display = 'iter'
    elif verbose > 2:
        options.Display = 'final'
    else:
        options.Display = 'off'
    # initializing mu (the posterior prob) using majority voting counting
    mu = np.zeros((len(label_domain), no_tasks))
    for i in range(no_tasks):
        neib = worker_indices_per_task[i]
        labs = L[i, neib]
        for k in range(len(label_domain)):
            mu[k, i] = (labs == label_domain[k]).sum()/len(labs)
    wX = np.zeros((class_counts, class_counts, no_tasks))
    alpha = np.zeros((class_counts, class_counts, no_workers))
    logp_task = np.zeros((class_counts, no_tasks))
    cal_truth = False
    if hasattr(Model, 'true_labels'):
        cal_truth = True
        true_labels = Model.true_labels
        dx_with_ans = np.argwhere(np.isfinite(true_labels))
        prob_err_Vec = np.zeros(maxIter)
    # main iteration
    err = np.nan
    tic = time()
    for iter_ in range(maxIter):
        # M step
        for inner_iter in range(inner_maxIter):
            # M step: update alpha (confusion matrix)
            if update_alpha:
                for j in range(no_workers):
                    neib = task_indices_per_worker[j]
                    labs = L[neib, j].T
                    mu_neibj = mu[:, neib]
                    wX_neibj = wX[:, :, neib]
                    lambda_alpha_v = lambda_alpha_Vec[j]

                    def obj_handle(unknown): return obj_update_alpha_minimaxent_per_worker(
                        unknown, wX_neibj, mu_neibj, labs, label_domain, lambda_alpha_v, offdiag)
                    alpha_tmp = minFunc(obj_handle, np.reshape(
                        alpha[:, :, j].T, (-1, 1)), options)
                    alpha[:, :, j] = np.reshape(
                        alpha_tmp, (class_counts, class_counts)).T
            # M step: update wx (tasks confusion)
            if update_w:
                for i in range(no_tasks):
                    neib = worker_indices_per_task[i]
                    labs = L[i, neib]
                    mui = mu[:, i]
                    alpha_neibi = alpha[:, :, neib]
                    lambda_w_v = lambda_w_Vec[i]
                    def obj_handle(wxi): return obj_update_wx_per_task(
                        wxi, mui, alpha_neibi, labs, label_domain, lambda_w_v)
                    wX_tmp = minFunc(obj_handle, np.reshape(
                        wX[:, :, i].T, (-1, 1)), options)
                    wX[:, :, i] = np.reshape(
                        wX_tmp, (class_counts, class_counts)).T
        # E step: update posterior distribution of the labels (mu)
        old_mu = mu.copy()
        for i in range(no_tasks):
            neib = worker_indices_per_task[i]
            labs = L[i, neib]
            logp_task[:, i] = np.log(prior_tasks[:, i])
            for jdx in range(len(neib)):
                j = neib[jdx]
                tmp = alpha[:, :, j] + wX[:, :, i]
                logz = logsumexp2_stable(tmp)
                logp_task[:, i] = logp_task[:, i] + \
                    tmp[:, labs[jdx]-1] - logz.flatten()
            eps = np.spacing(1)
            tmp = logp_task[:, i] + damping * np.log(old_mu[:, i]+eps)
            mu[:, i] = np.exp(tmp - np.max(tmp))
            mu[:, i] = mu[:, i]/np.sum(mu[:, i])
        # check the convergence error
        err = np.max(np.abs(old_mu-mu))
        if verbose >= 2:
            printstr = '%s: iter=%d, congerr=%f, ' % (
                main.__file__, iter_, err)
        # evaluate ground truth
        if cal_truth:
            # noise = (1+np.random.uniform(0,1,size=mu.shape))*eps
            # mxdx = np.argmax(mu*noise, axis=0) # add some random noise to break ties.
            mxdx = np.argmax(mu, axis=0)
            ans_labels = label_domain[mxdx]
            prob_err_Vec[iter_] = np.mean(
                ans_labels[dx_with_ans] != true_labels[dx_with_ans])
            if verbose >= 2:
                printstr = printstr + 'err_rate=%f, ' % prob_err_Vec[iter_]
        if verbose >= 2:
            printstr = printstr + 'time=%f' % (time()-tic)
        if verbose >= 2:
            print(printstr)
        if err < TOL:
            break
    # check the error rate
    Key = Dummy()
    # noise = (1+np.random.uniform(0,1,size=mu.shape))*eps
    # mxdx = np.argmax(mu*noise, axis=0) # add some random noise to break ties.
    mxdx = np.argmax(mu, axis=0)
    Key.ans_labels = label_domain[mxdx]
    if cal_truth:
        Key.error_rate = np.mean(Key.ans_labels != true_labels)
    Key.soft_labels = mu
    Key.parameter_worker = alpha
    Key.parameter_task = wX
    Key.converge_error = err
    # Print out final information
    if verbose >= 1:
        printstr = '%s:\n\t-- break at %dth iteration, congerr=%f\n' % (
            main.__file__, iter_, err)
        if cal_truth:
            printstr = printstr + '\t-- error rate = %f' % Key.error_rate
        print(printstr)
    return Key


def MinimaxEntropy_crowd_model(label_matrix, true_labels):
    Model = crowd_model(label_matrix, true_labels=true_labels)
    # Set parameters:
    lambda_worker = 0.25*Model.class_counts**2
    lambda_task = lambda_worker * (np.mean(Model.no_tasks_per_worker)/np.mean(
        Model.no_workers_per_task))  # regularization parameters
    opts = {'lambda_worker': lambda_worker, 'lambda_task': lambda_task, 'maxIter': 50,
            'TOL': 5*1e-3, 'verbose': 1}
    result = MinimaxEntropy_Categorical_crowd_model(Model, opts)
    return result.ans_labels, result.error_rate


if __name__ == '__main__':
    np.random.seed(4)
    data = np.random.randint(3, size=(800, 10)) + 1
    true_labels = np.random.randint(3, size=800) + 1
    print(data.shape, true_labels.shape)

    Model = crowd_model(data, true_labels=true_labels)
    # Majority voting:
    mv = MajorityVote_crowd_model(Model)
    print("Majority voting", mv.ans_labels)
    # Dawid & Sknene:
    ds = DawidSkene_crowd_model(Model)
    print("Dawid & Skene", ds.ans_labels)

    labels, error_rate = MinimaxEntropy_crowd_model(data, true_labels)
