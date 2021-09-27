# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pp
from matplotlib import cm
import pdb
import numpy as np
import pandas as pd
from vihds import utils

def plot_prediction_summary(device_names, signal_names, times, OBS, MU, STD, 
    device_ids, predict_style, fixYaxis = False):
    '''Compare the simulation against the data for the highest weighted sample'''

    nplots = MU.shape[1]
    unique_devices = np.unique(device_ids)
    ndevices = len(unique_devices)    
    
    f, axs = pp.subplots(ndevices, nplots, sharex=True, figsize=(10, 2*ndevices))
    for iu,device_id in enumerate(unique_devices):
        locs = np.where(device_ids == device_id)[0]
        for idx in range(nplots):
            if ndevices > 1:
                ax = axs[iu,idx]
            else:
                ax = axs[idx]
            w_mu = MU[locs, idx, :]
            w_std = STD[locs, idx, :]

            for mu,std in zip(w_mu, w_std):
                ax.fill_between(times, mu-2*std, mu+2*std, color='grey', alpha=0.1)

            ax.plot(times, OBS[locs,idx,:].T, 'r-', lw=1, alpha=1)
            ax.plot(times, w_mu.T, predict_style, lw=1, alpha=0.75, color='k')
            if fixYaxis: ax.set_ylim(-0.2,1.2)

            if iu == ndevices-1: ax.set_xlabel('Time (h)')
            if iu == 0: ax.set_title(signal_names[idx])
            if idx == 0: ax.set_ylabel(device_names[device_id])
    pp.tight_layout()
    sns.despine()

    return f

def plot_weighted_theta(theta_names, TR_iws, TR_theta, TR_device_ids, VL_iws, VL_theta, VL_device_ids, columns2use, sample = True, nsamples=100):
    # make a dataframe so we can call seaborn scatter plot
    order_ids = np.argsort(theta_names)

    n_train,n_train_samples = TR_iws.shape
    n_val,n_val_samples = VL_iws.shape

    # resample with replacement
    TR_samples = []
    for iws in TR_iws:
        if sample:
            # sub-sample according to iws
            samples = np.random.choice(n_train_samples, nsamples, p = iws)
        else:
            # sub-sample uniformly
            samples =  np.random.choice(n_train_samples, nsamples)
        TR_samples.append(samples)

    VL_samples = []
    for iws in VL_iws:
        if sample:
            # sub-sample according to iws
            samples = np.random.choice(n_val_samples, nsamples, p = iws)
        else:
            # sub-sample uniformly
            samples =  np.random.choice(n_val_samples, nsamples)
        VL_samples.append(samples)

    TR_devices = np.tile(TR_device_ids.reshape((n_train,1)), [1,nsamples])
    VL_devices = np.tile(VL_device_ids.reshape((n_val,1)), [1,nsamples])

    names = []
    train_thetas = []
    val_thetas = []
    for theta_idx in order_ids:
        theta_name  = theta_names[ theta_idx ]
        train_theta = []
        #TR_theta[ theta_idx ].flatten()
        val_theta   = [] #VL_theta[ theta_idx ].flatten()

        for samples, values in zip(TR_samples, TR_theta[ theta_idx ]):
            train_theta.append(values[samples])
        for samples, values in zip(VL_samples, VL_theta[ theta_idx ]):
            val_theta.append(values[samples])

        names.append(theta_name)
        train_thetas.append(np.array(train_theta).flatten())
        val_thetas.append(np.array(val_theta).flatten())

    #names.append("weight")
    names.append("device")
    #train_thetas.append(TR_iws.flatten())
    train_thetas.append(TR_devices.flatten())
    #val_thetas.append(VL_iws.flatten())
    val_thetas.append(VL_devices.flatten())

    train_thetas = np.array(train_thetas, dtype=float).T
    #val_thetas = np.array(val_thetas, dtype=float).T
    tr_df = pd.DataFrame(train_thetas, columns = names)
    #vl_df = pd.DataFrame(val_thetas, columns = names)

    #f = pp.figure(figsize=(16,16))
    #ax = f.add_subplot(111)
    sns.set(style="ticks")

    #g = sns.pairplot(tr_df, vars=columns2use,hue="device",height=2.0, plot_kws=dict(s=20, edgecolor="k",linewidth=0.1))
    g = sns.PairGrid(tr_df, hue="device",  vars=columns2use)#, hue_kws={"cmap": ["Blues", "Greens", "Reds"]})
    g = g.map_diag(sns.kdeplot, shade=True, alpha=0.5)
    #g = g.map_lower(sns.jointplot, kind='hex',gridsize=2) #bw=0.5, n_levels=3)

    g = g.map_offdiag(sns.scatterplot, s=20, alpha=0.25, edgecolor='k', linewidth=0.5)
    g = g.add_legend()
    #g = g.map_offdiag(sns.sactter,  kind='hex' kwargs={})
    return g.fig

def species_summary(species_names, treatments, device_ids, times, iw_states, devices, settings, normalise = True):
    '''Plot the simulated latent species'''
    ndevices = len(devices)
    nplots = iw_states.shape[1]
    fs = 14
    treat_max = treatments.max()
    colors = 'grbcmyk'
    
    divisors = [np.max(iw_states[:, idx, :]) if normalise else 1.0 for idx in range(nplots)]
    
    f, axs = pp.subplots(ndevices, nplots, sharex=True, sharey=normalise, figsize=(14, 2*ndevices) )
    for iu,device_id in enumerate(devices):
        for idx in range(nplots):
            if ndevices is 1:
                ax = axs[idx]
            else:
                ax = axs[iu,idx]
            if settings.separate_conditions is True:
                for i,_ in enumerate(settings.conditions):
                    locs = np.where((device_ids == device_id) & (treatments[:,i] > 0.0))[0]
                    mus = iw_states[locs, idx, :] / divisors[idx]
                    #alphas = treatments[locs,i] / treat_max
                    alphas = 0.5
                    ax.plot(np.tile(times, [len(locs), 1]).T, mus.T, '-', lw=1, alpha=alphas, color=colors[i])
            else:
                locs = np.where(device_ids == device_id)[0]
                mus = iw_states[locs, idx, :] / divisors[idx]
                ax.plot(np.tile(times, [len(locs), 1]).T, mus.T, '-', lw=1, color='k')
            if normalise: ax.set_ylim(-0.1,1.1)
            if iu == 0:
                if idx < len(species_names): 
                    ax.set_title(species_names[idx])
                else:
                    ax.set_title("Latent %d"%(idx-len(species_names)))
            ax.set_xticks([0,4,8,12,16])
        if ndevices is 1:
            ax = axs[0]
        else:
            ax = axs[iu,0]
        ax.set_ylabel(settings.pretty_devices[device_id], labelpad=20, fontweight='bold', fontsize=fs)
    sns.despine()
    pp.tight_layout()
        
    # Global axis labels: add a big axis, then hide frame
    f.add_subplot(111, frameon=False)
    pp.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    pp.xlabel("Time (h)", fontsize=fs, labelpad=7)
    if ndevices > 1:
        pp.ylabel("Normalized output", fontsize=fs, labelpad=0)
    else:
        pp.ylabel("Norm. output", fontsize=fs, labelpad=0)
    
    return f

def xval_treatments(res, devices):
    '''Compare the final simulated points against the equivalent data-points to establish functional response'''
    nplots = len(res.settings.signals)
    ndev = len(devices)
    
    ms = 5
    fs = 14
    obs_mk = 'x'
    pred_mk = 'o'
    colors = ['g','r','b']
    edges = ['darkgreen','darkred','darkblue']
    
    f, axs = pp.subplots(ndev, nplots, sharex=True, sharey=True, figsize=(9, 2.2*ndev))    
    for iu,device_id in enumerate(devices):        
        locs = np.where(res.devices == device_id)[0]
        input_values = []
        for ci, _ in enumerate(res.settings.conditions):
            vs = np.exp(res.treatments[:,ci])-1
            input_values.append(vs[locs])

        for j,signal in enumerate(res.settings.signals):
            if ndev > 1:
                ax = axs[iu,j]
            else:
                ax = axs[j]
            mu  = res.iw_predict_mu[locs, j, -1]
            std = res.iw_predict_std[locs, j, -1]
            for ci, cvalues in enumerate(input_values):
                ax.errorbar(cvalues, mu, yerr=std, fmt=pred_mk, ms=ms, lw=1, mec=edges[ci], color=colors[ci], zorder=ci)
                ax.semilogx(cvalues, res.X_obs[locs,j,-1], 'k'+obs_mk, ms=ms, lw=1, color=edges[ci], zorder=ci+20)
            ax.set_ylim(-0.1,1.1)
            ax.tick_params(axis='both', which='major', labelsize=fs)
            ax.set_xticks(np.logspace(0,4,3))
            if j is 0: 
                ax.set_ylabel(res.settings.devices[iu], labelpad=25, fontweight='bold', fontsize=fs)
            if iu is 0: 
                ax.set_title(signal,fontsize=fs)
    
    # Add legend to one of the panels
    if (ndev>1):
        ytext = "Normalized fluorescence"
        ax = axs[0,nplots-1]
    else:
        ytext = "Norm. fluorescence"
        ax = axs[nplots-1]
    dstr = list(map(lambda s: s + " (data)", res.settings.conditions)) 
    mstr = list(map(lambda s: s + " (model)", res.settings.conditions))    
    ax.legend(labels=dstr + mstr)
    
    # Global axis labels: add a big axis, then hide frame
    f.add_subplot(111, frameon=False)
    pp.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    pp.xlabel(' / '.join(res.settings.conditions), fontsize=fs, labelpad=7)
    #pp.xlabel("C$_6$ / C$_{12}$ (nM)", fontsize=fs, labelpad=7)
    pp.ylabel(ytext, fontsize=fs, labelpad=7)
    sns.despine()
    
    return f

def xval_fit_summary(res, device_id, separatedInputs=False):
    '''Summary plot of model-data fit for cross-validation results'''
    nplots = len(res.settings.signals)
    fs = 14
    
    all_locs = []
    if separatedInputs is True:
        nrows = len(res.settings.conditions)
        for i in range(nrows):
            dev_locs = np.where((res.devices == device_id) & (res.treatments[:, i] > 0.0))[0]
            _, indices = np.unique(res.treatments[dev_locs, i], return_index=True)
            all_locs.append(dev_locs[indices])
        f, axs = pp.subplots(nrows, nplots, sharex=True, sharey=True, figsize=(2.2*nplots, 1.6*nrows+1.2))
    else:
        nrows = 1
        dev_locs = np.where(res.devices == device_id)[0]
        _, indices = np.unique(res.treatments[dev_locs, :], return_index=True, axis=0)
        all_locs.append(dev_locs[indices])
        f, axs = pp.subplots(1, nplots, sharey=True, figsize=(2.2*nplots, 2.8))
    
    for i, locs in enumerate(all_locs):
        colors = [ cm.rainbow(x) for x in np.linspace(0, 1, np.shape(locs)[0]) ]    # pylint: disable=no-member
        for idx in range(nplots):
            if nrows > 1:
                ax = axs[i,idx]
            else:
                ax = axs[idx]

            w_mu = res.iw_predict_mu[locs, idx, :]
            w_std = res.iw_predict_std[locs, idx, :]
            ax.set_prop_cycle('color', colors)
            for mu, std in zip(w_mu, w_std):
                ax.fill_between(res.times, mu-2*std, mu+2*std, alpha=0.1)
            ax.plot(res.times, res.X_obs[locs, idx, :].T, '.', alpha=1, markersize=2)
            ax.plot(res.times, w_mu.T, '-', lw=2, alpha=0.75)
            ax.set_xlim(0.0, 17)
            ax.set_xticks([0, 5, 10, 15])
            ax.set_ylim(-0.2, 1.2)
            if (idx == 0) & (nrows > 1):
                ax.set_ylabel(res.settings.conditions[i] + " dilution", labelpad=25, fontweight='bold', fontsize=fs)
            if i == 0:
                ax.set_title(res.settings.signals[idx], fontsize=fs)          

    # Global axis labels: add a big axis, then hide frame
    f.add_subplot(111, frameon=False)
    pp.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    pp.xlabel("Time (h)", fontsize=fs, labelpad=7)
    pp.ylabel("Normalized output", fontsize=fs, labelpad=7)
    pp.tight_layout()
    sns.despine()

    return f

def gen_treatment_str(conditions, treatments, unit=None):
    vstr_list = []
    for k, v in zip(conditions, treatments):
        val = np.exp(v) - 1.0
        if (val > 0.0) & (val < 1.0):
            vstr = '%s = %1.1f'%(k,val)
        else:
            vstr = '%s = %1.0f'%(k,val)
        if unit is not None:
            vstr = '%s %s'%(vstr,unit)
        vstr_list.append(vstr)
    return '\n'.join(vstr_list)

def xval_individual(res, device_id):
    nplots = res.X_obs.shape[1]
    colors = ['tab:gray','r','y','c']
    maxs = np.max(res.X_obs, axis=(0,2))

    fs = 14
    locs = np.where(res.devices == device_id)[0]
    ids = np.argsort(res.ids[locs])
    locs = locs[ids]
    ntreatments = len(locs)
    nrows = int(np.ceil(ntreatments / 2.0))
    f = pp.figure(figsize=(12, 1.2*nrows))
    for col in range(2):
        left = 0.1+col*0.5
        bottom = 0.4/nrows
        width = 0.33/nplots
        dx = 0.38/nplots
        dy = (1-bottom)/nrows
        height = 0.8*dy
        for i in range(nrows):
            loc = locs[i+col*nrows]
            treatment_str = gen_treatment_str(res.settings.conditions, res.treatments[loc])

            for idx, maxi in enumerate(maxs):
                ax = f.add_subplot(nrows, 2*nplots, col*nplots+(nrows-i-1)*2*nplots+idx+1)
                ax.set_position([left+idx*dx, bottom+(nrows-i-1)*dy, width, height])

                mu = res.iw_predict_mu[loc, idx, :]
                std = res.iw_predict_std[loc, idx, :]

                ax.fill_between(res.times, (mu-2*std)/maxi, (mu+2*std)/maxi, alpha=0.25, color=colors[idx])
                ax.plot(res.times, res.X_obs[loc,idx,:]/maxi, 'k.', markersize=2)
                ax.plot(res.times, mu/maxi, '-', lw=2, alpha=0.75, color=colors[idx])
                ax.set_xlim(0.0,17)
                ax.set_xticks([0,5,10,15])
                ax.set_ylim(-0.2,1.2)
                ax.tick_params(axis='both', which='major', labelsize=fs)

                if i==0:
                    pp.title(res.settings.signals[idx], fontsize=fs)
                #if i<nrows-1:
                ax.set_xticklabels([])
                if idx==0:
                    ax.set_ylabel(treatment_str,labelpad=25,fontsize=fs-2)
                else:
                    ax.set_yticklabels([])


        # Add labels
        f.text(left-0.35*dx,0.5, "Normalized output", ha="center", va="center", rotation=90, fontsize=fs)
        f.text(left+2*dx,0,"Time (h)", ha="center", va="bottom", fontsize=fs)

    sns.despine()
    return f

def xval_individual_2treatments(res, device_id):
    '''Multi-panel plot for each sample, with treatments separated into 2 groups'''
    nplots = res.X_obs.shape[1]
    colors = ['tab:gray','r','y','c']
    maxs = np.max(res.X_obs, axis=(0,2))
    
    fs = 14
    both_locs = []
    for col in range(2):
        all_locs = np.where((res.devices == device_id) & (res.treatments[:,col] > 0.0))[0]
        indices = np.argsort(res.treatments[all_locs,col])
        both_locs.append(all_locs[indices])
    
    ntreatments = max(map(len,both_locs))
    f = pp.figure(figsize=(12, 1.5*ntreatments))
    for col,locs in enumerate(both_locs):
        left = 0.1+col*0.5
        bottom = 0.4/ntreatments
        width = 0.33/nplots
        dx = 0.38/nplots
        dy = (1-bottom)/ntreatments
        height = 0.8*dy
        for i,loc in enumerate(locs[:ntreatments]):
            #TODO(ndalchau): Incorporate units into conditions specification (here we assume nM)
            treatment_str = gen_treatment_str(res.settings.conditions, res.treatments[loc], unit='nM')

            for idx, maxi in enumerate(maxs):
                ax = f.add_subplot(ntreatments, 2*nplots, col*nplots+(ntreatments-i-1)*2*nplots+idx+1)
                ax.set_position([left+idx*dx, bottom+(ntreatments-i-1)*dy, width, height])

                mu = res.iw_predict_mu[loc, idx, :]
                std = res.iw_predict_std[loc, idx, :]

                ax.fill_between(res.times, (mu-2*std)/maxi, (mu+2*std)/maxi, alpha=0.25, color=colors[idx])
                ax.plot(res.times, res.X_obs[loc,idx,:]/maxi, 'k.', markersize=2)
                ax.plot(res.times, mu/maxi, '-', lw=2, alpha=0.75, color=colors[idx])
                ax.set_xlim(0.0,17)
                ax.set_xticks([0,5,10,15])
                ax.set_ylim(-0.2,1.2)
                ax.tick_params(axis='both', which='major', labelsize=fs)

                if i==0:
                    pp.title(res.settings.signals[idx], fontsize=fs)
                if i<ntreatments-1:
                    ax.set_xticklabels([])
                if idx==0:
                    ax.set_ylabel(treatment_str,labelpad=25,fontsize=fs-2)
                else:
                    ax.set_yticklabels([])

                sns.despine()

        # Add labels
        f.text(left-0.35*dx,0.5, "Normalized output", ha="center", va="center", rotation=90, fontsize=fs)
        f.text(left+2*dx,0,"Time (h)", ha="center", va="bottom", fontsize=fs)

    return f

def combined_treatments(results, devices):
    '''Compare model-data functional responses to inputs for multiple models'''
    ndev = len(devices)
    nres = len(results)

    ms = 5
    fs = 14
    obs_mk = 'x'
    pred_mk = 'o'

    width = 0.2
    lefts = [0.05,0.57]
    bottom = 0.3/ndev
    dx = 0.23
    dy = (1-bottom)/ndev
    height = 0.9*dy
    c6_idx = 1
    c12_idx = 0
    ids = [2,3]
    colors = ['y','c']
    f, ax = pp.subplots(ndev, 2*nres, sharex=True, figsize=(9, 2.2*ndev+0.5))
    for iu,device_id in enumerate(devices):
        if ndev==1:
            row = ax
            ytext = "Norm. fluorescence"
        else:
            row = ax[iu]
            ytext = "Normalized fluorescence"
        row[0].set_ylabel(results[0].pretty_devices[iu], labelpad=25, fontweight='bold', fontsize=fs)
        for ir,res in enumerate(results):
            locs = np.where(res.devices == device_id)[0]
            OBS = np.transpose(res.X_obs[locs,-1,:],[1,0])
            IW = res.importance_weights[locs]
            PREDICT = np.transpose(res.PREDICT[locs,:],[2,0,1])
            STD = np.transpose(res.STD[locs,:],[2,0,1])
            all_C6 = np.exp(res.treatments[:,c6_idx])-1
            all_C12 = np.exp(res.treatments[:,c12_idx])-1
            C6  = all_C6[locs]
            C12 = all_C12[locs]

            for j,color in zip(ids,colors):
                mu  = np.sum(IW*PREDICT[j], 1)
                var = np.sum(IW*(PREDICT[j]**2 + STD[j]**2), 1) - mu**2
                std = np.sqrt(var)
                for k,(id,C) in enumerate(zip(ids,[C6,C12])):
                    ic = ir+k*nres
                    row[ic].errorbar(C, mu, yerr=std, fmt=pred_mk, mec='k', ms=ms, lw=1, color=color)
                    row[ic].semilogx(C, OBS[id], obs_mk, ms=ms, lw=1, color=color)
            
            if ir>0:
                row[ir].set_yticklabels([])
                row[ir+nres].set_yticklabels([])     
            for k in range(2):
                ic = ir+k*nres
                row[ic].set_position([lefts[k]+ir*dx, bottom+(ndev-iu-1)*dy, width, height])            
                row[ic].set_xticks(np.logspace(0,4,3))
                row[ic].set_ylim(-0.1,1.1)
                row[ic].set_yticks([0.0,0.5,1.0])
                row[ic].tick_params(axis='both', which='major', labelsize=fs)
                if iu==0:
                    row[ic].set_title(res.label,fontsize=fs)
       
    # Global axis labels: add a big axis, then hide frame
    xlabels = ['C$_6$ (nM)','C$_{12}$ (nM)']
    for k,xlabel in enumerate(xlabels):
        f.add_subplot(1,2,k+1, frameon=False, position=[lefts[k], bottom, width+(nres-1)*dx, height+(ndev-1)*dy])
        pp.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        pp.xlabel(xlabel, fontsize=fs, labelpad=10)
        pp.ylabel(ytext, fontsize=fs, labelpad=8)
    
    sns.despine()

    return f

def xval_variable_parameters(res, ncols=2):
    ndata = len(res.ids)
    qs = dict(zip(res.q_names, res.q_values))
    
    devices = np.unique(res.devices)
    indexes = np.unique([n.split('.')[0] for n in res.q_names], return_index=True)[1]
    all_ps = [[n.split('.')[0] for n in res.q_names][index] for index in sorted(indexes)]
    
    ps = []
    for i,p in enumerate(all_ps):
        if p+'.mu' in qs:
            if np.shape(qs[p + '.mu'])[0] == ndata:
                ps.append(p)
    
    if utils.is_empty(ps):
        print("- No variables parameters: not producing plot")
        return
    print('- ', ps)
    
    # Define data and device-dependent colours
    cdict = dict(zip(devices, sns.color_palette()))
    
    # Define geometry and figures
    nrows = np.ceil(len(ps) / ncols).astype(int)
    f, axs = pp.subplots(nrows, ncols, sharex=True, figsize=(6*ncols,2*nrows))
    f.suptitle('Local parameters', fontsize=14)
    for i in range(nrows):
        for j in range(ncols):
            if nrows > 1:
                ax = axs[i,j]
            else:
                ax = axs[j]
            if (j+i*ncols) < len(ps):
                name = ps[j+i*ncols]
                for di in devices:
                    locs = np.where(res.devices == di)
                    x = res.ids[locs]
                    y = np.squeeze(qs['%s.mu'%name][locs])
                    err = np.squeeze(1 / qs['%s.prec'%name][locs])
                    ax.errorbar(x, y, err, fmt='.', color=cdict[di])
                    ax.set_title(name)
                if i == (nrows-1):
                    ax.set_xlabel('Data instance')
            else:
                if i > 0:
                    axs[i-1,j].set_xlabel('Data instance')
                ax.set_visible(False)            
        if nrows > 1:
            axs[i,0].set_ylabel('Parameter value')
        else:
            axs[0].set_ylabel('Parameter value')
    f.tight_layout(rect=(0,0,1,0.97))
    sns.despine()

    return f

def xval_global_parameters(res, ncols=6):
    ndata = len(res.ids)
    nfolds = len(res.chunk_sizes)
    qs = dict(zip(res.q_names, res.q_values))
    
    indexes = np.unique([n.split('.')[0] for n in res.q_names], return_index=True)[1]
    all_ps = [[n.split('.')[0] for n in res.q_names][index] for index in sorted(indexes)]

    ps = []
    for i,p in enumerate(all_ps):
        if p+'.mu' in qs:
            if np.shape(qs[p + '.mu'])[0] < ndata:
                ps.append(p)
    print('- ', ps)
    if utils.is_empty(ps):
        print("- No global parameters: not producing plot")
        return
    
    # Define geometry and figures
    n = len(ps)
    if n < ncols:
        ncols = n
    nrows = np.ceil(n / ncols).astype(int)
    f, axs = pp.subplots(nrows, ncols, figsize=(2*ncols,2*nrows))
    f.suptitle('Global parameters', fontsize=14)
    for i in range(nrows):
        for j in range(ncols):
            if nrows > 1:
                ax = axs[i,j]
            else:
                ax = axs[j]
            if (j+i*ncols) < len(ps):
                name = ps[j+i*ncols]
                ax.errorbar(np.linspace(1,nfolds,nfolds),qs['%s.mu'%name], 1 / qs['%s.prec'%name], fmt='.')
                ax.set_title(name)
                ax.set_xlim([0.5, nfolds + 0.5])
                ax.set_xticks(range(1,nfolds+1))
                if i == (nrows-1):
                    ax.set_xlabel('Fold')
            else:
                if i > 0:
                    axs[i-1,j].set_xlabel('Fold')
                ax.set_visible(False)
        if nrows > 1:
            axs[i,0].set_ylabel('Parameter value')
        else:
            axs[0].set_ylabel('Parameter value')
    f.tight_layout(rect=(0,0,1,0.96))
    sns.despine()

    return f