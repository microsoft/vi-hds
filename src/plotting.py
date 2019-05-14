# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pp
import matplotlib.cm as cmx
import pdb
import numpy as np
import pandas as pd

def plot_prediction_summary(procdata, names, times, OBS, PREDICT, PRECISIONS, device_ids, log_ws, predict_style, fixYaxis = False):
    '''Compare the simulation against the data for the highest weighted sample'''

    nplots = PREDICT.shape[-1]
    unique_devices = np.unique(device_ids)
    ndevices = len(unique_devices)
    log_ws = log_ws[:, :, None, None]
    importance_weights = np.exp(np.squeeze(log_ws))
    
    # NEW: assume PREC.shape = n x samples x time x species
    # if len(PRECISIONS.shape) == 3:
    #     PREC = PRECISIONS[:,:,np.newaxis,:]
    #     PREC = np.tile(PREC, [1,1,PREDICT.shape[2],1])
    # else:
    #     PREC = PRECISIONS
    #pdb.set_trace()
    PREC = PRECISIONS
    STD = 1.0 / np.sqrt(PREC)

    f, ax =pp.subplots(ndevices, nplots, sharex=True, figsize=(10, 2*ndevices))
    for iu,u in enumerate(unique_devices):
        locs = np.where(device_ids == u)[0]
        device = procdata.device_lookup[u]
        for idx in range(nplots):
            w_mu = np.sum(importance_weights[locs,:,np.newaxis]*PREDICT[locs, :, :, idx], 1)
            w_var =  np.sum(importance_weights[locs,:,np.newaxis]*(PREDICT[locs, :, :, idx]**2 + STD[locs, :, :, idx]**2), 1) - w_mu**2
            w_std = np.sqrt(w_var)

            for mu,std in zip(w_mu, w_std):
                ax[iu,idx].fill_between(times, mu-2*std, mu+2*std, color='grey', alpha=0.1)

            ax[iu,idx].plot(times, OBS[locs,:,idx].T, 'r-', lw=1, alpha=1)
            ax[iu,idx].plot(times, w_mu.T, predict_style, lw=1, alpha=0.75, color='k')
            if fixYaxis: ax[iu,idx].set_ylim(-0.2,1.2)

            if iu == ndevices-1: ax[iu,idx].set_xlabel('Time (h)')
            if iu == 0: ax[iu,idx].set_title(names[idx])
            if idx == 0: ax[iu,idx].set_ylabel(device)
            pp.tight_layout()
            sns.despine()

    return f

def plot_weighted_theta(procdata, theta_names, TR_iws, TR_theta, TR_device_ids, VL_iws, VL_theta, VL_device_ids, columns2use, sample = True, nsamples=100):
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

def xval_species_summary(res, procdata, devices, pretty_devices, nplots, fixYaxis = False):
    '''Plot the simulated latent species'''    
    ndevices = len(devices)
    colors = ['r','g']    
    fs = 14
    treat_max = np.max(res.treatments)
    
    mus = []
    maxs = []
    for idx in range(nplots):
        mus.append([])
        maxs_i = []
        for iu,u in enumerate(devices):
            mus[idx].append([])
            for i,color in enumerate(colors):
                locs = np.where((res.devices == u) & (res.treatments[:,i] > 0.0))[0]        
                mu = np.sum( res.importance_weights[locs,:,np.newaxis]*res.X_sample[locs, :, :, idx], 1)
                mus[idx][iu].append(mu)
            maxs_i.append(np.max(mus[idx][iu]))
        maxs.append(np.max(maxs_i))
    
    f, ax = pp.subplots(ndevices, nplots, sharex=True, sharey=True, figsize=(14, 2*ndevices) )
    for iu,u in enumerate(devices):
        for idx in range(nplots):
            for i,color in enumerate(colors):
                locs = np.where((res.devices == u) & (res.treatments[:,i] > 0.0))[0]
                for iloc,loc in enumerate(locs):
                    alpha = res.treatments[loc,i] / treat_max
                    ax[iu,idx].plot(res.times, mus[idx][iu][i][iloc].T/maxs[idx], '-', lw=1, alpha=alpha, color=color )
            if fixYaxis: ax[iu,idx].set_ylim(-0.1,1.1)
            if iu == 0:
                if idx < len(res.names): 
                    ax[iu,idx].set_title(res.names[idx])
                else:
                    ax[iu,idx].set_title("Latent %d"%(idx-len(res.names)))
            ax[iu,idx].set_xticks([0,4,8,12,16])
        ax[iu,0].set_ylabel(pretty_devices[iu],labelpad=20,fontweight='bold',fontsize=fs)
    sns.despine()
    pp.tight_layout()            
        
    # Global axis labels: add a big axis, then hide frame
    f.add_subplot(111, frameon=False)
    pp.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    pp.xlabel("Time (h)", fontsize=fs, labelpad=7)
    pp.ylabel("Normalized output", fontsize=fs, labelpad=7)
    
    return f

def xval_treatments(res, data, devices, pretty_devices):
    '''Compare the final simulated points against the equivalent data-points to establish functional response'''
    nplots = len(data.signals)
    ndev = len(devices)
    
    ms = 5
    fs = 14
    obs_mk = 'x'
    pred_mk = 'o'
    colors = ['g','r']
    edges = ['darkgreen','darkred']
    c6_idx = 1
    c12_idx = 0     

    f, ax = pp.subplots(ndev, nplots, sharex=True, sharey=True, figsize=(9, 2.2*ndev))    
    for iu,u in enumerate(devices):        
        locs = np.where(res.devices == u)[0]        
        C6                 = np.exp(res.treatments[:,c6_idx])-1
        C12                = np.exp(res.treatments[:,c12_idx])-1
        device_C6          = C6[locs]
        device_C12         = C12[locs]
        device_OBS         = np.transpose(res.X_obs[locs,-1,:],[1,0])
        device_IW          = res.importance_weights[locs]
        device_PREDICT     = np.transpose(res.PREDICT[locs,:],[2,0,1])
        device_STD         = np.transpose(res.STD[locs,:],[2,0,1])

        for j,signal in enumerate(data.signals):           
            mu  = np.sum(device_IW*device_PREDICT[j], 1)
            std = np.sqrt(np.sum(device_IW*(device_PREDICT[j]**2 + device_STD[j]**2 ), 1) - mu**2)
            
            ax[iu,j].errorbar( device_C6, mu, yerr=std, fmt=pred_mk, ms=ms, lw=1, mec=edges[0], color=colors[0],zorder=0)
            ax[iu,j].semilogx( device_C6, device_OBS[j], 'k'+obs_mk, ms=ms, lw=1, color=edges[0], zorder=20)
            ax[iu,j].errorbar( device_C12, mu, yerr=std, fmt=pred_mk, ms=ms, lw=1, mec=edges[1], color=colors[1], zorder=10)
            ax[iu,j].semilogx( device_C12, device_OBS[j], 'k'+obs_mk, ms=ms, lw=1, color=edges[1], zorder=30)

            ax[iu,j].set_ylim(-0.1,1.1)
            ax[iu,j].tick_params(axis='both', which='major', labelsize=fs)
            ax[iu,j].set_xticks(np.logspace(0,4,3))
        
        ax[iu,0].set_ylabel(pretty_devices[iu], labelpad=25, fontweight='bold', fontsize=fs)
    for j in range(nplots):
        ax[0,j].set_title(signal,fontsize=fs)

    if (ndev>1):
        ytext = "Normalized fluorescence"
    else:
        ytext = "Norm. fluorescence"
    # Global axis labels: add a big axis, then hide frame
    f.add_subplot(111, frameon=False)
    pp.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    pp.xlabel("C$_6$ / C$_{12}$ (nM)", fontsize=fs, labelpad=7)
    pp.ylabel(ytext, fontsize=fs, labelpad=7)
    ax[0,nplots-1].legend(labels=['C$_6$ (data)','C$_{12}$ (data)','C$_6$ (model)','C$_{12}$ (data)'])
    sns.despine()
    
    return f

def xval_fit_summary(res, data, device):
    '''Summary plot of model-data fit for cross-validation results'''
    nplots = res.X_post_sample.shape[-1]    
    prec = res.precisions
    stdev = 1.0 / np.sqrt(prec)
    titles = ['C$_{12}$ dilution', 'C$_6$ dilution']
    u = data.device_map[device]

    fs = 14
    f, ax = pp.subplots(2, nplots, sharex=True, sharey=True, figsize=(2.2*nplots, 4.4))
    for i in range(2):
        all_locs = np.where((res.devices == u) & (res.treatments[:, i] > 0.0))[0]
        _, indices = np.unique(res.treatments[all_locs, i], return_index=True)
        locs = all_locs[indices]
        colors = [ cmx.rainbow(x) for x in np.linspace(0, 1, np.shape(locs)[0]) ]
        for idx in range(nplots):
            w_mu = np.sum(res.importance_weights[locs, :, np.newaxis] * res.X_post_sample[locs, :, :, idx], 1)
            w_var =  np.sum(res.importance_weights[locs, :, np.newaxis] * (res.X_post_sample[locs, :, :, idx]**2 + stdev[locs, :, :, idx]**2), 1) - w_mu**2
            w_std = np.sqrt(w_var)

            ax[i,idx].set_prop_cycle('color', colors)
            for mu, std in zip(w_mu, w_std):
                ax[i,idx].fill_between(res.times, mu-2*std, mu+2*std, alpha=0.1)
            ax[i,idx].plot(res.times, res.X_obs[locs, :, idx].T, '.', alpha=1, markersize=2)
            ax[i,idx].plot(res.times, w_mu.T, '-', lw=2, alpha=0.75)
            ax[i,idx].set_xlim(0.0, 17)
            ax[i,idx].set_xticks([0, 5, 10, 15])
            ax[i,idx].set_ylim(-0.2, 1.2)
            if idx == 0:
                ax[i,idx].set_ylabel(titles[i], labelpad=25, fontweight='bold', fontsize=fs)
    for idx in range(nplots):
        ax[0,idx].set_title(res.names[idx], fontsize=fs)          

    # Global axis labels: add a big axis, then hide frame
    f.add_subplot(111, frameon=False)
    pp.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    pp.xlabel("Time (h)", fontsize=fs, labelpad=7)
    pp.ylabel("Normalized output", fontsize=fs, labelpad=7)
    pp.tight_layout()
    sns.despine()

    return f

def xval_fit_individual(res, data, u):
    nplots = res.X_post_sample.shape[-1]
    colors = ['tab:gray','r','y','c']
    PREC = res.precisions
    STD = 1.0 / np.sqrt(PREC)

    fs = 14
    both_locs = []
    for col in range(2):
        all_locs = np.where((res.devices == u) & (res.treatments[:,1-col] > 0.0))[0]
        indices = np.argsort(res.treatments[all_locs,1-col])
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
            for idx in range(nplots):
                C6  = np.exp(res.treatments[loc][1]) - 1.0
                if (C6 > 0.0) & (C6 < 1.0):
                    C6str = '%1.1f'%C6
                else:
                    C6str = '%1.0f'%C6
                C12 = np.exp(res.treatments[loc][0]) - 1.0
                if (C12 > 0.0) & (C12 < 1.0):
                    C12str = '%1.1f'%C12
                else:
                    C12str = '%1.0f'%C12
                treatment_str = 'C6 = %s nM\nC12 = %s nM'%(C6str,C12str)

                ax = f.add_subplot(ntreatments, 2*nplots, col*nplots+(ntreatments-i-1)*2*nplots+idx+1)
                ax.set_position([left+idx*dx, bottom+(ntreatments-i-1)*dy, width, height])

                mu  = np.sum(res.importance_weights[loc,:,np.newaxis]*res.X_post_sample[loc, :, :, idx], 0)
                var = np.sum(res.importance_weights[loc,:,np.newaxis]*(res.X_post_sample[loc, :, :, idx]**2 + STD[loc, :, :, idx]**2), 0) - mu**2
                std = np.sqrt(var)

                ax.fill_between(res.times, mu-2*std, mu+2*std, alpha=0.25, color=colors[idx])
                ax.plot(res.times, res.X_obs[loc,:,idx], 'k.', markersize=2)
                ax.plot(res.times, mu, '-', lw=2, alpha=0.75, color=colors[idx])
                ax.set_xlim(0.0,17)
                ax.set_xticks([0,5,10,15])
                ax.set_ylim(-0.2,1.2)
                ax.tick_params(axis='both', which='major', labelsize=fs)

                if i==0:
                    pp.title(res.names[idx], fontsize=fs)
                if i<ntreatments-1:
                    ax.set_xticklabels([])
                if idx==0:
                    ax.set_ylabel(treatment_str,labelpad=25,fontsize=fs-2)
                else:
                    ax.set_yticklabels([])

                sns.despine()

        # Add labels
        f.text(left-0.5*dx,0.5, "Normalized output", ha="center", va="center", rotation=90, fontsize=fs)
        f.text(left+2*dx,0,"Time (h)", ha="center", va="bottom", fontsize=fs)

    return f

def combined_treatments(procdata, results, devices, pretty_devices):
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
    for iu,u in enumerate(devices):
        if ndev==1:
            row = ax
            ytext = "Norm. fluorescence"
        else:
            row = ax[iu]
            ytext = "Normalized fluorescence"
        row[0].set_ylabel(pretty_devices[iu], labelpad=25, fontweight='bold', fontsize=fs)
        for ir,res in enumerate(results):
            locs = np.where(res.devices == u)[0]
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