import tqdm.auto as tqdm
import numpy as np


def _get_penultimate_layer(model):
    pass

def _get_activations_from_layer(layer, sdataloader):
    from ..preprocessing import decodeDNA
    activations = []
    sequences = []
    for i_batch, batch in tqdm(enumerate(sdataloader)):
        ID, x, x_rev_comp, y = batch
        sequences.append(decodeDNA(x.transpose(2,1).detach().cpu().numpy()))
        activations.append(F.relu(layer(x)).detach().cpu().numpy())
        np_act = np.concatenate(activations)


def _get_layer_activators(activations, sequences, layer):
    pass


def layer_activations(model, sdata, copy=False):
    sdata = sdata.copy() if copy else sdata
    #Do some stuff to sdata
    return sdata if copy else None
