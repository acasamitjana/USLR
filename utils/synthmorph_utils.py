import pdb
import os

import tensorflow as tf
import h5py

import numpy as np
import neurite as ne
from voxelmorph import layers
from voxelmorph import utils as vxm_utils
import voxelmorph as vxm
import nibabel as nib

SpatialTransformer = layers.SpatialTransformer
compose_transforms = vxm_utils.compose
constant_layer = ne.layers.Constant
def load_weights(model, weights):
    """Load weights into model or submodel.

    Attempts to load (all) weights into a model or one of its submodels. If
    that fails, `model` may be a submodel of what we got weights for, and we
    attempt to load the weights of a submodel (layer) into `model`.

    Parameters
    ----------
    model : TensorFlow model
        Model to initialize.
    weights : str or pathlib.Path
        Path to weights file.

    Raises
    ------
    ValueError
        If unsuccessful at loading any weights.

    """
    # Extract submodels.
    models = [model]
    i = 0
    while i < len(models):
        layers = [f for f in models[i].layers if isinstance(f, tf.keras.Model)]
        models.extend(layers)
        i += 1

    # Add models wrapping a single model in case this was done in training.
    # Requires list expansion or Python will get stuck.
    models.extend([tf.keras.Model(m.inputs, m(m.inputs)) for m in models])

    # Attempt to load all weights into one of the models.
    for mod in models:
        try:
            mod.load_weights(weights)
            return
        except ValueError as e:
            pass

    # Assume `model` is a submodel of what we got weights for.
    with h5py.File(weights, mode='r') as h5:
        layers = h5.attrs['layer_names']
        weights = [list(h5[lay].attrs['weight_names']) for lay in layers]

        # Layers with weights. Attempt loading.
        layers, weights = zip(*filter(lambda f: f[1], zip(layers, weights)))
        for lay, wei in zip(layers, weights):
            try:
                model.set_weights([h5[lay][w] for w in wei])
                return
            except ValueError as e:
                if lay is layers[-1]:
                    raise e

def warp(arr_flo, arr_def):

    if len(arr_flo.shape) == 3:
        arr_flo = arr_flo[..., np.newaxis]
        
    tf_arr_flo = tf.cast(arr_flo[np.newaxis], tf.float32)
    tf_arr_def = tf.cast(arr_def[np.newaxis], tf.float32)

    prop = dict(shift_center=False, fill_value=0, shape=tf_arr_def.shape[1:-1])
    mov_1 = SpatialTransformer(**prop)((tf_arr_flo, tf_arr_def))

    return np.squeeze(np.array(mov_1))

def integrate_svf(svf, orig_shape, scaling_factor=2, int_steps=7):
    num_dim = len(svf.shape) - 1

    def scale(fact):
        mat = np.diag((*[fact] * num_dim, 1))
        return ne.layers.Constant(mat)([])

    aff_1 = scale(scaling_factor)
    fw_def = layers.VecInt(method='ss', int_steps=int_steps)(svf[np.newaxis])
    fw_def = (aff_1, fw_def, scale(1 / scaling_factor), aff_1)
    fw_def = layers.ComposeTransform(shift_center=False)(fw_def)

    down = layers.AffineToDenseShift(orig_shape, shift_center=False)(scale(1 / scaling_factor))
    fw_def = layers.ComposeTransform()((fw_def, down))

    return np.squeeze(np.array(fw_def))


def synthmorph_register(imageref_file, imageflo_file, reg_param=0.5):
    # subprocess.call(
    #     ['mri_synthmorph', 'register', '-m', 'deform', '-t', 'prova_sm_def.nii.gz', '-o', 'prova_sm_mov.nii.gz', imageflo_file.path,
    #      imageref_file.path], stdout=subprocess.DEVNULL)
    #
    # pdb.set_trace()

    if isinstance(imageref_file, nib.Nifti1Image):
        proxyref = imageref_file
    elif isinstance(imageref_file, str):
        proxyref = nib.load(imageref_file)
    else:
        proxyref = nib.load(imageref_file.path)

    if isinstance(imageflo_file, nib.Nifti1Image):
        proxyflo = imageflo_file
    elif isinstance(imageflo_file, str):
        proxyflo = nib.load(imageflo_file)
    else:
        proxyflo = nib.load(imageflo_file.path)

    ref_image = np.array(proxyref.dataobj)
    flo_image = np.array(proxyflo.dataobj)

    ref_image -= np.min(ref_image)
    ref_image /= np.max(ref_image)
    ref_image = ref_image[np.newaxis, ..., np.newaxis]

    flo_image -= np.min(flo_image)
    flo_image /= np.max(flo_image)
    flo_image = flo_image[np.newaxis, ..., np.newaxis]

    inputs = (tf.constant([reg_param]), tf.cast(flo_image, tf.float32), tf.cast(ref_image, tf.float32))

    prop = dict(in_shape=ref_image.shape, bidir=True, int_steps=7, skip_affine=True, return_svf=True)
    model = vxm.networks.HyperVxmJoint(**prop)
    load_weights(model, weights=os.path.join(os.environ.get('FREESURFER_HOME'), 'models', 'synthmorph.deform.3.h5'))

    _, _, fw_svf, _ = tuple(map(tf.squeeze, model(inputs)))

    return np.squeeze(np.array(fw_svf))
