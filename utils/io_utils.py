import pdb
from datetime import datetime, date
from os.path import join, exists, basename, dirname
from os import makedirs, listdir
import os
import gc
import subprocess
import json
import shutil

import nibabel as nib
from skimage.transform import rescale
import numpy as np
import csv

def save_nii(im, v2r, filepath):
    if im.dtype in [np.bool_]:
        im = im.astype('uint8')
    elif im.dtype in [np.int64]:
        im = im.astype('float')

    create_dir(dirname(filepath))
    img = nib.Nifti1Image(im, v2r)
    nib.save(img, filepath)

def check_nan(v):
    if v in ['', 'nan', 'NaN'] or np.isnan(v):
        return True
    else:
        return False

def create_dir(path, subdirs=None):
    if subdirs is None:
        if not exists(path):
            makedirs(path)
    else:
        if isinstance(subdirs, str):
            subdirs = [subdirs]

        for subdir in subdirs:
            if not exists(join(path, subdir)):
                makedirs(join(path, subdir))

def remove_dir(path):
    subprocess.call(['rm', '-rf', path])

def create_derivative_dir(path, description):
    if not os.path.exists(path): os.makedirs(path)
    data_descr_path = os.path.join(path, 'dataset_description.json')

    if not os.path.exists(data_descr_path):
        data_descr = {}
        data_descr['Name'] = os.path.basename(path)
        data_descr['BIDSVersion'] = '1.0.2'
        data_descr['GeneratedBy'] = [{'Name': os.path.basename(path)}]
        data_descr['Description'] = description
        json_object = json.dumps(data_descr, indent=4)
        with open(data_descr_path, 'w') as outfile:
            outfile.write(json_object)

def write_json_derivatives(pixdim, volshape, filename, extra_kwargs={}):
    im_json = {
        "Resolution": {
            "R": str(pixdim[0]),
            "A": str(pixdim[1]),
            "S": str(pixdim[2])
        },
        "ImageShape": {
            "X": str(volshape[0]),
            "Y": str(volshape[1]),
            "Z": str(volshape[2])
        }}

    im_json = {**im_json, **extra_kwargs}

    json_object = json.dumps(im_json, indent=4)
    with open(filename, 'w') as outfile:
        outfile.write(json_object)

def create_results_dir(results_dir, subdirs=None):
    if subdirs is None:
        subdirs = ['checkpoints', 'results']
    if not exists(results_dir):
        for sd in subdirs:
            makedirs(join(results_dir, sd))
    else:
        for sd in subdirs:
            if not exists(join(results_dir, sd)):
                makedirs(join(results_dir, sd))

def mri_convert_nifti_directory(directory, extension='.nii.gz'):

    files = listdir(directory)
    for f in files:
        if extension in f:
            continue
        elif '.nii.gz' in f:
            new_f = f[:-6] + extension

        elif '.nii' in f:
            new_f = f[:-4] + extension

        elif '.mgz' in f:
            new_f = f[:-4] + extension

        else:
            continue

        subprocess.call(['mri_convert', join(directory, f), join(directory, new_f)])

def read_lta(file):
    lta = np.zeros((4,4))
    with open(file, 'r') as txtfile:
        lines = txtfile.readlines()
        for it_row, l in enumerate(lines[5:9]):
            aff_row = l.split(' ')[:-1]
            lta[it_row] = [float(ar) for ar in aff_row]

    return lta

def read_tsv(path, k=None, delimiter='\t', codec_read=False):

    t = {}
    with open(path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter=delimiter)
        if k is None: k = csvreader.fieldnames[0]
        if not isinstance(k, list): k = [k]

        for it_row, row in enumerate(csvreader):
            t['_'.join([row[i] for i in k])] = row

    return t

def write_affine_matrix(path, affine_matrix):
    with open(path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        for it_row in range(4):
            csvwriter.writerow(affine_matrix[it_row])

def read_affine_matrix(path, full=False):
    with open(path, 'r') as csvfile:
        rotation_matrix = np.zeros((3, 3))
        translation_vector = np.zeros((3,))
        csvreader = csv.reader(csvfile, delimiter=' ')
        for it_row, row in enumerate(csvreader):
            rotation_matrix[it_row, 0] = float(row[0])
            rotation_matrix[it_row, 1] = float(row[1])
            rotation_matrix[it_row, 2] = float(row[2])
            translation_vector[it_row] = float(row[3])
            if it_row == 2:
                break

    if full:
        affine_matrix = np.zeros((4,4))
        affine_matrix[:3, :3] = rotation_matrix
        affine_matrix[:3, 3] = translation_vector
        affine_matrix[3, 3] = 1
        return affine_matrix

    else:
        return rotation_matrix, translation_vector

def load_array_if_path(var, load_as_numpy=True):
    """If var is a string and load_as_numpy is True, this function loads the array writen at the path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), 'No such path: %s' % var
        var = np.load(var)
    return var

def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 5>3"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels

def get_ras_axes(aff, n_dims=3):
    """This function finds the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    :param aff: affine matrix Can be a 2d numpy array of size n_dims*n_dims, n_dims+1*n_dims+1, or n_dims*n_dims+1.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: two numpy 1d arrays of lengtn n_dims, one with the axes corresponding to RAS orientations,
    and one with their corresponding direction.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    for i in range(n_dims):
        if i not in img_ras_axes:
            unique, counts = np.unique(img_ras_axes, return_counts=True)
            incorrect_value = unique[np.argmax(counts)]
            img_ras_axes[np.where(img_ras_axes == incorrect_value)[0][-1]] = i

    return img_ras_axes

def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None, return_copy=True):
    """This function aligns a volume to a reference orientation (axis and direction) specified by an affine matrix.
    :param volume: a numpy array
    :param aff: affine matrix of the floating volume
    :param aff_ref: (optional) affine matrix of the target orientation. Default is identity matrix.
    :param return_aff: (optional) whether to return the affine matrix of the aligned volume
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume. If not provided, n_dims will be
    inferred from the input volume.
    :return: aligned volume, with corresponding affine matrix if return_aff is True.
    """

    # work on copy
    new_volume = volume.copy() if return_copy else volume
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)

    # extract ras axes
    if n_dims is None:
        n_dims, _ = get_dims(new_volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            new_volume = np.swapaxes(new_volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            new_volume = np.flip(new_volume, axis=i)
            aff_flo[:, i] = - aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (new_volume.shape[i] - 1)

    if return_aff:
        return new_volume, aff_flo
    else:
        return new_volume

def reformat_to_list(var, length=None, load_as_numpy=False, dtype=None):
    """This function takes a variable and reformat it into a list of desired
    length and type (int, float, bool, str).
    If variable is a string, and load_as_numpy is True, it will be loaded as a numpy array.
    If variable is None, this funtion returns None.
    :param var: a str, int, float, list, tuple, or numpy array
    :param length: (optional) if var is a single item, it will be replicated to a list of this length
    :param load_as_numpy: (optional) whether var is the path to a numpy array
    :param dtype: (optional) convert all item to this type. Can be 'int', 'float', 'bool', or 'str'
    :return: reformated list
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(var, (int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64)):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = [var[0]]
        else:
            var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    elif isinstance(var, bool):
        var = [var]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError('if var is a list/tuple/numpy array, it should be of length 1 or {0}, '
                                 'had {1}'.format(length, var))
    else:
        raise TypeError('var should be an int, float, tuple, list, numpy array, or path to numpy array')

    # convert items type
    if dtype is not None:
        if dtype == 'int':
            var = [int(v) for v in var]
        elif dtype == 'float':
            var = [float(v) for v in var]
        elif dtype == 'bool':
            var = [bool(v) for v in var]
        elif dtype == 'str':
            var = [str(v) for v in var]
        else:
            raise ValueError("dtype should be 'str', 'float', 'int', or 'bool'; had {}".format(dtype))
    return var

def load_volume(path_volume, im_only=True, squeeze=True, dtype=None, aff_ref=None):
    """
    Load volume file.
    :param path_volume: path of the volume to load. Can either be a nii, nii.gz, mgz, or npz format.
    If npz format, 1) the variable name is assumed to be 'vol_data',
    2) the volume is associated with a identity affine matrix and blank header.
    :param im_only: (optional) if False, the function also returns the affine matrix and header of the volume.
    :param squeeze: (optional) whether to squeeze the volume when loading.
    :param dtype: (optional) if not None, convert the loaded volume to this numpy dtype.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
    The returned affine matrix is also given in this new space. Must be a numpy array of dimension 4x4.
    :return: the volume, with corresponding affine matrix and header if im_only is False.
    """
    assert path_volume.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume

    if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
        x = nib.load(path_volume)
        if squeeze:
            volume = np.squeeze(x.get_fdata())
        else:
            volume = x.get_fdata()
        aff = x.affine
        header = x.header
    else:  # npz
        volume = np.load(path_volume)['vol_data']
        if squeeze:
            volume = np.squeeze(volume)
        aff = np.eye(4)
        header = nib.Nifti1Header()
    if dtype is not None:
        if 'int' in dtype:
            volume = np.round(volume)
        volume = volume.astype(dtype=dtype)

    # align image to reference affine matrix
    if aff_ref is not None:
        n_dims, _ = get_dims(list(volume.shape), max_channels=10)
        volume, aff = align_volume_to_ref(volume, aff, aff_ref=aff_ref, return_aff=True, n_dims=n_dims)

    if im_only:
        return volume
    else:
        return volume, aff, header

def save_volume(volume, aff, header, path, res=None, dtype=None, n_dims=3):
    """
    Save a volume.
    :param volume: volume to save
    :param aff: affine matrix of the volume to save. If aff is None, the volume is saved with an identity affine matrix.
    aff can also be set to 'FS', in which case the volume is saved with the affine matrix of FreeSurfer outputs.
    :param header: header of the volume to save. If None, the volume is saved with a blank header.
    :param path: path where to save the volume.
    :param res: (optional) update the resolution in the header before saving the volume.
    :param dtype: (optional) numpy dtype for the saved volume.
    :param n_dims: (optional) number of dimensions, to avoid confusion in multi-channel case. Default is None, where
    n_dims is automatically inferred.
    """

    if not exists(os.path.dirname(path)): makedirs(os.path.dirname(path))
    if '.npz' in path:
        np.savez_compressed(path, vol_data=volume)
    else:
        if header is None:
            header = nib.Nifti1Header()
        if isinstance(aff, str):
            if aff == 'FS':
                aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        elif aff is None:
            aff = np.eye(4)
        nifty = nib.Nifti1Image(volume, aff, header)
        if dtype is not None:
            if 'int' in dtype:
                volume = np.round(volume)
            volume = volume.astype(dtype=dtype)
            nifty.set_data_dtype(dtype)
        if res is not None:
            if n_dims is None:
                n_dims, _ = get_dims(volume.shape)
            res = reformat_to_list(res, length=n_dims, dtype=None)
            nifty.header.set_zooms(res)
        nib.save(nifty, path)

def get_volume_info(path_volume, return_volume=False, aff_ref=None, max_channels=10):
    """
    Gather information about a volume: shape, affine matrix, number of dimensions and channels, header, and resolution.
    :param path_volume: path of the volume to get information form.
    :param return_volume: (optional) whether to return the volume along with the information.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
    All info relative to the volume is then given in this new space. Must be a numpy array of dimension 4x4.
    :return: volume (if return_volume is true), and corresponding info. If aff_ref is not None, the returned aff is
    the original one, i.e. the affine of the image before being aligned to aff_ref.
    """
    # read image
    im, aff, header = load_volume(path_volume, im_only=False)

    # understand if image is multichannel
    im_shape = list(im.shape)
    n_dims, n_channels = get_dims(im_shape, max_channels=max_channels)
    im_shape = im_shape[:n_dims]

    # get labels res
    if '.nii' in path_volume:
        data_res = np.array(header['pixdim'][1:n_dims + 1])
    elif '.mgz' in path_volume:
        data_res = np.array(header['delta'])  # mgz image
    else:
        data_res = np.array([1.0] * n_dims)

    # align to given affine matrix
    if aff_ref is not None:
        ras_axes = get_ras_axes(aff, n_dims=n_dims)
        ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
        im = align_volume_to_ref(im, aff, aff_ref=aff_ref, n_dims=n_dims)
        im_shape = np.array(im_shape)
        data_res = np.array(data_res)
        im_shape[ras_axes_ref] = im_shape[ras_axes]
        data_res[ras_axes_ref] = data_res[ras_axes]
        im_shape = im_shape.tolist()

    # return info
    if return_volume:
        return im, im_shape, aff, n_dims, n_channels, header, data_res
    else:
        return im_shape, aff, n_dims, n_channels, header, data_res

def get_run(file):
    for f in file.split('_'):
        if 'run' in f: return f.split('-')[1]
    return ''

def load_chunk(ps, chunk, ps_id):
    return np.asarray(ps.dataobj[chunk[0][0]:chunk[0][1], chunk[1][0]:chunk[1][1], chunk[2][0]:chunk[2][1]])[np.newaxis], ps_id

def load_chunk_rescale(ps, chunk, ps_id, factor):
    svf = np.asarray(ps.dataobj)
    svf = rescale(svf, [factor]*3  + [1])
    return svf[chunk[0][0]:chunk[0][1], chunk[1][0]:chunk[1][1], chunk[2][0]:chunk[2][1]][np.newaxis], ps_id

def get_age(age):
    if age == 'nan': age = None
    try:
        age = float(age)
    except:
        age = None

    return age

def get_dx_bl(subject):
    if 'dx' in subject.sbj_dict.keys():
        dx = subject.sbj_dict['dx']
    elif 'diagnosis' in subject.sbj_dict.keys():
        dx = subject.sbj_dict['diagnosis']
    else:
        dx = DX_DICT[get_dx_long([tp.sess_metadata['dx'] if 'dx' in tp.sess_metadata.keys() else '' for tp in subject.sessions])]

    return dx

def get_dx(dx):
    if dx == '':
        dx = -1
    elif dx == '-1':
        dx = -1
    elif dx in ['CN', 'Control', 'CTR']:
        dx = 0
    elif dx == 'MCI':
        dx = 1
    elif dx in ['Dementia', 'AD']:
        dx = 2
    elif dx in ['FTD', 'DFT']:
        dx = 3
    elif dx in ['PortadorGENFI']:
        dx = 4

    return dx

def get_dx_long(dx_list):
    if len(dx_list) == 1:
        return get_dx(dx_list[0])

    dx_bl = get_dx(dx_list[0])
    dx_last = get_dx(dx_list[-1])


    if dx_bl == -1 or dx_last == -1:
        dx_flag = [get_dx(d) != -1 for d in dx_list]

        if sum(dx_flag) == 0:
            return -1

        elif sum(dx_flag) == 1:
            idx = int(np.where(np.array(dx_flag)==1)[0])
            return get_dx(dx_list[idx])

        else:
            idx = np.where(np.array(dx_flag)==1)[0]
            idx_min = int(np.min(idx))
            idx_max = int(np.min(idx))
            dx_bl = get_dx(dx_list[idx_min])
            dx_last = get_dx(dx_list[idx_max])

    if dx_bl == 0:
        if dx_last == 0:
            dx = 3# CN stable
        elif dx_last == 1:
            dx = 4# CN-MCI converter
        elif dx_last == 2:
            dx = 5# CN-AD converter
        elif dx_last == 3:
            dx = 11
        else:
            dx = 9
    elif dx_bl == 1:
        if dx_last == 1:
            dx = 6# MCI_stable
        elif dx_last == 2:
            dx = 7# MCI-AD converter
        else:
            dx = 9
    elif dx_bl == 2:
        if dx_last == 2:
            dx = 8# AD stable
        else:
            dx = 9
    elif dx_bl == 3:
        if dx_last == 3:
            dx = 10# AD stable
        else:
            dx = 9
    elif dx_bl == 4:
        if dx_last == 4:
            dx = 12  # GENFI
        else:
            dx = 9
    else:
        dx = 9# Reversed diagnosis

    return dx

DX_DICT = {
    -1: 'Unknown',
    0: 'CN_cross',
    1: 'MCI_cross',
    2: 'Dementia_cross',
    3: 'CN_stable',
    4: 'CN_MCI_converter',
    5: 'CN_AD_converter',
    6: 'MCI_stable',
    7: 'MCI_AD_converter',
    8: 'AD_stable',
    9: 'Reversed_diagnosis/Unknown',
    10: 'FTD_stable',
    11: 'FTD_converter',
    12: 'GENFI',

}

def read_FS_volumes(file, labs=None):


    etiv = 0
    fs_vols = {}
    start = False
    with open(file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ')
        for row in csvreader:
            row_cool = list(filter(lambda x: x != '', row))
            if start is True:
                fs_vols[row_cool[1]] = float(row_cool[3])

            if 'ColHeaders' in row_cool and start is False:
                start = True
            elif 'EstimatedTotalIntraCranialVol,' in row_cool:
                etiv = float(row_cool[-2].split(',')[0])

    # vols = {**{lab: float(fs_vols[lab_str]) for lab, lab_str in labs.items() if 'Thalamus' not in lab_str},
    #         **{lab: float(fs_vols[lab_str + '-Proper']) for lab, lab_str in labs.items() if 'Thalamus' in lab_str}}

    return fs_vols, etiv

def get_float_vol(v):
    try:
        return float(v)
    except:
        return -4

