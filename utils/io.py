import numpy as np
import sys
import re
import imageio
import matplotlib.pyplot as plt


def load_LIU_data(root, pair, mode, item):
    wta, scale =        readPFM(root + pair + '/left_initial_disparity/' + item + '.pfm')
    if mode == 'saab':
        output, scale = readPFM(root + pair + '/left_initial_disparity/' + item + '.pfm')
    elif mode == 'sdr':
        output, scale = readPFM(root + pair + '/left_output_sdr/' + item + '.pfm')
    elif mode == 'inpaint':
        output = imageio.imread(root + pair + '/left_output_inpaint/' + item + '.png')
        output = scale_disp(output)
    elif mode == 'nconv':
        output = imageio.imread(root + pair + '/left_output_nconv/' + item + '.png')
        output = scale_disp(output)
    else:
        raise Exception

    gt = imageio.imread(root + pair + '/left_gt/' + item + '.png')
    gt = np.abs(scale_disp(gt))
    rgb = imageio.imread(root + pair + '/left_rgb/' + item + '.png')

    assert_data(wta, output, gt)
    return wta, output, gt, rgb

def load_MiddV3_data(root, folder, mode, *args):
    wta, scale =        readPFM(root + folder + '/disp_Initial_Saab.pfm')
    if mode == 'saab':
        output, scale = readPFM(root + folder + '/disp_Initial_Saab.pfm')
    elif mode == 'sdr':
        output, scale = readPFM(root + folder + '/disp0FDR.pfm')
    elif mode == 'inpaint':
        output = imageio.imread(root + folder + '/im0_inpaint_disp.png')
        output = np.abs(scale_disp(output))
    else:
        raise Exception
    
    gt, _ = readPFM(root + folder + '/disp0GT.pfm')
    gt[gt == np.inf] = 0
    rgb = imageio.imread(root + folder + '/im0.png')

    assert_data(wta, output, gt)
    return wta, output, gt, rgb


def assert_data(wta, output, gt):
    assert(wta.mean() > 0)
    assert(gt.mean() > 0)
    assert(output.mean() > 0)
    assert((wta== 0).sum() > 0)
    assert((gt == 0).sum() > 0)

def write_eval(path, details_string, results_string):
    with open(path, 'w') as f:
        f.write(details_string)
        f.write(results_string)


def scale_disp(d):
    return -d.astype('float64') / 64 + 350


def make_pfm(src_str, dst_str):
    d = imageio.imread(src_str)
    d = scale_disp(d)
    if d.max() == 0:
        d = -d # flip to positive disps

    writePFM(dst_str, d)
    print('Wrote {}. max={}'.format(dst_str, d.max()))


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    
    # If header is of type bytes not string, make it string
    try:
        header = header.decode('UTF-8')
    except:
        print('header was not in bytes. No conversion to string needed.')
        
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
      
    image = np.flipud(image)  

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    image.tofile(file)
