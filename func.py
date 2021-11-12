import json
import os
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import csv


def parse_configs(json_name="default", configs_root=""):
    json_file = os.path.join(configs_root, json_name)
    with open(json_file, 'r') as f:
        args = json.load(f)
    return args


def jacobian_derterminant(disp, spacing):
    """

    :param disp: in the form of [x,y,z,3]
    :return:
    """
    dx = torch_gradient(disp[:, :, :, 0], spacing)
    dy = torch_gradient(disp[:, :, :, 1], spacing)
    dz = torch_gradient(disp[:, :, :, 2], spacing)
    detPhi = (1+dx[..., 0]) * (1+dy[..., 1]) * (1+dz[..., 2]) + \
              dx[..., 1] * dy[..., 2] * dz[..., 0] + \
              dx[..., 2] * dy[..., 0] * dz[..., 1] - \
              dx[..., 2] * (1+dy[..., 1]) * dz[..., 0] - \
              dx[..., 1] * dy[..., 0] * (1+dz[..., 2]) - \
             (1+dx[..., 0]) * dy[..., 2] * dz[..., 1]
    return detPhi


def torch_gradient(arr, spacing=[1,1,1]):
    """
    :param arr: shape of [x,y,z]
    :param spacing:
    :param direction:
    :return: [x,y,z,3]
    """
    grad_list = []
    gradx = arr[1:, :, :] - arr[:-1, :, :]

    grad_list.append(F.pad(gradx.unsqueeze(0).unsqueeze(0), (0,0,0,0,0,1), mode='replicate').squeeze()*spacing[0])
    grady = arr[:, 1:, :] - arr[:, :-1, :]
    grad_list.append(F.pad(grady.unsqueeze(0).unsqueeze(0), (0,0,0,1,0,0), mode='replicate').squeeze()*spacing[1])
    gradz = arr[:, :, 1:] - arr[:, :, :-1]
    grad_list.append(F.pad(gradz.unsqueeze(0).unsqueeze(0), (0,1,0,0,0,0), mode='replicate').squeeze()*spacing[2])

    return torch.stack(grad_list, dim=-1)

def neg_Jac_percent(disp_channel_last, disp_spacing, mask=None):
    jac_disp = jacobian_derterminant(disp_channel_last, disp_spacing)
    img_size = list(jac_disp.shape)
    num_pixels = np.prod(img_size)
    if mask is not None:
        mask[mask < 1] = 0
        num_pixels = mask.sum()
        if jac_disp.is_cuda:
            mask = torch.tensor(mask).cuda(jac_disp.device)
        jac_disp = jac_disp*mask

    num_neg_pixels = torch.sum(jac_disp < 0)
    if num_neg_pixels.device is not None:
        num_neg_pixels = num_neg_pixels.cpu()
    num_neg_pixels = float(num_neg_pixels.numpy())
    neg_Jac_perc = round(100.0 * num_neg_pixels / num_pixels, 4)
    return neg_Jac_perc


def create_table(case_range, seq_range=range(1,6)):
    Results_table = [['case_id']] + [[case_id] for case_id in case_range]
    for mov_idx in seq_range:
        Results_table[0].append('TRE_mean_' + str(mov_idx))
        Results_table[0].append('TRE_std_' + str(mov_idx))
        Results_table[0].append('ori_TRE_mean_' + str(mov_idx))
        Results_table[0].append('ori_TRE_std_' + str(mov_idx))

        if mov_idx == 5:
            Results_table[0].append('TRE300_mean_' + str(mov_idx))
            Results_table[0].append('TRE300_std_' + str(mov_idx))
            Results_table[0].append('ori_TRE300_mean_' + str(mov_idx))
            Results_table[0].append('ori_TRE300_std_' + str(mov_idx))

        Results_table[0].append('neg_Jac_perc_' + str(mov_idx))
        Results_table[0].append('time_' + str(mov_idx))
    return Results_table


def grid_sample_without_grid(inp, displacement_field, batch_regular_grid=None, padding_mode="border",
                             interp_mode='bilinear', align_corners=True):
    """
    no grid but flow
    :param inp: [batch, 1, x,y,z]
    :param displacement_field: [batch, 3, x,y,z]
    :param padding_mode:
    :param interp_mode:
    :return: [N,C,x,y,z]
    """
    if batch_regular_grid is None:
        batch_size = len(inp)
        batch_regular_grid = create_batch_regular_grid(batch_size, inp.shape[2:], displacement_field.device)
    else:
        if batch_regular_grid.device != displacement_field.device:
            batch_regular_grid = batch_regular_grid.cuda(displacement_field.device)

    grid_channel_last = batch_regular_grid + displacement_field.permute(0,2,3,4,1)
    output = grid_sample_with_grid(inp, grid_channel_last, padding_mode=padding_mode, interp_mode=interp_mode,
                                   align_corners=align_corners)
    return output

def grid_sample_with_grid(inp, deformation_field, padding_mode="border", interp_mode='bilinear',align_corners=True):
    """
    :param inp: [batch, 1, x,y,z]
    :param deformation_field:  [batch, x,y,z, 3]
    :param padding_mode:
    :param interp_mode:
    :return:
    """
    # print("grid xyz: ", inp.shape, grid.shape)
    assert (inp.shape[1] == 1 or inp.shape[1]==3) and len(inp.shape) == 5
    assert deformation_field.shape[-1] == 3 and len(deformation_field.shape) == 5

    # print("type: ", grid.type(), inp.type())

    grid_rev = torch.flip(deformation_field, [-1])  # flip the dim
    output_tensor = F.grid_sample(inp, grid_rev, padding_mode=padding_mode, mode=interp_mode,
                                  align_corners=align_corners)
    # output_tensor is [N,C,x,y,z]
    return output_tensor

def create_batch_regular_grid(batch_size, img_size, cuda_device):
    """

    :param batch_size:
    :param img_size:
    :param cuda_device:
    :return: channel last regular grid [batch, x,y,z,3]
    """
    D, H, W = img_size
    x_range = torch.tensor(list([i * 2 / (D - 1) - 1 for i in range(D)]), device=cuda_device)
    y_range = torch.tensor(list([i * 2 / (H - 1) - 1 for i in range(H)]), device=cuda_device)
    z_range = torch.tensor(list([i * 2 / (W - 1) - 1 for i in range(W)]), device=cuda_device)

    regular_grid_list = torch.meshgrid(x_range, y_range, z_range)
    regular_grid = torch.stack(regular_grid_list, dim=-1)
    batch_regular_grid = regular_grid.repeat(batch_size, 1, 1, 1, 1)
    return batch_regular_grid

def scale_displacement(disp, output_channel_last=None):
    """
    change a displacement of range [-1, 1] to its original range
    :param disp: [x,y,z,3] or [3,x,y,z].
    :param channel_last: None means we do not need to do any changes.
    True means we need to change the displacement into channel last form.
    :return:
    """
    disp_shape = disp.shape
    assert len(disp_shape) == 4
    ori_channel_last = True if disp_shape[-1] == 3 else False
    new_disp = disp.clone()
    if output_channel_last == None:
        output_channel_last = ori_channel_last
    else:
        if output_channel_last == True and ori_channel_last == False:
            new_disp = new_disp.permute(1, 2, 3, 0)
        elif output_channel_last == False and ori_channel_last == True:
            new_disp = new_disp.permute(3, 0, 1, 2)

    for idx in range(3):
        if output_channel_last:
            img_size = new_disp.shape[:-1]
            new_disp[..., idx] = new_disp[..., idx] / 2 * (img_size[idx] - 1)
        else:
            img_size = new_disp.shape[1:]
            new_disp[idx] = new_disp[idx]/ 2 * (img_size[idx] - 1)

    return new_disp

def eval_TRE(displacement, fix_idx, mov_idx, info_case, resampling_spacing):
    TRE_row = []

    # get transformed landmarks (after resampling and cropping)
    fix_mov_list = [info_case['rs_crop_point75_phase' + str(fix_idx)],
                    info_case['rs_crop_point75_phase' + str(mov_idx)]]

    TRE_mean, TRE_std = TRE_by_disp(displacement, fix_mov_list, resampling_spacing, round_num=2)
    ori_TRE_mean, ori_TRE_std = TRE_by_disp(np.zeros(list(displacement.shape)), fix_mov_list, resampling_spacing, round_num=2)

    TRE_row.append(TRE_mean)
    TRE_row.append(TRE_std)
    TRE_row.append(ori_TRE_mean)
    TRE_row.append(ori_TRE_std)

    if mov_idx == 5:
        fix_idx = 0
        fix_mov_list = [info_case['rs_crop_point300_phase' + str(fix_idx)],
                        info_case['rs_crop_point300_phase' + str(mov_idx)]]
        TRE_mean, TRE_std = TRE_by_disp(displacement, fix_mov_list, resampling_spacing, round_num=2)
        ori_TRE_mean, ori_TRE_std = TRE_by_disp(np.zeros(list(displacement.shape)), fix_mov_list, resampling_spacing, round_num=2)

        TRE_row.append(TRE_mean)
        TRE_row.append(TRE_std)
        TRE_row.append(ori_TRE_mean)
        TRE_row.append(ori_TRE_std)
    return TRE_row


def TRE_by_disp(disp_list, landmarks_list, spacing, round_num=-1):
    """
    given a displacement, calculate its TRE by the landmarks and spacing.
    :param disp_list: in the shape of [x,y,z,3], channel last
    :param landmarks_list: in the order of [fix, mov]
    :return:
    """
    warp_landmarks = get_warped_landmarks(landmarks_list[0], disp_list)
    TRE_mean, TRE_std = compute_tre(landmarks_list[-1], warp_landmarks, spacing)
    if round_num>-1:
        TRE_mean = round(TRE_mean, round_num)
        TRE_std = round(TRE_std, round_num)
    return TRE_mean, TRE_std

def get_warped_landmarks(fix_landmarks, displacement):
    """
    they are with the same spacing, i.e., displacement must not in [-1,1]
    :param fix_landmarks: shape of [n, 3]
    :param displacement: channel last, in the shape of [x, y, z, 3]
    :return: warp_landmarks
    """
    warp_landmarks = np.array(fix_landmarks).copy()
    for i in range(len(fix_landmarks)):
        wi, hi, di = [int(round(fix_landmarks[i][j])) for j in range(3)]
        for j in range(3):
            warp_landmarks[i][j] += float(displacement[wi, hi, di][j])
    return warp_landmarks

def write_csv(datas, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in datas:
            writer.writerow(row)

def compute_tre(mov_lmk, ref_lmk, spacing):
    #TRE, unit: mm
    mov_lmk, ref_lmk,spacing = np.array(mov_lmk), np.array(ref_lmk), np.array(spacing)
    diff = (ref_lmk - mov_lmk) * spacing

    tre = np.sqrt(np.sum(diff**2, 1))

    return np.mean(tre), np.std(tre)


def get_image_array_from_name(filename, orientation="xyz"):
    image_array = sitk.GetArrayFromImage(sitk.ReadImage(filename)).astype(np.float32)
    # The coordinate order of the image data read by SimpleITK is ZYX,
    # that is,  how many slices, width, height
    if orientation == "xyz":
        image_array = np.swapaxes(image_array, 0, 2)
    return image_array

