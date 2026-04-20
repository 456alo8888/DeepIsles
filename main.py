# Author: Ezequiel de la Rosa (ezequieldlrosa@gmail.com)
# 03.04.2023

import os
import sys
import argparse
import subprocess

cwd = os.getcwd()
sys.path.append(cwd)
from src.isles22_ensemble import IslesEnsemble


# python main.py \
#     --dwi_file_name '/mnt/disk1/SOOP_TRACE_STRIPPED/sub-3_rec-TRACE_dwi.nii.gz' \
#     --adc_file_name '/mnt/disk1/SOOP_ADC_STRIPPED/sub-3_ADC_dwi.nii.gz' \
#     --flair_file_name '/mnt/disk1/SOOP_FLAIR_STRIPPED/sub-3_FLAIR.nii.gz' \
#     --fast --save_team_outputs --skull_strip --parallelize --results_mni

def main():
    parser = argparse.ArgumentParser(description='Isles22 Ensemble Algorithm')
    # parser.add_argument('--dwi_file_name', type=str, required=True, help='Name of DWI image (required)' , default = '/mnt/disk1/SOOP_TRACE_STRIPPED/sub-3_rec-TRACE_dwi.nii.gz')
    # parser.add_argument('--adc_file_name', type=str, required=True, help='Name of ADC image (required)' , default= '/mnt/disk1/SOOP_ADC_STRIPPED/sub-3_ADC_dwi.nii.gz')
    # parser.add_argument('--flair_file_name', type=str, help='Name of FLAIR image (optional)' , default = '/mnt/disk1/SOOP_FLAIR_STRIPPED/sub-3_FLAIR.nii.gz')
    # parser.add_argument('--fast', action='store_true', help='Run only the best isles22 algorithm')
    # parser.add_argument('--save_team_outputs', action='store_true', help='Save individual team outputs')
    # parser.add_argument('--skull_strip', action='store_true', help='Run skull stripping')
    # parser.add_argument('--parallelize', action='store_false', help='Run inference in parallel')
    # parser.add_argument('--results_mni', action='store_true', help='Save results in MNI space')

    # args = parser.parse_args()

    # if args.dwi_file_name is None:
    #     raise ValueError('Please provide a DWI image file name')

    # if args.adc_file_name is None:
    #     raise ValueError('Please provide an ADC image file name')

    # INPUT_FLAIR = None
    # if args.flair_file_name:
        # INPUT_FLAIR = os.path.join('/app', 'data', args.flair_file_name)  # path-to-FLAIR
    # INPUT_DWI = os.path.join('/app', 'data', args.dwi_file_name)  # pat-t-DWI
    # INPUT_ADC = os.path.join('/app', 'data', args.adc_file_name)  # path-to-ADC

    INPUT_FLAIR = '/mnt/disk1/SOOP_FLAIR_STRIPPED/sub-3_FLAIR.nii.gz'
    INPUT_DWI = '/mnt/disk1/SOOP_TRACE_STRIPPED/sub-3_rec-TRACE_dwi.nii.gz'
    INPUT_ADC = '/mnt/disk1/SOOP_ADC_STRIPPED/sub-3_rec-ADC_dwi.nii.gz'

    if INPUT_FLAIR is not None:
        if os.path.exists(INPUT_FLAIR) is False:
            raise FileNotFoundError(f'{INPUT_FLAIR} does not exist')

    if os.path.exists(INPUT_DWI) is False:
        raise FileNotFoundError(f'{INPUT_DWI} does not exist')

    if os.path.exists(INPUT_ADC) is False:
        raise FileNotFoundError(f'{INPUT_ADC} does not exist')

    # OUTPUT_PATH = os.path.join('/app', 'data', 'results')  # path-to-output
    OUTPUT_PATH = '/mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code/code/baseline_encoder/DeepIsles/toy_data/SOOP_toy/model_mask'
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    PATH_DEEPISLES = '/mnt/disk1/hieupc/4gpus-Stroke-outcome-prediction-code/code/baseline_encoder/DeepIsles'
    stroke_segm = IslesEnsemble()
    stroke_segm.predict_ensemble(ensemble_path=PATH_DEEPISLES,
                                input_dwi_path=INPUT_DWI,
                                input_adc_path=INPUT_ADC,
                                input_flair_path=INPUT_FLAIR,
                                output_path=OUTPUT_PATH,
                                skull_strip=False,
                                fast=False,
                                save_team_outputs=False,
                                results_mni=False,
                                parallelize=True
                                )

    subprocess.run(['chmod', '777', OUTPUT_PATH], check=True)


if __name__ == '__main__':
    main()