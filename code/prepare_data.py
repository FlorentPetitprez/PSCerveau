import numpy as np
import pylab as pl
import os
import glob
import re

from nisl.io import NiftiMasker


if not "WORD_DECODING_DATA" in os.environ.keys():
    raise Exception("Please set the environment variable WORD_DECODING_DIR" + \
                    " to where the imaging data is using the command\n" + \
                    "export WORD_DECODING_DATA=/home/.../where_the_data_is")

wd_dir = os.environ["WORD_DECODING_DATA"]


# helper function that returns a function which scans a string for a specified
# regular expression and returns None if not found or a match object if found
def re_filter(regex):
    def fil(string):
        return re.search(regex, string)
    return fil


# finding the folder names associated with subjects
subject_regex = r'(\w{2}\d{6})$'
subject_filter = re_filter(subject_regex)
subject_folders = filter(subject_filter,
                         glob.glob(os.path.join(
                             wd_dir, "mri/*")))

subjects = [subject_filter(folder).groups()[0] for folder in subject_folders]
subject_folders = dict(zip(subjects, subject_folders))


def get_nii_data_from_folder(folder, use_verbs=False):

    # inside the folder, search only for betasXX.nii files
    beta_map_regex = r'betas(\d{2}).nii$'
    beta_map_filter = re_filter(beta_map_regex)

    beta_files = filter(beta_map_filter,
                        glob.glob(
                            os.path.join(folder, "*")))

    beta_files = sorted(beta_files)

    # get mask file. It must be in same folder
    mask_file = os.path.join(folder, "mask.nii")

    masker = NiftiMasker(mask_file)

    masker.fit(beta_files[0])

    masked_nii_data = [masker.transform(beta_file)
                       for beta_file in beta_files]

    # this returns a 3D array with dimensions
    # (sessions, trials, voxels)
    masked_nii_data = np.array(masked_nii_data)

    # return only the useful values: The last 6 are
    # drift regressors and are thrown away

    masked_nii_data = masked_nii_data[:, :-6, :]

    if use_verbs:
        return masked_nii_data
    else:
        # In case we do not want the verbs (default case)
        # we need to remove the lines corresponding to verbs

        masked_nii_data = masked_nii_data.reshape(-1,
                                                  masked_nii_data.shape[-1])

        _, verbs, _, _, _ = parse_stimuli()

        return masked_nii_data[verbs == False]


def get_nii_data(subject, normalize_sessions=True):
    folder = os.path.join(subject_folders[subject],
                          "glm")

    data = get_nii_data_from_folder(folder)

    if normalize_sessions:
        data = data.reshape(6, -1, data.shape[-1])
        data = data - data.mean(1)[:, np.newaxis, :]
        data = data / np.sqrt((data ** 2).sum(1))[:, np.newaxis, :]

    data = data.reshape(-1, data.shape[-1])

    return data


def load_stimuli_raw():
    folder = os.path.join(wd_dir, "stimuli")
    stim_file_re = r'lexique_nletters_4_block_(\d)\.txt$'
    stim_file_fil = re_filter(stim_file_re)

    stim_file_names = filter(stim_file_fil, glob.glob(
        os.path.join(folder, "*")))

    stim_file_names = sorted(stim_file_names)

    return np.concatenate([pl.csv2rec(filename, delimiter="\t")
                           for filename in stim_file_names])


def parse_stimuli():
    stimuli = load_stimuli_raw()

    words = stimuli['1_ortho']

    verbs = np.logical_and(np.isnan(stimuli['pseudo']),
                           np.isnan(stimuli['high_freq']))

    bar_names = stimuli.dtype.names[1:-2]

    bars = np.hstack([stimuli[bar][:, np.newaxis]
                      for bar in bar_names])

    rest = stimuli[['pseudo', 'high_freq']]

    return words, verbs, bar_names, bars, rest


def load_stimuli(use_verbs=False):

    words, verbs, bar_names, bars, rest = parse_stimuli()

    bar_names = list(bar_names)

    if use_verbs:
        return words, verbs, bar_names, bars
    else:
        words = words[verbs == False]
        bars = bars[verbs == False]

        used_bars = (bars.sum(0) != 0)

        bars = bars[:, used_bars]

        #        bar_names = [bar_name if used
        #             for used, bar_name in zip(used_bars, bar_name)]

        for un_used, bar_name in zip(used_bars == False, bar_names):
            if un_used:
                bar_names.remove(bar_name)

        return words, bar_names, bars




if __name__ == "__main__":
    pass

