#########################################
## CS 209b MICROBIOME T1D UTILS
#########################################
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#########################################
## DATA LOADING
#########################################


def load_data(path): 
    path_16s = os.path.join(path, "data_16s_patients.csv")
    path_shotgun = os.path.join(path, "shotgun_seq1.csv")
    # read in the data for 16s sequencing from csv 
    data_16s = pd.read_csv(path_16s,header=None).transpose()

    # read in the data for shotgun sequencing from csv 
    data_shotgun = pd.read_csv(path_shotgun, header=None).transpose() 
    bacteria_start_col_16s = 60 
    bacteria_start_col_shotgun = 2 

    # fixing indices 
    data_16s = data_16s.rename(columns=data_16s.iloc[0]).drop(data_16s.index[0])
    data_shotgun = data_shotgun.rename(columns=data_shotgun.iloc[0]).drop(data_shotgun.index[0])

    # remove duplicate columns if there are any 
    data_16s = data_16s.loc[:,~data_16s.columns.duplicated()]
    data_shotgun = data_shotgun.loc[:,~data_shotgun.columns.duplicated()]

    # setting to numeric for sortiing later
    data_16s["Age_at_Collection"] = pd.to_numeric(data_16s["Age_at_Collection"])

    # return the dataframe with the entire data + split based on patient/bacteria only data 
    patients = data_16s.iloc[:, list(range(0, bacteria_start_col_16s))] 
    patients.index = pd.MultiIndex.from_arrays([patients["Subject_ID"], patients["sample"]], 
                                names=('subject_id', 'sample_index'))

    # same patient dataframe for both sequencing techniques 
    patients = patients.sort_values(["Subject_ID", "Age_at_Collection"])

    # returning dataframes with bacteria data only + including sample id for grouping later 
    bacteria_16s = data_16s.iloc[:, list(range(bacteria_start_col_16s, len(data_16s.columns)))] 
    bacteria_16s.loc[:,"sample"] = data_16s["sample"]

    bacteria_shotgun = data_shotgun.iloc[:, list(range(bacteria_start_col_shotgun, len(data_shotgun.columns)))] 
    bacteria_shotgun.loc[:,"sample"] = data_shotgun["Sample_ID"]

    return data_16s, data_shotgun, bacteria_16s, bacteria_shotgun, patients 

# use:
# import utils
# data_16s, data_shotgun, bacteria_16s, bacteria_shotgun, patients = utils.load_data()

#########################################
## BACTERIAL DF PROCESSING
#########################################

# Example on how to only access the rows at a given Taxonomy level (Genus in this case)
# index names : 'Kingdom','Phylum','Class','Order','Family','Genus','Species','Strain'

def pad_reindex(data_16s_df):
    """
    Create dataframe with only bacteria measurements from the original data16s dataframe

    Function returns a dataframe where columns are each sample ID and the row index 
    is a padded multiIndex of all of the taxonomy levels
    """
    bacteria_names = [col for col in data_16s_df.columns if col.startswith('k__Bacteria')]

    # Create dataset of bacteria measurements only for each sample
    col_names = ['sample'] + bacteria_names

    df = data_16s_df[col_names]

    df = df.transpose()
    df.columns = df.loc['sample']
    df.drop('sample', inplace=True)

    def pad_lists(arr, length=8, pad_token='NA'):
        # Post pads up to length
        # Truncates after len
        return [seq[:length] + [pad_token for i in range(length - len(seq))] for seq in arr]

    padded_names = pad_lists(df.index.str.split('|'))

    multi_index = pd.MultiIndex.from_arrays(np.array(padded_names).transpose(), names= ['Kingdom','Phylum','Class','Order','Family','Genus','Species','Strain'])

    df.set_index(multi_index, verify_integrity=True, inplace=True)

    return df


def taxonomy_reshaping(df, taxonomy_level, remove_taxonomy_path=True):
    """
    Parameters
    -----------
    return_single_level_col: Boolean that determines wheter to truncate the previous taxonomy levels from the column names
    """
    taxonomy = ['Kingdom','Phylum','Class','Order','Family','Genus','Species','Strain']
    assert taxonomy_level in taxonomy, "Taxonomy level must be one of ['Kingdom','Phylum','Class','Order','Family','Genus','Species','Strain']"

    processed_df = df.copy()

    # Genus-only measurements are those for which the "Genus" column is not NA but the following 'Species' column is NA
    level_not_NA_filter = processed_df.index.get_level_values(taxonomy_level) != 'NA'
    processed_df = processed_df.loc[level_not_NA_filter]

    if taxonomy_level == 'Strain':
        # The strain level is the lowest in the taxonomy and therefore, no next levels must be removed
        print("Dataframe is already at the strain level, It remains unchanged")
    
    else :
        current_level_index = taxonomy.index(taxonomy_level)

        next_level_idx = current_level_index + 1
        next_level_NA_filter = processed_df.index.get_level_values(taxonomy[next_level_idx]) == 'NA'

        # filter out all measurements that are not at the current taxonomy level
        processed_df = processed_df.loc[next_level_NA_filter]

        # Remove index levels that are no longer relevant after the current taxonomy level (all NA)
        processed_df.reset_index(level=taxonomy[next_level_idx:], inplace=True, drop=True)

    
    # Rename indeces
    if remove_taxonomy_path:
        # Remove previous indeces of the path and only 
        processed_df.reset_index(level=taxonomy[:current_level_index], inplace=True, drop=True)
    else:
        processed_df.index = ["|".join(idx) for idx in processed_df.index]

    processed_df = processed_df.transpose()

    return processed_df


def numerize_df(df):
    """
    Convert all columns in dataframe to numeric
    """
    for column in df.columns:
        df[column] = pd.to_numeric(df[column])
    return df

#########################################
## PATIENT DF PROCESSING
#########################################


def compute_seropositivity(df):
    """
    Compute seropositivity mask of a dataframe defined as being positive 
    for at least two of the five autoantibodies analyzed in this study:
    Dataframe must contain columns 
    'GADA_Positive'
    'IAA_Positive'
    'ICA_Positive'
    'IA-2A_Positive'
    'ZNT8A_Positive'
    
    """
    for col in ['GADA_Positive', 'IAA_Positive', 'ICA_Positive' ,'IA-2A_Positive', 'ZNT8A_Positive']:
        assert col in df.columns

    return (df['GADA_Positive'].astype(int) + 
            df['IAA_Positive'].astype(int) + 
            df['ICA_Positive'].astype(int) + 
            df['ZNT8A_Positive'].astype(int) + 
            df['IA-2A_Positive'].astype(int)) >= 2 

def compute_group(input_df):
    
    df = input_df.copy() 

    # defined as being positive for at least two of the five autoantibodies analyzed 
    # in this study which are IAA, GADA, IA-2A, ZNT8A, ICA
    T1D_diagnosed_filt = df['T1D_Diagnosed'] == 1
    control_filt = df["Case"] == 0

    # Control: Nonconverters
    df.loc[control_filt ,'Group'] = 0

    # Seroconverters AND T1D
    df.loc[~control_filt & T1D_diagnosed_filt,'Group'] = 1

    # Seroconverters Non T1D
    df.loc[~control_filt & ~T1D_diagnosed_filt,'Group'] = 2


    # non-converter (not seroconverted, not T1D), 
    # seroconverted (seroconverters, not diagnosed with T1D), 
    # and T1D cases (seroconverted subjects also diagnosed with T1D)
    

    print("There are " + str(control_filt.sum()) + " T1D diagnosed samples")
    print("There are " + str(((~control_filt) & T1D_diagnosed_filt).sum()) + " Seropositive samples")
    print("There are " + str((~control_filt & ~T1D_diagnosed_filt).sum()) + " Non-Converter samples")

    assert (df.loc[control_filt,'Subject_ID'].nunique() + 
            df.loc[(~control_filt) & T1D_diagnosed_filt,'Subject_ID'].nunique() +
            df.loc[~control_filt & ~T1D_diagnosed_filt,'Subject_ID'].nunique()) == df['Subject_ID'].nunique(), "The number of Subjects doesn't match"

    return df['Group']

def preprocess_patients(patients_input): 
    patients = patients_input.copy() 
    categorical_cols = ['AbxAtCollection', 'AbxPreCollection', 'DNA_Yield_Class', 'Flowcell', 'HLA_Risk_Class', 'Read_Depth_Class', 'IllnessAtCollection']
    string_cols = ['sample', 'Subject_ID', 'Container_1', 'Container_2'] 
    level_cols = ['ZNT8A_Level', 'ICA_Level', 'IAA_Level', 'IA-2A_Level', 'GADA_Level']
    ignore_cols = ["Delivery_Route", "Gender", "Case_Control", "Country", "Collection_Location"]
    
    for column in patients.columns: 
        if column in categorical_cols: 
            patients = pd.get_dummies(patients, columns = [column], drop_first=True)
        elif (column in string_cols) or (column in ignore_cols): 
            # we want the value to remain as strings 
            continue 
        elif column in level_cols: 
            patients.loc[patients[column] == "neg", column] = 0 
            patients.loc[patients[column] == "zero", column] = 1
            patients.loc[patients[column] == "pos", column] = 2 
        else: 
            # convert to numeric 
            patients.loc[patients[column] == "engine='openpyxl'", column] = "0" 
            patients[column] = pd.to_numeric(patients[column])

    # map directly Delivery_Route, Country, Gender
    patients.replace({"Delivery_Route": {"vaginal": 1, "cesarian": 0},
                        "Gender": {"female": 1, "male":0}, "Country":  {"Finland": 1, "Estonia":0}, 
                        "Case_Control" : {"case": 1, "control": 0}, 
                        "Collection_Location": {"Jorvi": 1, "Tarto": 0}}, 
                        inplace=True)
    patients.rename(columns={"Delivery_Route": "Delivery_Route_Vaginal", 
                        "Gender": "Gender_Female", "Country": "Country_Finland", 
                        "Case_Control": "Case", 
                        "Collection_Location": "Collection_Location_Jorvi"}, inplace=True)
    patients["Antibody_Score"] = (patients["IA-2A_Positive"] + patients["IAA_Positive"] + patients["ICA_Positive"] + 
                patients["ZNT8A_Positive"] + patients["GADA_Positive"])

    patients['Group'] = compute_group(patients)
    
    return patients 
            

