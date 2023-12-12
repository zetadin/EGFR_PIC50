#! /usr/bin/python3

# This is a prediction pipeline only.
# Training happened in Step3_TrainSimpleModels.ipynb.
# Models, feature selection, and normalization was saved to pickle files which we load here.

import argparse
import pickle
import rdkit
import numpy as np

from tqdm import tqdm, trange
from rdkit import Chem
from rdkit.Chem import rdmolops
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC

from CustomMolDataset import CustomMolDataset, dataBlocks

# only run code if executing current file, not if importing it for some reason
if __name__ == "__main__":
    
    # parse comand line arguments
    parser = argparse.ArgumentParser(
        description='Classifies molecules as active or inactive with respect to binding to EGFR with pIC50>8',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('-i', dest='input', type=str,
                        default="predTest.smi",
                        help='Input file with SMILES strings, one per line.')
    parser.add_argument('-o', dest='output', type=str,
                        default=None,
                        help='Output file for SMILES strings of predicted active compounds. None prints to screen.')
    parser.add_argument('--maxcache', dest='maxcache', type=float,
                        default=512,
                        help='Maximum requested size (in MB) of the in-memory cache for storing features of all the molecules.'
                        ' Increase this if running predictions for a large number of molecules.')
    args = parser.parse_args()
    
    # show help
    parser.print_help()
    print("\n\n") # add some space after help
       
    # load SMILES strings from input file
    smis = []
    with open(args.input, 'r') as inpfile:
        lines=inpfile.readlines()
        for l in lines:
            l = l.strip() # clean white space
            # ignore comment and empty lines
            if(len(l)==0):
                continue
            elif(l[0]==';'):
                continue
            smis.append(l)
            
    # build RDKit molecules from the strings
    molecules=[]
    for smi in tqdm(smis, desc="Building molecules from SMILES"): # add a progress bar
        mol = Chem.MolFromSmiles(smi) # build a molecule
        mol = rdmolops.AddHs(mol) # add hydrogens
        mol.SetProp("ID", smi)    # set ID
        molecules.append(mol)
        
    # load the feature filter containing indeces of useful 2D descriptors.
    # Used to eliminate duplicate and non-finite descriptors
    with open("X_filt_no3D.pickle", 'rb') as f:
        X_filt = pickle.load(f)
    print(f"Loaded a feature filter that retains {len(X_filt)} features.")
        
    # create a CustomMolDataset instance that will calculate all the 2D descriptors
    descriptorBlocknames = ["MACCS", "rdkitFP", "MorganFP2", "MorganFP3",
                            "Descriptors", "EState_FP", "Graph_desc",
                            "MOE", "MQN", "AUTOCORR2D", "PEOE_VSA",
                            "SMR_VSA", "SlogP_VSA"]
    descriptorFlags = [int(dataBlocks(i).name in descriptorBlocknames) for i in range(len(dataBlocks))]
    ds = CustomMolDataset(molecules,
                          representation_flags = descriptorFlags, # encodes selection of descriptor types to use
                          X_filter = X_filt,        # apply the feature filter
                          normalize_x = False,      # do not normalize the features, we will load normalization factors later
                          use_hdf5_cache = True,    # cache results to an hdf5 file
                          name = "predTest",        # name of the hdf5 cache file
                          internal_cache_maxMem_MB = args.maxcache # size of the requested cache
                         )
    
    # compute 2D descriptors now
    for i in trange(0, len(molecules), desc="Computing 2D descriptors"):
        _, _ = ds[i]
        
    # load feature normalization constants
    # This will force the CustomMolDataset to normalize descriptors for the new molecules the same way as during training
    ds.read_normalization_factors("EGFR_set_all_features_NormFactors_6740_Features.pickle")
    
    # load models
    with open("models_for_inference.pickle", 'rb') as f:
        (pls, classifier) = pickle.load(f)
        
    # get the features as a numpy array
    ds.build_internal_filtered_cache() # build the in-memory cache
    allFeatures = ds.internal_filtered_cache[0] # reference it
        
    # apply PLS transform with 8 components
    transformedFeatures = pls.transform(allFeatures)
    
    # predict actives with SVC using a radial basis function kernel
    predictedClass = classifier.predict(transformedFeatures)
    
    # output predicted actives
    actives_idxs = np.where(predictedClass==1)[0]
    print(f"Found {len(actives_idxs)} active compounds among {len(smis)} evaluated ones.")
    
    if(args.output): # output file specified
        print(f"Writing them to \"{args.output}\".")
        with open(args.output, "w") as of:
            for idx in actives_idxs:
                of.write(smis[idx]+"\n")
    
    else: # no output file, so print to screen
        print("They are listed below:\n\n")
        for idx in actives_idxs:
            print(smis[idx])
    
    # report the end    
    print("\n\nDone.")
    exit(0)
    