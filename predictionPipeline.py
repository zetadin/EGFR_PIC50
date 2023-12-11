#! /usr/bin/python3
import argparse
import pickle
import rdkit

from tqdm import tqdm, trange
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdmolops
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC

from CustomMolDataset import CustomMolDataset, dataBlocks

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
    args = parser.parse_args()
    
    parser.print_help()
       
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
        
    # load the feature filter index for 2D descriptors.
    # Used to eliminate duplicate and non-finite descriptors
    with open("X_filt_no3D.pickle", 'rb') as f:
        X_filt = pickle.load(f)
        
    # create a dataset instance that will calculate all the 2D descriptors
    descriptorBlocknames = ["MACCS", "rdkitFP", "MorganFP2", "MorganFP3",
                            "Descriptors", "EState_FP", "Graph_desc",
                            "MOE", "MQN", "AUTOCORR2D", "PEOE_VSA",
                            "SMR_VSA", "SlogP_VSA"]
    descriptorFlags = [int(dataBlocks(i).name in descriptorBlocknames) for i in range(len(dataBlocks))]
    ds = CustomMolDataset(molecules,
                          representation_flags = descriptorFlags,
                          X_filter = X_filt,        # apply the feature filter
                          normalize_x = False,    # do not normalize the features, we will load normalization factors later
                          use_hdf5_cache = True,  # cache results to an hdf5 file
                          name = "predTest"       # name of the hdf5 cache file
                         )
    
    # compute 2D descriptors now
    for i in trange(0, len(molecules), desc="Computing 2D descriptors"):
        _, _ = ds[i]
        
    # load feature normalization constants
    
    
    # load model
    with open("models_for_inference.pickle", 'rb') as f:
        (pls, classifier) = pickle.load(f)
        
    # apply PLS transform
    
    
    # predict actives with SVC
    
    
    # output predicted actives
        
        
    # report end    
    print("Done.")
    