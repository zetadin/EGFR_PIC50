"""
Implements a CustomMolDataset class that computes, caches,
normalizes, and filters RDKit descriptos for molecules.
This is a pytorch Dataset subclas originally designed
for fast dataloading when training neural networks on
descriptor-derrived features.
"""


import os
import gc
import pickle
from enum import Enum, auto
from inspect import getmembers, isfunction, getfullargspec

import h5py
import numpy as np
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, ChemicalFeatures, rdmolops
from rdkit.DataStructs import cDataStructs
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Generate
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.EState import Fingerprinter
import rdkit.Chem.EState.EState_VSA
import rdkit.Chem.GraphDescriptors



class DataBlocks(Enum):
    """
    Enumerator for types of supported RDKit Descriptors.
    Alows them to be identified via their name strings or indeces.
    """
    MACCS = 0
    rdkitFP = auto()
    minFeatFP = auto() # Slow, causes triangle inequality violations
    MorganFP2 = auto()
    MorganFP3 = auto()
    
    Descriptors = auto()
    EState_FP = auto()
    Graph_desc = auto()
    
    #extras
    MOE = auto()
    MQN = auto()
    GETAWAY = auto() # Slow for some ligands
    AUTOCORR2D = auto()
    AUTOCORR3D = auto()
    BCUT2D = auto()  # has a problem with ionic Zn^2+
    WHIM = auto()
    RDF = auto()
    USR = auto()
    USRCUT = auto()
    PEOE_VSA = auto()
    SMR_VSA = auto()
    SlogP_VSA = auto()
    MORSE = auto()
    
    def __int__(self):
        return self.value
    
    
def wiener_index(m):
    """
    Computes the Wiener index, which is not included in RDKit.
    Part of graph descriptors group.
    """
    res = 0
    amat = rdmolops.GetDistanceMatrix(m)
    num_atoms = m.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(i+1,num_atoms):
            res += amat[i][j]
    return res
    
def bit_vect_to_array(vec):
    """Conversion function from RDKit's bit vectors to a numpy array"""
    txt=cDataStructs.BitVectToText( vec )
    arr=np.zeros(len(txt), dtype=np.uint8)
    for j,c in enumerate(txt):
        if c=="1":
            arr[j]=1
    return arr

    
# The Dataset subclass that generates molecular features on first access to the molecule
class CustomMolDataset(Dataset):
    """
    A pytorch Dataset subclass that computes RDKit Descriptors for molecules.
    It then caches them in an hdf5 file grouped by descriptor type for fast access.
    It also supports automatic normalization (cached separately) of features, and
    feature filtering via an index array (to remove low quaity pre-computed features).
    """
    def __init__(self,
                 ligs, # list of RDKit molecules. Expects "ID" and "pIC50" properties set.
                       # "ID" -> unique identifier "pIC50" -> Y-values (nan if empty)
                 name="unnamed", # name of the cache file
                 representation_flags=[1]*(len(DataBlocks)), # list of booleans
                       #which marks DataBlocks of features that should be calculated, default: all
                 work_folder=os.path.split(os.path.realpath(__file__))[0], # folder of input files
                 cache_folder=os.path.split(os.path.realpath(__file__))[0], # folder of cache file
                 normalize_x=False, # should we normalize the data?
                 X_filter=None,     # numpy array of indeces of the features we want, None for all
                 verbose=False,     # be noisy? Debug output.
                 use_hdf5_cache=True, # use the hdf5 cache file? True/False/"read_only"
                 internal_cache_max_mem_MB=512 # max size of in-memory cache in MB
                 ):
        self.representation_flags=representation_flags
        self.active_flags=np.where(self.representation_flags)[0]
        self.work_folder=work_folder
        self.normalize_x=normalize_x
        self.norm_mu=None
        self.norm_width=None
        self.X_filter=None
        self.internal_filtered_cache=None
        self.verbose=verbose
        self.use_hdf5_cache=use_hdf5_cache
        self.ligs=ligs
        
        # used for a fast in-memory cache as a numpy array
        self._internal_cache_max_mem=internal_cache_max_mem_MB*1024*1024 # 512 MB by default
        
        # set up for feature filtering
        if X_filter is not None:
            if isinstance(X_filter, np.ndarray):
                if X_filter.ndim!=1:
                    raise ValueError("X_filter should a 1D nunpy array"
                                     " or a filename of a pickled 1D array.")
                self.X_filter=X_filter
            elif not os.path.exists(X_filter):
                raise IOError(f"No such file: {X_filter}")
            else:
                with open(X_filter, 'rb') as f:
                    self.X_filter=pickle.load(f)
        
        
        # load configuration for SigFactory: binned custom feature definitions of RDKit (SLOW!)
        if self.representation_flags[int(DataBlocks.minFeatFP)]:
            fdef_name = self.work_folder+'/MinimalFeatures.fdef'
            feat_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
            self.sig_factory = SigFactory(feat_factory,
                                         minPointCount=2,
                                         maxPointCount=3,
                                         trianglePruneBins=False)
            self.sig_factory.SetBins([(0,2),(2,5),(5,8)])
            self.sig_factory.Init()

        # open the cache file
        if self.use_hdf5_cache:
            self.cache_folder=cache_folder
            self.name=name
            self.cache_fn=f"{self.cache_folder}/{self.name}.hdf5"
            if not os.path.exists(self.cache_folder): #make sure the folder exists
                os.makedirs(self.cache_folder)
                
            if(self.use_hdf5_cache in ["read-only", "r"]):
                # cache in read-only mode requested
                
                if os.path.exists(self.cache_fn): # requested read-only, but no chache file found
                    self.cache_fp=h5py.File(self.cache_fn, "r")
                else:
                    print(f"{self.cache_fn} does not exist yet."
                          " Switchinhg to append mode and will create one.")
                    self.use_hdf5_cache=True
                    self.cache_fp=h5py.File(self.cache_fn, "a")
                    
            else:
                # cache in append mode requested
                self.cache_fp=h5py.File(self.cache_fn, "a")
                            
    def __del__(self):
        # close hdf5 cache file
        if(self.use_hdf5_cache and self.cache_fp):
            self.cache_fp.close()
                    
    def find_ranges(self):
        """Computes max and min for features in the dataset."""
        all_x=np.array([entry[0] for entry in self])
        all_range=np.zeros((all_x.shape[1],2))
        all_range[:,0]=np.min(all_x, axis=0) # axis=0 loops ove all the ligands
        all_range[:,1]=np.max(all_x, axis=0)
        return all_range

    def find_normalization_factors(self):
        """Computes normalization of features in the dataset."""
        self.normalize_x=False # temporarily disable normalization so we can get raw values
        all_x=np.array([entry[0] for entry in self])
        self.norm_mu=np.mean(all_x, axis=0)
        
        # std sometimes overflows from very large feature values, so temporarily scale them
        ranges = self.find_ranges()
        scales = ranges[:,1]-ranges[:,0]
        self.norm_width=np.std(all_x/scales, axis=0)*scales
        self.norm_width[self.norm_width<1e-7]=1.0 # if standard deviation is 0, don't scale
        self.normalize_x=True
        
        if self.verbose:
            print(f"Generating normalization factors for a {all_x.shape} dataset")
                
        # build in-memory cache
        self.build_internal_filtered_cache()
        _=gc.collect()
        
        # save normalization factors for later reuse in prediction pipeline
        if self.use_hdf5_cache:
            # note: this will ovewrite the the normalization factors
            # in read-only mode of the cache, as they are in a different file
            
            # find number of features, to distinguish different cache files
            # with different X_filters and representations
            if self.X_filter is not None: # that is size of the X_filter, if set
                n_features = self.X_filter.shape[0]
            else: # or size of transform() output if not
                n_features = self.transform(0).shape[0]
                
            # then pickle factors into appropriately named file
            with open(f"{self.name}_NormFactors_{n_features}_Features.pickle", 'wb') as f:
                pickle.dump((self.norm_mu, self.norm_width), f)

                
    def read_normalization_factors(self, fname):
        """Reads normalization of features from a pickle file."""
        # read previously saved normalization factors from a file
        with open(fname, 'rb') as f:
            (mu, width) = pickle.load(f)
            
        # sanity checks
        if mu.shape != width.shape:
            raise ValueError(f"Mutually incompatible normalization factors loaded from {fname}!")
            
        # check if loaded factors have same width as current data representation
        # that is size of the X_filter, if set
        if self.X_filter is not None:
            features_shape = self.X_filter.shape
        # or size of transform() output if not
        else:
            features_shape = self.transform(0).shape # get all features for molecule 0
        if(mu.shape[0] != features_shape[0] or width.shape[0] != features_shape[0]):
            raise ValueError("Shapes of loaded normalization factors"
                             f" {mu.shape} and {width.shape} loaded from {fname}"
                             " do not match curent molecular representation"
                             f" shape {features_shape}!")
        
        # if everything passes, store the factors for use
        self.norm_mu = mu
        self.norm_width = width
            
        # Set the normalize flag
        self.normalize_x = True


    # copies normalization factors from another dataset. 
    # Eg. for making a test set that has the same normalization as all whole dataset.
    def copy_normalization_factors(self, other):
        """Copies normalization of features from a another CustomMolDataset."""
        # sanity checks
        # compare X_filters, if set
        if(self.X_filter is not None and
           other.X_filter is not None and
           not np.array_equal(self.X_filter,other.X_filter)
           ):
            raise ValueError("Mismatching X_filters in Datasets!")
        
        # compare feature array sizes
        if self.X_filter is not None: # if X_filter set
            m_features = self.X_filter.shape[0]
        else: # read number of features from transform instead
            m_features = self.transform(0).shape[0]
            
        # same for the other Dataset
        if other.X_filter is not None: # if X_filter set
            o_features = other.X_filter.shape[0]
        else: # read number of features from transform instead
            o_features = other.transform(0).shape[0]
        
        if m_features != o_features:
            raise ValueError("Mismatched number of used features in Datasets!")
        
        self.norm_mu=other.norm_mu
        self.norm_width=other.norm_width
       
    # builds the in-memory cache
    def build_internal_filtered_cache(self):
        """Builds the in-memory cache in the form of numpy arrays for
           rapid access of filtered features."""
        if(self.norm_mu is None and self.normalize_x):
            raise RuntimeError("call build_internal_filtered_cache() only after normalization!")
        needed_mem=len(self)*(self[0][0].shape[0]+self[0][1].shape[0])*self[0][1].itemsize
        if needed_mem>self._internal_cache_max_mem:
            raise MemoryError("Building the internal_filtered_cache needs"
                              f" {needed_mem/1024/1024} MB, more than the"
                              f" {self._internal_cache_max_mem/1024/1024} MB limit.")
        all_x=[]
        all_y=[]
        for entry in self: # loop over self only once
            all_x.append(entry[0])
            all_y.append(entry[1])
        all_x=np.array(all_x)
        all_y=np.array(all_y)
        self.internal_filtered_cache=(all_x, all_y)
        if self.verbose:
            print("Creating an in-memory filtered & normalized cache"
                  f" of shape ({self.internal_filtered_cache[0].shape},"
                  f"{self.internal_filtered_cache[1].shape})")


    # normalizes features of a single ligand
    def _normalize_input(self,x):
        """Normalizes features for a single molecule.
           Should only be called internally."""
        if self.norm_mu is None:
            self.find_normalization_factors()
        return (x-self.norm_mu)/self.norm_width



    def __len__(self):
        return len(self.ligs)

    # main acess method for data in Dataset, eg through Dataset[index]
    def __getitem__(self, idx):
        """Acessor for elements of the dataset via []."""
        lig = self.ligs[idx]

        # if there is no in-memory cache
        if self.internal_filtered_cache is None:
        
            # load data from HDD or compute it
            x = self.transform(idx).astype(np.float32)
            y = np.array([float(lig.GetProp('pIC50')) 
                          if lig.HasProp('pIC50') else np.nan])
           
            # apply feature filter
            if not self.X_filter is None:
                x=x[self.X_filter]
            # normalize
            if self.normalize_x:
                x=self._normalize_input(x)
                
        # otherwize read the in-memory cache
        else:
            x=self.internal_filtered_cache[0][idx]
            y=self.internal_filtered_cache[1][idx]
        
        return x, y
            
            
    # Compute the different descriptors in blocks
    def generate_data_block(self, lig, block_id):
        """Computes descriptors in a DataBlock for a single molecule."""
        block_id = DataBlocks(block_id)
        
        if block_id==DataBlocks.MACCS:
            Chem.GetSymmSSSR(lig)
            return bit_vect_to_array(rdMolDescriptors.GetMACCSKeysFingerprint(lig))
        
        if block_id==DataBlocks.MorganFP2:
            Chem.GetSymmSSSR(lig)
            return bit_vect_to_array(rdMolDescriptors.GetMorganFingerprintAsBitVect(lig, 2))
            
        if block_id==DataBlocks.MorganFP3:
            Chem.GetSymmSSSR(lig)
            return bit_vect_to_array(rdMolDescriptors.GetMorganFingerprintAsBitVect(lig, 3))
        
        if block_id==DataBlocks.rdkitFP:
            Chem.GetSymmSSSR(lig)
            return bit_vect_to_array(Chem.rdmolops.RDKFingerprint(lig))
        
        if block_id==DataBlocks.minFeatFP:
            Chem.GetSymmSSSR(lig)
            return bit_vect_to_array(Generate.Gen2DFingerprint(lig, self.sig_factory))
    
        if block_id==DataBlocks.Descriptors:
            nms=[x[0] for x in Descriptors._descList]
            calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
            des = np.array(calc.CalcDescriptors(lig))
            return des
        
        if block_id==DataBlocks.EState_FP:
            electric_state=Fingerprinter.FingerprintMol(lig)
            funcs=getmembers(rdkit.Chem.GraphDescriptors, isfunction)
            funcs=[f[1] for f in funcs if f[0][0]!='_' and len(getfullargspec(f[1])[0])==1]
            electric_state_VSA=np.array([f(lig) for f in funcs])
            finger_print=np.concatenate((electric_state[0],electric_state[1],electric_state_VSA))
            return finger_print
    
        if block_id==DataBlocks.Graph_desc:
            funcs=getmembers(rdkit.Chem.GraphDescriptors, isfunction)
            funcs=[f[1] for f in funcs if f[0][0]!='_' and len(getfullargspec(f[1])[0])==1]
            funcs+=[wiener_index]
            graph_desc=np.array([f(lig) for f in funcs])
            return graph_desc
    
        
        # extras
        if block_id==DataBlocks.MOE:
            funcs=getmembers(rdkit.Chem.MolSurf, isfunction)
            funcs=[f[1] for f in funcs if f[0][0]!='_' and len(getfullargspec(f[1])[0])==1]
            return np.array([f(lig) for f in funcs])
        if block_id==DataBlocks.MQN:
            return np.array(rdMolDescriptors.MQNs_(lig))
        if block_id==DataBlocks.GETAWAY:
            return np.array(rdMolDescriptors.CalcGETAWAY(lig))
        if block_id==DataBlocks.AUTOCORR2D:
            return np.array(rdMolDescriptors.CalcAUTOCORR2D(lig))
        if block_id==DataBlocks.AUTOCORR3D:
            return np.array(rdMolDescriptors.CalcAUTOCORR3D(lig))
        if block_id==DataBlocks.BCUT2D:
            return np.array(rdMolDescriptors.BCUT2D(lig))
        if block_id==DataBlocks.WHIM:
            return np.array(rdMolDescriptors.CalcWHIM(lig))
        if block_id==DataBlocks.RDF:
            return np.array(rdMolDescriptors.CalcRDF(lig))
        if block_id==DataBlocks.USR:
            return np.array(rdMolDescriptors.GetUSR(lig))
        if block_id==DataBlocks.USRCUT:
            return np.array(rdMolDescriptors.GetUSRCAT(lig))
        if block_id==DataBlocks.PEOE_VSA:
            return np.array(rdMolDescriptors.PEOE_VSA_(lig))
        if block_id==DataBlocks.SMR_VSA:
            return np.array(rdMolDescriptors.SMR_VSA_(lig))
        if block_id==DataBlocks.SlogP_VSA:
            return np.array(rdMolDescriptors.SlogP_VSA_(lig))
        if block_id==DataBlocks.MORSE:
            return np.array(rdMolDescriptors.CalcMORSE(lig))
            
        raise NotImplementedError(f"Unsupported dataBlock requested: {block_id}")
        
        
            
    # load descriptors from HDD or compute them via transform()
    def transform(self, lig_idx):
        """Computes all active descriptors for a molecule."""
        vecs=[]
                
        # iterate through requested blocks of descriptors
        for i in self.active_flags:
            
            # first try reading from cache
            if self.use_hdf5_cache:
                lig_id = self.ligs[lig_idx].GetProp("ID")
                node = f"{lig_id}/{DataBlocks(i).name}"
                # is there an entry for this ligand & dataBlock?
                if node in self.cache_fp.keys():
                    x_block_rep = self.cache_fp[node] # then read it
                    vecs.append(x_block_rep) #and add it to the overall representation
                    continue # next dataBlock
                    
            #if not cashed, compute it
            x_block_rep = self.generate_data_block(self.ligs[lig_idx], i)
            vecs.append(x_block_rep) #add to overall representation
                  
            # and cache it if in append mode
            if(self.use_hdf5_cache and 
               not (self.use_hdf5_cache in ["read-only","r"])
              ):
                
                lig_id = self.ligs[lig_idx].GetProp("ID")
                node = f"{lig_id}/{DataBlocks(i).name}"
                self.cache_fp.create_dataset(node, data=x_block_rep, dtype='f')
                
        return(np.concatenate(tuple(vecs), axis=0)) # flatten into a 1D array
