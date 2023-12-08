import errno
import numpy as np
import os
import gc
from enum import Enum, auto
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdmolfiles, rdMolAlign, rdmolops, rdchem, rdMolDescriptors, ChemicalFeatures
from rdkit.Chem import PeriodicTable, GetPeriodicTable
from rdkit import RDConfig
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit.DataStructs import cDataStructs
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Generate
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.EState import Fingerprinter
import rdkit.Chem.EState.EState_VSA
import rdkit.Chem.GraphDescriptors
from inspect import getmembers, isfunction, getfullargspec
import h5py
import hashlib


try:
    import cPickle as pickle
except:
    import pickle

from rdkit.Chem import rdRGroupDecomposition as rdRGD
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class dataBlocks(Enum):   
    MACCS = 0
    rdkitFP = auto()
    minFeatFP = auto() # causes triangle inequality violations and hence unequal number of features for some entries in the FreeSolv set
    MorganFP2 = auto()
    MorganFP3 = auto()
    
    Descriptors = auto()
    EState_FP = auto()
    Graph_desc = auto()
    
    #extras
    MOE = auto()
    MQN = auto()
    GETAWAY = auto()
    AUTOCORR2D = auto()
    AUTOCORR3D = auto()
    BCUT2D = auto()
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
    
    
    
# helper functions that used to be in a separate file
def wiener_index(m):
    res = 0
    amat = Chem.GetDistanceMatrix(m)
    num_atoms = m.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(i+1,num_atoms):
            res += amat[i][j]
    return res

def get_feature_score_vector(lig_fmap, xray_fmap):
    #http://rdkit.blogspot.com/2017/11/using-feature-maps.html
    #https://link.springer.com/article/10.1007/s10822-006-9085-8
    vec=np.zeros(xray_fmap.GetNumFeatures())
    for f in range(len(vec)):
        xray_feature=xray_fmap.GetFeature(f)
        vec[f]=np.sum([lig_fmap.GetFeatFeatScore(lig_fmap.GetFeature(f_l), xray_feature) for f_l in range(lig_fmap.GetNumFeatures())])
    return(vec)


    
# The Dataset subclass that generates molecular features on first access to the molecule
class CustomMolDataset(Dataset):
    def __init__(self,
                 ligs, # list of RDKit molecules. Expects "ID" and "dG" properties set.
                       # "ID" -> unique identifier "dG" -> Y-values
                 name="unnamed", # name of the cahce file
                 representation_flags=[1]*(len(dataBlocks)-1), # list of booleans for the dataBlocks of features that should be calculated, default: all
                 work_folder=os.path.split(os.path.realpath(__file__))[0], # look for config files in same folder as this file
                 cachefolder=os.path.split(os.path.realpath(__file__))[0], # save cache in same folder as this file
                 normalize_x=False, # should we normalize the data?
                 X_filter=None,     # numpy array of indeces of the features we actually want, None for all
                 verbose=False,     # be noisy? Debug output.
                 use_hdf5_cache=True, # other options are False and "read-only", refers to a cache file in the FS
                 internal_cache_maxMem_MB=512 # max size of in-memory cache
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
        
        # used for a fast in-memory cache as a numpy array
        self._internal_cache_maxMem=internal_cache_maxMem_MB*1024*1024 # 512 MB by default
        
        # set up for feature filtering
        if(X_filter is not None):
            if type(X_filter) is np.ndarray:
                if(X_filter.ndim!=1):
                    raise ValueError("X_filter should a 1D nunpy array or a filename of a pickled 1D array.")
                self.X_filter=X_filter
            elif(not os.path.exists(X_filter)):
                raise(Exception(f"No such file: {X_filter}"))
            else:
                with open(X_filter, 'rb') as f:
                    self.X_filter=pickle.load(f)
                    
        # set up for feature calculation
        self.ligs=ligs
        if(self.representation_flags[int(dataBlocks.minFeatFP)]):
            fdefName = self.work_folder+'/MinimalFeatures.fdef'
            featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)
            self.sigFactory = SigFactory(featFactory,minPointCount=2,maxPointCount=3, trianglePruneBins=False)
            self.sigFactory.SetBins([(0,2),(2,5),(5,8)])
            self.sigFactory.Init()
        

            

        #open the cache file
        if(self.use_hdf5_cache):
            self.cachefolder=cachefolder
            self.name=name
            self.cache_fn=f"{self.cachefolder}/{self.name}.hdf5"
            if not os.path.exists(self.cachefolder): #make sure the folder exists
                os.makedirs(self.cachefolder)
                
            if(self.use_hdf5_cache=="read-only" or self.use_hdf5_cache=="r"): # cache in read-only mode requested
                if(os.path.exists(self.cache_fn)): # requested read-only, but no chache file found
                    self.cache_fp=h5py.File(self.cache_fn, "r")
                else:
                    print(f"{cache_fn} does not exist yet. Switchinhg to append mode and will create one.")
                    self.use_hdf5_cache=True
                    self.cache_fp=h5py.File(self.cache_fn, "a")
                    
            else: # cache in append mode requested
                sef.cache_fp=h5py.File(self.cache_fn, "a")
                            
    def __del__(self):
        # close hdf5 cache file
        if(self.use_hdf5_cache):
            self.cache_fp.close()
                    
    def find_ranges(self):
        allX=np.array([entry[0] for entry in self])
        allrange=np.zeros((allX.shape[1],2))
        allrange[:,0]=np.min(allX, axis=0) # axis=0 loops ove all the ligands
        allrange[:,1]=np.max(allX, axis=0)
        return(allrange)

    def find_normalization_factors(self):
        # read the normalization cache if it was previusly saved
        filt_spec="_no_X_filter"
        fn_no_filt=f"{self.cachefolder}/normalization_factors_{filt_spec}.dat"
        if(self.X_filter is not None):
            filt_hash=hashlib.md5(np.packbits(np.array(self.X_filter, dtype=bool)).tobytes()).hexdigest()
            filt_spec="_fiter_hash_"+filt_hash
        fn=f"{self.cachefolder}/normalization_factors_{filt_spec}.dat"
        if(os.path.exists(fn)):
            temp=np.loadtxt(fn)
            temp=temp.astype(np.float32) # defaults to float64, which translates to torch's double and is incompatible with linear layers
            self.norm_mu=temp[0,:]
            self.norm_width=temp[1,:]
            if(self.verbose):
                print(f"Reading normalization factors for a {norm_mu.shape} dataset")
        elif(os.path.exists(fn_no_filt) and self.X_filter is not None):
            temp=np.loadtxt(fn_no_filt)
            temp=temp.astype(np.float32) # defaults to float64, which translates to torch's double and is incompatible with linear layers
            self.norm_mu=temp[0,self.X_filter]
            self.norm_width=temp[1,self.X_filter]
            if(self.verbose):
                print(f"Reading normalization factors for a {norm_mu.shape} dataset")
        else:
            self.normalize_x=False # temporarily disable normalization so we can get raw values
            allX=np.array([entry[0] for entry in self])
            self.norm_mu=np.mean(allX, axis=0)
            self.norm_width=np.std(allX, axis=0)
            self.norm_width[self.norm_width<1e-7]=1.0 # if standard deviation is 0, don't scale
            self.normalize_x=True
            
            # save normalization factors
            if not os.path.exists(self.cachefolder): #make sure the folder exists
                os.makedirs(self.cachefolder, exist_ok=True)
            np.savetxt(fn, np.vstack((self.norm_mu, self.norm_width)))
            
            if(self.verbose):
                print(f"Generating normalization factors for a {allX.shape} dataset")
                
        # build in-memory cache
        self.build_internal_filtered_cache()
        _=gc.collect()


    # copies normalization factors from another dataset. Eg. for making a test set that has the same normalization as all whole dataset.
    def copy_normalization_factors(self, other):
        if(not np.array_equal(self.X_filter,other.X_filter)):
            raise(Exception("Mismatching X_filters!"))
        self.norm_mu=other.norm_mu
        self.norm_width=other.norm_width
       
    # builds the in-memory cache
    def build_internal_filtered_cache(self):
        if(self.norm_mu is None and self.normalize_x):
            raise(Exception("call build_internal_filtered_cache() only after normalization!"))
        neededMem=len(self)*(self[0][0].shape[0]+self[0][1].shape[0])*self[0][1].itemsize
        if(neededMem>self._internal_cache_maxMem):
            print(f"Building the internal_filtered_cache needs {neededMem/1024/1024} MB, more than the {self._internal_cache_maxMem/1024/1024} MB limit. SKIPPING and will read samples from HDD each time instead.")
            return()
        allX=[]
        allY=[]
        for entry in self: # loop over self only once
            allX.append(entry[0])
            allY.append(entry[1])
        allX=np.array(allX)
        allY=np.array(allY)
        self.internal_filtered_cache=(allX, allY)
        if(self.verbose):
            print(f"Creating an in-memory filtered & normalized cache of shape ({self.internal_filtered_cache[0].shape},{self.internal_filtered_cache[1].shape})")


    # normalizes features of a single ligand
    def normalize_input(self,x):
        if(self.norm_mu is None):
            self.find_normalization_factors()
        return((x-self.norm_mu)/self.norm_width)



    def __len__(self):
        return len(self.ligs)

    # main acess method for data in Dataset, eg through Dataset[index]
    def __getitem__(self, idx):
        lig = self.ligs[idx]

        # if there is no in-memory cache
        if(self.internal_filtered_cache is None):
        
            # load data from HDD or compute it
            X = self.transform(idx).astype(np.float32)
            Y = np.array([float(lig.GetProp('dG')) if lig.HasProp('dG') else np.nan]) # kcal/mol
           
            # apply feature filter
            if(not self.X_filter is None):
                X=X[self.X_filter]
            # normalize
            if(self.normalize_x):
                X=self.normalize_input(X)
                
        # otherwize read the in-memory cache
        else:
            X=self.internal_filtered_cache[0][idx]
            Y=self.internal_filtered_cache[1][idx]
        
        return X, Y
            
    # Compute the different descriptors in blocks
    def generate_DataBlock(self, lig, blockID):
        blockID=dataBlocks(blockID)
        
        if(blockID==dataBlocks.MACCS):
            Chem.GetSymmSSSR(lig)
            MACCS_txt=cDataStructs.BitVectToText(rdMolDescriptors.GetMACCSKeysFingerprint(lig))
            MACCS_arr=np.zeros(len(MACCS_txt), dtype=np.uint8)
            for j in range(len(MACCS_txt)):
                if(MACCS_txt[j]=="1"):
                    MACCS_arr[j]=1;
            return(MACCS_arr)
        
        elif(blockID==dataBlocks.MorganFP2):
            Chem.GetSymmSSSR(lig)
            Morgan_txt=cDataStructs.BitVectToText(rdMolDescriptors.GetMorganFingerprintAsBitVect(lig, 2))
            Morgan_arr=np.zeros(len(Morgan_txt), dtype=np.uint8)
            for j in range(len(Morgan_txt)):
                if(Morgan_txt[j]=="1"):
                    Morgan_arr[j]=1;
            return(Morgan_arr)
        elif(blockID==dataBlocks.MorganFP3):
            Chem.GetSymmSSSR(lig)
            Morgan_txt=cDataStructs.BitVectToText(rdMolDescriptors.GetMorganFingerprintAsBitVect(lig, 3))
            Morgan_arr=np.zeros(len(Morgan_txt), dtype=np.uint8)
            for j in range(len(Morgan_txt)):
                if(Morgan_txt[j]=="1"):
                    Morgan_arr[j]=1;
            return(Morgan_arr)
        
        elif(blockID==dataBlocks.rdkitFP):
            Chem.GetSymmSSSR(lig)
            rdkitFingerprint_txt=cDataStructs.BitVectToText(Chem.rdmolops.RDKFingerprint(lig))
            rdkitFingerprint_arr=np.zeros(len(rdkitFingerprint_txt), dtype=np.uint8)
            for j in range(len(rdkitFingerprint_txt)):
                if(rdkitFingerprint_txt[j]=="1"):
                    rdkitFingerprint_arr[j]=1;
            return(rdkitFingerprint_arr)
        
        elif(blockID==dataBlocks.minFeatFP):
           Chem.GetSymmSSSR(lig)
           minFeatFingerprint_txt=cDataStructs.BitVectToText(Generate.Gen2DFingerprint(lig, self.sigFactory))
           minFeatFingerprint_arr=np.zeros(len(minFeatFingerprint_txt), dtype=np.uint8)
           for j in range(len(minFeatFingerprint_txt)):
               if(minFeatFingerprint_txt[j]=="1"):
                   minFeatFingerprint_arr[j]=1;
           return(minFeatFingerprint_arr)
    
        elif(blockID==dataBlocks.Descriptors):
            nms=[x[0] for x in Descriptors._descList]
            calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
            des = np.array(calc.CalcDescriptors(lig))
            return(des)
        
        elif(blockID==dataBlocks.EState_FP):
            ES=Fingerprinter.FingerprintMol(lig)
            funcs=getmembers(rdkit.Chem.GraphDescriptors, isfunction)
            funcs=[f[1] for f in funcs if f[0][0]!='_' and len(getfullargspec(f[1])[0])==1]
            ES_VSA=np.array([f(lig) for f in funcs])
            ES_FP=np.concatenate((ES[0],ES[1],ES_VSA))
            return(ES_FP)
    
        elif(blockID==dataBlocks.Graph_desc):
            funcs=getmembers(rdkit.Chem.GraphDescriptors, isfunction)
            funcs=[f[1] for f in funcs if f[0][0]!='_' and len(getfullargspec(f[1])[0])==1]
            funcs+=[wiener_index]
            graph_desc=np.array([f(lig) for f in funcs])
            return(graph_desc)
    
        
        #extras
        elif(blockID==dataBlocks.MOE):
            funcs=getmembers(rdkit.Chem.MolSurf, isfunction)
            funcs=[f[1] for f in funcs if f[0][0]!='_' and len(getfullargspec(f[1])[0])==1]
            MOE=np.array([f(lig) for f in funcs])
            return(MOE)
        elif(blockID==dataBlocks.MQN):
            return(np.array(rdMolDescriptors.MQNs_(lig) ))
        elif(blockID==dataBlocks.GETAWAY):
            return(np.array(rdMolDescriptors.CalcGETAWAY(lig) ))
        elif(blockID==dataBlocks.AUTOCORR2D):
            return(np.array(rdMolDescriptors.CalcAUTOCORR2D(lig) ))
        elif(blockID==dataBlocks.AUTOCORR3D):
            return(np.array(rdMolDescriptors.CalcAUTOCORR3D(lig) ))
        elif(blockID==dataBlocks.BCUT2D):
            return(np.array(rdMolDescriptors.BCUT2D(lig) ))
        elif(blockID==dataBlocks.WHIM):
            return(np.array(rdMolDescriptors.CalcWHIM(lig) ))
        elif(blockID==dataBlocks.RDF):
            return(np.array(rdMolDescriptors.CalcRDF(lig) ))
        elif(blockID==dataBlocks.USR):
            return(np.array(rdMolDescriptors.GetUSR(lig) ))
        elif(blockID==dataBlocks.USRCUT):
            return(np.array(rdMolDescriptors.GetUSRCAT(lig) ))
        elif(blockID==dataBlocks.PEOE_VSA):
            return(np.array(rdMolDescriptors.PEOE_VSA_(lig) ))
        elif(blockID==dataBlocks.SMR_VSA):
            return(np.array(rdMolDescriptors.SMR_VSA_(lig) ))
        elif(blockID==dataBlocks.SlogP_VSA):
            return(np.array(rdMolDescriptors.SlogP_VSA_(lig) ))
        elif(blockID==dataBlocks.MORSE):
            return(np.array(rdMolDescriptors.CalcMORSE(lig) ))
            
        else:
            raise(Exception(f"Unsupported dataBlock requested: {blockID}"))
        
        
            
    # load descriptors from HDD or compute them via transform()
    def transform(self, lig_idx):
        vecs=[]
                
        # iterate through requested blocks of descriptors
        for i in self.active_flags:
            
            # first try reading from cache
            if self.use_hdf5_cache:
                lig_ID = self.ligs[lig_idx].GetProp("ID")
                node = f"{lig_ID}/{dataBlocks(i).name}"
                # is there an entry for this ligand & dataBlock?
                if(node in self.cache_fp.keys()):
                    X_block_rep = self.cache_fp[node] # then read it
                    vecs.append(X_block_rep) #and add it to the overall representation
                    continue # next dataBlock
                    
            #if not cashed, compute it
            X_block_rep = self.generate_DataBlock(self.ligs[lig_idx], i)
            vecs.append(X_block_rep) #add to overall representation
                  
            # and cache it if in append mode
            if(self.use_hdf5_cache and not (self.use_hdf5_cache=="read-only" or self.use_hdf5_cache=="r")):
                lig_ID = self.ligs[lig_idx].GetProp("ID")
                node = f"{lig_ID}/{dataBlocks(i).name}"
                f.create_dataset(node, data=X_block_rep, dtype='f')
                
        return(np.concatenate(tuple(vecs), axis=0)) # flatten into a 1D array




