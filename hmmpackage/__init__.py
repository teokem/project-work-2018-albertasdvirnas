# example of how to make this package :https://python-packaging.readthedocs.io/en/latest/minimal.html

import numpy as np
import scipy.ndimage.filters as fi
import scipy.stats as st
import matplotlib.pyplot as plt
import os
from math import floor
from math import exp



def cb_transfer_matrix_method(ntSeq, NETROPSINconc = 6E-6,YOYO1conc = 4E-8,yoyo1BindingConstant = 1E10,    netropsinBindingConstant = np.array([5E5,1E8]), untrustedRegion = 1000):

    #from cb import DnaUtil
    Dna = DnaUtil()
    ntIntTemp= DnaUtil.seq_to_intarr(Dna,ntSeq);
    ntIntSeq = np.append(ntIntTemp[len(ntIntTemp)-untrustedRegion:],ntIntTemp)
    ntIntSeq = np.append(ntIntSeq,ntIntTemp[0:untrustedRegion])

    nSize = len(ntIntSeq)

    probBinding = np.zeros(len(ntIntSeq))

    ntNetrospinSeq = np.zeros(nSize);

    was = 0
    for i in range(nSize-1,-1,-1):#1 = 'A' %2 = 'C', 3 = 'G', 4 = 'T'
        if ntIntSeq[i] == 1 or ntIntSeq[i] == 4 :
            if was == 3:
                ntNetrospinSeq[i] = 1
            else:
                was = was + 1
                ntNetrospinSeq[i] = 0
        else:
            was = 0
            ntNetrospinSeq[i] = 0 

    bitsmartTranslationArr = np.uint8((32, 8, 4, 2, 1, 10, 5, 3, 12, 6, 9, 7, 11, 13, 14, 15, 16))

    choice = NETROPSINconc*netropsinBindingConstant

    leftVec = np.zeros((nSize+1,9))
    rightVec = np.zeros((nSize+1,9))
    maxEltLeft = np.zeros(nSize)
    maxEltRight = np.zeros(nSize)

    leftVec[0,:] = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    leftVec[0,:] = np.divide(leftVec[0,:],np.linalg.norm(leftVec[0,:]))
    rightVec[nSize,:] = (1,0,0,0,0,0,0,0,0)
    yoyoConst = YOYO1conc*yoyo1BindingConstant

    for i in range(0,nSize):
        leftVecPrev = leftVec[i,:]
        leftVecNext = np.array((leftVecPrev[0]+leftVecPrev[1]+leftVecPrev[5], leftVecPrev[2],  leftVecPrev[3],leftVecPrev[4]*choice[int(ntNetrospinSeq[i])],leftVecPrev[0]+leftVecPrev[1]+leftVecPrev[5], leftVecPrev[6],leftVecPrev[7],  leftVecPrev[8]*yoyoConst,leftVecPrev[0]+leftVecPrev[1]+leftVecPrev[5]))

        rightVecPrev = rightVec[nSize-i,:]
        rightVecNext = np.array((rightVecPrev[0]+rightVecPrev[4]+rightVecPrev[8],
                     rightVecPrev[0]+rightVecPrev[4]+rightVecPrev[8],
                     rightVecPrev[1], rightVecPrev[2],
                     rightVecPrev[3]*choice[int(ntNetrospinSeq[nSize-i-1])],
                     rightVecPrev[0]+rightVecPrev[4]+rightVecPrev[8],
                     rightVecPrev[5], rightVecPrev[6], rightVecPrev[7]*yoyoConst))
        maxEltLeft[i] = np.linalg.norm(leftVecNext)
        maxEltRight[nSize-i-1] = np.linalg.norm(rightVecNext)

        leftVec[i+1,:] = leftVecNext/maxEltLeft[i]

        rightVec[nSize-i-1,:] = rightVecNext/maxEltRight[nSize-i-1]


    maxVecDiv =  np.zeros(nSize)
    maxVecDiv[0] = np.divide(maxEltLeft[0],maxEltRight[0])

    for i in range(1,nSize-1):
        maxVecDiv[i] = maxVecDiv[i-1]*maxEltLeft[i]/maxEltRight[i]

    denominator = np.dot(leftVec[0,:],np.transpose(rightVec[0,:]))

    oMat = np.diag((0,0,0,0,0,1,1,1,1)) # this selects yoyo1 probability binding vec.

    probBinding[0] = np.dot(np.dot(leftVec[0,:],oMat),np.transpose(rightVec[0,:]))/denominator;

    for i in range(1,nSize-1):
        probBinding[i] = np.dot(np.dot(leftVec[i,:],oMat),rightVec[i,:])*maxVecDiv[i-1]/denominator;

    if untrustedRegion > 0:
        probBinding = probBinding[untrustedRegion:len(probBinding)-untrustedRegion];

    return probBinding


def gaussian_kernel(n, sigma ):

    if n%2 == 0:
        k = range(-n/2+1,n/2)
    else:
        k = range(-n/2,n/2)


    #print exp(-np.power(k,2)/(2*sigma**2))
    kernel = np.fft.fftshift(exp(-np.power(k,2)/(2*sigma**2)))

    kernel = kernel/sum(kernel)
    return kernel


class DnaUtil:
    def __init__(self):
        self.nt2int_dct = {'A':1, 'C':2, 'G':3, 'T':4, 'N':4}
        #self.int2nt_arr = ['A', 'C', 'G', 'T', 'N']
        #self.acgt_set = set(['A', 'C', 'G', 'T'])

    def is_acgt_only(self, seq):
        return all(nt in self.acgt_set for nt in seq)

    def nt2int(self, nt):
        #return '4' for unknown/gap
        return self.nt2int_dct.get(nt, 4)

    def seq_to_intarr(self, seq):
        import numpy as np
        arr = np.zeros((len(seq)))
        for i in range(0,len(seq)):
            arr[i] = self.nt2int(seq[i])
        return arr

    def intarr_to_seq(self, intarr):
        return '' . join([self.int2nt(i) for i in intarr])

    def int2nt(self, i):
        return self.int2nt_arr[i]

    
    

def find_txts_in_dir(directory):
    """ Find txts in a directory

    This function finds txts in a directory

    Args:
        directory (str): string of where to look for txts

    Returns:
        txts: all the txts that were found

    """

    txts = []

    for filename in os.listdir(directory):
        if filename.endswith("txt"):
            txts.append(os.path.join(directory, filename))
        else:
            continue
    return txts


def open_txt(name):
    f = open(name, 'r')
    x = f.readlines()
    f.close()
    y = [u.split() for u in x]

    seq1 = np.array(y[1]).astype(float)
    seq1 = np.reshape(seq1, (-1, len(seq1)))

    return seq1


def sim_txt(name):
    f = open(name, 'r')
    x = f.readlines()
    f.close()
    y = [u.split() for u in x]

    seq1 = np.array(y[0]).astype(float)
    seq1 = np.reshape(seq1, (-1, len(seq1)))
    seq2 = np.array(y[1]).astype(float)
    seq2 = np.reshape(seq2, (-1, len(seq2)))
    return seq1,seq2


def gkern(kernlen, nsig):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)


def seq_to_timeseries(seq, barLen, kerLen ):
    ker = np.roll(gkern(barLen, kerLen), int(round((barLen + 1) / 2)))
    #plt.plot(ker)

    conVec = np.conj(np.fft.fft(ker, barLen))

    return np.real(np.fft.ifft(np.fft.fft(seq) * conVec)) #/ (barLen - 1)

           
def compute_pcc_one(vec1,vec2):
    vec1 =(vec1-np.mean(vec1))/np.std(vec1, ddof=1)
    vec2 =(vec2-np.mean(vec2))/np.std(vec2, ddof=1)
    return sum(vec1*vec2)/len(vec1)


# method 1 for computing CC values
def compute_pcc(vec1,vec2,len1):
    vec1 =(vec1-np.mean(vec1))/np.std(vec1, ddof=1)

    from scipy.signal import fftconvolve

    longer = np.hstack([vec2, vec2[0:len1 - 1]])

    ccForward = fftconvolve(longer, vec1[::-1], mode='valid') / (len1 - 1)

    ccBackward = fftconvolve(longer, vec1, mode='valid') / (len1 - 1)

    movMean = fftconvolve(longer, np.ones(vec1.shape), mode='valid') / (len1)

    movStd = fftconvolve(longer ** 2, np.ones(vec1.shape), mode='valid')

    stdForward = np.sqrt((movStd - len1 * movMean ** 2) / (len1 - 1))

    ccForward = ccForward / stdForward
    ccBackward = ccBackward / stdForward
    return ccForward,ccBackward



def discretize_barcode(z_barcode):
    # we make barcodes to be in range [-30,30]
    discrete_barcode = np.round(z_barcode/np.max([abs(np.max(z_barcode)),abs(np.min(z_barcode))])*30)


    return discrete_barcode


def build_profile(discrete_barcode):

    length = len(discrete_barcode[0])
    discretePoints = 61

    # SET UP/VALUE INITIALIZATION, COULD BE TRAINED?

    # probability of moving from E to J
    # p_EJ = 0.5

    # probability of moving from Mi to E
    # TODO: maybe 1/100 kb
    p_ME = 1./(length)
    # probability of going from MN to M1
    p_MM = 1-p_ME

    # probability of moving from E to C, E to J
    p_EC = 1./3
    p_EJ = 1-p_EC

    #  probability of moving from J to J,J to B
    p_JJ = 1./20 # 1/20
    p_JB = 1- p_JJ

    #  probability of moving from N to N, N to B
    p_NN = 1./20
    p_NB = 1-p_NN

    # TODO: move to one parameter with N. Maybe 1/100
    #  probability of moving from C to C
    p_CC = 1./20
    p_CT = 1-p_CC


    p1_profile = {}
    p1_profile['len'] = length
    Em = {}
    Em['Mf'] = np.zeros((length+1,discretePoints))
    Em['Mr'] = np.zeros((length+1,discretePoints))
    p1_profile['Em'] = Em

    Tr = {}
    Tr['Mf'] = np.zeros((length+1,2))
    Tr['Mr'] = np.zeros((length+1,2))
    Tr['B'] =  np.zeros((2,length))

    p1_profile['Tr'] = Tr

    points = np.linspace(-(discretePoints-1)/2,(discretePoints-1)/2,discretePoints)
    distfun = lambda px : st.norm.pdf(points,px,2)

    v = np.zeros((len(discrete_barcode[0]),discretePoints))
    for i in range(0,len(discrete_barcode[0])):
        pdv = distfun(discrete_barcode[0][i])
        v[i,:] = np.log10(pdv/np.sum(pdv))

    Em['Mf'][1:,:] = v
    Em['Mr'][1:,:] = np.flipud(v)

    Tr['Mf'][:,0] = np.log10(p_MM)
    Tr['Mf'][:, 1] = np.log10(p_ME)

    # redundant
    Tr['Mr'][:,0] = np.log10(p_MM)
    Tr['Mr'][:, 1] = np.log10(p_ME)

    Tr['B'][:,:] = np.log10(1./(2*length))
    Tr['E'] = np.log10([p_EJ,p_EC])
    Tr['N']= np.log10([p_NB, p_NN])
    Tr['J'] = np.log10([p_JB,p_JJ])
    Tr['C'] = np.log10([p_CT ,p_CC ])

    # import matplotlib.pyplot as plt
    # f, axarr = plt.subplots(1, 1)
    # axarr.plot(points,v[1])

    return p1_profile


def msv_algorithm_nonlinear(p1_profile, sequence):

    # readability
    L = len(sequence[0])
    M = p1_profile['len']
    tr = p1_profile['Tr']
    em = p1_profile['Em']

    # initialization of MSV

    #  initialize M forward matrices and M backward matrices to -inf
    #  2*M+1 - N, 2*M+2 - B, 2*M+3 - E, 2*M+4 - J, 2*M+5- C
    N = 2*M+1
    B = 2*M+2
    E = 2*M+3
    J = 2*M+4
    C = 2*M+5
    W = np.zeros((L+1,2*M+1+5))
    W[:] = -np.inf

    # initialize N
    W[0,N] = 0

    #  initialize B, i.e. transition probability of going from N to B
    W[0,B] = tr['N'][0]


    for i in range(1,L+1):

        # Mf
        W[i, 1] =  em['Mf'][1, int(30+sequence[0][i-1])]+np.maximum(W[i-1,M]+tr['Mf'][M,0], W[i-1,B]+tr['B'][0,0])
        # Mb
        W[i, M+1] = em['Mr'][1, int(30+sequence[0][i-1])]+np.maximum(W[i-1,2*M]+tr['Mr'][M,0], W[i-1,B]+tr['B'][0,0])

        # E
        W[i,E] = np.maximum(np.maximum(W[i,1]+tr['Mf'][M,1], W[i,M+1]+tr['Mr'][M,1]), W[i,E])

        for k in range(2,M+1):
            W[i, k] = em['Mf'][k, int(30+sequence[0][i-1])]+np.maximum(W[i-1,k-1]+tr['Mf'][M,0], W[i-1,B]+tr['B'][0,0])
            W[i, k+M] = em['Mr'][k, int(30+sequence[0][i-1])]+np.maximum(W[i-1,M+k-1]+tr['Mr'][M,0], W[i-1,B]+tr['B'][0,0])
            W[i, E] = np.maximum(np.maximum(W[i,k]+tr['Mf'][M,1], W[i,k+M]+tr['Mr'][M,1]),  W[i, E])
        #  from which state it\s biggest probability to get to E?
        # W[i,E] = np.max(W[i, 1:M])

        # Prob of going from N state to N state again
        W[i, N] = W[i-1,N]+ tr['N'][1]


        # Prob of being in state J
        W[i, J] = np.maximum(W[i-1,J]+tr['J'][1], W[i,E]+tr['E'][0])

        #  Prob of being in state C
        W[i, C] = np.maximum(W[i-1,C]+tr['C'][1], W[i,E]+tr['E'][1])

        # Prob of being in state B
        W[i,B] = np.maximum(W[i,N]+tr['N'][0], W[i,J]+tr['J'][0])
        # MrMx = np.zeros((L+1,M+1))
    # MrMx[:] = -np.inf

    # termination

    score = W[L,C]+tr['C'][0]

    return W, score




def msv_traceback(p1_profile, sequence, W, score, M, L):
    # readability
    L = len(sequence[0])
    M = p1_profile['len']
    tr = p1_profile['Tr']
    em = p1_profile['Em']

    N = 2*M+1
    B = 2*M+2
    E = 2*M+3
    J = 2*M+4
    C = 2*M+5

    prevSt = C


    # traceback vector..
    def fn(W, i, prevSt, M,tr):
        N = 2 * M + 1
        B = 2 * M + 2
        E = 2 * M + 3
        J = 2 * M + 4
        C = 2 * M + 5
        return {
            N : [0][np.argwhere([W[i-1,N]+ tr['N'][1]==  W[i, prevSt]])[0][0]],

        }[prevSt]
    # traceback vector..
    def fc(W, i, prevSt, M,tr):
        N = 2 * M + 1
        B = 2 * M + 2
        E = 2 * M + 3
        J = 2 * M + 4
        C = 2 * M + 5
        return {
            C: [E,C][np.argwhere([W[i , E] + tr['E'][1] == W[i, prevSt], W[i - 1, C] + tr['C'][1] == W[i, prevSt]])[0][0]],
        }[prevSt]

    # traceback vector..
    def fj(W, i, prevSt, M,tr):
        N = 2 * M + 1
        B = 2 * M + 2
        E = 2 * M + 3
        J = 2 * M + 4
        C = 2 * M + 5
        return {
            J : [E,J][np.argwhere([W[i,E]+tr['E'][0]== W[i, prevSt], W[i-1,J]+tr['J'][1] == W[i, prevSt]])[0][0]],
        }[prevSt]


    #  initial point L
    i = L

    # first we go from C state to E state
    prevSt = C

    vitTraceback = np.array([[i,C]])

    while fc(W, i, prevSt, M, tr) == C:
        i = i-1
        vitTraceback = np.concatenate((vitTraceback,[[i,C]]))
    # i=i+1

    # we then move to state E
    prevSt = E
    vitTraceback = np.concatenate((vitTraceback, np.array([[i, prevSt]])))

    # now, while previous state is not N
    while prevSt != N:

        # if previous state was E
        if prevSt == E:
            try:
                prevSt = np.argwhere([W[i, k] + tr['Mf'][M, 1] == W[i, prevSt] for k in range(0, M + 1)])[0][0]
            except:
                prevSt = M+np.argwhere([W[i, k+M] + tr['Mf'][M, 1] == W[i, prevSt] for k in range(0, M + 1)])[0][0]

            vitTraceback = np.concatenate((vitTraceback, np.array([[i, prevSt]])))

        while prevSt != B:
            try:
                while W[i - 1, prevSt - 1] + tr['Mf'][M, 0] + em['Mf'][prevSt, int(30 + sequence[0][i - 1])] == W[i, prevSt]:
                    i = i - 1
                    prevSt = prevSt - 1
                    vitTraceback = np.concatenate((vitTraceback, np.array([[i, prevSt]])))
            except:
                while W[i - 1, prevSt - 1] + tr['Mr'][M, 0] + em['Mr'][prevSt-M, int(30 + sequence[0][i - 1])] == W[i, prevSt]:
                    i = i - 1
                    prevSt = prevSt - 1
                    vitTraceback = np.concatenate((vitTraceback, np.array([[i, prevSt]])))

            # i = i+1
            try:
                prevSt = [B][np.argwhere(W[i - 1, B] + tr['B'][0, 0] + em['Mf'][prevSt, int(30 + sequence[0][i - 1])] == W[i, prevSt])[0][0]]
            except:
                try:
                    prevSt =[B][np.argwhere(W[i - 1, B] + tr['B'][0, 0] + em['Mr'][prevSt-M, int(30 + sequence[0][i - 1])] == W[i, prevSt])[0][0]]
                except:
                    try:
                        prevSt = [M][np.argwhere(W[i - 1, M] + tr['Mf'][M, 0] + em['Mf'][prevSt, int(30 + sequence[0][i - 1])] == W[i, prevSt])[0][0]]
                    except:
                        prevSt = [2*M][np.argwhere(W[i - 1, 2*M] + tr['Mf'][M, 0] + em['Mr'][prevSt-M, int(30 + sequence[0][i - 1])] == W[i, prevSt])[0][0]]
                    i = i-1
            # just fix index inside loop
            # i=i-1
            vitTraceback = np.concatenate((vitTraceback, np.array([[i, prevSt]])))

        # now we are in B state. from this we either go to N state, or to J state
        try:
            prevSt = [N][np.argwhere(W[i, N] + tr['N'][0] ==  W[i, prevSt])[0][0]]
        except:
            prevSt = [J][np.argwhere( W[i, J] + tr['J'][0] ==  W[i, prevSt])[0][0]]
        vitTraceback = np.concatenate((vitTraceback, np.array([[i, prevSt]])))

        if prevSt == J:
            while fj(W, i, prevSt, M, tr) == J:
                i = i - 1
                vitTraceback = np.concatenate((vitTraceback, np.array([[i, prevSt]])))

            prevSt = E
            vitTraceback = np.concatenate((vitTraceback, np.array([[i, prevSt]])))

    while fn(W, i, prevSt, M, tr) == N:
        i = i - 1
        vitTraceback = np.concatenate((vitTraceback, np.array([[i, prevSt]])))

    vitTraceback = np.concatenate((vitTraceback, np.array([[i-1, prevSt]])))
    return vitTraceback



def compute_example(bar, bar2):
    z_barcode = st.zscore(bar, axis=1, ddof=0)
    z_barcode2 = st.zscore(bar2, axis=1, ddof=0)

    discrete_barcode = discretize_barcode(z_barcode)
    p1_profile = build_profile(discrete_barcode)
    discrete_barcode2 = discretize_barcode(z_barcode2)

    W, score = msv_algorithm_nonlinear(p1_profile, discrete_barcode2)

    M = len(discrete_barcode[0])
    L = len(discrete_barcode2[0])
    # sequence = discrete_barcode2
    vitTraceback = msv_traceback(p1_profile, discrete_barcode2, W, score, M, L)
#     plot_traceback(vitTraceback, W, M, L)
    return vitTraceback, W, score




def parse_vtrace(vitTraceback, M):
    resTable = []
    vec1 = []
    vec2 = []
    for i in range(0, len(vitTraceback)):
        if vitTraceback[i, 1] == 2 * M + 2:
            if vec1 != []:
                resTable.append([vec1, vec2])
                vec1 = []
                vec2 = []

        if vitTraceback[i, 1] < 2 * M + 1:
            # curVec.append(vitTraceback[i, :])
            vec1.append(vitTraceback[i, 0])
            if vitTraceback[i, 1] > M:
                vec2.append(2*M+1-vitTraceback[i, 1])
            else:
                vec2.append(vitTraceback[i, 1])

    return resTable





def vit_to_table(seq1,seq2):
    # # Example 1: circular shift
    vitTraceback, W, score = compute_example(seq1, seq2)

    M = len(seq1[0])
    resTable = parse_vtrace(vitTraceback, M)
    segScores = []
    for i in range(0,len(resTable)):
        segScores.append(W[resTable[i][0][-1],2*M+2]-W[resTable[i][0][0],2*M+3])


    if resTable[-1][1][-1] == resTable[0][1][0] - M + 1:
        resTable[0][0].append(resTable[-1][0])
        resTable[0][1].append(resTable[-1][1])
        resTable[-1][0] = []
        resTable[-1][1] = []
        segScores[0] = segScores[0]+segScores[-1]
        segScores = segScores[:-1]

    return resTable,segScores
