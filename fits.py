
import numpy as np
import lmoments as lm
from scipy import stats
ma=np.ma
import scipy.stats.distributions as ssd
import scipy as _sp
import scipy.special as _spsp

def cdfnor(x,para):
    if _is_numeric(x)==False:
        x = _sp.array(x)

    if para[1] < 0:
        print("Invalid Parameters")
    cdfnor = 0.5+0.5*_spsp.erf((x-para[0])/para[1]*1.0/_sp.sqrt(2))
    return(cdfnor)


def cdfgam(x,para):

    CDFGAM=0
    Alpha=para[0]
    Beta=para[1]
    if Alpha <= 0 or Beta <= 0:
        print("Parameters Invalid")
        return
    if _is_numeric(x)==True:
        if x <= 0:
            print("x Parameter Invalid")
            return
    else:
        for i in x:
            if i <= 0:
                print("One X Parameter in list is Invalid")
                return
        x = _sp.array(x)
    CDFGAM = _spsp.gammainc(Alpha,x/Beta)
    return(CDFGAM)


def _comb(N,k):
    if (k > N) or (N < 0) or (k < 0):
        return 0
    val = 1
    for j in range(min(k, N-k)):
        val = (val*(N-j))//(j+1)
    return val


def _is_numeric(obj):
    try:
        obj+obj, obj-obj, obj*obj, obj**obj, obj/obj
    except ZeroDivisionError:
        return True
    except Exception:
        return False
    else:
        try:
            a = len(obj)
            if a == 1:
                return True
            return False
        except:
            return True


def _samlmusmall(x,nmom=5):
    checkx = []
    for i in x:
        if _is_numeric(i):
            checkx.append(i)
    x = checkx
    if nmom <= 0:
        return("Invalid number of Sample L-Moments")

    x = sorted(x)
    n = len(x)

    if n < nmom:
        return("Insufficient length of data for specified nmoments")
    ##Calculate first order
    ##Pretty efficient, no loops
    coefl1 = 1.0/_comb(n,1)
    suml1 = sum(x)
    l1 = coefl1*suml1

    if nmom == 1:
        ret = l1
        return(ret)

    ##Calculate Second order

    #comb terms appear elsewhere, this will decrease calc time
    #for nmom > 2, and shouldn't decrease time for nmom == 2
    #comb(x,1) = x
    #for i in range(1,n+1):
    ##        comb1.append(_comb(i-1,1))
    ##        comb2.append(_comb(n-i,1))
    #Can be simplifed to comb1 = range(0,n)

    comb1 = range(0,n)
    comb2 = range(n-1,-1,-1)

    coefl2 = 0.5 * 1.0/_comb(n,2)
    xtrans = []
    for i in range(0,n):
        coeftemp = comb1[i]-comb2[i]
        xtrans.append(coeftemp*x[i])


    l2 = coefl2 * sum(xtrans)

    if nmom  ==2:
        ret = [l1,l2]
        return(ret)

    ##Calculate Third order
    #comb terms appear elsewhere, this will decrease calc time
    #for nmom > 2, and shouldn't decrease time for nmom == 2
    #comb3 = comb(i-1,2)
    #comb4 = comb3.reverse()
    comb3 = []
    comb4 = []
    for i in range(0,n):
        combtemp = _comb(i,2)
        comb3.append(combtemp)
        comb4.insert(0,combtemp)


    coefl3 = 1.0/3 * 1.0/_comb(n,3)
    xtrans = []
    for i in range(0,n):
        coeftemp = (comb3[i]-
                    2*comb1[i]*comb2[i] +
                    comb4[i])
        xtrans.append(coeftemp*x[i])

    l3 = coefl3 *sum(xtrans) /l2

    if nmom  ==3:
        ret = [l1,l2,l3]
        return(ret)

    ##Calculate Fourth order
    #comb5 = comb(i-1,3)
    #comb6 = comb(n-i,3)
    comb5 = []
    comb6 = []
    for i in range(0,n):
        combtemp = _comb(i,3)
        comb5.append(combtemp)
        comb6.insert(0,combtemp)

    coefl4 = 1.0/4 * 1.0/_comb(n,4)
    xtrans = []
    for i in range(0,n):
        coeftemp = (comb5[i]-
                    3*comb3[i]*comb2[i] +
                    3*comb1[i]*comb4[i] -
                    comb6[i])
        xtrans.append(coeftemp*x[i])

    l4 = coefl4 *sum(xtrans)/l2
    if nmom  ==4:
        ret = [l1,l2,l3,l4]
        return(ret)

    ##Calculate Fifth order
    comb7 = []
    comb8 = []
    for i in range(0,n):
        combtemp = _comb(i,4)
        comb7.append(combtemp)
        comb8.insert(0,combtemp)

    coefl5 = 1.0/5 * 1.0/_comb(n,5)
    xtrans = []
    for i in range(0,n):
        coeftemp = (comb7[i]-
                    4*comb5[i]*comb2[i] +
                    6*comb3[i]*comb4[i] -
                    4*comb1[i]*comb6[i] +
                    comb8[i])
        xtrans.append(coeftemp*x[i])

    l5 = coefl5 *sum(xtrans)/l2

    if nmom ==5:
        ret = [l1,l2,l3,l4,l5]
        return(ret)

def samlmu(x,nmom=5):
    if nmom <= 5:
        var = _samlmusmall(x,nmom)
        return(var)
    else:
        var = _samlmularge(x,nmom)
        return(var)


def pelgam(xmom):
    A1 = -0.3080
    A2 = -0.05812
    A3 = 0.01765
    B1 = 0.7213
    B2 = -0.5947
    B3 = -2.1817
    B4 = 1.2113

    if xmom[0] <= xmom[1] or xmom[1]<= 0:
        print("L-Moments Invalid")
        return
    CV = xmom[1]/xmom[0]
    if CV >= 0.5:
        T = 1-CV
        ALPHA =T*(B1+T*B2)/(1+T*(B3+T*B4))
    else:
        T=_sp.pi*CV**2
        ALPHA=(1+A1*T)/(T*(1+T*(A2+T*A3)))

    para = [ALPHA,xmom[0]/ALPHA]
    return(para)
    




def pearsonfit(data):
    data=np.array(data)
    nozero=len(data.nonzero()[0])
    pze=1-float(nozero)/len(data)
    para=lm.pelpe3(lm.samlmu(data[data!=0],3))
    p3= np.array([lm.cdfpe3(i,para) for i in  data])

    p3=stats.norm.ppf(p3)
    return p3




def gamma_cdf(aseries):  
    """
    Returns the CDF values for aseries.
    
    -Parameters
    aseries : TimeSeries
        Annual series of data (one column per period)
    """
    # Mask the months for which no precipitations were recorded
    
    # Get the proportion of 0 precipitation for each period (MM/WW)
    nozero=np.count_nonzero(aseries) 
    zero=len(aseries)-nozero
    pzero=float(zero)/len(aseries)
    aseries_ = ma.masked_values(aseries, 0.0)
    # Mask outside the reference period
    #aseries_._mask |= condition._data
    mean_rain = aseries_.mean(axis=0)
    aleph = np.ma.log(mean_rain) - np.ma.log(aseries_).mean(axis=0) 
    alpha = (1. + ma.sqrt(1.+4./3*aleph)) / (4.*aleph)

    beta = mean_rain/alpha
    # Get the Gamma CDF (per month)
    cdf = pzero + (1.-pzero) * ssd.gamma.cdf(aseries,alpha,scale=beta)
    pn=stats.norm.ppf(cdf.astype(float))
    pn[np.where(pn<-4)[0]]=-4
    
    
    return pn



def gammafit(Data):
    data=np.array(Data)
    index1=np.where(data==0)[0]
    index2=np.where(data!=0)[0]
    pze=float(len(index1))/len(data)
    if pze>=(1/16.):
        indx=np.where(data==0)[0]
        data[indx[0]]=0.001

    para=pelgam(samlmu(data[data!=0],2))
    
    gam= np.array([cdfgam(i,para) for i in data])
#    ngam = []
#    for gami in gam:
#        if gami is not None :
#            ngam.append(gami.astype(float))
#        else:
#            ngam.append(0)
#    gam = np.array(ngam)
    gam=stats.norm.ppf(gam.astype(float))
    cdf= np.zeros(shape=len(gam))#([pze+(1-pze)*lm.cdfnor(i,[0,1]) for i in gam])
    cdf[index1]=pze
    cdf[index2]=np.array([pze+(1-pze)*cdfnor(i,[0,1]) for i in gam[index2]])
    pn=stats.norm.ppf(cdf.astype(float))
    pn[np.where(pn<-4)[0]]=-4
    del data
    return pn

def glo(data):
    """Generalized Logistic Generalized Logistic distribution function."""
    para=lm.pelglo(lm.samlmu(data,4))

    p3= np.array([lm.cdfglo(i,para) for i in  data])
    
    p3=stats.norm.ppf(p3.astype(float))
    pn=stats.norm.ppf(lm.cdfnor(p3,[0,1]))
    return pn
def gamma3(data):
    para=stats.gamma._fitstart(data)
    a,loc,scale=para
    cdf=stats.gamma.cdf(data,a,loc=loc,scale=scale)
    ppf=stats.norm.ppf(cdf)
    return ppf
    
if __name__ =="__main__":
    import pandas as pd
    #df = pd.read_csv('data/11.csv')
    df = pd.read_csv('data/clearn_train_spi.csv')
    data = []
    for i in range(len(df)):
        print (df.loc[i].t)
        data.append(np.array([df.loc[i].t,df.loc[i].rain_six_hour]))
    data = np.array(data)
    print (data)
    spi=gammafit(data[:,1])
    df['spi'] = spi
    df.to_csv('spi_train.csv',index=False)
