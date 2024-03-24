# __author__ = 'koushik'

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.naive_bayes import MultinomialNB

import dateutils as du
import pickle as pk
import quandl as quandl
import scipy as sp
import functools as ft

#import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#import matplotlib.pyplot as plt
#from sklearn import KFold
#import sys


class const():
    @staticmethod
    def loanIndxMax():
        return 100000
    @staticmethod
    def maxTerm():
        return 61
    @staticmethod
    def pi():
        return 3.14159
    @staticmethod
    def ageKnots():
        return 5
    @staticmethod
    def maxTrees():
        return 100
    @staticmethod
    def maxDepth():
        return None
    @staticmethod
    def maxLeaf():
        return 1
    @staticmethod
    def chartFields():
        return ['term','MOB','orig_fico','vintage','month','loan_amnt','coupon','purpose','emp_length','region','y_mod','y_act']
    @staticmethod
    def charFields():
        return ['id','loan_amnt','sub_grade','emp_length','verification_status','purpose','addr_state',
                'dti','initial_list_status','last_credit_pull_d','collections_12_mths_ex_med',
                'mths_since_last_major_derog','policy_code','application_type','total_rev_hi_lim','acc_open_past_24mths',
                'bc_open_to_buy','bc_util','chargeoff_within_12_mths','mo_sin_old_il_acct','mo_sin_old_rev_tl_op',
                'mo_sin_rcnt_rev_tl_op','mo_sin_rcnt_tl','mort_acc','mths_since_recent_bc','mths_since_recent_bc_dlq','mths_since_recent_inq',
                'mths_since_recent_revol_delinq','num_accts_ever_120_pd','num_actv_bc_tl','num_actv_rev_tl','num_bc_sats','num_bc_tl',
                'num_il_tl','num_op_rev_tl','num_rev_accts','num_rev_tl_bal_gt_0','num_sats','num_tl_120dpd_2m','num_tl_30dpd',
                'num_tl_90g_dpd_24m','num_tl_op_past_12m','pct_tl_nvr_dlq','percent_bc_gt_75','pub_rec_bankruptcies','tax_liens',
                'tot_hi_cred_lim','total_bal_ex_mort','total_bc_limit','total_il_high_credit_limit']
    @staticmethod
    def pmtFields():
        return ['LOAN_ID','PBAL_BEG_PERIOD','PRNCP_PAID','INT_PAID','FEE_PAID','DUE_AMT','RECEIVED_AMT','PERIOD_END_LSTAT',
                'MONTH','PBAL_END_PERIOD','MOB','COAMT','MONTHLYCONTRACTAMT','InterestRate','IssuedDate','HomeOwnership','MonthlyIncome',
                'EarliestCREDITLine','OpenCREDITLines','TotalCREDITLines','RevolvingCREDITBalance','RevolvingLineUtilization',
                'Inquiries6M','DQ2yrs', 'MonthsSinceDQ','PublicRec','MonthsSinceLastRec','EmploymentLength','term','grade',
                'APPL_FICO_BAND', 'Last_FICO_BAND','PCO_RECOVERY','PCO_COLLECTION_FEE']
    @staticmethod
    def regressorFields():
        return ['MOB','HomeOwnership','MonthlyIncome','OpenCREDITLines','TotalCREDITLines','RevolvingCREDITBalance',
                'RevolvingLineUtilization','Inquiries6M','DQ2yrs','MonthsSinceDQ','PublicRec','MonthsSinceLastRec','EmploymentLength',
                'term','inception_year','loan_amnt','emp_length','verification_status','purpose','dti','initial_list_status',
                'collections_12_mths_ex_med','mths_since_last_major_derog','policy_code','application_type','total_rev_hi_lim',
                'acc_open_past_24mths','bc_open_to_buy','bc_util','chargeoff_within_12_mths','mo_sin_old_il_acct',
                'mo_sin_old_rev_tl_op','mo_sin_rcnt_rev_tl_op','mo_sin_rcnt_tl','mort_acc','mths_since_recent_bc','mths_since_recent_bc_dlq','mths_since_recent_inq',
                'mths_since_recent_revol_delinq','num_accts_ever_120_pd','num_actv_bc_tl','num_actv_rev_tl','num_bc_sats',
                'num_bc_tl','num_il_tl','num_op_rev_tl','num_rev_accts','num_rev_tl_bal_gt_0','num_sats','num_tl_120dpd_2m',
                'num_tl_30dpd','num_tl_90g_dpd_24m','num_tl_op_past_12m','pct_tl_nvr_dlq','percent_bc_gt_75','pub_rec_bankruptcies',
                'tax_liens','tot_hi_cred_lim','total_bal_ex_mort','total_bc_limit','total_il_high_credit_limit','loan_month','coupon',
                'never_late_indicator',	'region','earliest_year','last_pull_year']
    @staticmethod
    def startStates():
        return np.array(['C','D3','D6','D6+'])
    @staticmethod
    def endStates():
        return np.array(['C','D3','D6','D6+','D','P'])
    @staticmethod
    def allTransitions():
        return np.array(['CtoC', 'CtoD3', 'CtoD6', 'CtoD6+', 'CtoD', 'CtoP',
               'D3toC', 'D3toD3', 'D3toD6', 'D3toD6+', 'D3toD', 'D3toP',
               'D6toC', 'D6toD3', 'D6toD6', 'D6toD6+', 'D6toD', 'D6toP',
               'D6+toC', 'D6+toD3', 'D6+toD6', 'D6+toD6+', 'D6+toD', 'D6+toP'])
    @staticmethod
    def modeledTransitions():
        return np.array(['CtoC', 'CtoD3', 'CtoP',
               'D3toC', 'D3toD3', 'D3toD6', 'D3toD', 'D3toP',
               'D6toC', 'D6toD3', 'D6toD6', 'D6toD6+', 'D6toD',
               'D6+toC', 'D6+toD3', 'D6+toD6', 'D6+toD6+', 'D6+toD'])
    @staticmethod
    def curveHeaders():
        return ['NewOrig','BegBal','PrinPaid','PrinPrepaid','Defs','EndBal','IntPaid','CPR','CDR']
    @staticmethod
    def priceHeaders():
        return ['LoanID','MOB','Term','Status','Amount','Coupon','ParSpread','ParYield','ModYield','AvgYield','ActYield','ModPrice','AvgPrice','ActPrice']
    @staticmethod
    def cumulHeaders():
        return ['CPR_mod','CPR_act','CDR_mod','CDR_act']
    @staticmethod
    def quandlKey():
        return 'oJQ8bM4ArnvxyLUSNjgf'

def read_pmt_data(pmtStr):
# Purpose: Reads in pmts and char files

    dtPmts = pd.read_csv(pmtStr, header=0, index_col=False)
    dtPmts = dtPmts.sort_values(by=['LOAN_ID','MOB'])
    dtPmts = dtPmts.reset_index(drop=True)
    dtPmts = dtPmts[const.pmtFields()]

    # Pick random 1000 loans from 2012
    dtPmts['IssuedDate'] = pd.to_datetime(dtPmts['IssuedDate'], format='%b%Y')
    #dtPmts = dtPmts.iloc[np.where(dtPmts['IssuedDate'].dt.year == 2011)]
    pmtIndx = np.where(np.logical_and(dtPmts['IssuedDate'].dt.year == 2012,dtPmts['term'] == 36))
    loanIndx = np.where(np.in1d(dtPmts['LOAN_ID'],dtPmts['LOAN_ID'].iloc[pmtIndx].sample(10000)))
    dtPmts = dtPmts.iloc[loanIndx]

    print('\nReading in LC payments data...')

    return dtPmts

def read_dict(path):
# Purpose: Reads in dict from the path.  File contains lookups to append to dataframe.

    dtD = pd.read_csv('Inputs/'+path, header=0, index_col=False)
    print('Read in file from path: %s ...' %path)
    return dtD

def read_rates():

    dtR = quandl.get('FED/SVENY', rows=1).ix[:,:5]
    fRates = sp.interpolate.interp1d([0,12,24,36,48,60],np.insert(dtR.values,0,0),kind='cubic')

    return fRates(range(1,const.maxTerm()))

def pickle_save(dtPmts,saveStr):
# Purpose: Stores pandas dataframe as a pickled .p file

    pk.dump(dtPmts, open('Pickled/'+saveStr, 'wb'))
    print('Pickled file ...')

def pickle_load(loadStr):
# Purpose: Loads object from a pickled .p file

    dtOut = pd.DataFrame()

    for l in loadStr:
        print('Loaded pickled file from: %s...' %l)
        dtOut = pd.concat([dtOut,pk.load(open('Pickled/'+l, 'rb'))],axis=0)

    return dtOut

def clean_pmt_data(dtPmts):
# Purpose: Cleans data from LC pmts file so inter- and intra-temporal pmts word

    # Fix: Round numerical columns
    roundCols = ['LOAN_ID', 'PBAL_BEG_PERIOD', 'PRNCP_PAID', 'INT_PAID', 'FEE_PAID', 'DUE_AMT',
                 'RECEIVED_AMT', 'PBAL_END_PERIOD','COAMT']

    for r in roundCols:
        dtPmts[r] = ((100*dtPmts[r]).round(0))/100
    print('\nFix: Rounded numerical columns ...')

    # Fix: Convert month dates to datetimes
    dtPmts['MONTH'] = pd.to_datetime(dtPmts['MONTH'],format='%b%Y')
    dtPmts['EarliestCREDITLine'] = pd.to_datetime(dtPmts['EarliestCREDITLine'],format='%b%Y')
    dtPmts['IssuedDate'] = pd.to_datetime(dtPmts['IssuedDate'], format='%b%Y')
    dtPmts['inception_year'] = dtPmts['IssuedDate'].dt.year
    dtPmts = dtPmts.iloc[np.where(np.logical_and(dtPmts['inception_year'] > 2011,dtPmts['inception_year'] < 2014))]
    print('\nFix: Converted dates to datetimes ...')

    # Fix: Convert PCO's from NaN to 0
    dtPmts['PCO_RECOVERY'] = dtPmts['PCO_RECOVERY'].fillna(0)
    dtPmts['PCO_COLLECTION_FEE'] = dtPmts['PCO_COLLECTION_FEE'].fillna(0)
    dtPmts['grade'] = dtPmts['grade'].fillna('NR')
    dtPmts['MONTH'] = dtPmts['MONTH'].fillna('3000-13-32')
    dtPmts['RevolvingLineUtilization'] = dtPmts['RevolvingLineUtilization'].fillna(-1)
    #dtPmts['PBAL_BEG_PERIOD'] = dtPmts['PBAL_BEG_PERIOD'].fillna(0)
    #dtPmts['PBAL_END_PERIOD'] = dtPmts['PBAL_END_PERIOD'].fillna(0)
    #dtPmts['PRNCP_PAID'] = dtPmts['PRNCP_PAID'].fillna(0)
    #dtPmts['COAMT'] = dtPmts['COAMT'].fillna(0)
    print('\nFix: Removed NaNs from PCO, grade, month, balances, payments fields ...')

    # Fix: Average out FICO's
    dtPmts['appl_FICO'] = 0.5 * (dtPmts['APPL_FICO_BAND'].apply(lambda x: float(x[:3])) +
                             dtPmts['APPL_FICO_BAND'].apply(lambda x: float(x[-3:])))

    dtPmts['Last_FICO_BAND'].iloc[np.where(dtPmts['Last_FICO_BAND'] == 'LOW-499')] = '450-499'
    dtPmts['Last_FICO_BAND'].iloc[np.where(dtPmts['Last_FICO_BAND'] == '845-HIGH')] = '845-850'
    dtPmts['Last_FICO_BAND'].iloc[np.where(dtPmts['Last_FICO_BAND'] == 'MISSING')] = '190-210'
    dtPmts['last_FICO'] = 0.5 * (dtPmts['Last_FICO_BAND'].apply(lambda x: float(x[:3])) +
                             dtPmts['Last_FICO_BAND'].apply(lambda x: float(x[-3:])))

    dtPmts = dtPmts.drop(['APPL_FICO_BAND','Last_FICO_BAND'],axis=1)
    print('\nFix: Created avg fico column ...')

    # Fix: Make End Bal = 0, when charge off occurs
    coIndx = np.where(dtPmts['COAMT'] > 0)
    dtPmts['PBAL_END_PERIOD'].iloc[coIndx] = 0
    print('\nFix: Zeroed out end bal when loans charged off...')

    # Fix: Find non-consecutive pmt months and insert a row
    pmtIndx = np.where(np.logical_and((12 * dtPmts['MONTH'].iloc[1:].dt.year + dtPmts['MONTH'].iloc[1:].dt.month).values -
                                      (12 * dtPmts['MONTH'].iloc[:-1].dt.year + dtPmts['MONTH'].iloc[:-1].dt.month).values == 2,
                                      dtPmts['LOAN_ID'].iloc[1:].values == dtPmts['LOAN_ID'].iloc[:-1].values))
    dtNew = dtPmts.iloc[pmtIndx].copy(deep=True)
    dtNew[['PRNCP_PAID','INT_PAID','FEE_PAID','RECEIVED_AMT','DUE_AMT','COAMT']] = 0
    dtNew['PBAL_BEG_PERIOD'] = dtNew['PBAL_END_PERIOD']
    dtNew['MONTH'] = dtNew['MONTH'].apply(lambda x: x + du.relativedelta(months=1))
    dtPmts = pd.concat([dtPmts,dtNew],axis=0)

    dtPmts = dtPmts.sort_values(['LOAN_ID','MONTH'],ascending=[True,True])
    dtPmts = dtPmts.reset_index(drop=True)
    print('\nFix: Inserted zero row for non-consecutive payments ...')

    # Fix: Find multiple payments in one month and delete a row
    pmtIndx = np.where(np.logical_and(dtPmts['MONTH'].iloc[1:].values == dtPmts['MONTH'].iloc[:-1].values,
                                        dtPmts['LOAN_ID'].iloc[1:].values == dtPmts['LOAN_ID'].iloc[:-1].values))[0]

    dtPmts[['PRNCP_PAID','INT_PAID','FEE_PAID','RECEIVED_AMT','DUE_AMT','COAMT']].iloc[pmtIndx] += dtPmts[['PRNCP_PAID','INT_PAID','FEE_PAID','RECEIVED_AMT','DUE_AMT','COAMT']].iloc[pmtIndx+1]
    dtPmts['PBAL_END_PERIOD'].iloc[pmtIndx] = dtPmts['PBAL_END_PERIOD'].iloc[pmtIndx+1].values
    dtPmts = dtPmts.drop(dtPmts.index[pmtIndx+1])

    dtPmts = dtPmts.sort_values(['LOAN_ID','MONTH'],ascending=[True,True])
    dtPmts = dtPmts.reset_index(drop=True)
    print('\nFix: Consolidated payments in same month for same loan ...')

    # Fix: Ensure a zero payment exists when difference between MOB 1 and the issue date

    # Fix: Insert zero payment for most recent month and last month all active loans

    lastMonth = np.sort(dtPmts['MONTH'].unique())[-1]
    prevMonth = np.sort(dtPmts['MONTH'].unique())[-2]
    prevpmtIndx = np.where(np.logical_and(dtPmts['MONTH'] == prevMonth,dtPmts['PBAL_END_PERIOD'] > 1))
    lastpmtIndx = np.where(dtPmts['MONTH'] == lastMonth)

    loanIndx = dtPmts['LOAN_ID'].iloc[prevpmtIndx].unique()
    loanIndx = loanIndx[np.where(np.in1d(loanIndx,dtPmts['LOAN_ID'].iloc[lastpmtIndx]) == False)]
    prevpmtIndx = prevpmtIndx[0][np.in1d(dtPmts['LOAN_ID'].iloc[prevpmtIndx],loanIndx)]

    dtNew = dtPmts.iloc[prevpmtIndx].copy(deep=True)
    dtNew[['PRNCP_PAID','INT_PAID','FEE_PAID','RECEIVED_AMT','DUE_AMT','COAMT']] = 0
    dtNew['PBAL_BEG_PERIOD'] = dtNew['PBAL_END_PERIOD']
    dtNew['MONTH'] = lastMonth
    dtPmts = pd.concat([dtPmts,dtNew],axis=0)

    dtPmts = dtPmts.sort_values(['LOAN_ID','MONTH'],ascending=[True,True])
    dtPmts = dtPmts.reset_index(drop=True)
    print('\nFix: Added zero rows for loans missing last reported months data ...')

    # Fix: Check to ensure AMT PAID = INT PAID + FEE PAID, if not modify

    pmtIndx = np.where(dtPmts['RECEIVED_AMT'] - dtPmts['PRNCP_PAID'] - dtPmts['INT_PAID'] - dtPmts['FEE_PAID'] != 0)
    dtPmts['RECEIVED_AMT'].iloc[pmtIndx] = dtPmts['PRNCP_PAID'].iloc[pmtIndx] + dtPmts['INT_PAID'].iloc[pmtIndx] + dtPmts['FEE_PAID'].iloc[pmtIndx]
    print('\nFix: Added up received amount ...')

    # Fix: Make sure charge offs only occur once
    pmtIndx = np.where(np.logical_and(dtPmts['PERIOD_END_LSTAT'].iloc[:-1] == 'Charged Off',
                                     dtPmts['PERIOD_END_LSTAT'].iloc[1:] == 'Charged Off'))
    dtPmts = dtPmts.drop(dtPmts.index[pmtIndx[0]+1])

    dtPmts = dtPmts.sort_values(['LOAN_ID','MONTH'],ascending=[True,True])
    dtPmts = dtPmts.reset_index(drop=True)

    print('\nFix: Dropped multiple charged off statuses ...')

    # Fix: Find where next opening balance is greater than previous opening balance for same loan
    pmtIndx = np.where(np.logical_and(dtPmts['PBAL_BEG_PERIOD'].iloc[1:].values > dtPmts['PBAL_BEG_PERIOD'].iloc[:-1].values,
                                      dtPmts['LOAN_ID'].iloc[1:].values == dtPmts['LOAN_ID'].iloc[:-1].values))
    loanIndx = np.unique(dtPmts['LOAN_ID'].iloc[pmtIndx])
    for l in loanIndx:
    #l = loanIndx[14]
        rIndx = np.where(dtPmts['LOAN_ID'] == l)
        for r in range(0,len(rIndx[0])-1):
            if (dtPmts['PBAL_BEG_PERIOD'].iloc[rIndx[0][-1]-r] > dtPmts['PBAL_BEG_PERIOD'].iloc[rIndx[0][-1]-r-1]):
                dtPmts['PBAL_BEG_PERIOD'].iloc[rIndx[0][-1]-r-1] = dtPmts['PBAL_BEG_PERIOD'].iloc[rIndx[0][-1]-r]

    print('\nFix: Made sure opening bal is decreasing over time ...')

    # Fix: Find loans where next opening balance does not match previous closing balance

    pmtIndx = np.where(np.logical_and(dtPmts['PBAL_BEG_PERIOD'].iloc[1:].values != dtPmts['PBAL_END_PERIOD'].iloc[:-1].values,
                                    dtPmts['LOAN_ID'].iloc[1:].values == dtPmts['LOAN_ID'].iloc[:-1].values))
    dtPmts['PBAL_END_PERIOD'].iloc[pmtIndx[0]] = dtPmts['PBAL_BEG_PERIOD'].iloc[pmtIndx[0]+1].values

    print('\nFix: Made sure consecutive opening and closing bals match ...')

    # Fix: Check to ensure PBAL BEG - PRIN PAID - COAMT = PBAL END, if not modify PRIN PAID

    pmtIndx = np.where(dtPmts['PBAL_BEG_PERIOD']-dtPmts['PRNCP_PAID']-dtPmts['COAMT']-dtPmts['PBAL_END_PERIOD'] != 0)
    dtPmts['PRNCP_PAID'].iloc[pmtIndx] = dtPmts['PBAL_BEG_PERIOD'].iloc[pmtIndx] - dtPmts['COAMT'].iloc[pmtIndx] - dtPmts['PBAL_END_PERIOD'].iloc[pmtIndx]
    print('\nFix: Reconciled balances and payments ...')

    # Fix: Renumber MOB's

    startIndx = np.where(dtPmts['LOAN_ID'].iloc[:-1].values != dtPmts['LOAN_ID'].iloc[1:].values)[0] + 1
    startIndx = np.insert(startIndx,0,0)
    subIndx = pd.DataFrame(data=np.arange(0,dtPmts.shape[0]))
    subIndx.iloc[np.where(np.in1d(subIndx,startIndx)==False)[0]] = np.nan
    subIndx = subIndx.ffill()
    dtPmts['MOB'] = np.subtract(np.arange(0,dtPmts.shape[0]),subIndx.values.transpose()).transpose() + 1

    print('\nFix: Added MOBs ...')

    # Fix: Add unscheduled and principals paid
    dtPmts['unscheduled_principal'] = np.maximum(dtPmts['RECEIVED_AMT'] - dtPmts['DUE_AMT'], 0)
    dtPmts['scheduled_principal'] = dtPmts['PRNCP_PAID'] - dtPmts['unscheduled_principal']

    print('\nFix: Added unscheduled and scheduled principal paid ...')

    # Fix: Add new origination amount
    dtPmts['new_orig'] = 0
    newOrigIndx = np.where(dtPmts['MOB'] == 1)
    dtPmts['new_orig'].iloc[newOrigIndx] = dtPmts['PBAL_BEG_PERIOD'].iloc[newOrigIndx].values
    print('\nFix: Added new originations ...')

    # replace when add check for last month in clean_data
    #dropIndx = np.where(dtPmts['MONTH'] == '2016-12-01')
    #dtPmts = dtPmts.drop(dtPmts.index[dropIndx])
    #dtPmts = dtPmts.reset_index(drop=True)

    return dtPmts

def check_pmt_data(dtPmts):
# Purpose: Checks to ensure inter-, intra-temporal pmts, and MOBs are aligned

    # Check 1: Make sure walk works
    walkIndx = np.where(np.logical_and(np.abs(np.subtract(dtPmts['PBAL_BEG_PERIOD'].iloc[1:].values,dtPmts['PBAL_END_PERIOD'].iloc[:-1].values)) > 1,
                     dtPmts['LOAN_ID'].iloc[1:].values == dtPmts['LOAN_ID'].iloc[:-1].values))

    if len(walkIndx[0]) == 0:
        print('\nWalk check works...')
    else:
        print('\nProblem with walk check, but fixed it...')
        dropIndx = np.where(np.in1d(dtPmts['LOAN_ID'],np.unique(dtPmts['LOAN_ID'].iloc[walkIndx])))
        dtPmts = dtPmts.drop(dtPmts.index[dropIndx])

    # Check 2: Make sure months are consecutive
    monthIndx = np.where(np.logical_and.reduce((dtPmts['LOAN_ID'].iloc[1:].values == dtPmts['LOAN_ID'].iloc[:-1].values,
                    pd.DatetimeIndex(dtPmts['MONTH'].iloc[:-1]) + pd.offsets.MonthBegin(1) != pd.DatetimeIndex(dtPmts['MONTH'].iloc[1:]),
                    dtPmts['PERIOD_END_LSTAT'].iloc[1:] != 'Charged Off')))

    if len(monthIndx[0]) == 0:
        print('\nMonth check works...')
    else:
        print('\nProblem with month check, but fixed it ...')
        dropIndx = np.where(np.in1d(dtPmts['LOAN_ID'], np.unique(dtPmts['LOAN_ID'].iloc[monthIndx])))
        dtPmts = dtPmts.drop(dtPmts.index[dropIndx])

    # Check 3: Make sure MOBs are consecutive
    mobIndx = np.where(np.logical_and.reduce((dtPmts['LOAN_ID'].iloc[1:].values == dtPmts['LOAN_ID'].iloc[:-1].values,
                    np.subtract(dtPmts['MOB'].iloc[:-1].values,dtPmts['MOB'].iloc[1:].values) > 1)))

    if len(mobIndx[0]) == 0:
        print('\nMOB check works...')
    else:
        print('\nProblem with MOB check, but fixed it...')
        dropIndx = np.where(np.in1d(dtPmts['LOAN_ID'], np.unique(dtPmts['LOAN_ID'].iloc[mobIndx])))
        dtPmts = dtPmts.drop(dtPmts.index[dropIndx])

    return dtPmts

def insert_status(dtPmts):
# Purpose: Inserts status number as column

    loan_status = np.zeros(shape=(dtPmts.shape[0],1))

    loan_status[:,0] = np.round((dtPmts['DUE_AMT'].values - dtPmts['RECEIVED_AMT'].values)/dtPmts['MONTHLYCONTRACTAMT'].values,
                           decimals=0) # calculate number of pmts missed (not based on time)
    loan_status[np.where(loan_status > 3),0] = 3 # max amount missed is 3 payment cycles

    loan_status[np.where(np.logical_or(dtPmts['PERIOD_END_LSTAT'] == 'Current', # current: status = current or rec amt > due amt
                                       dtPmts['RECEIVED_AMT'] >= dtPmts['DUE_AMT'])),0] = 0

    loan_status[np.where(dtPmts['COAMT'] > 0),0] = 4 # defaulted or charged off

    #loan_status[np.where(np.logical_or(dtPmts['PERIOD_END_LSTAT'] == 'Default', # defaulted or charged off
    #                                   dtPmts['PERIOD_END_LSTAT'] == 'Charged Off')),0] = 4

    loan_status[np.where(np.logical_and(dtPmts['PERIOD_END_LSTAT'] == 'Fully Paid', # prepaid: fully paid before term is up
                                       dtPmts['MOB'] <= dtPmts['term'])),0] = 5

    loan_status[np.where(np.logical_or(dtPmts['PERIOD_END_LSTAT'] == 'In Grace Period',  # other
                                       dtPmts['PERIOD_END_LSTAT'] == 'Issued')), 0] = -1

    #loan_status = pd.DataFrame(data=loan_status,index=dtPmts.index,columns=['loan_status'])
    dtPmts['loan_status'] = loan_status

    print('\nFinished adding status column...')

    return dtPmts

def insert_delq_indicator(dtPmts):
# Purpose: To be deprecated

    firstDelqStatus = np.zeros(shape=(dtPmts.shape[0], 1))
    secondDelqStatus = np.zeros(shape=(dtPmts.shape[0], 1))

    firstDelqStatus[np.in1d(dtPmts['LOAN_ID'],dtPmts['LOAN_ID'].iloc[np.where(dtPmts['loan_status'] == 1)].unique()),0] = 1
    secondDelqStatus[np.in1d(dtPmts['LOAN_ID'],dtPmts['LOAN_ID'].iloc[np.where(dtPmts['loan_status'] == 2)].unique()),0] = 1

    dtPmts['first_delq'] = firstDelqStatus
    dtPmts['second_delq'] = secondDelqStatus

    return dtPmts

def read_stats_data(dtPmts):
# Purpose: Reads in stats or characteristics file

    dtChars = pd.DataFrame()
    charFiles = ['3a.csv', '3b.csv', '3c.csv', '3d.csv', '2016Q1.csv', '2016Q2.csv','2016Q3.csv']
    for c in charFiles:
        print('Reading in file: %s...' % c)
        dtChars = pd.concat([dtChars, pd.read_csv('LC Data/' + c, header=1, index_col=False)])

    #dtChars.to_csv('Loan Stats.csv')
    dtChars = dtChars[const.charFields()]
    #dropIndx = np.where(pd.isnull(dtChars['purpose']))
    #dtChars = dtChars.drop(dtChars.index[dropIndx])

    dtPmts = pd.merge(dtPmts, dtChars, how='inner', left_on='LOAN_ID', right_on='id', copy=False)
    dtPmts = dtPmts.drop('id',axis=1)

    print('Merging in stats purpose ...')

    return dtPmts

def clean_stats_data(dtPmts):
# Purpose: Shifts certain column formats

    #dtPmts['RevolvingLineUtilization'] = dtPmts['RevolvingLineUtilization'].str.rstrip('%').astype('float64').fillna(-100) / 100
    dtPmts['loan_month'] = dtPmts['MONTH'].dt.month
    print('Added loan_month ... ')

    dtPmts['coupon'] = dtPmts['InterestRate'] * 100
    print('Added coupon ... ')

    lateIndx = np.where(np.logical_and(dtPmts['loan_status'] > 0,dtPmts['loan_status'] < 5))[0]
    endIndx = np.append(np.where(dtPmts['MOB'] == 1)[0][1:]-1,dtPmts.shape[0]-1)
    endIndx = endIndx[np.searchsorted(endIndx,lateIndx,'left')]

    lateRows = []
    for i in range(0,len(lateIndx)):
        lateRows.append(np.arange(lateIndx[i],endIndx[i]+1))
    lateRows = np.unique(np.concatenate(lateRows).ravel())

    dtPmts['never_late_indicator'] = 1
    dtPmts['never_late_indicator'].iloc[lateRows] = 0
    print('Added never_late_indicator ... ')

    # Add region
    dtRegion = read_dict('states.csv')
    dtPmts = pd.merge(dtPmts, dtRegion, how='left', left_on='addr_state', right_on='state', copy=False)
    #dtPmts = dtPmts.drop(['state'], axis=1)
    print('Added field: region ...')

    # Add years for date fields
    dtPmts['earliest_year'] = dtPmts['EarliestCREDITLine'].dt.year
    dtPmts['last_credit_pull_d'] = pd.to_datetime(dtPmts['last_credit_pull_d'], format='%b-%Y')
    dtPmts['last_pull_year'] = dtPmts['last_credit_pull_d'].dt.year
    print('Added field: earliest_year, last_pull_year ...')

    dtPmts = dtPmts.drop(['InterestRate','addr_state','state','EarliestCREDITLine','last_credit_pull_d'], axis=1)

    print('\nProcessed data for LC payments and borrower data...')

    return dtPmts


def generate_regressors(dt):
# Converts input data to classification columns and I tried to drop the columns containing 20% of the non-missing values

    cols = []
    dtOut = np.zeros(shape=(dt.shape[0],0))
    sparseCount = dt.count(axis=0)/dt.shape[0] # percentage of non-nans in columns

    topTolerance = .9
    bottomTolerance = .2
    freqTolerance = .1
    sparseTolerance = .2
    binTolerance = 10
    bins = range(0,int((1-sparseTolerance)*100),binTolerance)

    for i in range(0,dt.shape[1]):


        #print(i)
        freqTable = dt.iloc[:,i].value_counts()/dt.iloc[:,i].count()

        if (freqTable.shape[0] > 0):

            if (freqTable.iloc[0] >= topTolerance) & (sparseCount[i] > sparseTolerance) & (freqTable.shape[0] > 1):
            # Case 0: If most frequent value is > 90% and column is not sparse, then (is frequent value, is missing)
                dtTemp = np.zeros(shape=(dt.shape[0],1))
                cols.append(dt.columns.values[i] + "_is" + str(dt.iloc[:, i].value_counts().index[0]))
                dtTemp[np.where(dt.iloc[:, i] == dt.iloc[:, i].value_counts().index[0]),0] = 1
                dtOut = np.concatenate((dtOut, dtTemp), axis=1)
                print('Case 0: For col %s, freq val > 90 & col is NOT sparse & more than 1 val' %dt.columns.values[i])

            elif (bottomTolerance <= freqTable.iloc[0] < topTolerance) & (sparseCount[i] > sparseTolerance) & (freqTable.shape[0] > 1):
            # Case 1: If most frequent value is in between tolerances and column is not sparse, then (is multiple vals above threshold, is missing)

                uniqCols = freqTable.index[(freqTable > freqTolerance).nonzero()[0]]
                if uniqCols.shape[0] == freqTable.shape[0]:
                # if uniqCols contains all values then drop last one
                    uniqCols = uniqCols[0:uniqCols.shape[0]-1]

                dtTemp = np.zeros(shape=(dt.shape[0],uniqCols.shape[0]))
                # pulls values for columns

                for j,u in enumerate(uniqCols): # parse possible values
                    cols.append(dt.columns.values[i] + "_is" + str(u))
                    dtTemp[np.where(dt.iloc[:,i] == u),j] = 1
                    #print '\ni = %d, j = %d' % (i, j)

                dtOut = np.concatenate((dtOut,dtTemp),axis=1)
                print('Case 1: For col %s, freq val 90 - 20 & col is NOT sparse & more than 1 val' %dt.columns.values[i])

            elif (sparseCount[i] > sparseTolerance) & (freqTable.shape[0] > 1):
            # Case 2: Else numbers are numerical
                dtTemp = np.zeros(shape=(dt.shape[0],len(bins)))
                # divide column into deciles and exclude last 20%
                for j,u in enumerate(bins):
                    #print(j,u)
                    dtTemp[np.where(np.logical_and(dt.iloc[:,i]>=np.percentile(dt.iloc[:,i],u),dt.iloc[:,i]<np.percentile(dt.iloc[:,i],u+binTolerance))),j] = 1
                    cols.append(dt.columns.values[i] + "_bin" + str(u))

                dtOut = np.concatenate((dtOut,dtTemp),axis=1)
                print('Case 2: For col %s, values are numerical' %dt.columns.values[i])

            else:
            # Case 3: Values are either one value or numbers are too sparse - turns out only one column falls into this case, which was dropped
                print('Case 3: For col %s values are either one type or too sparse' %dt.columns.values[i])

            if (dt.iloc[:,i].isnull().sum() > 0):
            # Case 3: Add a column if some values are missing or if there is 1 value and blanks
                dtTemp = np.zeros(shape=(dt.shape[0],1))
                dtTemp[np.where(dt.iloc[:,i].isnull()),0] = 1
                cols.append(dt.columns.values[i] + "_isnull")
                dtOut = np.concatenate((dtOut,dtTemp),axis=1)
                #print('Adding col for missing vals, for col %s' %dt.columns.values[i])
        else:
            print('Col %s is empty' %dt.columns.values[i])

    print('Finished processing input data...')
    #temp = pd.DataFrame(data=dtOut, index=dt.index, columns=cols)
    #temp.to_csv('temp.csv')
    return pd.DataFrame(data=dtOut,index=dt.index,columns=cols)

def generate_responses(dtPmts):
# Purpose: Generates dataseries of transitions and transition counts
# 0: current, 1: 1m late, 2: 2m late, 3: 2+m late, 4: default, 5: prepaid entirely

    matCount = np.zeros(shape=(len(const.startStates()),len(const.endStates())),dtype='float')
    dtResp = np.zeros(shape=(dtPmts.shape[0],len(const.allTransitions())),dtype='int')

    for i in range(0,len(const.startStates())):
        for j in range(0,len(const.endStates())):
            respIndx = np.where(np.logical_and.reduce((np.array(dtPmts['loan_status'].iloc[1:] == j),
                                                   np.array(dtPmts['loan_status'].iloc[:-1] == i),
                                                   np.array(dtPmts['LOAN_ID'].iloc[1:].values == dtPmts['LOAN_ID'].iloc[:-1].values))))
            dtResp[respIndx,len(const.startStates())*i+j] = 1
            matCount[i,j] = len(respIndx[0])
            print('Finished calculating responses for transition state: %s ...' %const.allTransitions()[len(const.endStates())*i+j])

    return pd.DataFrame(data=dtResp,columns=const.allTransitions(),index=dtPmts.index), pd.DataFrame(data=matCount,columns=const.endStates(),index=const.startStates())

def test_train_split(dtPmts,term,testSize):
# Purpose: Splits population into train and test sets for a particular term

    #termIndx = np.where(np.in1d(dtPmts['term'],term))
    #origIndx = dtPmts['LOAN_ID'].iloc[termIndx]
    #origIndx = np.where(np.logical_and(np.in1d(dtPmts['LOAN_ID'],origIndx),dtPmts['MOB']==1))
    X_train, X_test, y_train, y_test = train_test_split(dtPmts, np.ones(shape=(dtPmts.shape[0])), test_size=testSize, random_state=0)

    #return np.where(np.in1d(dtPmts['LOAN_ID'],dtPmts['LOAN_ID'].iloc[X_train.index])),np.where(np.in1d(dtPmts['LOAN_ID'],dtPmts['LOAN_ID'].iloc[X_test.index]))
    return X_train.index, X_test.index

def class_weights(startState):

    return {'C':{0:.05,1:.7,2:.25},
                 'D3':'balanced',}[startState]

def analyze_model(model,X_test,y_test,startState):
# Purpose: Provides summary metrics on model performance

#model = modelList[i]
#X_test = dtRegr.iloc[testingIndx]
#y_test = dtResp.iloc[testingIndx]
#startState = 'C'

    # print model accuracy
    print('\nModel overall accuracy is: %.2f for startState: %s ...' \
          % (model.score(X_test,y_test),startState))

    # print OOB error rate
    if (startState == 'GG'):
        print('Model OOB error rate is: %.2f for startState: %s ...' \
             % (1-model.oob_score_,startState))

    # print precision and recall scores
    y_mod = model.predict(X_test)
    print('Model precision and recall scores are: %.4f and %.4f for Pr(%s) ...' \
          % (metrics.precision_score(y_test, y_mod, average='macro'),
             metrics.recall_score(y_test, y_mod, average='macro'), startState))

    # print model vs actual score for each transition
    for c in np.sort(y_test.unique()):
        print('Model : Actual are %.4f : %.4f for Pr(%s) ...' \
              % (float((y_mod == c).sum())/float(y_test.shape[0]),float((y_test == c).sum())/float(y_test.shape[0]),const.allTransitions()[int(c)]))

def generate_transition_models(dtRegr,dtResp,trainIndx,testIndx):
# Purpose: Create random forest models for each transition state

    modelList = [LogisticRegression for i in const.startStates()]

    for startState in const.startStates():

startState = 'D3'

        startStateCols = const.allTransitions()[np.where([colName[0:colName.index('t')] == startState
                                                    for colName in const.allTransitions()])[0]]
        modStateCols = const.modeledTransitions()[np.where([colName[0:colName.index('t')] == startState
                                                  for colName in const.modeledTransitions()])[0]]

        trainingIndx = testingIndx = []
        #testingIndx = []
        startIndx = np.where(dtResp[startStateCols].sum(axis=1) == 1)
        dtSample = pd.Series(data=np.matmul(dtResp[modStateCols],np.arange(0,modStateCols.shape[0],1)),
                                         index=dtResp.index)

        while (dtSample.loc[trainingIndx].value_counts().shape[0] < modStateCols.shape[0]) or \
                (dtSample.loc[testingIndx].value_counts().shape[0] < modStateCols.shape[0]):
            trainingIndx = np.intersect1d(dtSample.sample(const.loanIndxMax(), replace=True).index,
                                          dtSample.iloc[startIndx].index)
            testingIndx = np.intersect1d(dtSample.sample(const.loanIndxMax(), replace=True).index,
                                          dtSample.iloc[startIndx].index)
            #trainingIndx,testingIndx,_,_ = train_test_split(dtSample.loc[sampleIndx],np.ones(shape=(dtSample.loc[sampleIndx].shape[0])),test_size=.33)
            #trainingIndx = trainingIndx.index
            #testingIndx = testingIndx.index

        model = LogisticRegression(penalty='l1',multi_class='multinomial',solver='saga',max_iter=1000,class_weight=class_weights(startState))
        model = RandomForestClassifier(n_estimators=const.maxTrees(), max_depth=const.maxDepth(), max_features='auto',bootstrap=True,
                                        oob_score=True, random_state=531, min_samples_leaf=const.maxLeaf(),class_weight=class_weights(startState))
        #model = GradientBoostingClassifier()
        #model = MultinomialNB(class_prior=[.96,.01,.03])
        modelList[np.where(const.startStates()==startState)[0][0]] = model.fit(dtRegr.loc[trainingIndx],np.ravel(dtSample.loc[trainingIndx]))


kk = modelList[np.where(const.startStates()==startState)[0][0]].predict(dtRegr.loc[testingIndx])
gg = pd.DataFrame(data=np.c_[kk,dtSample.loc[testingIndx]],columns=['pred','act'])
gg['pred'].value_counts()
gg['act'].value_counts()


    return modelList


def increment_regressor(X,mob):
# Purpose: Change the input values as we model down the time axis for a loan

    dtModRegr = X.copy(deep=True)
    lastCol = 0

    for i in range(0,10):
        if  (mob >= (i+1)*const.ageKnots()) and (mob < (i+2)*const.ageKnots()):
            dtModRegr.iloc[i] = max(mob - (i+1)*const.ageKnots(),0)
        else:
            dtModRegr.iloc[i] = 0
        lastCol = lastCol+1
        #print(lastCol)

    lastCol = np.where(np.in1d(X.index.values,'age_k0.term36'))[0][0]
    # indicators for term * age linear spline
    for j in range(0,10):
        dtModRegr.iloc[lastCol] = np.multiply(np.absolute(dtModRegr.iloc[11]-1),dtModRegr.iloc[j])
        lastCol = lastCol+1
        #print(lastCol)

    # indicators for term * age linear spline for term=60
    for j in range(0,10):
        dtModRegr.iloc[lastCol] = np.multiply(dtModRegr.iloc[11],dtModRegr.iloc[j])
        lastCol = lastCol+1
        #print(lastCol)

    # loan month
    # determines start month
    lastCol = np.where(np.in1d(X.index.values,'m2'))[0][0]
    if np.where(dtModRegr.iloc[lastCol:lastCol+11]==1)[0].sum()==0:
        month = 1
    else:
        month = np.where(dtModRegr.iloc[lastCol:lastCol+11]==1)[0][0]+2

    # determines new month based on mob
    dtModRegr.iloc[lastCol:lastCol+11] = 0
    if mob + month > 12:
        month = (mob + month) % 12
    else:
        month = mob + month

    # specifies column
    if month > 1:
        dtModRegr.iloc[lastCol+month-2] = 1

    return dtModRegr

def calculate_transition_matrix(X,modelList):
# Purpose: Create the switching probability matrix for a loan payment

    #X = dtRegrOrig.iloc[0]


    # returns cumulative transition prob matrix
    trMat = np.zeros(shape=(len(const.startStates())+2,len(const.endStates())),dtype=float)
    for i in range(0,len(const.startStates())):
        #print(i)
        trMat[i,:] = modelList[i].predict_proba(X.reshape(1,-1))[0]

    trMat[len(const.endStates())-2,len(const.endStates())-2] = 1 # for defs, end in defs
    trMat[len(const.endStates())-1,len(const.endStates())-1] = 1 # for prepays, move to def

    # trMat[0,:] = [.978,.985,.985,.985,.985,1]
    # trMat[1,:] = [.176,.324,.976,.977,.992,1]
    # trMat[2,:] = [.027,.055,.142,.945,.997,1]
    # trMat[3,:] = [.01,.013,.024,.56,998,1]

    #print trMat

    return trMat.cumsum(axis=1)

def simulate_cashflows(coupon,term,loanAmt,xRegr,modelList,stState,nSims):
# Purpose: Simulates CF's for a loan

    np.set_printoptions(precision=4)
    balPaid = np.zeros(shape=(nSims, term), dtype=float)  # cumulative and includes defaults
    prinPrepaid = np.zeros(shape=(nSims, term), dtype=float)  # not cumulative
    prinDef = np.zeros(shape=(nSims, term), dtype=float)  # not cumulative
    intPaid = np.zeros(shape=(nSims, term), dtype=float)
    cfMat = np.zeros(shape=(nSims,term),dtype=float)
    dtState = np.zeros(shape=(nSims,term),dtype=int)

    # Don't do simulation for loans that have paid off or defaulted already
    if (stState < 4):

        if (stState == 0):
            cfMat[:,0] = np.pmt(coupon / 1200, term, loanAmt, 0,0)  # at t=0 all loans are assumed to be current
            balPaid[:,0] = np.ppmt(coupon / 1200, 1, term, loanAmt, 0, 0)
            intPaid[:,0] = np.ipmt(coupon / 1200, 1, term, loanAmt, 0, 0)
            dtState[:,0] = stState

        for j in range(1,term): # j is the time index
            #print(j)
            trMat = calculate_transition_matrix(increment_regressor(xRegr,j),modelList)
            balPaid[:,j] = balPaid[:,j-1]
            stateVar = np.random.uniform(0,1,nSims).reshape(1,-1).transpose() # stateVar are the random uniforms
            dtState[:,j] = np.argmax(np.less(np.tile(stateVar,(1,trMat.shape[1])),trMat[dtState[:,j-1],:]),axis=1)

            if (j == term - 1):
                # terminal condition for current loans or loans prepaid in last time period
                # if current then will be prepaid, if delinquent then will be defaulted
                dtState[np.where(dtState[:,j] == 0),j] = 5
                dtState[np.where(np.logical_and(dtState[:,j] > 0,dtState[:,j] < 4)),j] = 4

            # For prepays, cashFlow = intPaid + prinPaid + rest of loan amt (prinPrepaid)
            prIndx = np.where(np.logical_and(dtState[:,j]==5,dtState[:,j-1]!=5)) # prepay index
            cfMat[prIndx,j] = np.ipmt(coupon/1200,j+1,term,loanAmt,0,0) - loanAmt - balPaid[prIndx,j]
            prinPrepaid[prIndx,j] = -loanAmt - np.ppmt(coupon/1200,j+1,term,loanAmt,0,0) - balPaid[prIndx,j]
            intPaid[prIndx,j] = np.ipmt(coupon/1200,j+1,term,loanAmt,0,0)
            balPaid[prIndx,j] = -loanAmt

            # For defaults, cf = intPaid = prinPrepaid = 0, prinPaid = full loan amount, prinDef = rest of loan amount
            defIndx = np.where(np.logical_and(dtState[:,j]==4,dtState[:,j-1]!=4))
            prinDef[defIndx,j] = -loanAmt - balPaid[defIndx,j]
            balPaid[defIndx, j] = -loanAmt

            # For unequal, states need to add up payments, includes current pmts
            # cf = delta * pmts, prinPaid = delta * ppmt, intPaid = delta * ipmt, prinDef = 0
            lsIndx = np.where(np.logical_and(dtState[:,j]<=dtState[:,j-1],dtState[:,j]<4)) # states are unequal and not prepay or default
            cfMat[lsIndx,j] = np.multiply(np.pmt(coupon/1200,term,loanAmt,0,0),dtState[lsIndx,j-1]-dtState[lsIndx,j]+1)
            balPaid[lsIndx,j] += [np.ppmt(coupon/1200,np.arange(j-dtState[lsIndx[0][l],j-1]+dtState[lsIndx[0][l],j],j+1)+1,term,loanAmt,0,0)[0] \
                                    for l in range(0,lsIndx[0].shape[0])]
            intPaid[lsIndx, j] = [np.ipmt(coupon / 1200, np.arange(j-dtState[lsIndx[0][l],j-1]+dtState[lsIndx[0][l],j],j+1)+1,term,loanAmt,0,0)[0] \
                                    for l in range(0, lsIndx[0].shape[0])]

    return cfMat,prinPrepaid,prinDef,intPaid,balPaid,dtState


def calculate_par_yield(dtSummary,dtOrig,dtRegrOrig,modelList,nSims):
# Purpose: Calculates par credit spread from modelled CF's for a loan (with a flat yield curve)

#dtOrig = dt.iloc[pmtIndx]
#dtRegrOrig = dtR.iloc[pmtIndx]

    loans = np.sort(dtSummary['LoanID'].iloc[np.where(dtSummary['MOB'] == 1)])
    for loan in loans:   # i is the loan index

        i = np.where(dtSummary['LoanID'] == loan)[0][0]
        r = np.where(np.logical_and(dtOrig['LOAN_ID']==loan,dtOrig['MOB']==1))[0][0]

        cfMat = simulate_cashflows(dtSummary['Coupon'].iloc[i],dtSummary['Term'].iloc[i].astype(int),
                                   dtSummary['Amount'].iloc[i],dtRegrOrig.iloc[r],modelList,0,nSims)[0]
        cfMat = np.c_[-dtSummary['Amount'].iloc[i]*np.ones(shape=(cfMat.shape[0])),-1*cfMat]
        summIndx = np.where(dtSummary['LoanID'] == loan)
        dtSummary['ParYield'].iloc[summIndx] = 100*((1+np.irr(cfMat.mean(axis=0)))**12-1)
        print('For loan: %.0f coupon : par yield is %.2f : %.2f ...' % \
              (loan,dtSummary['Coupon'].iloc[summIndx[0][0]],dtSummary['ParYield'].iloc[summIndx[0][0]]))

    return dtSummary

def calculate_actual_yield(dtSummary,dtC,initSwitch):
# Purpose: Calculates the actual yield of a loan
#dtC = dt.iloc[pmtIndx]

    for i in range(0,dtSummary.shape[0]):
        if (initSwitch == True):
            loanIndx = np.where(dtC['LOAN_ID'] == dtSummary['LoanID'].iloc[i])
            loanField = 'loan_amnt'
        else:
            loanIndx = np.where(np.logical_and(dtC['LOAN_ID'] == dtSummary['LoanID'].iloc[i],
                                               dtC['MOB'] > dtSummary['MOB'].iloc[i]))
            loanField = 'PBAL_BEG_PERIOD'

        if (len(loanIndx[0]) > 1):
            cfMat = np.zeros(shape=(len(loanIndx[0]) + 1), dtype=float)
            cfMat[0] = -dtC[loanField].iloc[loanIndx[0][0]]
            cfMat[1:cfMat.shape[0]] = dtC['RECEIVED_AMT'].iloc[loanIndx]
            dtSummary['ActYield'].iloc[i] = 100*((1+np.irr(cfMat))**12-1)
            print('For loan: %.0f @ MOB: %d the actual yield is: %.2f  ...'
                %(dtSummary['LoanID'].iloc[i],dtSummary['MOB'].iloc[i],dtSummary['ActYield'].iloc[i]))

    return dtSummary

def calculate_actual_curves(dtC):
# Purpose: Produces actual CDR and CPR curves for a set of loans

#dtC = dt.iloc[np.where(dt['LOAN_ID'] == dtSummary['LoanID'].iloc[i])]

    dtAct = pd.DataFrame(data=np.zeros(shape=(dtC['term'].iloc[0].astype(int),len(const.curveHeaders())),dtype=float),
                      index=np.arange(1,dtC['term'].iloc[0]+1),columns=const.curveHeaders())
    sumFields = ['PBAL_BEG_PERIOD','scheduled_principal','unscheduled_principal','COAMT','PBAL_END_PERIOD','INT_PAID']
    actFields = ['BegBal','PrinPaid','PrinPrepaid','Defs','EndBal','IntPaid']
    actIndx = np.arange(dtC['MOB'].min().astype(int),dtC['MOB'].max().astype(int)+1)

    dtSumm = dtC.groupby(['MOB'])[sumFields].sum()
    for f in range(0,len(sumFields)):
        dtAct[actFields[f]].loc[actIndx] = dtSumm[sumFields[f]].loc[actIndx].values

    dtAct['NewOrig'].ix[dtC['MOB'].min().astype(int)] = dtAct['BegBal'].ix[dtC['MOB'].min().astype(int)]

    return dtAct

def calculate_performance(dtSummary,dtLast,dtRegrLast,modelList,nSims,initSwitch):
# Purpose: Calculates loan price from par yield

#dtLast = dt.iloc[pmtIndx]
#dtRegrLast = dtR.iloc[pmtIndx]

    #np.set_printoptions(precision=8)
    dtMod = pd.DataFrame(data=np.zeros(shape=(dtLast['term'].max(),len(const.curveHeaders())),dtype=float),
                          index=np.arange(1,dtLast['term'].max()+1),columns=const.curveHeaders())
    dtAct = pd.DataFrame(data=np.zeros(shape=(dtLast['term'].max(),len(const.curveHeaders())),dtype=float),
                          index=np.arange(1,dtLast['term'].max()+1),columns=const.curveHeaders())
    dtCumul = pd.DataFrame(data=np.zeros(shape=(dtLast['term'].max(),len(const.cumulHeaders()))),
                           index=np.arange(1,dtLast['term'].max()+1),columns=const.cumulHeaders())

    for i in range(0,dtSummary.shape[0]):   # i is the loan index

        # if initSwitch then predict all forward CF's, else just ones from current mob + 1 to end
        if (initSwitch == True): # assumes dtLast are only origIndx
            firstMOB = 0
            loanField = 'loan_amnt'
        else:
            firstMOB = dtSummary['MOB'].iloc[i].astype(int)
            loanField = 'PBAL_BEG_PERIOD'

        loanIndx = np.where(np.logical_and(dtLast['LOAN_ID'] == dtSummary['LoanID'].iloc[i],
                                           dtLast['MOB'] == firstMOB + 1))[0]

        if (dtSummary['Term'].iloc[i] - dtSummary['MOB'].iloc[i] > 1) and (len(loanIndx) > 0):
            mobIndx = np.arange(firstMOB+1,dtSummary['Term'].iloc[i].astype(int)+1)
            loanAmt = dtLast[loanField].iloc[loanIndx[0]]
            dtAct = dtAct.add(calculate_actual_curves(dtLast.iloc[np.where(np.logical_and(dtLast['LOAN_ID'] == dtSummary['LoanID'].iloc[i],
                                                                                          dtLast['MOB'] > firstMOB))]))

            cfMat,cfPrepaid,cfDefs,cfInt,cfPaid,dtState = simulate_cashflows(dtSummary['Coupon'].iloc[i],len(mobIndx),
                 loanAmt,dtRegrLast.iloc[loanIndx[0]],modelList,dtSummary['Status'].iloc[i],nSims)

            dtSummary['ParSpread'].iloc[i] = dtSummary['ParYield'].iloc[i] - read_rates()[dtSummary['Term'].iloc[i].astype(int) - 1]
            dtSummary['ModYield'].iloc[i] = 100 * ((np.irr(np.insert(-cfMat.mean(axis=0),0,-loanAmt)) + 1)**12 - 1)
            dtSummary['AvgYield'].iloc[i] = 100 * np.mean([((np.irr(np.insert(-cfMat[row,:],0,-loanAmt)) + 1)**12 - 1) for row in range(0,nSims)])

            parYieldCurve = (.01 * (dtSummary['ParSpread'].iloc[i] + read_rates()[0:len(mobIndx)]) + 1) ** (1.0 / 12)
            discFactor = np.power(parYieldCurve,-np.arange(1,len(mobIndx)+1))

            dtSummary['ModPrice'].iloc[i] = 100 * np.dot(-cfMat.mean(axis=0),discFactor) / loanAmt
            dtSummary['AvgPrice'].iloc[i] = 100 * np.multiply(-cfMat,np.tile(discFactor,(cfMat.shape[0],1))).sum(axis=1).mean() / loanAmt
            dtSummary['ActPrice'].iloc[i] = 100 # temporary until get actual pricing

            dtMod['NewOrig'].loc[firstMOB+1] += loanAmt
            dtMod['PrinPrepaid'].loc[mobIndx] += -cfPrepaid.mean(axis=0) # prepays made positive
            dtMod['Defs'].loc[mobIndx] += -cfDefs.mean(axis=0) # defaults made positive
            dtMod['IntPaid'].loc[mobIndx] += -cfInt.mean(axis=0) # intPaid
            dtMod['PrinPaid'].loc[mobIndx] += np.c_[-cfPaid[:,0],np.diff(-cfPaid,n=1,axis=1)].mean(axis=0) + cfPrepaid.mean(axis=0) + cfDefs.mean(axis=0) # sch prinPaid
            dtMod['EndBal'].loc[mobIndx] += loanAmt + cfPaid.mean(axis=0) # endBal

            if dtMod['EndBal'].iloc[-1] > 1000:
                print('Problem with loan: %d MOB: %d....' %(dtSummary['LoanID'].iloc[i],dtSummary['MOB'].iloc[i]))
                arrs = [cfMat, cfPrepaid, cfDefs, cfInt, cfPaid, dtState]
                names = ['cfMat', 'cfPrepaid', 'cfDefs', 'cfInt', 'cfPaid', 'dtState']
                for k, a in enumerate(arrs):
                    pd.DataFrame(a, columns=mobIndx).to_csv(names[k] + '.csv')

            print('Mod price: avg price is %.2f : %.2f for loan: %d @mob: %d' %(dtSummary['ModPrice'].iloc[i],dtSummary['AvgPrice'].iloc[i],
                                                                                dtSummary['LoanID'].iloc[i],dtSummary['MOB'].iloc[i]))

    dtMod['BegBal'] = dtMod[['EndBal','PrinPaid','PrinPrepaid','Defs']].sum(axis=1)  # next begBal is last endBal
    dtMod['CPR'] = (dtMod['PrinPrepaid']/dtMod['BegBal']).fillna(0)
    dtMod['CDR'] = (dtMod['Defs']/dtMod['BegBal']).fillna(0)
    dtAct['CPR'] = (dtAct['PrinPrepaid']/dtAct['BegBal']).fillna(0)
    dtAct['CDR'] = (dtAct['Defs']/dtAct['BegBal']).fillna(0)
    dtComp = pd.merge(dtMod,dtAct,how='outer',left_index=True,right_index=True,suffixes=('_m','_a'))
    dtCumul[['CPR_mod','CDR_mod']] = 100 * dtMod[['PrinPrepaid','Defs']].cumsum(axis=0) / dtMod['NewOrig'].sum(axis=0)
    dtCumul[['CPR_act','CDR_act']] = 100 * dtAct[['PrinPrepaid','Defs']].cumsum(axis=0) / dtAct['NewOrig'].sum(axis=0)

    return dtSummary,dtComp,dtCumul

def calculate_summary(dt,dtR,numLoans,term,modelList,nSims,initSwitch):
# Purpose: Outputs modelled vs actual CPR and CDR curves as well as prices and par yields

# numLoans=1
# term=60
# nSims=100
# dt = dtPmts.iloc[testIndx]
# dtR = dtRegr.iloc[testIndx]
# initSwitch = False

    sampleLoans = dt['LOAN_ID'].iloc[np.where(np.logical_and(dt['MOB']==1,dt['term']==term))].sample(numLoans).values
    pmtIndx = np.where(np.logical_and(np.in1d(dt['LOAN_ID'],sampleLoans),dt['MOB']<=dt['term']))
    origIndx = np.where(np.logical_and(np.in1d(dt['LOAN_ID'],sampleLoans),dt['MOB']==1))

    if (initSwitch == True):
        summIndx = origIndx
    else:
        summIndx = pmtIndx

    dtSummary = pd.DataFrame(np.zeros(shape=(len(summIndx[0]), len(const.priceHeaders())), dtype=float),
                            index=dt.iloc[summIndx].index, columns=const.priceHeaders())
    dtSummary[['LoanID','MOB','Term','Status','Amount','Coupon']] = dt[['LOAN_ID','MOB','term','loan_status','loan_amnt','coupon']].iloc[summIndx].values

    dtSummary = calculate_par_yield(dtSummary,dt.iloc[pmtIndx],dtR.iloc[pmtIndx],modelList,nSims)
    dtSummary = calculate_actual_yield(dtSummary,dt.iloc[pmtIndx],initSwitch)
    dtSummary,dtComp,dtCumul = calculate_performance(dtSummary,dt.iloc[pmtIndx],dtR.iloc[pmtIndx],modelList,nSims,initSwitch)

    dtSummary.to_csv('Curves/summ' + du.datetime.today().strftime('%Y%m%d') + '.csv')
    dtComp.to_csv('Curves/comp' + du.datetime.today().strftime('%Y%m%d') + '.csv')
    dtCumul.to_csv('Curves/cum' + du.datetime.today().strftime('%Y%m%d') + '.csv')

    return dtSummary,dtComp,dtCumul


def main(argv = sys.argv):

quandl.ApiConfig.api_key = const.quandlKey()
dtPmts = read_pmt_data('LC Data/PMTHIST_ALL_20170117_V1.csv')
dtPmts = clean_pmt_data(dtPmts)
dtPmts = check_pmt_data(dtPmts)
dtPmts = insert_status(dtPmts)
dtPmts = read_stats_data(dtPmts)
dtPmts = clean_stats_data(dtPmts)

dtRegr = generate_regressors(dtPmts[const.regressorFields()])
dtResp, matCount = generate_responses(dtPmts)

trainIndx, testIndx = test_train_split(dtPmts,[36,60],.33)
modelList, dtCharts = generate_transition_models(dtRegr,dtResp,dtCharts,trainIndx,testIndx)

    dtSummary,compCurves,cumCurves = calculate_summary(dtPmts.iloc[testIndx],dtRegr.iloc[testIndx],10,60,modelList,100,True)
    dtSummary,compCurves,cumCurves = calculate_summary(dtPmts.iloc[testIndx],dtRegr.iloc[testIndx],10,60,modelList,100,False)

if __name__ == "__main__":
    sys.exit(main())

