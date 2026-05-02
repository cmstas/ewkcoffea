import argparse
import pickle
import gzip
import json
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import copy

import ewkcoffea.modules.plotting_tools as plt_tools
import topcoffea.modules.MakeLatexTable as mlt


HTML_PC = "/home/users/kmohrman/ref_scripts/html_stuff/index.php"

#CLR_LST = ['#d55e00', '#e69f00', '#f0e442', '#009e73', '#0072b2', '#56b4e9', '#cc79a7', '#6e3600', '#a17500'] #, '#a39b2f', '#00664f', '#005d87', '#999999', '#8c5d77']
CLR_LST = ['#d55e00', '#e69f00']

UNBLIND_CATS = [
    "all_events",
    "2lOSSF_1fjx",
    "3l",
]

CAT_LST = [

    #"all_events",

    ### 2l OS SF 1FJ ###

    #"2l",
    #"2lOS",
    #"2lOSSF",
    #"2lOSSF_1fj",
    "2lOSSF_1fjx",
    #"2lOSSF_1fjx_2j",
    "2lOSSF_1fjx_HFJ",
    "2lOSSF_1fjx_HFJtag",
    #"2lOSSF_1fjx_HFJtag_nj2",
    "2lOSSF_1fjx_HFJtag_nj2_mjj600",
    "2lOSSF_1fjx_HFJtag_nj2_mjj600_nbm0",
    #"2lOSSF_1fjx_HFJtag_nj2_mjj600_nbm0_onZ",
    #"2lOSSF_1fjx_HFJtag_nj2_mjj600_nbm0_offZ",

    ### 3l ###

    #"3l",
    #"3l_chsum3",
    #"3l_chsum3_mjj400",
    #"3l_chsum3_mjj400_b0p4",
    #"3l_chsum1",
    #"3l_chsum1_mll12",
    #"3l_chsum1_mll12_sfos0",
    #"3l_chsum1_mll12_sfos0_mjj400",
    #"3l_chsum1_mll12_sfos0_mjj400_b0p4",
    #"3l_chsum1_mll12_sfos1",
    #"3l_chsum1_mll12_sfos1_mjj400",
    #"3l_chsum1_mll12_sfos1_mjj400_jf0pt50",
    #"3l_chsum1_mll12_sfos2",
    #"3l_chsum1_mll12_sfos2_mjj400",
    #"3l_chsum1_mll12_sfos2_mjj400_jf0pt50",

]


GRP_DICT_FULL_R3 = {
    "Signal": [
        "VBSWWH_OS_c2v1p0_c3_1p0",
        "VBSWWH_SS_c2v1p0_c3_1p0",
        "VBSWZH_c2v1p0_c3_1p0",
        "VBSZZH_c2v1p0_c3_1p0",
        #"VBSWWH_OS_c2v1p0_c3_10p0",
        #"VBSWWH_SS_c2v1p0_c3_10p0",
        #"VBSWZH_c2v1p0_c3_10p0",
        #"VBSZZH_c2v1p0_c3_10p0",
        #"VBSWWH_OS_c2v1p5_c3_1p0",
        #"VBSWWH_SS_c2v1p5_c3_1p0",
        #"VBSWZH_c2v1p5_c3_1p0",
        #"VBSZZH_c2v1p5_c3_1p0",
    ],
    "Data":[
        "Muon",
        "MuonEG",
        "EGamma",
    ],
    "QCD": [
        "QCD_Bin-PT-1000to1500_TuneCP5_13p6TeV",
        "QCD_Bin-PT-120to170_TuneCP5_13p6TeV",
        "QCD_Bin-PT-1500to2000_TuneCP5_13p6TeV",
        "QCD_Bin-PT-170to300_TuneCP5_13p6TeV",
        "QCD_Bin-PT-2000to2500_TuneCP5_13p6TeV",
        "QCD_Bin-PT-2500to3000_TuneCP5_13p6TeV",
        "QCD_Bin-PT-3000_TuneCP5_13p6TeV",
        "QCD_Bin-PT-300to470_TuneCP5_13p6TeV",
        "QCD_Bin-PT-470to600_TuneCP5_13p6TeV",
        "QCD_Bin-PT-50to80_TuneCP5_13p6TeV",
        "QCD_Bin-PT-600to800_TuneCP5_13p6TeV",
        "QCD_Bin-PT-800to1000_TuneCP5_13p6TeV",
        "QCD_Bin-PT-80to120_TuneCP5_13p6TeV",
    ],
    "ttbar": [
        "TTto2L2Nu_TuneCP5_13p6TeV",
        "TTto4Q_TuneCP5_13p6TeV",
        "TTtoLNu2Q_TuneCP5_13p6TeV",
    ],
    "single-t": [
        "TBbarQto2Q-t-channel-4FS_TuneCP5_13p6TeV",
        "TBbarQtoLNu-t-channel-4FS_TuneCP5_13p6TeV",
        "TBbartoLNu-s-channel_TuneCP5_13p6TeV",
        "TWminusto2L2Nu_TuneCP5_13p6TeV",
        "TWminusto4Q_TuneCP5_13p6TeV",
        "TWminustoLNu2Q_TuneCP5_13p6TeV",
        "TZQB-Zto2L-4FS_Bin-MLL-30_TuneCP5_13p6TeV",
        "TbarBQto2Q-t-channel-4FS_TuneCP5_13p6TeV",
        "TbarBQtoLNu-t-channel-4FS_TuneCP5_13p6TeV",
        "TbarBtoLNu-s-channel_TuneCP5_13p6TeV",
        "TbarWplusto2L2Nu_TuneCP5_13p6TeV",
        "TbarWplusto4Q_TuneCP5_13p6TeV",
        "TbarWplustoLNu2Q_TuneCP5_13p6TeV",
    ],
    "ttX": [
        "TTH-Hto2B_Par-M-125_TuneCP5_13p6TeV",
        "TTH-HtoNon2B_Par-M-125_TuneCP5_13p6TeV",
        "TTLL_Bin-MLL-4to50_TuneCP5_13p6TeV",
        "TTLL_Bin-MLL-50_TuneCP5_13p6TeV",
        "TTLNu-1Jets_TuneCP5_13p6TeV",
        "TTW-WtoQQ-1Jets_TuneCP5_13p6TeV",
    ],
    "rare-top": [
        "TTWW_TuneCP5_13p6TeV",
        "TTWZ_TuneCP5_13p6TeV",
    ],
    "Vjets": [
        "Wto2Q-3Jets_Bin-HT-100to400_TuneCP5_13p6TeV",
        "Wto2Q-3Jets_Bin-HT-1500to2500_TuneCP5_13p6TeV",
        "Wto2Q-3Jets_Bin-HT-2500_TuneCP5_13p6TeV",
        "Wto2Q-3Jets_Bin-HT-400to800_TuneCP5_13p6TeV",
        "Wto2Q-3Jets_Bin-HT-800to1500_TuneCP5_13p6TeV",
        "WtoLNu-2Jets_Bin-1J-PTLNu-100to200_TuneCP5_13p6TeV",
        "WtoLNu-2Jets_Bin-1J-PTLNu-200to400_TuneCP5_13p6TeV",
        "WtoLNu-2Jets_Bin-1J-PTLNu-400to600_TuneCP5_13p6TeV",
        "WtoLNu-2Jets_Bin-1J-PTLNu-40to100_TuneCP5_13p6TeV",
        "WtoLNu-2Jets_Bin-1J-PTLNu-600_TuneCP5_13p6TeV",
        "WtoLNu-2Jets_Bin-2J-PTLNu-100to200_TuneCP5_13p6TeV",
        "WtoLNu-2Jets_Bin-2J-PTLNu-200to400_TuneCP5_13p6TeV",
        "WtoLNu-2Jets_Bin-2J-PTLNu-400to600_TuneCP5_13p6TeV",
        "WtoLNu-2Jets_Bin-2J-PTLNu-40to100_TuneCP5_13p6TeV",
        "WtoLNu-2Jets_Bin-2J-PTLNu-600_TuneCP5_13p6TeV",
        "WtoLNu-4Jets_Bin-1J_TuneCP5_13p6TeV",
        "WtoLNu-4Jets_Bin-2J_TuneCP5_13p6TeV",
        "WtoLNu-4Jets_Bin-3J_TuneCP5_13p6TeV",
        "WtoLNu-4Jets_Bin-4J_TuneCP5_13p6TeV",
        "Zto2Q-4Jets_Bin-HT-100to400_TuneCP5_13p6TeV",
        "Zto2Q-4Jets_Bin-HT-1500to2500_TuneCP5_13p6TeV",
        "Zto2Q-4Jets_Bin-HT-2500_TuneCP5_13p6TeV",
        "Zto2Q-4Jets_Bin-HT-400to800_TuneCP5_13p6TeV",
        "Zto2Q-4Jets_Bin-HT-800to1500_TuneCP5_13p6TeV",
    ],
    "DY": [
        #"DYto2L-2Jets_Bin-1J-MLL-50-PTLL-100to200",
        #"DYto2L-2Jets_Bin-1J-MLL-50-PTLL-200to400",
        #"DYto2L-2Jets_Bin-1J-MLL-50-PTLL-400to600",
        #"DYto2L-2Jets_Bin-1J-MLL-50-PTLL-40to100",
        #"DYto2L-2Jets_Bin-1J-MLL-50-PTLL-600",
        #"DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200",
        #"DYto2L-2Jets_Bin-2J-MLL-50-PTLL-200to400",
        #"DYto2L-2Jets_Bin-2J-MLL-50-PTLL-400to600",
        #"DYto2L-2Jets_Bin-2J-MLL-50-PTLL-40to100",
        #"DYto2L-2Jets_Bin-2J-MLL-50-PTLL-600",
        "DYto2E_Bin-MLL-10to50_TuneCP5_13p6TeV",
        "DYto2Mu_Bin-MLL-10to50_TuneCP5_13p6TeV",
        "DYto2Tau_Bin-MLL-10to50_TuneCP5_13p6TeV",
        "DYto2Mu-2Jets_Bin-MLL-50_TuneCP5_13p6TeV",
        "DYto2E-2Jets_Bin-MLL-50_TuneCP5_13p6TeV",
        "DYto2Tau-2Jets_Bin-MLL-50_TuneCP5_13p6TeV",
    ],
    #"ggHtoZZ": [
    #    "GluGluH-Hto2Zto4L_Par-M-125_TuneCP5_13p6TeV",
    #],
    #"ggVV": [
    #    "GluGluToContinto2Zto2E2Mu_TuneCP5_13p6TeV",
    #    "GluGluToContinto2Zto2E2Tau_TuneCP5_13p6TeV",
    #    "GluGluToContinto2Zto2Mu2Tau_TuneCP5_13p6TeV",
    #    "GluGlutoContinto2Zto4E_TuneCP5_13p6TeV",
    #    "GluGlutoContinto2Zto4Mu_TuneCP5_13p6TeV",
    #    "GluGlutoContinto2Zto4Tau_TuneCP5_13p6TeV",
    #],
    "VV": [
        "WWJJto2L2Nu-OS-noTop-EWK_TuneCP5_13p6TeV",
        "WWJJto2L2Nu-SS-noTop-EWK_TuneCP5_13p6TeV",
        "WW_TuneCP5_13p6TeV",
        "WWto2L2Nu_TuneCP5_13p6TeV",
        "WWto4Q_TuneCP5_13p6TeV",
        "WWtoLNu2Q_TuneCP5_13p6TeV",
        "WZ_TuneCP5_13p6TeV",
        "WZto2L2Q_TuneCP5_13p6TeV",
        "WZto3LNu_TuneCP5_13p6TeV",
        "WZtoL3Nu_TuneCP5_13p6TeV",
        "WZtoLNu2Q_TuneCP5_13p6TeV",
        "ZZJJto4L-EWK_TuneCP5_13p6TeV",
        "ZZJJto4L-QCD_TuneCP5_13p6TeV",
        "ZZ_TuneCP5_13p6TeV",
        "ZZto2L2Nu_TuneCP5_13p6TeV",
        "ZZto2L2Q_TuneCP5_13p6TeV",
        "ZZto2Nu2Q_TuneCP5_13p6TeV",
        "ZZto4L_TuneCP5_13p6TeV",
        "ZZto4Q-1Jets_TuneCP5_13p6TeV",
    ],
    "ewkVV": [
        "VBS-SSWW-LL_TuneCP5_13p6TeV",
        "VBS-SSWW-TL_TuneCP5_13p6TeV",
    ],
    "VH": [
        "WminusH-HtoNon2B_Par-M-125_TuneCP5_13p6TeV",
        "WminusH-Wto2Q-Hto2B_Par-M-125_TuneCP5_13p6TeV",
        "WminusH-WtoLNu-Hto2B_Par-M-125_TuneCP5_13p6TeV",
        "WplusH-HtoNon2B_Par-M-125_TuneCP5_13p6TeV",
        "WplusH-Wto2Q-Hto2B_Par-M-125_TuneCP5_13p6TeV",
        "WplusH-WtoLNu-Hto2B_Par-M-125_TuneCP5_13p6TeV",
        "ZH-HtoNon2B_Par-M-125_TuneCP5_13p6TeV",
        "ZH-Zto2L-Hto2B_Par-M-125_TuneCP5_13p6TeV",
        "ZH-Zto2Q-Hto2B_Par-M-125_TuneCP5_13p6TeV",
        "GluGluZH-Zto2L-Hto2B_Par-M-125_TuneCP5_13p6TeV",
    ],
    "VVV": [
        "WWW-4F_TuneCP5_13p6TeV",
        "WWZ-4F_TuneCP5_13p6TeV",
        "WZZ-5F_TuneCP5_13p6TeV",
        "ZZZ-5F_TuneCP5_13p6TeV",
    ],
}


GRP_DICT_FULL_R2 = {
    "Data" : [
        #"data",
        "DoubleMuon",
        "MuonEG",
        "DoubleEG",
        "SingleMuon",
        "SingleElectron",
        "EGamma",
    ],
    "Signal" : [
        "VBSWWH_SS_c2v1p0_c3_1p0",
        "VBSWWH_OS_c2v1p0_c3_1p0",
        "VBSWZH_c2v1p0_c3_1p0",
        "VBSZZH_c2v1p0_c3_1p0",
        #"VBSWWH_SS_VBSCuts_13TeV",
        #"VBSWWH_OS_VBSCuts_13TeV",
        #"VBSWZH_VBSCuts_13TeV",
        #"VBSZZH_VBSCuts_13TeV",
    ],
    "VBSWWH_SS": ["VBSWWH_SS_c2v1p0_c3_1p0"],
    "VBSWWH_OS": ["VBSWWH_OS_c2v1p0_c3_1p0"],
    "VBSWZH": ["VBSWZH_c2v1p0_c3_1p0"],
    "VBSZZH": ["VBSZZH_c2v1p0_c3_1p0"],

    "QCD" : [
        "QCD_HT50to100_TuneCP5_PSWeights_13TeV",
        "QCD_HT100to200_TuneCP5_PSWeights_13TeV",
        "QCD_HT200to300_TuneCP5_PSWeights_13TeV",
        "QCD_HT300to500_TuneCP5_PSWeights_13TeV",
        "QCD_HT500to700_TuneCP5_PSWeights_13TeV",
        "QCD_HT700to1000_TuneCP5_PSWeights_13TeV",
        "QCD_HT1000to1500_TuneCP5_PSWeights_13TeV",
        "QCD_HT1500to2000_TuneCP5_PSWeights_13TeV",
        "QCD_HT2000toInf_TuneCP5_PSWeights_13TeV",
    ],
    "ttbar" : [
        "TTTo2L2Nu_TuneCP5_13TeV",
        "TTToSemiLeptonic_TuneCP5_13TeV",
        "TTToHadronic_TuneCP5_13TeV",
    ],
    "single-t" : [
        "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV",
        "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV",
        "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV",
        "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV",
        "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV",
    ],
    "ttX" : [
        "TTWJetsToLNu_TuneCP5_13TeV",
        "TTWJetsToQQ_TuneCP5_13TeV",
        "TTWW_TuneCP5_13TeV",
        "TTWZ_TuneCP5_13TeV",
        "TTZToLLNuNu_M-10_TuneCP5_13TeV",
        "TTbb_4f_TTTo2L2Nu",
        "TTbb_4f_TTToSemiLeptonic",
        "TTbb_4f_TTToHadronic",
        "ttHToNonbb_M125_TuneCP5_13TeV",
        "ttHTobb_M125_TuneCP5_13TeV",
        "ttWJets_TuneCP5_13TeV",
        "ttZJets_TuneCP5_13TeV",
    ],
    "rare-top" : [
        "TWZToLL_tlept_Wlept_5f_DR_TuneCP5_13TeV",
        "tZq_ll_4f_ckm_NLO_TuneCP5_13TeV",
    ],
    "Vjets" : [
        "WJetsToLNu_HT-70To100_TuneCP5_13TeV",
        "WJetsToLNu_HT-100To200_TuneCP5_13TeV",
        "WJetsToLNu_HT-200To400_TuneCP5_13TeV",
        "WJetsToLNu_HT-400To600_TuneCP5_13TeV",
        "WJetsToLNu_HT-600To800_TuneCP5_13TeV",
        "WJetsToLNu_HT-800To1200_TuneCP5_13TeV",
        "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV",
        "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV",
        "WJetsToLNu_TuneCP5_13TeV",
        "WJetsToQQ_HT-200to400_TuneCP5_13TeV",
        "WJetsToQQ_HT-400to600_TuneCP5_13TeV",
        "WJetsToQQ_HT-600to800_TuneCP5_13TeV",
        "WJetsToQQ_HT-800toInf_TuneCP5_13TeV",
        "ZJetsToQQ_HT-200to400_TuneCP5_13TeV",
        "ZJetsToQQ_HT-400to600_TuneCP5_13TeV",
        "ZJetsToQQ_HT-600to800_TuneCP5_13TeV",
        "ZJetsToQQ_HT-800toInf_TuneCP5_13TeV",
        "WminusH_HToBB_WToLNu_M-125_TuneCP5_13TeV",
        "WminusH_HToBB_WToQQ_M-125_TuneCP5_13TeV",
        "WplusH_HToBB_WToLNu_M-125_TuneCP5_13TeV",
        "WplusH_HToBB_WToQQ_M-125_TuneCP5_13TeV",
    ],
    "DY" : [
        "DYJetsToLL_M-10to50_TuneCP5_13TeV",
        "DYJetsToLL_M-50_TuneCP5_13TeV",
    ],
    "ewkV" : [
        "EWKWMinus2Jets_WToLNu_M-50_TuneCP5_withDipoleRecoil_13TeV",
        "EWKWPlus2Jets_WToLNu_M-50_TuneCP5_withDipoleRecoil_13TeV",
        "EWKWminus2Jets_WToQQ_dipoleRecoilOn_TuneCP5_13TeV",
        "EWKWplus2Jets_WToQQ_dipoleRecoilOn_TuneCP5_13TeV",
        "EWKZ2Jets_ZToLL_M-50_TuneCP5_withDipoleRecoil_13TeV",
        "EWKZ2Jets_ZToNuNu_M-50_TuneCP5_withDipoleRecoil_13TeV",
        "EWKZ2Jets_ZToQQ_dipoleRecoilOn_TuneCP5_13TeV",
        "WWJJToLNuLNu_EWK_noTop_TuneCP5_13TeV",
    ],
    "VV" : [
        "GluGluToContinToZZTo2e2mu_TuneCP5_13TeV",
        "GluGluToContinToZZTo2e2tau_TuneCP5_13TeV",
        "GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV",
        "GluGluToContinToZZTo4e_TuneCP5_13TeV",
        "GluGluToContinToZZTo4mu_TuneCP5_13TeV",
        "GluGluToContinToZZTo4tau_TuneCP5_13TeV",
        "GluGluZH_HToWWTo2L2Nu_TuneCP5_13TeV",
        "GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV",
        "WZ_TuneCP5_13TeV",
        "WZJJ_EWK_InclusivePolarization_TuneCP5_13TeV",
        "WZTo1L1Nu2Q_4f_TuneCP5_13TeV",
        "WZTo1L3Nu_4f_TuneCP5_13TeV",
        "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV",
        "WZTo3LNu_TuneCP5_13TeV",
        "WW_TuneCP5_13TeV",
        "ZZ_TuneCP5_13TeV",
        "WWTo4Q_4f_TuneCP5_13TeV",
        "WWTo1L1Nu2Q_4f_TuneCP5_13TeV",
        "WWTo2L2Nu_TuneCP5_13TeV",
        "GluGluHToZZTo4L",
        "ZZJJTo4L_TuneCP5_13TeV",
        "ZZTo2Nu2Q_5f_TuneCP5_13TeV",
        "ZZTo4Q_5f_TuneCP5_13TeV",
        "ZZJJTo4L_EWKnotop_TuneCP5_13TeV",
        "ZZTo2L2Nu_TuneCP5_13TeV",
        "ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV",
        "ZZTo4L_M-1toInf_TuneCP5_13TeV",
    ],
    "ewkVV" : [
        "SSWW",
    ],
    "VH" : [
        "VBFWH_HToBB_WToLNu_M-125_dipoleRecoilOn_TuneCP5_13TeV",
        "VHToNonbb_M125_TuneCP5_13TeV",
        "ggZH_HToBB_ZToLL_M-125_TuneCP5_13TeV",
        "ggZH_HToBB_ZToBB_M-125_TuneCP5_13TeV",
        "ggZH_HToBB_ZToNuNu_M-125_TuneCP5_13TeV",
        "ggZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV",
        "ZH_HToBB_ZToLL_M-125_TuneCP5_13TeV",
        "ZH_HToBB_ZToBB_M-125_TuneCP5_13TeV",
        "ZH_HToBB_ZToNuNu_M-125_TuneCP5_13TeV",
        "ZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV",
        "HZJ_HToWWTo2L2Nu_ZTo2L_M-125_TuneCP5_13TeV",
    ],
    "VVV" : [
        "WWW_4F_TuneCP5_13TeV",
        "WWZ_4F_TuneCP5_13TeV",
        "WZZ_TuneCP5_13TeV",
        "ZZZ_TuneCP5_13TeV",
    ],

}




########################
### Helper functions ###

# Append the years to sample names dict
def append_years(sample_dict_base,year_lst):
    out_dict = {}
    for proc_group in sample_dict_base.keys():
        out_dict[proc_group] = []
        for proc_base_name in sample_dict_base[proc_group]:
            for year_str in year_lst:
                out_dict[proc_group].append(f"{year_str}_{proc_base_name}")
    return out_dict


# Get sig and bkg yield in all categories
def get_yields_per_cat(histo_dict,var_name,grp_dict,year_name_lst_to_prepend, lepflav_bin=None):
    out_dict = {}

    # Get the initial grouping dict
    #grouping_dict = append_years(grp_dict,year_name_lst_to_prepend) # For fromnano
    grouping_dict = copy.deepcopy(grp_dict)

    # Get list of all of the backgrounds together
    bkg_lst = []
    for grp in grouping_dict:
        if (grp != "Signal") and (grp != "Data") and (grp not in ["VBSWWH_SS","VBSWWH_OS","VBSWZH","VBSZZH"]):
            bkg_lst = bkg_lst + grouping_dict[grp]

    # Make the dictionary to get yields for, it includes what's in grouping_dict, plus the backgrounds grouped as one
    groups_to_get_yields_for_dict = copy.deepcopy(grouping_dict)
    groups_to_get_yields_for_dict["Background"] = bkg_lst

    # Loop over cats and fill dict of sig and bkg
    for cat in CAT_LST:
        if cat not in plt_tools.get_axis_cats(histo_dict[var_name],"category"): continue
        else: out_dict[cat] = {}

        # Get the hist for the given categroy
        if lepflav_bin is None:
            histo_base = histo_dict[var_name][{"systematic":"nominal", "category":cat}]
        elif lepflav_bin == "all":
            histo_base = histo_dict[var_name][{"systematic":"nominal", "category":cat, "lepflav":sum}]
        elif isinstance(lepflav_bin,int):
            histo_base = histo_dict[var_name][{"systematic":"nominal", "category":cat, "lepflav": lepflav_bin}]
        else:
            raise Exception(f"Unknown lep flav handling: {lepflav_bin}")
        #histo_base = histo_dict[var_name][{"systematic":"nominal", "category":cat}] # For fromnano
        #histo_base = histo_dict[var_name][{"systematic":"nominal", "category":cat, "year": sum}] # If have years

        # Get values per proc
        for group_name,group_lst in groups_to_get_yields_for_dict.items():
            histo = plt_tools.group(histo_base,"process","process",{group_name:group_lst})
            yld = sum(sum(histo.values(flow=True)))
            var = sum(sum(histo.variances(flow=True)))
            out_dict[cat][group_name] = [yld,(var)**0.5]

        # Blind
        if cat not in UNBLIND_CATS:
            out_dict[cat]["Data"] = [1e-6,1e-6]

        # Get the metric
        sig = out_dict[cat]["Signal"][0]
        bkg = out_dict[cat]["Background"][0]
        metric = sig/(bkg)**0.5
        out_dict[cat]["metric"] = [metric,None] # Don't bother propagating error
        out_dict[cat]["metricX100"] = [metric*100,None] # Don't bother propagating error

        # Get the data/MC
        mc = sig+bkg
        data = out_dict[cat]["Data"][0]
        out_dict[cat]["Data/MC"] = [data/mc,None] # Don't bother propagating error

    return out_dict


# Make the figures for the vvh study
def make_vvh_fig(histo_mc,histo_mc_sig,histo_mc_bkg,histo_dat=None,title="test",axisrangex=None):

    # Create the figure
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(7,10),
        gridspec_kw={"height_ratios": (3, 1, 1, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.07)

    # Plot the stack plot
    histo_mc.plot1d(
        stack=True,
        histtype="fill",
        color=CLR_LST,
        ax=ax1,
        zorder=10,
    )

    # Plot the data
    if histo_dat is not None:
        histo_dat.plot1d(
            stack=False,
            histtype="errorbar",
            color="k",
            ax=ax1,
            w2=histo_dat.variances(),
            w2method="sqrt",
            #w2method="poisson",
            zorder=11,
        )

    # Get the errs on MC and plot them by hand on the stack plot
    histo_mc_sum = histo_mc[{"process_grp":sum}]
    mc_arr = histo_mc_sum.values()
    mc_err_arr = np.sqrt(histo_mc_sum.variances())
    err_p = np.append(mc_arr + mc_err_arr, 0)
    err_m = np.append(mc_arr - mc_err_arr, 0)
    bin_edges_arr = histo_mc_sum.axes[0].edges
    bin_centers_arr = histo_mc_sum.axes[0].centers
    ax1.fill_between(bin_edges_arr,err_m,err_p, step='post', facecolor='none', edgecolor='gray', alpha=0.5, linewidth=0.0, label='MC stat', hatch='/////', zorder=11)


    ## Draw the normalized shapes ##

    # Get normalized hists of sig and bkg
    yld_sig = sum(sum(histo_mc_sig.values(flow=True)))
    yld_bkg = sum(sum(histo_mc_bkg.values(flow=True)))
    metric = yld_sig/(yld_bkg**0.5)
    histo_mc_sig_scale_to_bkg = plt_tools.scale(copy.deepcopy(histo_mc_sig), "process_grp", {"Signal":yld_bkg/yld_sig})
    histo_mc_sig_norm         = plt_tools.scale(copy.deepcopy(histo_mc_sig), "process_grp", {"Signal":1.0/yld_sig})
    histo_mc_bkg_norm         = plt_tools.scale(copy.deepcopy(histo_mc_bkg), "process_grp", {"Background":1.0/yld_bkg})

    histo_mc_sig_scale_to_bkg.plot1d(color=["red"], ax=ax1, zorder=100)
    histo_mc_sig_norm.plot1d(color="red",  ax=ax2, zorder=100)
    histo_mc_bkg_norm.plot1d(color="gray", ax=ax2, zorder=100)


    ## Draw the significance ##

    # Get the sig and bkg arrays (Not including flow bins here, overflow should already be handled, and if we have underflow, why?)
    yld_sig_arr = sum(histo_mc_sig.values())
    yld_bkg_arr = sum(histo_mc_bkg.values())

    # Get the cumulative signifiance, starting from left
    yld_sig_arr_cum = np.cumsum(yld_sig_arr)
    yld_bkg_arr_cum = np.cumsum(yld_bkg_arr)
    metric_cum = yld_sig_arr_cum/np.sqrt(yld_bkg_arr_cum)
    metric_cum = np.nan_to_num(metric_cum,nan=0,posinf=0) # Set the nan (from sig and bkg both being 0) to 0

    # Get the cumulative signifiance, starting from right
    yld_sig_arr_cum_ud = np.cumsum(np.flipud(yld_sig_arr))
    yld_bkg_arr_cum_ud = np.cumsum(np.flipud(yld_bkg_arr))
    metric_cum_ud = np.flipud(yld_sig_arr_cum_ud/np.sqrt(yld_bkg_arr_cum_ud))
    metric_cum_ud = np.nan_to_num(metric_cum_ud,nan=0,posinf=0) # Set the nan (from sig and bkg both being 0) to 0
    yld_sig_arr_cum_ud = np.flipud(yld_sig_arr_cum_ud) # Flip back so the order is as expected for later use
    yld_bkg_arr_cum_ud = np.flipud(yld_bkg_arr_cum_ud) # Flip back so the order is as expected for later use

    # Draw it on the third plot
    ax3.scatter(bin_centers_arr,metric_cum,   facecolor='none',edgecolor='black',marker=">",label="Cum. from left", zorder=100)
    ax3.scatter(bin_centers_arr,metric_cum_ud,facecolor='none',edgecolor='black',marker="<",label="Cum. from right", zorder=100)

    # Write the max values on the plot
    max_metric_from_left_idx  = np.argmax(metric_cum)
    max_metric_from_right_idx = np.argmax(metric_cum_ud)
    left_max_y  = metric_cum[max_metric_from_left_idx]
    right_max_y = metric_cum_ud[max_metric_from_right_idx]
    left_max_x  = bin_centers_arr[max_metric_from_left_idx]
    right_max_x = bin_centers_arr[max_metric_from_right_idx]
    left_s_at_max  = yld_sig_arr_cum[max_metric_from_left_idx]
    right_s_at_max = yld_sig_arr_cum_ud[max_metric_from_right_idx]
    left_b_at_max  = yld_bkg_arr_cum[max_metric_from_left_idx]
    right_b_at_max = yld_bkg_arr_cum_ud[max_metric_from_right_idx]
    plt.text(0.15,0.35, f"Max from left:  {np.round(left_max_y,3)} (at x={np.round(left_max_x,2)}, sig: {np.round(left_s_at_max,2)}, bkg: {np.round(left_b_at_max,1)})", fontsize=9, transform=fig.transFigure)
    plt.text(0.15,0.33, f"Max from right: {np.round(right_max_y,3)} (at x={np.round(right_max_x,2)} , sig: {np.round(right_s_at_max,2)}, bkg: {np.round(right_b_at_max,1)})", fontsize=9, transform=fig.transFigure)


    ## Draw on the fraction of signal retained ##
    yld_sig_arr_cum_frac    = np.cumsum(yld_sig_arr)/yld_sig
    yld_sig_arr_cum_frac_ud = np.flipud(np.cumsum(np.flipud(yld_sig_arr)))/yld_sig
    ax4.scatter(bin_centers_arr,yld_sig_arr_cum_frac,   facecolor='none',edgecolor='black',marker=">",label="Cum. from left", zorder=100)
    ax4.scatter(bin_centers_arr,yld_sig_arr_cum_frac_ud,facecolor='none',edgecolor='black',marker="<",label="Cum. from right", zorder=100)


    ## Legend, scale the axis, set labels, etc ##

    extr = ax1.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="12", frameon=False)
    extr = ax2.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="10", frameon=False)
    extr = ax3.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="10", frameon=False)
    extr = ax4.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="10", frameon=False)
    plt.text(0.15,0.85, f"Sig. yield: {np.round(yld_sig,2)}", fontsize = 11, transform=fig.transFigure)
    plt.text(0.15,0.83, f"Bkg. yield: {np.round(yld_bkg,2)}", fontsize = 11, transform=fig.transFigure)
    plt.text(0.15,0.81, f"Metric: {np.round(metric,3)}", fontsize = 11, transform=fig.transFigure)
    plt.text(0.15,0.79, f"[Note: sig. overlay scaled {np.round(yld_bkg/yld_sig,1)}x]", fontsize = 12, transform=fig.transFigure)
    if histo_dat is not None:
        yld_dat = sum(sum(histo_dat.values(flow=True)))
        plt.text(0.15,0.76, f"Data: {yld_dat}, data/mc = {np.round(yld_dat/(yld_sig+yld_bkg),2)}", fontsize = 12, transform=fig.transFigure)

    extt = ax1.set_title(title)
    ax1.set_xlabel(None)
    ax2.set_xlabel(None)
    extb = ax3.set_xlabel(None)
    # Plot a dummy hist on ax4 to get the label to show up
    histo_mc.plot1d(alpha=0, ax=ax4)

    extl = ax2.set_ylabel('Shapes')
    ax3.set_ylabel('Significance')
    ax4.set_ylabel('Signal kept (%)')
    ax1.tick_params(axis='y', labelsize=16)
    ax2.tick_params(axis='x', labelsize=16)
    ax3.axhline(0.0,linestyle="-",color="k",linewidth=0.5)
    ax4.axhline(0.0,linestyle="-",color="k",linewidth=0.5)
    #ax1.grid() # Note: grid does not respect z order :(
    #ax2.grid()
    ax3.grid()
    ax4.grid()

    shapes_ymax = max( max(sum(histo_mc_sig_norm.values(flow=True))) , max(sum(histo_mc_bkg_norm.values(flow=True))) )
    significance_max = max(max(metric_cum),max(metric_cum_ud))
    significance_min = 0-0.1*significance_max
    ax1.autoscale(axis='y')
    ax2.set_ylim(0.0,1.5*shapes_ymax)
    ax3.set_ylim(significance_min,2.5*significance_max)
    ax4.set_ylim(-0.1,1.2)
    #ax1.set_yscale('log')

    if axisrangex is not None:
        ax1.set_xlim(axisrangex[0],axisrangex[1])
        ax2.set_xlim(axisrangex[0],axisrangex[1])


    return (fig,(extt,extr,extb,extl))


# Dump to a datacard
# in_dict format should be {"cat":{"sig":x.x,"bkg":y.y},...}
def make_card(in_dict,out_name):

    obsblock_binname_str  = ""
    obsblock_obs_str      = ""
    expblock_binname_str  = ""
    expblock_procname_str = ""
    expblock_procnum_str  = ""
    expblock_rate_str     = ""
    for i,cat in enumerate(in_dict):
        sig = in_dict[cat]["sig"]
        bkg = in_dict[cat]["bkg"]
        obs = float(sig) + float(bkg)
        obsblock_binname_str  += f" {cat} "
        obsblock_obs_str      += f" {obs} "
        expblock_binname_str  += f" {cat} {cat} "
        expblock_procname_str += " sig bkg "
        expblock_procnum_str  += " 0 1"
        expblock_rate_str     += f" {sig} {bkg} "

    divider = "-------------------------------------------"

    # Build up the card
    out_str = ""
    out_str = out_str + f"\nimax *  number of channels"
    out_str = out_str + f"\njmax *  number of backgrounds"
    out_str = out_str + f"\nkmax *  number of nuisance parameters (sources of systematic uncertainties"
    out_str = out_str + f"\n{divider}"
    out_str = out_str + f"\nbin {obsblock_binname_str}"
    out_str = out_str + f"\nobservation {obsblock_obs_str}"
    out_str = out_str + f"\n{divider}"
    out_str = out_str + f"\nbin       {expblock_binname_str}"
    out_str = out_str + f"\nprocess   {expblock_procname_str}"
    out_str = out_str + f"\nprocess   {expblock_procnum_str}"
    out_str = out_str + f"\nrate      {expblock_rate_str}"
    out_str = out_str + f"\n"

    # Dump to screen
    print(out_str)

    text_file = open(f"dc_{out_name}.txt", "w")
    text_file.write(out_str)
    text_file.close()


##############################################################
### Wrapper functions for each of the main functionalities ###


### Sanity check of the different reweight points (for a hist that has extra axis to store that) ###
# Old
def check_rwgt(histo_dict):

    #pkl_file_path = "/home/users/kmohrman/vbs_vvh/ewkcoffea_for_vbs_vvh/ewkcoffea/analysis/vbs_vvh/histos/check_wgt_genw.pkl.gz"
    #pkl_file_path = "/home/users/kmohrman/vbs_vvh/ewkcoffea_for_vbs_vvh/ewkcoffea/analysis/vbs_vvh/histos/check_wgt_sm.pkl.gz"
    #pkl_file_path = "/home/users/kmohrman/vbs_vvh/ewkcoffea_for_vbs_vvh/ewkcoffea/analysis/vbs_vvh/histos/check_wgt_rwgtscan.pkl.gz"

    var_name = "njets"
    #var_name = "njets_counts"
    cat = "exactly1lep_exactly1fj_STmet1100"

    #cat_yld = sum(sum(histo_dict[var_name][{"systematic":"nominal", "category":cat}].values(flow=True)))
    #cat_err = (sum(sum(histo_dict[var_name][{"systematic":"nominal", "category":cat}].variances(flow=True))))**0.5
    #print(cat_yld, cat_err)
    #exit()

    wgts = []
    for i in range(120):
        idx_name = f"idx{i}"
        cat_yld = sum(sum(histo_dict[var_name][{"systematic":"nominal", "category":cat, "rwgtidx":idx_name}].values(flow=True)))
        cat_err = (sum(sum(histo_dict[var_name][{"systematic":"nominal", "category":cat, "rwgtidx":idx_name}].variances(flow=True))))**0.5
        wgts.append(cat_yld)
        print(i,cat_yld)

    print(min(wgts))



### Dumps the yields and counts for a couple categories into a json ###
# The output of this is used for the CI check
def dump_json_simple(histo_dict,out_name="vvh_yields_simple"):
    out_dict = {}
    hist_to_use = "njets"
    cats_to_check = ["all_events", "2lOSSF_1fjx", "3l"]
    for proc_name in histo_dict[hist_to_use].axes["process"]:
        out_dict[proc_name] = {}
        for cat_name in cats_to_check:
            if "lepflav" in histo_dict[hist_to_use].axes.name:
                yld = sum(sum(histo_dict[hist_to_use][{"systematic":"nominal", "category":cat_name, "lepflav":sum}].values(flow=True)))
            else:
                yld = sum(sum(histo_dict[hist_to_use][{"systematic":"nominal", "category":cat_name}].values(flow=True)))
            out_dict[proc_name][cat_name] = [yld,None]

    # Dump counts dict to json
    output_name = f"{out_name}.json"
    with open(output_name,"w") as out_file: json.dump(out_dict, out_file, indent=4)
    print(f"\nSaved json file: {output_name}\n")


### Dump a latex table of the yields
def print_latex_yields(histo_dict,grp_dict, tag="Yields", lepflav=None, print_begin_info=True,print_end_info=True):

    # Get ahold of the yields
    yld_dict    = get_yields_per_cat(histo_dict,"njets",grp_dict,None, lepflav_bin=lepflav)

    group_lst_order = ['Signal', 'VBSWWH_SS', 'VBSWWH_OS', 'VBSWZH', 'VBSZZH', 'Background', 'QCD', 'DY', 'ttbar', 'single-t', 'rare-top', 'ttX', 'Vjets', 'VV', 'ewkV', 'ewkVV', 'VH', 'VVV', 'Data', 'Data/MC', 'metricX100'] # R2
    #group_lst_order = ["Signal", "Background", "QCD", "ttbar", "single-t", "rare-top", "ttX", "Vjets", "VV", "DY", "ewkVV", "VH", "VVV","Data"] # R3

    mlt.print_latex_yield_table(
        yld_dict,
        subkey_order=group_lst_order,
        print_begin_info=print_begin_info,
        print_end_info=print_end_info,
        column_variable="keys",
        print_errs=True,
        tag=tag,
        hz_line_lst=[0,4,5,17,18,19],
        size="tiny",
    )

### Dump into datacard format
def dump_datacard(histo_dict,grp_dict,card_name):
    ##lepflav_lst = histo_dict["njets"].axes.name

    # Build up a yield dict for each non-zero lep combination
    yld_dict = {}
    lepflav_lst = plt_tools.get_axis_cats(histo_dict["njets"],"lepflav")
    for lepflav in lepflav_lst:
        yld_dict_flav = get_yields_per_cat(histo_dict,"njets",grp_dict,None, lepflav_bin=lepflav)
        for cat in yld_dict_flav:
            if cat != "2lOSSF_1fjx_HFJtag_nj2_mjj600_nbm0": continue
            if yld_dict_flav[cat]["Signal"][0] == 0: continue
            yld_dict[f"{cat}_{lepflav}"] = yld_dict_flav[cat]

    # Get just the info we need for card
    yld_dict_for_card = {}
    for cat in yld_dict:
        catname_for_card = f"bin_{cat}"
        yld_dict_for_card[catname_for_card] = {}
        sig = yld_dict[cat]["Signal"][0]
        bkg = yld_dict[cat]["Background"][0]
        yld_dict_for_card[catname_for_card]["sig"] = sig
        yld_dict_for_card[catname_for_card]["bkg"] = bkg

    make_card(yld_dict_for_card,card_name)



### Get the sig and bkg yields and print or dump to json ###
def print_yields(histo_dict,grp_dict,years_to_prepend,roundat=None,print_counts=False,dump_to_json=True,quiet=False,out_name="yields", lepflavbin=None):

    # Get ahold of the yields
    yld_dict    = get_yields_per_cat(histo_dict,"njets",grp_dict,years_to_prepend, lepflavbin)
    #counts_dict = get_yields_per_cat(histo_dict,"njets_counts",grp_dict,years_to_prepend)
    #yld_dict = counts_dict

    group_lst_order = ['Signal', 'VBSWWH_SS', 'VBSWWH_OS', 'VBSWZH', 'VBSZZH', 'Background', 'QCD', 'DY', 'ttbar', 'single-t', 'rare-top', 'ttX', 'Vjets', 'VV', 'ewkV', 'ewkVV', 'VH', 'VVV', 'Data', 'Data/MC', 'metricX100'] # R2
    #group_lst_order = ["Signal", "Background", "QCD", "ttbar", "single-t", "rare-top", "ttX", "Vjets", "VV", "DY", "ewkVV", "VH", "VVV","Data"] # R3

    # Print to screen
    if not quiet:

        ### Print readably ###
        print("\n--- Yields ---")
        for cat in yld_dict:
            print(f"\n{cat}")
            for group_name in group_lst_order:
                #if group_name not in ["Signal","Background"]: continue
                if group_name not in ["Signal","Background","Data"]: continue
                if group_name == "metric": continue
                if group_name == "metricX100": continue
                if group_name == "Data/MC": continue
                yld, err = yld_dict[cat][group_name]
                perr = 100*(err/yld)
                print(f"    {group_name}:  {np.round(yld,roundat)} +- {np.round(perr,2)}%")
            #print(f"    -> Metric: {np.round(yld_dict[cat]['metric'][0],3)}")
            #print(f"    -> For copy pasting: python dump_toy_card.py {yld_dict[cat]['Signal'][0]} {yld_dict[cat]['Background'][0]}")
        #exit()


        ### Print csv, build op as an out string ###

        # Append the header
        out_str = ""
        header = "cat name"
        for proc_name in group_lst_order:
            #header = header + f", {proc_name}"
            header = header + f", {proc_name}, pm, error"
        header = header + ", metric"
        out_str = out_str + header

        # Appead a line for each category, with yields and metric
        for cat in yld_dict:
            line_str = cat
            for group_name in group_lst_order:
                if group_name == "metric": continue
                yld, err = yld_dict[cat][group_name]
                if err is not None:
                    perr = 100*(err/yld)
                    #line_str = line_str + f" , {np.round(yld,roundat)} ± {np.round(perr,2)}%"
                    line_str = line_str + f" , {np.round(yld,roundat)} , ± , {np.round(err,roundat)}"
                else:
                    line_str = line_str + f" , {np.round(yld,roundat)} , ± , None"
            # And also append the metric
            metric = yld_dict[cat]["metric"][0]
            line_str = line_str + f" , {np.round(metric,3)}"
            # Append the string for this line to the out string
            out_str = out_str + f"\n{line_str}"

        # Print the out string to the screen
        print("\n\n--- Yields CSV formatted ---\n")
        print(out_str)


    '''
    # Dump directly to json
    if dump_to_json:
        out_dict = {"yields":yld_dict, "counts":counts_dict}
        output_name = f"{out_name}.json"
        with open(output_name,"w") as out_file: json.dump(out_dict, out_file, indent=4)
        if not quiet:
            print("\n\n--- Yields json formatted ---")
            print(f"\nSaved json file: {output_name}\n")
    '''




### Make the plots ###
def make_plots(histo_dict,grp_dict,year_name_lst_to_prepend):

    #cat_lst = ["exactly1lep_exactly1fj", "exactly1lep_exactly1fj550", "exactly1lep_exactly1fj550_2j", "exactly1lep_exactly1fj_2j"]

    #grouping_dict = append_years(grp_dict,year_name_lst_to_prepend) # For fromnano
    grouping_dict = copy.deepcopy(grp_dict)

    sample_group_names_lst_mc = []
    sample_group_names_lst_bkg = []
    for grp_name in grp_dict:
        if grp_name not in ["Data"]:
            sample_group_names_lst_mc.append(grp_name)
        if grp_name not in ["Data", "Signal"]:
            sample_group_names_lst_bkg.append(grp_name)

    cat_lst = CAT_LST
    var_lst = histo_dict.keys()
    #cat_lst = ["exactly1lep_exactly1fj_STmet1000"]
    #var_lst = ["scalarptsum_lepmet"]


    for cat in cat_lst:
        print("\nCat:",cat)
        for var in var_lst:
            print("\nVar:",var)
            if "fj1" in var: continue
            #if var not in ["njets","njets_counts","scalarptsum_lepmet"]: continue # TMP
            #if "truth" in var: continue
            if sum(histo_dict[var][{"systematic":sum, "category":sum, "process":sum}].values(flow=True)) == 0: continue

            histo = copy.deepcopy(histo_dict[var][{"systematic":"nominal", "category":cat}])

            # Clean up a bit (rebin, regroup, and handle overflow)
            if var not in ["njets","nleps","nbtagsl","nbtagsm","njets_counts","nleps_counts","nfatjets","njets_forward","njets_tot","n_ll_sfos","abs_ch_sum_3l","l0_truth","l1_truth","l2_truth", "nlep_truth_real", "nlep_truth_fake", "abs_pdgid_sum"]:
                histo = plt_tools.rebin(histo,6)
            #histo = plt_tools.group(histo,"process","process_grp",grouping_dict_mc)
            histo = plt_tools.group(histo,"process","process_grp",grouping_dict)
            histo = plt_tools.merge_overflow(histo)

            histo_mc  = histo[{"process_grp":sample_group_names_lst_mc}]
            histo_sig = histo[{"process_grp":["Signal"]}]
            histo_dat = histo[{"process_grp":["Data"]}]
            #histo_dat = None
            histo_bkg = plt_tools.group(histo,"process_grp","process_grp",{"Background": sample_group_names_lst_bkg})

            # Make the figure
            title = f"{cat}__{var}"
            fig,ext_tup = make_vvh_fig(
                histo_mc = histo_mc,
                histo_mc_sig = histo_sig,
                histo_mc_bkg = histo_bkg,
                histo_dat = histo_dat,
                title=title
            )

            # Save
            save_dir_path = "plots"
            if not os.path.exists("./plots"): os.mkdir("./plots")
            save_dir_path_cat = os.path.join(save_dir_path,cat)
            if not os.path.exists(save_dir_path_cat): os.mkdir(save_dir_path_cat)
            fig.savefig(os.path.join(save_dir_path_cat,title+".png"),bbox_extra_artists=ext_tup,bbox_inches='tight')
            shutil.copyfile(HTML_PC, os.path.join(save_dir_path_cat,"index.php"))



##################################### Main #####################################

def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file_path", help = "The path to the pkl file")
    parser.add_argument('-y', "--get-yields", action='store_true', help = "Get yields from the pkl file")
    parser.add_argument('-p', "--make-plots", action='store_true', help = "Make plots from the pkl file")
    parser.add_argument('-j', "--dump-json", action='store_true', help = "Dump some yield numbers into a json file")
    parser.add_argument('-d', "--datacard", action='store_true', help = "Dump yields into datacard")
    parser.add_argument('-o', "--output-name", default='vvh', help = "What to name the outputs")
    parser.add_argument('-r', "--run", default='r2', help = "Which year")
    args = parser.parse_args()

    # Get the dictionary of histograms from the input pkl file
    histo_dict = pickle.load(gzip.open(args.pkl_file_path))
    name = args.pkl_file_path.split("/")[1][:-7] # Drops leading histos/ and trailing .pkl.gz

    # Print total raw events
    #print(sum(histo_dict["njets_counts"][{"systematic":"nominal", "category":"all_events", "process":sum, "lepflav":sum}].values(flow=True)))
    #print(histo_dict["njets"])
    #tot = histo_dict["njets"][{"systematic":"nominal", "category":"2lOSSF_1fjx", "process":sum, "njets":sum, "lepflav":22}]
    #tot = sum(sum(histo_dict["njets"][{"systematic":"nominal", "category":"3l"}].values(flow=True)))
    #tot = sum(sum(histo_dict["njets"][{"systematic":"nominal", "category":"2lOSSF_1fjx"}].values(flow=True)))
    #print("Events:",tot)
    #exit()

    # Figure out the proc naming convention
    # Assume either fully Run 2 or fully Run 3, hists with both are not supported
    proc_name = plt_tools.get_axis_cats(histo_dict["njets"],"process")[0]
    if args.run == "r2":
        grp_dict = GRP_DICT_FULL_R2
        if proc_name.startswith("UL"):
            years_to_prepend = ["UL16APV","UL16","UL17","UL18"] # Looks like TOP-22-006 convention
        else: years_to_prepend = ["2016postVFP","2016preVFP","2017","2018"] # Otherwise from RDF convention
    elif args.run == "r3":
        grp_dict = GRP_DICT_FULL_R3
        years_to_prepend = ["2024"]
    else:
        raise Exception(f"Unknown year argument {args.r}")

    # Which main functionalities to run
    if args.dump_json:
        dump_json_simple(histo_dict,args.output_name)
    if args.get_yields:
        #print_yields(histo_dict,grp_dict,years_to_prepend,out_name=args.output_name+"_yields_sig_bkg",roundat=4,print_counts=False,dump_to_json=True, lepflavbin="all")
        #e = 4
        #m = 5
        print_latex_yields(histo_dict,grp_dict, lepflav="all", tag="2l, all flav", print_end_info=False)
        print_latex_yields(histo_dict,grp_dict, lepflav=22,    tag="2l, ee only",  print_begin_info=False,print_end_info=False)
        print_latex_yields(histo_dict,grp_dict, lepflav=26,    tag="2l, mm only",  print_begin_info=False)
        #print_latex_yields(histo_dict,grp_dict, lepflav="all", tag="3l (ele cutBased$\\geq${e}, mu pfIsoId$\\geq${m}), all flav", print_end_info=False)
        #print_latex_yields(histo_dict,grp_dict, lepflav=33,    tag="3l (ele cutBased$\\geq${e}, mu pfIsoId$\\geq${m}), eee only", print_begin_info=False,print_end_info=False)
        #print_latex_yields(histo_dict,grp_dict, lepflav=35,    tag="3l (ele cutBased$\\geq${e}, mu pfIsoId$\\geq${m}), eem only", print_begin_info=False,print_end_info=False)
        #print_latex_yields(histo_dict,grp_dict, lepflav=37,    tag="3l (ele cutBased$\\geq${e}, mu pfIsoId$\\geq${m}), emm only", print_begin_info=False,print_end_info=False)
        #print_latex_yields(histo_dict,grp_dict, lepflav=39,    tag="3l (ele cutBased$\\geq${e}, mu pfIsoId$\\geq${m}), mmm only", print_begin_info=False,print_end_info=True)
    if args.datacard:
        dump_datacard(histo_dict,grp_dict,name)
    if args.make_plots:
        make_plots(histo_dict,grp_dict,years_to_prepend)



if __name__ == '__main__':
    main()

