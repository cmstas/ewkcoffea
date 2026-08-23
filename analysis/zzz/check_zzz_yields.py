import argparse
import pickle
import gzip

# Copied verbatim from analysis/vbs_vvh/check_vvh_hists.py (GRP_DICT_FULL_R2)
GRP_DICT_FULL_R2 = {
    "Signal": ["VBSWWH_SS_c2v1p0_c3_1p0", "VBSWWH_OS_c2v1p0_c3_1p0", "VBSWZH_c2v1p0_c3_1p0", "VBSZZH_c2v1p0_c3_1p0"],
    "Data": ["DoubleMuon", "MuonEG", "DoubleEG", "SingleMuon", "SingleElectron", "EGamma"],
    "QCD": ["QCD_HT50to100_TuneCP5_PSWeights_13TeV", "QCD_HT100to200_TuneCP5_PSWeights_13TeV", "QCD_HT200to300_TuneCP5_PSWeights_13TeV", "QCD_HT300to500_TuneCP5_PSWeights_13TeV", "QCD_HT500to700_TuneCP5_PSWeights_13TeV", "QCD_HT700to1000_TuneCP5_PSWeights_13TeV", "QCD_HT1000to1500_TuneCP5_PSWeights_13TeV", "QCD_HT1500to2000_TuneCP5_PSWeights_13TeV", "QCD_HT2000toInf_TuneCP5_PSWeights_13TeV"],
    "ttbar": ["TTTo2L2Nu_TuneCP5_13TeV", "TTToSemiLeptonic_TuneCP5_13TeV", "TTToHadronic_TuneCP5_13TeV"],
    "single-t": ["ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV", "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV", "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV", "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV", "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV"],
    "ttX": ["TTWJetsToLNu_TuneCP5_13TeV", "TTWJetsToQQ_TuneCP5_13TeV", "TTbb_4f_TTTo2L2Nu", "TTbb_4f_TTToSemiLeptonic", "TTbb_4f_TTToHadronic", "ttHToNonbb_M125_TuneCP5_13TeV", "ttHTobb_M125_TuneCP5_13TeV", "ttWJets_TuneCP5_13TeV", "ttZJets_TuneCP5_13TeV"],
    "rare-top": ["TWZToLL_tlept_Wlept_5f_DR_TuneCP5_13TeV", "TTWZ_TuneCP5_13TeV", "TTWW_TuneCP5_13TeV", "tZq_ll_4f_ckm_NLO_TuneCP5_13TeV"],
    "Vjets": ["WJetsToLNu_HT-70To100_TuneCP5_13TeV", "WJetsToLNu_HT-100To200_TuneCP5_13TeV", "WJetsToLNu_HT-200To400_TuneCP5_13TeV", "WJetsToLNu_HT-400To600_TuneCP5_13TeV", "WJetsToLNu_HT-600To800_TuneCP5_13TeV", "WJetsToLNu_HT-800To1200_TuneCP5_13TeV", "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV", "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV", "WJetsToLNu_TuneCP5_13TeV", "WJetsToQQ_HT-200to400_TuneCP5_13TeV", "WJetsToQQ_HT-400to600_TuneCP5_13TeV", "WJetsToQQ_HT-600to800_TuneCP5_13TeV", "WJetsToQQ_HT-800toInf_TuneCP5_13TeV", "ZJetsToQQ_HT-200to400_TuneCP5_13TeV", "ZJetsToQQ_HT-400to600_TuneCP5_13TeV", "ZJetsToQQ_HT-600to800_TuneCP5_13TeV", "ZJetsToQQ_HT-800toInf_TuneCP5_13TeV", "WminusH_HToBB_WToLNu_M-125_TuneCP5_13TeV", "WminusH_HToBB_WToQQ_M-125_TuneCP5_13TeV", "WplusH_HToBB_WToLNu_M-125_TuneCP5_13TeV", "WplusH_HToBB_WToQQ_M-125_TuneCP5_13TeV"],
    "DY": ["DYJetsToLL_M-10to50_TuneCP5_13TeV", "DYJetsToLL_M-50_TuneCP5_13TeV"],
    "ewkV": ["EWKWMinus2Jets_WToLNu_M-50_TuneCP5_withDipoleRecoil_13TeV", "EWKWPlus2Jets_WToLNu_M-50_TuneCP5_withDipoleRecoil_13TeV", "EWKWminus2Jets_WToQQ_dipoleRecoilOn_TuneCP5_13TeV", "EWKWplus2Jets_WToQQ_dipoleRecoilOn_TuneCP5_13TeV", "EWKZ2Jets_ZToLL_M-50_TuneCP5_withDipoleRecoil_13TeV", "EWKZ2Jets_ZToNuNu_M-50_TuneCP5_withDipoleRecoil_13TeV", "EWKZ2Jets_ZToQQ_dipoleRecoilOn_TuneCP5_13TeV", "WWJJToLNuLNu_EWK_noTop_TuneCP5_13TeV"],
    # All VV (diboson) except ZZ -- WW, WZ, and the ZZJJ-EWK/GluGluZH oddments already in the
    # original grouping. Like "ZZ" below, only the exclusive-decay samples are used, not the
    # inclusive WW_TuneCP5_13TeV/WZ_TuneCP5_13TeV -- see EXCLUDED_OVERLAP.
    "VV-non-ZZ": ["GluGluZH_HToWWTo2L2Nu_TuneCP5_13TeV", "GluGluZH_HToWWTo2L2Nu_M-125_TuneCP5_13TeV", "WZJJ_EWK_InclusivePolarization_TuneCP5_13TeV", "WZTo1L1Nu2Q_4f_TuneCP5_13TeV", "WZTo1L3Nu_4f_TuneCP5_13TeV", "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV", "WZTo3LNu_TuneCP5_13TeV", "WWTo4Q_4f_TuneCP5_13TeV", "WWTo1L1Nu2Q_4f_TuneCP5_13TeV", "WWTo2L2Nu_TuneCP5_13TeV", "GluGluHToZZTo4L", "ZZTo2Nu2Q_5f_TuneCP5_13TeV", "ZZTo4Q_5f_TuneCP5_13TeV", "ZZJJTo4L_EWKnotop_TuneCP5_13TeV", "ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV"],
    # Standalone ZZ category: qq-initiated exclusive-decay samples (ZZTo4L, ZZTo2L2Nu -- the
    # only ZZ decay modes actually present in this sample set) plus the gg-initiated loop
    # continuum (a separate production mechanism, so no overlap with the qq-initiated ones).
    # The fully-inclusive "ZZ_TuneCP5_13TeV" sample is deliberately NOT included here since its
    # phase space is a superset of ZZTo4L/ZZTo2L2Nu -- see EXCLUDED_OVERLAP below.
    "ZZ": ["ZZTo4L_M-1toInf_TuneCP5_13TeV", "ZZTo2L2Nu_TuneCP5_13TeV",
           "GluGluToContinToZZTo2e2mu_TuneCP5_13TeV", "GluGluToContinToZZTo2e2tau_TuneCP5_13TeV",
           "GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV", "GluGluToContinToZZTo4e_TuneCP5_13TeV",
           "GluGluToContinToZZTo4mu_TuneCP5_13TeV", "GluGluToContinToZZTo4tau_TuneCP5_13TeV"],
    "ewkVV": ["SSWW"],
    "VH": ["VBFWH_HToBB_WToLNu_M-125_dipoleRecoilOn_TuneCP5_13TeV", "VHToNonbb_M125_TuneCP5_13TeV", "ggZH_HToBB_ZToLL_M-125_TuneCP5_13TeV", "ggZH_HToBB_ZToBB_M-125_TuneCP5_13TeV", "ggZH_HToBB_ZToNuNu_M-125_TuneCP5_13TeV", "ggZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV", "ZH_HToBB_ZToLL_M-125_TuneCP5_13TeV", "ZH_HToBB_ZToBB_M-125_TuneCP5_13TeV", "ZH_HToBB_ZToNuNu_M-125_TuneCP5_13TeV", "ZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV", "HZJ_HToWWTo2L2Nu_ZTo2L_M-125_TuneCP5_13TeV"],
    "VVV": ["WWW_4F_TuneCP5_13TeV", "WWZ_4F_TuneCP5_13TeV", "WZZ_TuneCP5_13TeV", "ZZZ_TuneCP5_13TeV"],
}

# Samples deliberately left out of every group above because they'd double-count phase
# space already covered by another sample we chose to use instead. Kept separate from
# "unmatched" so it's clear these are an intentional choice, not an oversight.
EXCLUDED_OVERLAP = {
    "ZZ_TuneCP5_13TeV": "inclusive ZZ (all decays) -- superseded by ZZTo4L_M-1toInf + ZZTo2L2Nu in the 'ZZ' group",
    "WW_TuneCP5_13TeV": "inclusive WW (all decays) -- superseded by WWTo2L2Nu in the 'VV-non-ZZ' group",
    "WZ_TuneCP5_13TeV": "inclusive WZ (all decays) -- superseded by WZTo3LNu in the 'VV-non-ZZ' group",
    "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV": "HT-binned M-50 DY -- superseded by inclusive DYJetsToLL_M-50 in the 'DY' group",
    "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV": "HT-binned M-50 DY -- superseded by inclusive DYJetsToLL_M-50 in the 'DY' group",
    "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV": "HT-binned M-50 DY -- superseded by inclusive DYJetsToLL_M-50 in the 'DY' group",
    "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV": "HT-binned M-50 DY -- superseded by inclusive DYJetsToLL_M-50 in the 'DY' group",
    "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV": "HT-binned M-50 DY -- superseded by inclusive DYJetsToLL_M-50 in the 'DY' group",
    "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV": "HT-binned M-50 DY -- superseded by inclusive DYJetsToLL_M-50 in the 'DY' group",
    "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV": "HT-binned M-50 DY -- superseded by inclusive DYJetsToLL_M-50 in the 'DY' group",
    "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV": "HT-binned M-50 DY -- superseded by inclusive DYJetsToLL_M-50 in the 'DY' group",
}

parser = argparse.ArgumentParser()
parser.add_argument("pkl_file", help="Path to the coffea output .pkl or .pkl.gz file")
parser.add_argument("--var", default="njets", help="Which histogram to use for yields")
parser.add_argument("--split", nargs="+", default=["VVV"], help="Group name(s) to break down by individual process instead of summing (VVV split by default)")
args = parser.parse_args()

opener = gzip.open if args.pkl_file.endswith(".gz") else open
with opener(args.pkl_file, "rb") as f:
    histo_dict = pickle.load(f)

h = histo_dict[args.var]
sel = {"systematic": "nominal"} if "systematic" in h.axes.name else {}

procs_in_hist = list(h.axes["process"]) if "process" in h.axes.name else None
if procs_in_hist is not None:
    matched = {p for grp in GRP_DICT_FULL_R2.values() for p in grp} | set(EXCLUDED_OVERLAP)
    unmatched = [p for p in procs_in_hist if p not in matched]

print(f"\n--- Yields ({args.var}) ---")
for cat in h.axes["category"]:
    h_cat = h[{**sel, "category": cat}]
    yld = h_cat.values(flow=True).sum()
    print(f"{cat:40s} {yld:.3f}")

    if procs_in_hist is not None:
        for grp_name, grp_procs in GRP_DICT_FULL_R2.items():
            procs_here = [p for p in grp_procs if p in procs_in_hist]
            if not procs_here:
                continue
            if grp_name in args.split:
                for p in procs_here:
                    p_yld = h_cat[{"process": p}].values(flow=True).sum()
                    print(f"    {grp_name + '/' + p:40s} {p_yld:.3f}")
            else:
                grp_yld = h_cat[{"process": procs_here}].values(flow=True).sum()
                print(f"    {grp_name:20s} {grp_yld:.3f}")
        if unmatched:
            other_yld = h_cat[{"process": unmatched}].values(flow=True).sum()
            print(f"    {'Other/unmatched':20s} {other_yld:.3f}")

if procs_in_hist is not None and unmatched:
    print(f"\n[Note: {len(unmatched)} process(es) in this file aren't in GRP_DICT_FULL_R2, lumped into 'Other/unmatched':]")
    for p in unmatched:
        print(f"    {p}")

excluded_here = [p for p in EXCLUDED_OVERLAP if procs_in_hist and p in procs_in_hist]
if excluded_here:
    print(f"\n[Note: {len(excluded_here)} process(es) deliberately excluded to avoid double-counting overlapping phase space:]")
    for p in excluded_here:
        print(f"    {p}: {EXCLUDED_OVERLAP[p]}")
