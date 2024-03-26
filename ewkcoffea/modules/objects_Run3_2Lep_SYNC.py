import awkward as ak
from ewkcoffea.modules.paths import ewkcoffea_path
from topcoffea.modules.get_param_from_jsons import GetParam as get_param
get_param = get_param(ewkcoffea_path("params/params.json"))

#def get_cleaned_collection(obj_collection_a,obj_collection_b,drcut=0.4):
#    obj_b_nearest_to_any_in_a , dr = obj_collection_b.nearest(obj_collection_a,return_metric=True)
#    mask = ak.fill_none(dr>drcut,True)
#    return mask

def is_tight_Run3_2Lep_ele(ele):
    mask = (
        (ele.pt                                >  15.0) &
        (abs(ele.eta)                          <  2.5) &
        (abs(ele.dxy)                          <  0.05) &
        (abs(ele.dz)                           <  0.1) &
        (ele.mvaIso_WP80)
    )
    return mask

def is_tight_Run3_2Lep_mu(mu):
    mask = (
        (mu.pt               >  10.0) &
        (abs(mu.eta)         <  2.4) &
        (abs(mu.dxy)         <  0.05) &
        (abs(mu.dz)          <  0.1) &
        (mu.pfIsoId          >=  4) &
        (mu.mediumId)
    )
    return mask

def is_presel_Run3_2Lep_jets(jets):
    mask = (
        (jets.pt               >  30.0) &
        (abs(jets.eta)         <  2.4)  &
        (jets.jetId            >= 4.0)
    )
    return mask
