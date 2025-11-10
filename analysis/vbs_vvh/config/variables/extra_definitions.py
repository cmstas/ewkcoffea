
import numpy as np
import awkward as ak

def VV_final_state(events):
    """Define per-event final state categories from daughter PDG IDs."""
    
    # Helper function
    def is_plep(pid):  return (pid == 11) | (pid == 13) | (pid == 15) | (pid == 17)
    def is_nlep(pid):  return (pid == -11) | (pid == -13) | (pid == -15) | (pid == -17)
    def is_had(pid):   return (abs(pid) <= 9) | (pid == 21)
    def is_MET(pid):   return (abs(pid) == 12) | (abs(pid) == 14) | (abs(pid) == 16)

    # Build all daughter-level flags
    daughters = ["tV1d1_id", "tV1d2_id", "tV2d1_id", "tV2d2_id"]
    plep = [is_plep(events[name]) for name in daughters]
    nlep = [is_nlep(events[name]) for name in daughters]
    had  = [is_had(events[name])  for name in daughters]
    met  = [is_MET(events[name])  for name in daughters]

    # Count per-event
    n_plep = sum(plep)
    n_nlep = sum(nlep)
    n_had  = sum(had)
    n_MET  = sum(met)
    n_lep  = n_plep + n_nlep

    # Define categories
    had_2lep   = (n_had == 2) & (n_lep == 2) & (n_MET == 0)
    had_1lep   = (n_had == 2) & (n_lep == 1) & (n_MET == 1)
    had_MET    = (n_had == 2) & (n_lep == 0) & (n_MET == 2)
    total_MET  = (n_MET == 4)
    lep_3      = (n_lep == 3) & (n_had == 0) & (n_MET == 1)
    lep_4      = (n_lep == 4) & (n_had == 0) & (n_MET == 0)
    nohad_oslep= (n_plep == 1) & (n_nlep == 1) & (n_MET == 2)
    ZZ_oslep   = nohad_oslep & (abs(events.tV1_id) == 23)
    WW_oslep   = nohad_oslep & (abs(events.tV1_id) == 24)
    WW_sslep   = ((n_plep == 2) | (n_nlep == 2)) & (n_MET == 2)
    MET_1lep   = (n_MET == 3) & (n_lep == 1)

    return {
        "had_2lep": had_2lep,
        "had_1lep": had_1lep,
        "had_MET": had_MET,
        "total_MET": total_MET,
        "lep_3": lep_3,
        "lep_4": lep_4,
        "nohad_oslep": nohad_oslep,
        "ZZ_oslep": ZZ_oslep,
        "WW_oslep": WW_oslep,
        "WW_sslep": WW_sslep,
        "MET_1lep": MET_1lep,
        "n_had": n_had,
        "n_lep": n_lep,
        "n_MET": n_MET,
    }

def Higgs_final_state(events):
    return 0


def VV_ndaughters(events):
    """Define per-event final state categories from daughter PDG IDs."""
    
    # Helper function
    def is_plep(pid):  return (pid == 11) | (pid == 13) | (pid == 15) | (pid == 17)
    def is_nlep(pid):  return (pid == -11) | (pid == -13) | (pid == -15) | (pid == -17)
    def is_had(pid):   return (abs(pid) <= 9) | (pid == 21)
    def is_MET(pid):   return (abs(pid) == 12) | (abs(pid) == 14) | (abs(pid) == 16)

    # Build all daughter-level flags
    daughters = ["tV1d1_id", "tV1d2_id", "tV2d1_id", "tV2d2_id"]
    plep = [is_plep(events[name]) for name in daughters]
    nlep = [is_nlep(events[name]) for name in daughters]
    had  = [is_had(events[name])  for name in daughters]
    met  = [is_MET(events[name])  for name in daughters]

    # Count per-event
    n_plep = sum(plep)
    n_nlep = sum(nlep)
    n_had  = sum(had)
    n_MET  = sum(met)
    n_lep  = n_plep + n_nlep

    return {
        "n_plep" :n_plep,
        "n_nlep" :n_nlep,
        "n_had"  :n_had,
        "n_MET"  :n_MET
    }
