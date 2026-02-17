import awkward as ak
from coffea.nanoevents.methods import vector
from hist import axis

ak.behavior.update(vector.behavior)
import numpy as np


def _select_by_score(good, field, has_jet, best_idx):
    return ak.fill_none(ak.where(has_jet, ak.firsts(good[field][best_idx]), None), -999)


obj = {
    "goodAK8Jets": lambda events: (
        lambda mask=(
            (events.fatjet_pt > 250)
            & (np.abs(events.fatjet_eta) <= 2.5)
            & (events.fatjet_msoftdrop > 40)
            & (events.fatjet_jetId > 0)
        ): ak.zip(
            {
                "pt": events.fatjet_pt[mask],
                "eta": events.fatjet_eta[mask],
                "phi": events.fatjet_phi[mask],
                "mass": events.fatjet_msoftdrop[mask],
                "HbbScore": (
                    events.fatjet_particleNetLegacy_Xbb[
                        mask
                    ]  # no MD? maybe default MD because we also have particleNetWithMass branches
                    / (
                        events.fatjet_particleNetLegacy_Xbb[mask]
                        + events.fatjet_particleNet_QCD[mask]
                    )
                ),
                "WqqScore": (
                    (
                        events.fatjet_particleNetLegacy_Xcc[mask]
                        + events.fatjet_particleNetLegacy_Xqq[mask]
                    )  # need to check this
                    / (
                        events.fatjet_particleNetLegacy_Xcc[mask]
                        + events.fatjet_particleNetLegacy_Xqq[mask]
                        + events.fatjet_particleNetLegacy_QCD[mask]
                    )
                ),
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )
    )(),
    "MET": lambda events: ak.zip(
        {
            "pt": events.PuppiMET_pt,
            "phi": events.PuppiMET_phi,
            "eta": ak.zeros_like(
                events.PuppiMET_pt
            ),  # Set to 0 to ensure valid LorentzVector
            "mass": ak.zeros_like(events.PuppiMET_pt),  # Same for mass
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    ),
}

other_objs = {  # reconstruction of all (good) jets and stuffs if needed
    "Higgs": lambda events, objects: (
    lambda good=objects["goodAK8Jets"],
           best_idx=ak.argmax(objects["goodAK8Jets"].HbbScore, axis=1, keepdims=True): ak.zip(
        {
            "pt":        ak.fill_none(ak.firsts(good.pt[best_idx]),        -999),
            "eta":       ak.fill_none(ak.firsts(good.eta[best_idx]),       -999),
            "phi":       ak.fill_none(ak.firsts(good.phi[best_idx]),       -999),
            "mass":      ak.fill_none(ak.firsts(good.mass[best_idx]),      -999),
            "score":     ak.fill_none(ak.firsts(good.HbbScore[best_idx]),  -999),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    )
)(),
    # "leadAK4": lambda events,objects: get_leading_jet(objects['goodAK4Jets']),
}


def get_leading_jet(jets):
    jets_sorted = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
    return ak.firsts(jets_sorted)


def get_len(arr):
    return ak.num(arr, axis=1)


dense_variables_config = {  # name of axis must be same as key
    "nGoodAK8": {
        "axis": axis.Regular(6, 0, 6, name="nGoodAK8", label="nGoodAK8"),
        "expr": lambda events, objects: get_len(objects["goodAK8Jets"].pt),
    },
    "Higgs_pt": {
        "axis": axis.Regular(50, 0, 2000, name="Higgs_pt", label="Higgs pt (GeV)"),
        "expr": lambda events, objects: objects["Higgs"].pt,
    },
    "Higgs_phi": {
        "axis": axis.Regular(50, -3.5, 3.5, name="Higgs_phi", label="Higgs phi"),
        "expr": lambda events, objects: objects["Higgs"].phi,
    },
    "Higgs_eta": {
        "axis": axis.Regular(50, 0, 3, name="Higgs_eta", label="Higgs eta"),
        "expr": lambda events, objects: objects["Higgs"].eta,
    },
    "Higgs_mass": {
        "axis": axis.Regular(50, 0, 400, name="Higgs_mass", label="Higgs mass (GeV)"),
        "expr": lambda events, objects: objects["Higgs"].mass,
    },
    "Higgs_score": {
        "axis": axis.Regular(50, 0, 1, name="Higgs_score", label="Higgs score"),
        "expr": lambda events, objects: objects["Higgs"].score,
    },
    "Met_pt": {
        "axis": axis.Regular(50, 0, 2000, name="Met_pt", label="MET pt (GeV)"),
        "expr": lambda events, objects: objects["MET"].pt,
    },
    "Met_pt_low": {
        "axis": axis.Regular(80, 0, 800, name="Met_pt_low", label="MET pt (GeV)"),
        "expr": lambda events, objects: objects["MET"].pt,
    },
}


def get_min(array):
    arr_sort = ak.pad_none(ak.sort(array, axis=1, ascending=True), 1, axis=1)
    return arr_sort[:, 0]


# seems delta_r is the only available function for coffea?
def deltaR(v1, v2):
    return v1.delta_r(v2)


def min_dR(v1, v2):
    dR = []
    for i in range(ak.num(v1, axis=0)):
        if ak.num(v1, axis=1) == 0 or ak.num(v2, axis=1) == 0:
            dR.append(None)
        else:
            k1, k2 = ak.unzip(ak.cartesian([v1, v2]))
            dR.append(get_min(k1.deltaR(k2)))
    return ak.flatten(dR)


def deltaPhi(v1, v2):
    phi1 = v1.phi
    phi2 = v2.phi
    abs_diff = np.abs(phi1 - phi2)
    dphi = ak.where(
        abs_diff < np.pi, abs_diff, 2 * np.pi - abs_diff
    )  # compare element-wise
    return dphi


def deltaPhi_1d(phi1, phi2):
    abs_diff = np.abs(phi1 - phi2)
    dphi = ak.where(
        abs_diff < np.pi, abs_diff, 2 * np.pi - abs_diff
    )  # compare element-wise
    return dphi


def deltaEta(v1, v2):
    eta1 = v1.eta
    eta2 = v2.eta
    return np.abs(eta1 - eta2)
