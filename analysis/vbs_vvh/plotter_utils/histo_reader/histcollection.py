# variable view ( sig+bkg+data per var )

from dataclasses import dataclass
from typing import Dict, List, Mapping


@dataclass
class HistCollection:
    views: Dict[str, Dict[str, object]]

    variables_sig_only: List[str]
    variables_mc_only: List[str]
    variables_all: List[str]
    proc_map: Mapping

    def has_data(self, var):
        return self.views[var]["data"] is not None