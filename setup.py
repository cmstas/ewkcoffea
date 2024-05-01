import setuptools

setuptools.setup(
    name='ewkcoffea',
    version='0.0.0',
    description='Analysis code for EWK analyses with coffea',
    packages=setuptools.find_packages(),
    # Include data files (Note: "include_package_data=True" does not seem to work)
    package_data={
        "ewkcoffea" : [
            "params/*",
            "data/topmva_lep_sf/*root",
            "data/topmva_lep_sf/*json",
            "data/wwz_zh_bdt/*json",
            "data/btag_eff/*.pkl.gz",
            "data/run3_pu/*/*json",
            "data/run3_lep_sf/electron_sf/*/*json",
            "data/run3_lep_sf/muon_sf/*/*json",
        ],
    }
)
