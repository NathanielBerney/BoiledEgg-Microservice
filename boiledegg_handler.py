from typing import List, Dict, Any, Optional
from boiled_egg.boiled_egg import BoiledEggGIA, BoiledEggBBB
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from shapely.geometry import Point, Polygon

class BoiledEggHandler:
    AVAILABLE_PROPERTIES = ["BBB", "GIA", "TPSA", "WLogP"]

    def __init__(self):
        self.models = {"BBB": None, "GIA": None, "TPSA": None, "WLogP":None}
        self._load_models()

    def _load_models(self) ->None:
        self.models["BBB"] = Polygon(BoiledEggBBB.ml_tensor)
        self.models["GIA"] = Polygon(BoiledEggGIA.ml_tensor)
    
    def process_multiple_properties(self, smi: str, property_list: List[str]) -> Dict[List[str, Any]]:
        if not smi:
            return []
    
        # Validate property list
        valid_props = [p for p in property_list if p in self.AVAILABLE_PROPERTIES]

       
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            print(f"⚠️ Skipping unparseable SMILES: {smi}")

        tpsa = round(Descriptors.TPSA(mol, includeSandP=True), 2)
        wlogp = round(Descriptors.MolLogP(mol), 2)
        point = Point(tpsa, wlogp)
        try:
            #Temporary storage for property predictions

            batch_preds = {}
            if "BBB" in valid_props:
                batch_preds["BBB"] = point.within(self.models["BBB"])
            if "GIA" in valid_props:
                batch_preds["GIA"] = point.within(self.models["GIA"])
            if "TPSA" in valid_props:
                batch_preds["TPSA"] = tpsa
            if "WLogP" in valid_props:
                batch_preds["WLogP"] = wlogp

            res_entry ={
                "smiles": smi,
                "status": "success",
                "results": {},
                "error": None
            }
            
            for prop in valid_props:
                res_entry["results"][prop] = {
                    "property": prop,
                    "status": "success",
                    "results": float(batch_preds[prop]),
                    "error": None
                }
            return res_entry

        except Exception as e:
            # Return error objects for the batch
            return [{"smiles": smi, "status": "error", "results": {}, "error": str(e)} ]
        
    def process_multiple_properties_batch(self, smiles_list: List[str], property_list: List[str]) -> List[Dict[str, Any]]:
        res_ls = []
        for smi in smiles_list:
            one_smi_res_obj = self.process_multiple_properties(smi, property_list)
            res_ls.append(one_smi_res_obj)
        return res_ls

if __name__ == "__main__":
    handler = BoiledEggHandler()
    print(handler.process_multiple_properties("CCCOCCCOCN", ["BBB", "GIA", "TPSA", "WLogP"]))

    

        