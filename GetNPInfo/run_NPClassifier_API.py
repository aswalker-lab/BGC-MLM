# -*- coding: utf-8 -*-
"""
Created on Wed May 13 21:33:51 2026

@author: Allison Walker
"""

import argparse
import time
import json
from typing import Any, Dict, Optional
from pathlib import Path
import pandas as pd
import requests


NPCLASSIFIER_URL = "https://npclassifier.gnps2.org/classify"

#change the in and output files to training or test datafiles
infile = open("test_mibig_20_40_smiles.txt")
outfile = open("test_mibig_20_40_smiles_npclassifier.txt",'w')


BGCcat_classes = ["Spirotetronate macrolides", "Aflatoxins", "Simple coumarins", "Methyl xanthones", "Open-chain polyketides", "Isocoumarins", "Polyene macrolides", "Tropane alkaloids", "Tylosins", "Ansa macrolides", "Macrolide lactones", "4-pyrone derivatives", "Orthosomycins", "Azaphilones", "Bafilomycins", "Macrolide lactams", "Naphthoquinones", "Erythromycins", "Polyether ionophores", "Enediynes", "Isoindole alkaloids", "Monacolins and Monacolin derivatives", "Zearalenones", "Fatty alcohols", "Anthraquinones and anthrones", "Depsidones", "Simple phenolic acids", "Phoslactomycins or Phosphazomycins", "Simple tetramate alkaloids", "Griseofulvins", "Lactam bearing macrolide lactones", "Cyclic peptides", "Pyridine alkaloids", "Piperidine alkaloids", "Prodigiosins", "Avermectins", "Benzophenones", "Phthalide derivatives", "Branched fatty acids", "Aminosugars", "Oligomycins", "Miscellaneous polyketides", "Fungal DPEs", "2-pyrone derivatives", "Pyrrole alkaloids", "Triketide meroterpenoids", "3-acyl tetramic acids", "Salinosporamides", "Decalins with 2-pyrones", "Chromones", "Tetracyclines", "Linear tetronates", "Other polyketide meroterpenoids", "Merosesquiterpenoids", "Cinnamic acids and derivatives", "Bryostatins", "Boromycins", "Linear polyenes", "Saxitoxins", "Simple indole alkaloids", "Anthracyclines", "Bisnaphthalenes", "Angucyclines", "Ericamycins", "Benastatins and derivatives", "Macrotetrolides", "Fasamycins and derivatives", "Acyl phloroglucinols", "Prenyl quinone meroterpenoids", "Amino cyclitols", "Pimarane and Isopimarane diterpenoids", "Aminoacids", "Pyrimidine nucleos(t)ides", "Vancomycins and Teicoplanins", "Thiodiketopiperazine alkaloids", "Other indole diketopiperazine alkaloids", "Imidazole alkaloids", "Actinomycins", "Aeruginosins", "Linear peptides", "Lipopeptides", "Depsipeptides", "Anabaenopeptins", "Pyrazine and Piperazine alkaloids", "Cephalosporins", "Cephamycins", "Indole diketopiperazine alkaloids (L-Trp, L-Ala)", "Tripeptides", "Simple amide alkaloids", "Ahp-containing cyclodepsipeptides", "Peptaibols", "p-Terphenyls", "Dipeptides", "Ergot alkaloids", "Isoquinoline alkaloids", "Ascomycins and Rapamycins", "Quinazoline alkaloids", "Azo and Azoxy alkaloids", "Tetrahydroisoquinoline alkaloids", "Monocyclic \u03b2-lactams", "Microcystins", "Penicillins", "Benzodiazepine alkaloids", "Thiazole alkaloids", "Mycosporine and Mycosporine-like amino acids", "Aminoglycosides", "Streptothricins and derivatives", "Carbazole alkaloids", "Polyamines", "RiPPs-Bottromycins", "RiPPs-Cyanobactins", "RiPPs-Lasso peptides", "RiPPs-Lanthipeptides", "RiPPs-Microcins", "RiPPs-Thiopeptides", "Carotenoids (C40, \u03b2-\u03b2)", "Botryane sesquiterpenoids", "Presilphiperfolane and Probotryane sesquiterpenoids", "Lactarane sesquiterpenoids", "Friedelane triterpenoids", "Pentalenane sesquiterpenoids", "Camphane monoterpenoids", "Zizaane sesquiterpenoids", "Phenoxazine alkaloids", "Hopane and Moretane triterpenoids", "Carotenoids (C40, \u03c0-\u03c0)", "Indole-Diterpenoid alkaloids (Penitrems)", "Norkaurane diterpenoids", "Eremophilane sesquiterpenoids", "Hapalindole alkaloids", "Polypodane triterpenoids", "Malabaricane triterpenoids", "Cycloartane triterpenoids", "Lanostane, Tirucallane and Euphane triterpenoids", "Norpimarane and Norisopimarane diterpenoids", "Cassane diterpenoids", "Kaurane and Phyllocladane diterpenoids", "Cadinane sesquiterpenoids", "Aphidicolane diterpenoids", "Tetracyclic diterpenoids", "Fusicoccane diterpenoids", "Quinoline alkaloids", "Tetraketide meroterpenoids", "Cycloeudesmane sesquiterpenoids", "Fusidane triterpenoids", "Labdane diterpenoids", "Monosaccharides", "Polysaccharides", "Glycerophosphoinositolglycans", "Cyanogenic glycosides", "Bactoprenols", "Pyrrolizidine alkaloids", "Carboline alkaloids", "Indole diketopiperazine alkaloids (L-Trp, L-Pro)", "Pyrroloindole alkaloids", "Catechols with side chains", "Clavams", "Carbapenems", "Lactones", "Simple diketopiperazine alkaloids", "Unsaturated fatty acids", "Glycerophosphoethanolamines", "Triacylglycerols", "Purine nucleos(t)ides", "Fatty acyl CoAs", "Bagremycins", "Shikimic acids and derivatives", "Marine-bacterial DPEs", "pteridine alkaloids", "Trichothecane sesquiterpenoids", "Tropolones and derivatives (Shikimate)", "Phenazine alkaloids", "N-acyl amines", "Neutral glycosphingolipids", "Antimycins", "Bleomycins", "Cytochalasan alkaloids", "Cryptophycins", "Cyclopiazonic acid-tpye tetramate alkaloids", "Tricyclic guanidine alkaloids", "Melithiazole and Myxothiazole derivatives", "DKXanthenes and derivatives", "Epothilones", "Macrocyclic tetramic acids", "3-Spirotetramic acids", "3-oligoenoyltetramic acids", "Sphingoid bases", "Dialkylresorcinols", "Elfamycins", "Miscellaneous meroterpenoids", "Cembrane diterpenoids", "Rhizoxins", "Streptogramins", "Platensimycin and Platencins", "Miscellaneous alkaloids", "Hydrocarbons", "Eudesmane sesquiterpenoids", "Noreudesmane sesquiterpenoids", "Colensane and Clerodane diterpenoids", "Aspidosperma type", "Benzoquinones", "Pyrrolidine alkaloids", "3-Decalinoyltetramic acids", "Naphthalenes and derivatives", "Decalins with side chains", "Fatty acyl glycosides of mono- and disaccharides", "Dicarboxylic acids", "Indolizidine alkaloids", "Flavanones", "Cucurbitane triterpenoids", "Lupane triterpenoids", "Santalane sesquiterpenoids", "Monocyclic monoterpenoids", "Nonadrides", "Furans", "Thapsane sesquiterpenoids", "Pradimicins", "Sorbicilinoids", "Acetate-derived alkaloids", "Long-Chain Bicyclic Phosphotriester", "Daucane sesquiterpenoids", "Acorane sesquiterpenoids", "Drimane sesquiterpenoids", "Betaestacin-type sesterterpenoids", "Flavonols", "Androstane steroids", "Duclauxin and derivatives", "Gibberellins", "Corynanthe type", "Africanane sesquiterpenoids", "Mangicol-type sesterterpenoids", "Pulvinones", "Ceramides", "Indole diketopiperazine alkaloids (L-Trp, L-Trp)", "Paulomycins and derivatives", "Tremulane sesquiterpenoids", "Tropolones and derivatives (PKS)", "Pyrroloquinoline alkaloids", "Morphinan alkaloids", "Humulane sesquiterpenoids", "Oxa-Bridged Macrolides", "Apocarotenoids(\u03b5-)", "Strobilurins and derivatives", "Monoalkylresorcinols", "Purine alkaloids", "Paraconic acids and derivatives", "RiPPs-Amatoxins and Phallotoxins", "Carbocyclic fatty acids", "Naphthalenones", "Dolabellane diterpenoids", "Phenylethanoids", "Longibornane sesquiterpenoids", "Acyclic monoterpenoids", "Cyathane diterpenoids", "Pyrrocidine tetramate alkaloids", "Caryophyllane sesquiterpenoids", "Steroidal alkaloids", "Merohemiterpenoids", "Protoilludane sesquiterpenoids", "Acetogenins", "Neolignans", "Phenylalanine-derived alkaloids", "Cyclopiane diterpenoids", "Pimprinine alkaloids", "Heterocyclic fatty acids", "Chalcones", "Bicyclogermacrane sesquiterpenoids", "Casbane diterpenoids", "Taxane diterpenoids", "Verticillane diterpenoids", "Abietane diterpenoids", "Cleistanthane diterpenoids", "Oleanane triterpenoids", "Fatty acid estolides", "Usnic acid and derivatives", "Bisabolane sesquiterpenoids", "Hydroxy fatty acids", "Silphinane sesquiterpenoids", "Depsides", "Valparane diterpenoids", "Microginins", "Acridone alkaloids", "Breviane diterpenoids", "Caryolane sesquiterpenoids", "Germacrane sesquiterpenoids", "Baccharane triterpenoids", "Simple aromatic polyketides", "Sativane sesquiterpenoids", "Copacamphane sesquiterpenoids", "Aromadendrane sesquiterpenoids", "Cericerane sesterterpenoids", "Oxo fatty acids", "Hydroxy-hydroperoxyeicosapentaenoic acids", "Other Docosanoids"]
#underrepresented = 2 or fewer in training set
underrepresented_classes = ['Aflatoxins', 'Methyl xanthones', 'Tropane alkaloids', '4-pyrone derivatives', 'Orthosomycins', 'Azaphilones', 'Isoindole alkaloids', 'Monacolins and Monacolin derivatives', 'Fatty alcohols', 'Phoslactomycins or Phosphazomycins', 'Simple tetramate alkaloids', 'Griseofulvins', 'Benzophenones', 'Phthalide derivatives', 'Miscellaneous polyketides', 'Fungal DPEs', 'Triketide meroterpenoids', 'Salinosporamides', 'Decalins with 2-pyrones', 'Chromones', 'Linear tetronates', 'Other polyketide meroterpenoids', 'Merosesquiterpenoids', 'Bryostatins', 'Linear polyenes', 'Bisnaphthalenes', 'Acyl phloroglucinols', 'Thiodiketopiperazine alkaloids', 'Other indole diketopiperazine alkaloids', 'Actinomycins', 'Aeruginosins', 'Cephamycins', 'Indole diketopiperazine alkaloids (L-Trp, L-Ala)', 'Ahp-containing cyclodepsipeptides', 'Peptaibols', 'p-Terphenyls', 'Ergot alkaloids', 'Isoquinoline alkaloids', 'Quinazoline alkaloids', 'Tetrahydroisoquinoline alkaloids', 'Monocyclic β-lactams', 'Penicillins', 'Mycosporine and Mycosporine-like amino acids', 'Carotenoids (C40, β-β)', 'Botryane sesquiterpenoids', 'Presilphiperfolane and Probotryane sesquiterpenoids', 'Lactarane sesquiterpenoids', 'Friedelane triterpenoids', 'Pentalenane sesquiterpenoids', 'Camphane monoterpenoids', 'Zizaane sesquiterpenoids', 'Phenoxazine alkaloids', 'Hopane and Moretane triterpenoids', 'Carotenoids (C40, π-π)', 'Indole-Diterpenoid alkaloids (Penitrems)', 'Norkaurane diterpenoids', 'Eremophilane sesquiterpenoids', 'Polypodane triterpenoids', 'Malabaricane triterpenoids', 'Cycloartane triterpenoids', 'Lanostane, Tirucallane and Euphane triterpenoids', 'Norpimarane and Norisopimarane diterpenoids', 'Cassane diterpenoids', 'Kaurane and Phyllocladane diterpenoids', 'Cadinane sesquiterpenoids', 'Aphidicolane diterpenoids', 'Tetracyclic diterpenoids', 'Fusicoccane diterpenoids', 'Tetraketide meroterpenoids', 'Cycloeudesmane sesquiterpenoids', 'Fusidane triterpenoids', 'Monosaccharides', 'Cyanogenic glycosides', 'Bactoprenols', 'Indole diketopiperazine alkaloids (L-Trp, L-Pro)', 'Pyrroloindole alkaloids', 'Clavams', 'Carbapenems', 'Lactones', 'Glycerophosphoethanolamines', 'Triacylglycerols', 'Fatty acyl CoAs', 'Trichothecane sesquiterpenoids', 'Tropolones and derivatives (Shikimate)', 'Neutral glycosphingolipids', 'Cytochalasan alkaloids', 'Cryptophycins', 'Cyclopiazonic acid-tpye tetramate alkaloids', 'Tricyclic guanidine alkaloids', 'DKXanthenes and derivatives', 'Epothilones', '3-Spirotetramic acids', 'Elfamycins', 'Miscellaneous meroterpenoids', 'Cembrane diterpenoids', 'Rhizoxins', 'Streptogramins', 'Miscellaneous alkaloids', 'Eudesmane sesquiterpenoids', 'Noreudesmane sesquiterpenoids', 'Colensane and Clerodane diterpenoids', 'Aspidosperma type', 'Benzoquinones', 'Pyrrolidine alkaloids', '3-Decalinoyltetramic acids', 'Naphthalenes and derivatives', 'Decalins with side chains', 'Fatty acyl glycosides of mono- and disaccharides', 'Dicarboxylic acids', 'Indolizidine alkaloids', 'Flavanones', 'Cucurbitane triterpenoids', 'Lupane triterpenoids', 'Santalane sesquiterpenoids', 'Monocyclic monoterpenoids', 'Nonadrides', 'Furans', 'Thapsane sesquiterpenoids', 'Pradimicins', 'Sorbicilinoids', 'Acetate-derived alkaloids', 'Daucane sesquiterpenoids', 'Acorane sesquiterpenoids', 'Drimane sesquiterpenoids', 'Betaestacin-type sesterterpenoids', 'Flavonols', 'Androstane steroids', 'Duclauxin and derivatives', 'Gibberellins', 'Corynanthe type', 'Africanane sesquiterpenoids', 'Mangicol-type sesterterpenoids', 'Pulvinones', 'Ceramides', 'Paulomycins and derivatives', 'Tremulane sesquiterpenoids', 'Morphinan alkaloids', 'Humulane sesquiterpenoids', 'Apocarotenoids(ε-)', 'Strobilurins and derivatives', 'Monoalkylresorcinols', 'Purine alkaloids', 'Paraconic acids and derivatives', 'RiPPs-Amatoxins and Phallotoxins', 'Naphthalenones', 'Dolabellane diterpenoids', 'Phenylethanoids', 'Longibornane sesquiterpenoids', 'Acyclic monoterpenoids', 'Cyathane diterpenoids', 'Pyrrocidine tetramate alkaloids', 'Caryophyllane sesquiterpenoids', 'Steroidal alkaloids', 'Merohemiterpenoids', 'Protoilludane sesquiterpenoids', 'Neolignans', 'Phenylalanine-derived alkaloids', 'Cyclopiane diterpenoids', 'Heterocyclic fatty acids', 'Chalcones', 'Bicyclogermacrane sesquiterpenoids', 'Casbane diterpenoids', 'Taxane diterpenoids', 'Verticillane diterpenoids', 'Abietane diterpenoids', 'Cleistanthane diterpenoids', 'Oleanane triterpenoids', 'Usnic acid and derivatives', 'Bisabolane sesquiterpenoids', 'Hydroxy fatty acids', 'Silphinane sesquiterpenoids', 'Depsides', 'Valparane diterpenoids', 'Acridone alkaloids', 'Breviane diterpenoids', 'Caryolane sesquiterpenoids', 'Germacrane sesquiterpenoids', 'Baccharane triterpenoids', 'Simple aromatic polyketides', 'Sativane sesquiterpenoids', 'Copacamphane sesquiterpenoids', 'Aromadendrane sesquiterpenoids', 'Cericerane sesterterpenoids', 'Oxo fatty acids', 'Hydroxy-hydroperoxyeicosapentaenoic acids', 'Other Docosanoids']
newBGCcat_classes = []
for c in BGCcat_classes:
    if c not in underrepresented_classes:
        newBGCcat_classes.append(c)
BGCcat_classes = newBGCcat_classes

# Point this at your cloned NP-Classifier repo
REPO = Path("/data/walker_lab/bin/NP-Classifier").resolve()

MODEL_DIR = REPO / "Classifier" / "models_folder" / "models"
ONTOLOGY_JSON = REPO / "Classifier" / "dict" / "index_v1.json"

ontology_dictionary = json.loads(ONTOLOGY_JSON.read_text())


def classify_smiles(
    smiles: str,
    session: requests.Session,
    timeout: int = 60,
    max_retries: int = 3,
    sleep_between_retries: float = 2.0,
) -> Dict[str, Any]:
    """
    Classify one SMILES string using the NPClassifier API.

    Returns a dictionary with:
        pathway_results
        superclass_results
        class_results
        isglycoside
        error
    """

    if not isinstance(smiles, str) or not smiles.strip():
        return {
            "pathway_results": [],
            "superclass_results": [],
            "class_results": [],
            "isglycoside": None,
            "error": "empty_smiles",
        }

    smiles = smiles.strip()

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(
                NPCLASSIFIER_URL,
                params={"smiles": smiles},
                timeout=timeout,
            )

            response.raise_for_status()

            result = response.json()

            return {
                "pathway_results": result.get("pathway_results", []),
                "superclass_results": result.get("superclass_results", []),
                "class_results": result.get("class_results", []),
                "isglycoside": result.get("isglycoside", None),
                "error": None,
            }

        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                return {
                    "pathway_results": [],
                    "superclass_results": [],
                    "class_results": [],
                    "isglycoside": None,
                    "error": f"request_error: {e}",
                }

            time.sleep(sleep_between_retries)

        except json.JSONDecodeError as e:
            return {
                "pathway_results": [],
                "superclass_results": [],
                "class_results": [],
                "isglycoside": None,
                "error": f"json_decode_error: {e}",
            }


outfile.write("bgc_id")
bgc_classifications = {}
number_of_classifications = 0
class_counts = {}
for c in BGCcat_classes:
    class_counts[c] = 1
for pathway in ontology_dictionary['Pathway']:
    #if pathway not in BGCcat_classes:
     #   continue
    class_counts[pathway] = 0
    number_of_classifications += 1
    pathway = pathway.replace(",","_")
    outfile.write("," + pathway)
    

for superclass in ontology_dictionary['Superclass']:
    if superclass not in BGCcat_classes:
        continue
    number_of_classifications += 1
    superclass = superclass.replace(",","_")
    outfile.write(",superclass_" + superclass)
    
for np_class in ontology_dictionary['Class']:
    if np_class not in BGCcat_classes:
        continue
    number_of_classifications += 1
    np_class = np_class.replace(",","_")
    outfile.write(",class_" + np_class)
    

#outfile.write(",is_glycoside")
outfile.write("\n")
session = requests.Session()
for line in infile:
    split_line = line.split(",")
    if len(split_line) <2:
        continue
    bgc_id = split_line[0]
    if bgc_id not in bgc_classifications:
        bgc_classifications[bgc_id] = {}
        bgc_classifications[bgc_id]["is_glycoside"] = False
        for pathway in ontology_dictionary['Pathway']:
            bgc_classifications[bgc_id][pathway] = False
        for superclass in ontology_dictionary['Superclass']:
            bgc_classifications[bgc_id]["superclass_" +superclass] = False
        for np_class in ontology_dictionary['Class']:
            bgc_classifications[bgc_id]["class_" + np_class] = False
    total_pathway_classifications = 0
    for i in range(1,len(split_line)):
        smiles = split_line[i]
        #print(json.dumps(classify_npclassifier_direct(smiles), indent=2))
        classifier_results = classify_smiles(smiles,session)
        #print(classifier_results["superclass_results"])
        for pathway in classifier_results["pathway_results"]:
            class_counts[pathway] += 1
            bgc_classifications[bgc_id][pathway] = True
            total_pathway_classifications += 1
        for superclass in classifier_results["superclass_results"]:
            if superclass in class_counts:
                class_counts[superclass] += 1
            bgc_classifications[bgc_id]["superclass_" +superclass] = True
        for np_class in classifier_results["class_results"]:
            if np_class in class_counts:
                class_counts[np_class] += 1
            bgc_classifications[bgc_id]["class_" + np_class] = True
        if classifier_results["isglycoside"]:
            bgc_classifications[bgc_id]["is_glycoside"] = True
    if total_pathway_classifications == 0:
        continue
    outfile.write(bgc_id)
    for pathway in ontology_dictionary['Pathway']:
        #if pathway not in BGCcat_classes:
         #   continue
        if bgc_classifications[bgc_id][pathway]:
            outfile.write(",1")
        else:
            outfile.write(",0")
    
    for superclass in ontology_dictionary['Superclass']:
        if superclass not in BGCcat_classes:
            continue
        if bgc_classifications[bgc_id]["superclass_"+superclass]:
            outfile.write(",1")
        else:
            outfile.write(",0")
        
    for np_class in ontology_dictionary['Class']:
        if np_class not in BGCcat_classes:
            continue
        if bgc_classifications[bgc_id]["class_"+np_class]:
            outfile.write(",1")
        else:
            outfile.write(",0")
    
    #if bgc_classifications[bgc_id]["is_glycoside"]:
     #   outfile.write(",1")
    #else:
     #   outfile.write(",0")
    outfile.write("\n")
outfile.close()

print("kept: " + str(number_of_classifications) + " classifications")

threshold = 3
under_threshold_list = []
for c in class_counts:
    if class_counts[c] <= threshold:
        under_threshold_list.append(c)
print(under_threshold_list)