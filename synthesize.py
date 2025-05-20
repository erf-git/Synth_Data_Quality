import os
import pandas as pd 
import numpy as np 

# SDV
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer

# Synthcity
import synthcity.logger as log
from synthcity.plugins import Plugins

# DataSynthesizer
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator


PATH = "/home/erf6575/Desktop/SynthData/data/"

#################### DATA ####################

print("Checking files...")

files = {
    "adult": PATH + "adult_original.csv",
    "bank": PATH + "bank_original.csv",
    "cancer": PATH + "cancer_original.csv",
    "card": PATH + "card_original.csv",
    "dermatology": PATH + "dermatology_original.csv",
    "diabetes": PATH + "diabetes_original.csv",
    "heart": PATH + "heart_original.csv",
    "iris": PATH + "iris_original.csv",
    "kidney": PATH + "kidney_original.csv",
    "titanic": PATH + "titanic_original.csv"
}


for name, file in files.items():
    if os.path.isfile(file):
        print("Located ", file)
    else:
        print("Missing or misspelled ", file)
print()


thresholds = {
    "adult": 42,
    "bank": 15,
    "cancer": 10,
    "card": 15,
    "depression": 15,
    "dermatology": 10,
    "diabetes": 10,
    "heart": 10,
    "iris": 10,
    "kidney": 10,
    "titanic": 10
}


#################### GENERATION FUNCIONS ####################

def SDV(name, dataset, num_generate):
    
    # SDV
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(dataset)
    metadata.validate()
    metadata.validate_data(data=dataset)
    
    # Gaussian Copula
    try:
        print("GaussianCopulaSynthesizer...")
        from sdv.single_table import GaussianCopulaSynthesizer
        gc_synthesizer = GaussianCopulaSynthesizer(metadata)
        gc_synthesizer.fit(dataset)
        gc_data = gc_synthesizer.sample(num_rows=num_generate)
        gc_data.to_csv(PATH + f"{name}_gc.csv", index=False)
        print("Complete")
    except Exception as e:
        print("Error running GaussianCopulaSynthesizer\n" + e)

    # CTGAN
    try:
        print("CTGANSynthesizer...")
        from sdv.single_table import CTGANSynthesizer
        ctgan_synthesizer = CTGANSynthesizer(metadata)
        ctgan_synthesizer.fit(dataset)
        ctgan_data = ctgan_synthesizer.sample(num_rows=num_generate)
        ctgan_data.to_csv(PATH + f"{name}_ctgan.csv", index=False)
        print("Complete")
    except Exception as e:
        print("Error running CTGANSynthesizer\n" + e)


# Synthcity
def Synthcity(name, dataset, num_generate):

    # PATE-GAN
    try: 
        print("PATE-GAN...")
        pate_synthesizer = Plugins().get("pategan", epsilon=1.0)
        pate_synthesizer.fit(dataset)
        pate_data = pate_synthesizer.generate(count=num_generate)
        pate_data.dataframe().to_csv(PATH + f"{name}_pate.csv", index=False)
        print("Complete")
    except Exception as e:
        print("Error running PATE-GAN\n" + e)
    
    # # DP-GAN
    # try:
    #     print("DP-GAN...")
    #     dpgan_synthesizer = Plugins().get("dpgan", epsilon=1.0)
    #     dpgan_synthesizer.fit(dataset)
    #     dpgan_data = dpgan_synthesizer.generate(count=num_generate)
    #     dpgan_data.dataframe().to_csv(PATH + f"dpgan_{name}.csv", index=False)
    #     print("Complete")
    # except Exception as e:
    #     print("Error running DP-GAN\n" + e)


# DataSynthesizer
def DataSynthesizer(name, dataset, num_generate):
    
    bn_description_path = PATH + f'datasynthesizer/bn_{name}_description.json'
    bn_generation_path = PATH + f'{name}_bn.csv'
    
    pb_description_path = PATH + f'datasynthesizer/pb_{name}_description.json'
    pb_generation_path = PATH + f'{name}_pb.csv'
    
    # # Bayessian-Network
    # try:
    #     print("Bayessian-Network...")
    #     bn_describer = DataDescriber(category_threshold=thresholds.get(name))
    #     bn_describer.describe_dataset_in_correlated_attribute_mode(dataset_file=files.get(name), 
    #                                                             epsilon=0, # No Differential Privacy 
    #                                                             k=4, # ( in paper is 4, but in code it 0 which means number of parents in network is automatically calculated 
    #                                                             # attribute_to_is_categorical=categorical_attributes.get(name)
    #                                                             )
    #     bn_describer.save_dataset_description_to_file(bn_description_path)
        
    #     bn_synthesizer = DataGenerator()
    #     bn_synthesizer.generate_dataset_in_correlated_attribute_mode(n=num_generate, description_file=bn_description_path)
    #     bn_synthesizer.save_synthetic_data(bn_generation_path)
    #     print("Complete")
        
    # except Exception as e:
    #     print("Error running Bayessian-Network\n" + e)
    
    # PrivBayes
    try:
        print("PrivBayes...")
        pb_describer = DataDescriber(category_threshold=thresholds.get(name))
        pb_describer.describe_dataset_in_correlated_attribute_mode(dataset_file=files.get(name), 
                                                                epsilon=1, # With Differential Privacy 
                                                                k=4, # ( in paper is 4, but in code it 0 which means number of parents in network is automatically calculated 
                                                                # attribute_to_is_categorical=categorical_attributes.get(name)
                                                                )
        pb_describer.save_dataset_description_to_file(pb_description_path)
        
        pb_synthesizer = DataGenerator()
        pb_synthesizer.generate_dataset_in_correlated_attribute_mode(n=num_generate, description_file=pb_description_path)
        pb_synthesizer.save_synthetic_data(pb_generation_path)
        print("Complete")
        
    except Exception as e:
        print("Error running PrivBayes\n"+ e)


# {"dermatology": PATH + "dermatology.csv"}.items()
for name, file in {"depression": PATH + "depression_original.csv"}.items():
    
    # Synthcity has trouble with NaN values
    dataset = pd.read_csv(file, na_filter=False)
    num_generate = len(dataset)
    print(f"Generating data for {name}")
    
    SDV(name, dataset, num_generate)
    Synthcity(name, dataset, num_generate)
    DataSynthesizer(name, dataset, num_generate)
    
    print()
