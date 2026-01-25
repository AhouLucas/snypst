import pandas as pd
import numpy as np
import pandoc
import shutil
import os
import argparse

from python.data.utils import (
    convert_latex_to_typst,
    ensure_empty_dir,
    convert_inkml_to_png,
    get_inkml_label,
    convert_latex_into_valid_expression
)

##########
# ARGPARSE
##########

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--test", help="Process only a few data for testing purposes", action="store_true")
TEST = arg_parser.parse_args().test


##########
# Path and Variables
##########

raw_path = "data/raw"
im2latex_path = "data/raw/im2latex"
handwritten_path = "data/raw/handwritten-raw"

converted_path = "data/converted"

d = {
    "img_path": [], # Path to the image
    "latex": [], # Corresponding Latex Formula
    "typst": [], # Corresponding Typst Formula if not failed, otherwise ""
    "type": [],  # 0 for compiled images or 1 for handwritten images
    "failed": [],   # Whether the conversion from Latex to Typst was successful (0) or failed (1)
}

##########
# IM2LATEX Dataset processor
##########

# Create folders and clear them they already exist
ensure_empty_dir(converted_path + "/img")

with open(im2latex_path + "/corresponding_png_images.txt") as img_file:
    img_filenames = img_file.read().splitlines()

with open(im2latex_path + "/final_png_formulas.txt") as formula_file:   
    latex_formulas = formula_file.read().splitlines()


N = len(img_filenames) if not TEST else 10
for i in range(N):

    # Copy .png to converted/img
    img_path = converted_path + f"/img/img2latex_{i}.png"
    shutil.copy(im2latex_path + f"/generated_png_images/{img_filenames[i]}", img_path)

    # Convert Latex to Typst using Pandoc
    latex_formula = convert_latex_into_valid_expression(latex_formulas[i])
    
    typst_formula, failed = convert_latex_to_typst(latex_formula)

    d["img_path"].append(img_path)
    d["latex"].append(latex_formula)
    d["typst"].append(typst_formula)
    d["type"].append(0)
    d["failed"].append(1 if failed else 0)


##########
# Handwritten Dataset processor
##########

# Get all image paths
img_filenames = [handwritten_path + "/train" + f"/{fp}" for fp in os.listdir(handwritten_path + "/train")]
img_filenames.extend([handwritten_path + "/synthetic" + f"/{fp}" for fp in os.listdir(handwritten_path + "/synthetic")])
img_filenames.extend([handwritten_path + "/valid" + f"/{fp}" for fp in os.listdir(handwritten_path + "/valid")])
img_filenames.extend([handwritten_path + "/test" + f"/{fp}" for fp in os.listdir(handwritten_path + "/test")])

N = len(img_filenames) if not TEST else 10
for i in range(N):
    # Convert .inkml file to .png
    img_path = converted_path + f"/img/handwritten_{i}.png"
    convert_inkml_to_png(img_filenames[i], img_path)
    
    # Get Original Latex expression stored inside the InkML file as annotation
    latex_formula = convert_latex_into_valid_expression(get_inkml_label(img_filenames[i]))

    # Convert Latex to Typst
    typst_formula, failed = convert_latex_to_typst(latex_formula)

    d["img_path"].append(img_path)
    d["latex"].append(latex_formula)
    d["typst"].append(typst_formula)
    d["type"].append(1)
    d["failed"].append(1 if failed else 0)


##########
# Write CSV
##########

csv_path = converted_path + "/description.csv"

df = pd.DataFrame(d)
df.to_csv(csv_path)