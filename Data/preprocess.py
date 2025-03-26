"""Preprocessing raw_data into format suitable for finetuning language models."""


import pandas as pd
import os
import sys
import glob
import json
from dataclasses import dataclass


def convert_xlsx_to_csv(input_file, output_file):
    try:
        # Read the Excel file
        df = pd.read_excel(input_file, usecols=[0, 1, 2])  # Ensure only 3 columns are read

        # Save as CSV
        df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error: {e}")


def get_problem(input_folder):
    assert os.path.isdir(input_folder), f"{input_folder} is not a valid directory"
    folder_name = os.path.basename(os.path.normpath(input_folder))

    with open(f"{input_folder}/{folder_name}.json", "r") as jsonfile:
        data = json.load(jsonfile)
    question, rules, examples = (data[key] for key in ["question", "rules", "examples"])
    rules = "\n".join([rule.strip() for rule in rules]) # Covert rules from list to text, each rule in a row

    return f"QUESTION: {question}\n\nRULES: {rules}\n\nEXAMPLES: {examples}"


def get_extension(input_folder):
    answer_file = glob.glob(f"{input_folder}/{folder_name}_*")[0]
    return answer_file[answer_file.rfind("."):]


def get_answer(input_folder, assistant):
    if assistant: # AI-generated answer
        answer_file = glob.glob(f"{input_folder}/{folder_name}_{assistant}.*")[0]
    else: # Human answer
        answer_file = f"{input_folder}/{folder_name}{get_extension(input_folder)}"

    with open(answer_file, "r") as data:
        return data.read()


if __name__ == "__main__":
    input_xlsx = sys.argv[1]
    output_csv = sys.argv[2]
    convert_xlsx_to_csv(input_xlsx, output_csv)

    with open(output_csv, "r") as data:
        samples = data.read().strip().split("\n")

    current_problem = {}
    human_samples = []
    for idx, sample in enumerate(samples[1:], start=1):
        folder_name, assistant, score = sample.split(",")
        score = float(score)
        input_folder = f"Data/dataset-source-codes/{folder_name}"
        if folder_name not in current_problem:
            # Get problem
            problem = get_problem(input_folder=input_folder)
            extension = get_extension(input_folder=input_folder)

            # Get human answer for each problem
            human_answer = get_answer(input_folder=input_folder, assistant="")
            human_samples.append(
                dict(
                    coding_problem_id = folder_name,
                    llm_answer_id = "",
                    label = score,
                    problem = problem,
                    answer = human_answer,
                    extension = extension
                )
            )

            # Avoid redundantly retrieve a problem (and the language of answer) multiple times
            current_problem[folder_name] = (problem, extension)

        samples[idx] = dict(
            coding_problem_id = folder_name,
            llm_answer_id = assistant,
            label = score,
            problem = current_problem[folder_name][0],
            answer = get_answer(input_folder=input_folder, assistant=assistant),
            extension = current_problem[folder_name][1]
        )

    samples.extend(human_samples)
    writer = open("Data/hf_data.json", "w")
    for sample in samples[1:]:
        writer.write(json.dumps(sample) + "\n")
    writer.close()
