import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input_path", default="evaluation/MergedEntityRecognizer/results_0_overlap-longest.csv", help="Path to the merged evaluation CSV file.")
parser.add_argument("--output_path", default="evaluation/MergedEntityRecognizer/results_0_overlap-longest_parsed.csv", help="Path to save the parsed evaluation CSV file.")

if __name__ == "__main__":
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    df = pd.read_csv(input_path)

    output_columns = [
        "BaseMethod",
        "AdvancedMethod",
        "ModelName",
        "MacroPrecision",
        "MacroRecall",
        "MacroF1"
    ]

    folderToMethod = {
        "BaselineEntityRecognizer": "Baseline",
        "HFEntityRecognizer": "TokenClass",
        "GLiNEREntityRecognizer": "GLiNER",
        "DspyEntityRecognizer": "Dspy",
        "LLMEntityRecognizer": "LLM",
    }

    advancedMethods = [
        "-fewshot-semantic-k4",
        "-fewshot-fixed-k4",
        "-zeroshot",
        "-finetuned",
    ]

    rename_models = {
        "gemma3-27b": "Gemma3-27B",
        "gpt-4o-deescalate": "GPT-4o",
        "gemma3-12b-it": "Gemma3-12B",
        "gemma3-4b": "Gemma3-4B",
        "bert-base-italian-xxl-cased": "BERT-Italian",
        "longest": "Longest",
        "gemma3-27b-it": "Gemma3-27B",
        "gpt-4o-mini-2": "GPT-4o Mini",
        "gliner-multi-v2.1-T0p50": "MultiV2p1",
        "gemma3-12b": "Gemma3-12B",
        "all": "All",
        "universalNerIta-T0p50": "universalNerIta",
        "SauerkrautLM-T0p50": "SauerkrautLM",
        "gemma3-4b-it": "Gemma3-4B",
    }

    model_names = set()

    dump_df = pd.DataFrame(columns=output_columns)

    for idx, row in df.iterrows():
        baseMethod = folderToMethod[row["source_folder"]]
        modelName = row["model_name"]
        advancedMethod = "None"
        for advMethod in advancedMethods:
            if modelName.find(advMethod) != -1:
                advancedMethod = advMethod[1:]
                modelName = modelName.replace(advMethod, "")
                break
        if modelName == "results":
            modelName = row["model_source"]
        if modelName == "epochs=5":
            modelName = row["language"]
        if modelName in rename_models:
            modelName = rename_models[modelName]
        macroPrecision = row["strict_precision"]
        macroRecall = row["strict_recall"]
        macroF1 = row["strict_f1"]
        dump_df.at[idx, "BaseMethod"] = baseMethod
        dump_df.at[idx, "AdvancedMethod"] = advancedMethod
        dump_df.at[idx, "ModelName"] = modelName
        dump_df.at[idx, "MacroPrecision"] = macroPrecision
        dump_df.at[idx, "MacroRecall"] = macroRecall
        dump_df.at[idx, "MacroF1"] = macroF1

    dump_df.to_csv(output_path, index=False, sep=";")
