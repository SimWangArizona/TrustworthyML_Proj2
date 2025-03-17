import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import csv


# 1. Load dataset
def load_aime_dataset_from_parquet(data_dir, split="train"):
    # setup dataset path
    dataset_path = os.path.join(data_dir, f"{split}-00000-of-00001.parquet")
    dataset = load_dataset("parquet", data_files=dataset_path)
    return dataset["train"]  
# 2. Load model
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True)
    return tokenizer, model

# 3. evaluate model and save results in csv
def evaluate_and_save_results(model, tokenizer, dataset, output_csv):
    correct = 0
    total = 0

    # write results
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Generated Answer", "Ground Truth Answer", "Is Correct"])

        for item in dataset:
            question = item['problem']
            ground_truth_answer = item['answer']
            print(f"ground turth answer is {ground_truth_answer}")
            inputs = tokenizer(question, return_tensors="pt", truncation=True)
            inputs = {key: value.cuda() for key, value in inputs.items()}  
            with torch.no_grad():
                outputs = model.generate(**inputs,
    eos_token_id=None,   
    pad_token_id=tokenizer.eos_token_id )

            predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"predicted_answer is {predicted_answer}")
            is_correct = predicted_answer.strip() == ground_truth_answer.strip()
            if is_correct:
                correct += 1
            total += 1

            writer.writerow([predicted_answer, ground_truth_answer, is_correct])

    # calculate acc
    accuracy = correct / total
    return accuracy


def main(args):
    # load dataset
    dataset = load_aime_dataset_from_parquet(args.data_dir, split="train")

    # load model
    model_name = args.model  
    tokenizer, model = load_model(model_name)

    # setup output path
    output_csv = os.path.join(args.save_dir, "llm_results.csv")

    # save results in csv
    accuracy = evaluate_and_save_results(model, tokenizer, dataset, output_csv)

    # print results
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    # setup parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5, help="Number of training examples")
    parser.add_argument("--data_dir", "-d", type=str, default="data", help="Directory containing the dataset")
    parser.add_argument("--save_dir", "-s", type=str, default="results", help="Directory to save results")
    parser.add_argument("--model", "-m", type=str, required=True, help="Name of the pre-trained model")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)