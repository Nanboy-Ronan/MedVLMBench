from transformers import TrainingArguments

args = TrainingArguments(output_dir="./")


print(args.get_process_log_level())
