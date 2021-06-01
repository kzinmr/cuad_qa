export SQUAD_DIR=/content/data

python run_qa_beam_search.py \
    --model_name_or_path roberta-base \
    --train_file data/train_separate_questions.json \
    --test_file data/test.json \
    --do_train \
    --do_eval \
    --version_2_with_negative \
    --learning_rate 1e-4 \
    --num_train_epochs 4 \
    --max_seq_length 512 \
    --max_answer_length 256 \
    --doc_stride 256 \
    --output_dir ./roberta_base_finetuned_cuad/ \
    --per_device_eval_batch_size=8  \
    --per_device_train_batch_size=8   \
    --save_steps 5000
