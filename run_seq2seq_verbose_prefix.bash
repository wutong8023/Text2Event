#!/usr/bin/env bash
# -*- coding:utf-8 -*-

EXP_ID=$(date +%F-%H-%M)
export CUDA_VISIBLE_DEVICES="1"
export batch_size="16"
export model_name=t5-base
#export model_name="models_trained/CF_2021_10_11_both/"
export data_name=one_ie_ace2005_subtype
export lr=5e-5
export task_name="event"
export seed="421"
export lr_scheduler=constant_with_warmup
export label_smoothing="0"
export epoch=30
export decoding_format='tree'
export eval_steps=500
export warmup_steps=0
export tuning_type="both"
export is_knowledge=""
export no_module=""
export do_train="--do_train"
export prefix_len=10
export output_dir="models/"
export constraint_decoding='--constraint_decoding'

OPTS=$(getopt -o b:d:m:i:t:s:l:f: --long batch:,device:,model:,data:,task:,seed:,lr:,lr_scheduler:,label_smoothing:,epoch:,format:,eval_steps:,no_train:,warmup_steps:,prefix_len:,tuning_type:,output_dir:,is_knowledge:,no_module:,wo_constraint_decoding -n 'parse-options' -- "$@")

if [ $? != 0 ]; then
  echo "Failed parsing options." >&2
  exit 1
fi

eval set -- "$OPTS"

while true; do
  case "$1" in
  -b | --batch)
    batch_size="$2"
    shift
    shift
    ;;
  -d | --device)
    CUDA_VISIBLE_DEVICES="$2"
    shift
    shift
    ;;
  -m | --model)
    model_name="$2"
    shift
    shift
    ;;
  -i | --data)
    data_name="$2"
    shift
    shift
    ;;
  -t | --task)
    task_name="$2"
    shift
    shift
    ;;
  -s | --seed)
    seed="$2"
    shift
    shift
    ;;
  -l | --lr)
    lr="$2"
    shift
    shift
    ;;
  -f | --format)
    decoding_format="$2"
    shift
    shift
    ;;
  --lr_scheduler)
    lr_scheduler="$2"
    shift
    shift
    ;;
  --label_smoothing)
    label_smoothing="$2"
    shift
    shift
    ;;
  --epoch)
    epoch="$2"
    shift
    shift
    ;;
  --prefix_len)
    prefix_len="$2"
    shift
    shift
    ;;
  --eval_steps)
    eval_steps="$2"
    shift
    shift
    ;;
  --is_knowledge)
    is_knowledge="--is_knowledge"
    shift
    shift
    ;;
  --no_train)
    do_train=""
    shift
    shift
    ;;
  --no_module)
    no_module="--no_module"
    shift
    shift
    ;;
  --warmup_steps)
    warmup_steps="$2"
    shift
    shift
    ;;
  --tuning_type)
    tuning_type="$2"
    shift
    shift
    ;;
  --output_dir)
    output_dir="$2"
    shift
    shift
    ;;
  --wo_constraint_decoding)
    constraint_decoding=""
    shift
    ;;
  --)
    shift
    break
    ;;
  *)
    echo "$1" not recognize.
    exit
    ;;
  esac
done

# google/mt5-base -> google_mt5-base
model_name_log=$(echo ${model_name} | sed -s "s/\//_/g")

model_folder=${output_dir}

data_folder=data/text2${decoding_format}/${data_name}

export TOKENIZERS_PARALLELISM=false


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python run_seq2seq_prefix.py \
    ${do_train} --do_eval --do_predict ${constraint_decoding} \
    --label_smoothing_sum=False \
    --use_fast_tokenizer=False \
    --evaluation_strategy steps \
    --predict_with_generate \
    --metric_for_best_model eval_role-F1 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --max_source_length=256 \
    --max_target_length=128 \
    --num_train_epochs=${epoch} \
    --task=${task_name} \
    --train_file=${data_folder}/train.json \
    --validation_file=${data_folder}/val.json \
    --test_file=${data_folder}/test.json \
    --event_schema=${data_folder}/event.schema \
    --per_device_train_batch_size=${batch_size} \
    --per_device_eval_batch_size=$((batch_size * 4)) \
    --output_dir=${output_dir} \
    --logging_dir=${output_dir}_log \
    --model_name_or_path=${model_name} \
    --learning_rate=${lr} \
    --lr_scheduler_type=${lr_scheduler} \
    --label_smoothing_factor=${label_smoothing} \
    --eval_steps ${eval_steps} \
    --decoding_format ${decoding_format} \
    --warmup_steps ${warmup_steps} \
    --source_prefix="${task_name}: " \
    --seed=${seed} \
    ${is_knowledge} \
    ${no_module} \
    --prefix_len=${prefix_len} \
    --tuning_type=${tuning_type}