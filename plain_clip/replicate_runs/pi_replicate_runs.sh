# Define dataset and mode arrays
datasets=("part_imagenet")
modes=(
    "gpt_descriptions" 
    "waffle"
    )
model_sizes=("ViT-B/32" "ViT-B/16" "ViT-L/14")
descriptor_paths=(
    # "../descriptors/part_imagenet/158_part_imagenet_descriptions.json"
    "../descriptors/part_imagenet/158_habitat_part_imagenet_descriptions.json"
)
seeds=(
    5
    6
    7
    8
    9
    10
)
# run CLIP + Habitat
# for dataset in "${datasets[@]}"; do
#     for model_size in "${model_sizes[@]}"; do
#     python ../main.py --dataset="$dataset" --mode="clip_habitat" --save_class_acc --model_size="$model_size" --descriptor_fnames "../descriptors/part_imagenet/158_habitat_part_imagenet_descriptions.json" --device cuda:0 --savename="${datasets[@]}/clip_habitat_accuracy"
#     done
# done

#run GPT descriptions and the waffle CLIP
for dataset in "${datasets[@]}"; do
    # for model_size in "${model_sizes[@]}"; do
    #     python ../main.py --dataset="$dataset" --mode="gpt_descriptions" --save_class_acc --model_size="$model_size" --descriptor_fnames "${descriptor_paths[@]}" --device cuda:0 --savename="normal_accuracy" --seeds "${seeds[@]}"
    # done

    for model_size in "${model_sizes[@]}"; do
        # python ../main.py --dataset="$dataset" --mode="waffle" --save_class_acc --model_size="$model_size" --descriptor_fnames "${descriptor_paths[@]}" --device cuda:0 --savename="${datasets[@]}/waffle_habitat_accuracy" --seeds "${seeds[@]}"

        python ../main.py --dataset="$dataset" --mode="waffle" --model_size="$model_size" --waffle_count 1 --descriptor_fnames "${descriptor_paths[@]}" --device cuda:2 --savename="${datasets[@]}/random_chars_accuracy" --seeds "${seeds[@]}"
        python ../main.py --dataset="$dataset" --mode="waffle_habitat" --model_size="$model_size" --waffle_count 1 --descriptor_fnames "${descriptor_paths[@]}" --device cuda:2 --savename="${datasets[@]}/random_chars_habitat_accuracy" --seeds "${seeds[@]}"
        python ../main.py --dataset="$dataset" --mode="waffle_habitat_only" --model_size="$model_size" --waffle_count 1 --descriptor_fnames "${descriptor_paths[@]}" --device cuda:2 --savename="${datasets[@]}/random_chars_habitat_only_accuracy"
    done
done
