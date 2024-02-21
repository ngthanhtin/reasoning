# Define dataset and mode arrays
datasets=("part_imagenet")
modes=(
    "gpt_descriptions" 
    "waffle"
    )
model_sizes=("ViT-B/32" "ViT-B/16" "ViT-L/14")
descriptor_paths=(
    "../descriptors/part_imagenet/78_part_imagenet_descriptions.json"
    "../descriptors/part_imagenet/78_habitat_part_imagenet_descriptions.json"
)
seeds=(
    1
    2
    3
    4
)
# Loop through each combination of dataset, mode, model size, and descriptor file
for dataset in "${datasets[@]}"; do
    for model_size in "${model_sizes[@]}"; do
        for descriptor_fname in "${descriptor_paths[@]}"; do
            # Run the Python script with the current combination
            python ../main.py --dataset="$dataset" --mode="gpt_descriptions" --model_size="$model_size" --descriptor_fname="$descriptor_fname" --device cuda:3 --savename="pi_accuracy"
        done
    done
    
    for model_size in "${model_sizes[@]}"; do
        for descriptor_fname in "${descriptor_paths[@]}"; do
            for seed in "${seeds[@]}"; do
                # Run the Python script with the current combination
                python ../main.py --dataset="$dataset" --mode="waffle" --model_size="$model_size" --descriptor_fname="$descriptor_fname" --device cuda:3 --savename="pi_accuracy" --seed=$seed
            done
        done
    done
done
