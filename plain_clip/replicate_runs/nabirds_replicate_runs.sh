# Define dataset and mode arrays
datasets=("nabirds")
modes=(
    "gpt_descriptions" 
    "waffle"
    )
model_sizes=("ViT-B/32" "ViT-B/16" "ViT-L/14")
descriptor_paths=(
    "../descriptors/nabirds/no_ann_sachit_descriptors_nabirds.json"
    "../descriptors/nabirds/habitat_no_ann_sachit_descriptors_nabirds.json"
    "../descriptors/nabirds/no_ann_chatgpt_descriptors_nabirds.json"
    "../descriptors/nabirds/habitat_no_ann_chatgpt_descriptors_nabirds.json"
    "../descriptors/nabirds/ssc_no_ann_descriptors_nabirds.json"
    "../descriptors/nabirds/ssch_no_ann_descriptors_nabirds.json"
)
seeds=(
    1
    2
    3
    4
)

# # run CLIP + Habitat
# for dataset in "${datasets[@]}"; do
#     for model_size in "${model_sizes[@]}"; do
#     python ../main.py --dataset="$dataset" --mode="clip_habitat" --save_class_acc --model_size="$model_size" --descriptor_fnames "../descriptors/nabirds/habitat_no_ann_sachit_descriptors_nabirds.json" --device cuda:2 --savename="${datasets[@]}/clip_habitat_accuracy"
#     done
# done

#run GPT descriptions and the waffle CLIP
for dataset in "${datasets[@]}"; do
    # for model_size in "${model_sizes[@]}"; do
    #     python ../main.py --dataset="$dataset" --mode="gpt_descriptions" --model_size="$model_size" --descriptor_fnames "${descriptor_paths[@]}" --device cuda:2 --savename="normal_nabirds_accuracy"
    # done
    
    for model_size in "${model_sizes[@]}"; do
        python ../main.py --dataset="$dataset" --mode="waffle" --model_size="$model_size" --descriptor_fnames "${descriptor_paths[@]}" --device cuda:2 --savename="waffle_nabirds_accuracy" --seeds "${seeds[@]}"
    done
done