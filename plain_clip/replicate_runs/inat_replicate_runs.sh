# Define dataset and mode arrays
datasets=("inaturalist")
modes=(
    "gpt_descriptions" 
    "waffle"
    )
model_sizes=("ViT-B/32" "ViT-B/16" "ViT-L/14")
descriptor_paths=(
    # "../descriptors/inaturalist2021/425_sachit_descriptors_inaturalist.json"
    # "../descriptors/inaturalist2021/habitat_425_sachit_descriptors_inaturalist.json"
    # "../descriptors/inaturalist2021/425_chatgpt_descriptors_inaturalist.json"
    # "../descriptors/inaturalist2021/habitat_425_chatgpt_descriptors_inaturalist.json"
    # "../descriptors/inaturalist2021/425_ssc_descriptors_inaturalist.json"
    "../descriptors/inaturalist2021/425_ssch_descriptors_inaturalist.json"
)
seeds=(
    1
    2
    3
    4
)

# run CLIP + Habitat
# for dataset in "${datasets[@]}"; do
#     for model_size in "${model_sizes[@]}"; do
#     python ../main.py --dataset="$dataset" --mode="clip_habitat" --save_class_acc --model_size="$model_size" --descriptor_fnames "../descriptors/inaturalist2021/habitat_425_sachit_descriptors_inaturalist.json" --device cuda:3 --savename="${datasets[@]}/clip_habitat_accuracy"
#     python ../main.py --dataset="$dataset" --sci2comm --mode="clip_habitat" --save_class_acc --model_size="$model_size" --descriptor_fnames "../descriptors/inaturalist2021/habitat_425_sachit_descriptors_inaturalist.json" --device cuda:3 --savename="${datasets[@]}/sci2comm_clip_habitat_accuracy"
#     done
# done

#run GPT descriptions and the waffle CLIP
for dataset in "${datasets[@]}"; do
    # for model_size in "${model_sizes[@]}"; do
    #     python ../main.py --dataset="$dataset" --mode="gpt_descriptions" --model_size="$model_size" --descriptor_fnames "${descriptor_paths[@]}" --device cuda:1 --savename="normal_inat_accuracy"
    # done
    
    for model_size in "${model_sizes[@]}"; do
        # python ../main.py --dataset="$dataset" --mode="waffle" --model_size="$model_size" --descriptor_fnames "${descriptor_paths[@]}" --device cuda:1 --savename="${datasets[@]}/waffle_inat_accuracy" --seeds "${seeds[@]}"

        python ../main.py --dataset="$dataset" --mode="waffle" --model_size="$model_size" --waffle_count 1 --descriptor_fnames "${descriptor_paths[@]}" --device cuda:7 --savename="${datasets[@]}/random_chars_accuracy" --seeds "${seeds[@]}"
        python ../main.py --dataset="$dataset" --mode="waffle_habitat" --model_size="$model_size" --waffle_count 1 --descriptor_fnames "${descriptor_paths[@]}" --device cuda:7 --savename="${datasets[@]}/random_chars_habitat_accuracy" --seeds "${seeds[@]}"
        python ../main.py --dataset="$dataset" --mode="waffle_habitat_only" --model_size="$model_size" --waffle_count 1 --descriptor_fnames "${descriptor_paths[@]}" --device cuda:7 --savename="${datasets[@]}/random_chars_habitat_only_accuracy"
    done
done
