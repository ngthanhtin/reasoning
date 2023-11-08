from load import *
import torchmetrics
from tqdm import tqdm
import cv2
from textwrap import wrap
import matplotlib.gridspec as gridspec

seed_everything(hparams['seed'])

def save_list_to_file(strings, filename):
    with open(filename, 'w') as file:
        for path, label, scores in strings:
            file.write(f"{path},{label},")
            for i, score in enumerate(scores):
                if i < len(scores) - 1:
                    file.write(f"{score:.4f},")
                else:
                    file.write(f"{score:.4f}\n")

def draw_chart(image, in_pred_classname, correct_pred_classname, in_scores, correct_scores, in_descriptions, correct_descriptions, color='dodgerblue', save_path='abc.pdf'):
    """
    image: np image
    predicted_classname: class name
    scores: scores of each description in descriptions [number_of_description]
    """
    
    def sorted_scores_and_descriptions(scores, descriptions):
        scores = [float(score) for score in scores]
        
        # remove which (is/has) in description
        refined_descriptions = []
        for i, des in enumerate(descriptions):
            if "which is" in des:
                index = des.find('which is')
                des = des[index + len('which is')+1:]
            elif "which has" in des:
                index = des.find('which has')
                des = des[index + len('which has')+1:]
            elif "which" in des:
                index = des.find('which')
                des = des[index + len('which')+1:]
            refined_descriptions.append(des)
        
        descriptions = refined_descriptions
        # sort scores and descriptions
        sorted_scores = sorted(scores, reverse=True)
        
        sorted_descriptions = []
        for ss in sorted_scores:
            for i, s in enumerate(scores):
                if ss == s:
                    # sorted_descriptions.append(descriptions[i].split(":")[0]) # show only the visual part type
                    sorted_descriptions.append(descriptions[i])
                    break
        
        sorted_descriptions = [ '\n'.join(wrap(l, 50)) for l in sorted_descriptions ]
        # add average score
        avg_score = np.mean(sorted_scores)
        sorted_descriptions.insert(0, "Average Score")
        sorted_scores.insert(0, avg_score)
        
        scores = sorted_scores
        scores = [s*100 for s in scores]
        descriptions = sorted_descriptions

        return scores, descriptions
    

    #
    plt.rcdefaults()
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig = plt.figure(figsize=(24, 9))

    gs = gridspec.GridSpec(1, 3, width_ratios=[2.6, 3.6, 3.8])

    # Create subplots in the specified grid cells
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # axis 1
    ax1.imshow(image)
    ax1.axis('off')

    # axis 2
    scores_2, descriptions_2 = sorted_scores_and_descriptions(in_scores, in_descriptions)
    y_pos = np.arange(len(descriptions_2))
    colors = ["darkblue"] + [color] * (len(scores_2)-1)
    bars = ax2.barh(y_pos, scores_2, align='center', color=colors)

    max_width = 0
    for index, bar in enumerate(bars):
        max_width =  bars[index].get_width()*0.6 if bars[index].get_width()*0.6 > max_width else max_width
        bars[index].set_width(bars[index].get_width() * 0.6)

    # Adding text inside the bars
    for i, (bar, score, description) in enumerate(zip(bars, scores_2, descriptions_2)):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, f'{score:.2f}',
                va='center', ha='right', color='white', fontweight='bold', fontsize=13)
        ax2.text(0.5, bar.get_y() + bar.get_height()/2, description,
            va='center', ha='left', color='white', fontweight='bold', fontsize=13)

    # axis 3
    scores_3, descriptions_3 = sorted_scores_and_descriptions(correct_scores, correct_descriptions)
    y_pos = np.arange(len(descriptions_3))
    colors = ["darkblue"] + [color] * (len(scores_3)-1)
    bars = ax3.barh(y_pos, scores_3, align='center', color=colors)

    max_width = 0
    for index, bar in enumerate(bars):
        max_width =  bars[index].get_width()*0.6 if bars[index].get_width()*0.6 > max_width else max_width
        bars[index].set_width(bars[index].get_width() * 0.6)

    # Adding text inside the bars
    for i, (bar, score, description) in enumerate(zip(bars, scores_3, descriptions_3)):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2, f'{score:.2f}',
                va='center', ha='right', color='white', fontweight='bold', fontsize=13)
        ax3.text(0.5, bar.get_y() + bar.get_height()/2, description,
            va='center', ha='left', color='white', fontweight='bold', fontsize=13)
        
    # modify axis
    ax2.axis('off')
    ax2.invert_yaxis()
    ax2.set_xlabel('')
    ax2.set_title(f'Wrong Prediction: {in_pred_classname}', fontsize=16)
    ax2.set_xlim(0, max_width)


    ax3.axis('off')
    ax3.invert_yaxis()
    ax3.set_xlabel('')
    ax3.set_title(f'Correct Prediction: {correct_pred_classname}', fontsize=16)
    ax3.set_xlim(0, max_width)

    # save figures
    # fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')

def generate_correct(path1, path2):
    for k, v in gpt_descriptions.items():
        v[-1] = v[-1].split('.')[0] + ', .etc'
        gpt_descriptions[k] = v[:12] + [v[-1]]

    # incorrect of no habitat and correct of habitat
    incorrects = []
    corrects = []
    with open(path1, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            path, pred = parts[0], parts[1]
            scores = parts[2:]
            incorrects.append([path, pred, scores])

    with open(path2, 'r') as file2:
        for line in file2:
            parts = line.strip().split(',')
            path, pred = parts[0], parts[1]
            scores = parts[2:]
            corrects.append([path, pred, scores])
    
    num_incorrect_correct = 0
    for incorrect in incorrects:
        in_path, in_pred, in_scores = incorrect
        for correct in corrects:
            path, pred, scores = correct

            if path == in_path:
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                num_incorrect_correct += 1
                for j, (h, l) in enumerate(gpt_descriptions.items()):
                    if j == int(in_pred):
                        in_pred_classname = h
                    if j == int(pred):
                        pred_classname = h
                        
                filename = in_path.split('/')[-1]
                draw_chart(image, in_pred_classname, pred_classname, in_scores, scores, gpt_descriptions[in_pred_classname], gpt_descriptions[pred_classname], color='dodgerblue', save_path=f"incorrect_correct_cub_figs/{filename}.pdf")
                break
    print(num_incorrect_correct) 



def inference():
    bs = hparams['batch_size']
    bs = 1
    dataloader = DataLoader(dataset, bs, shuffle=True, num_workers=16, pin_memory=True)

    print("Loading model...")

    device = torch.device(hparams['device'])
    # load model
    model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
    model.eval()
    model.requires_grad_(False)

    print("Encoding descriptions...")

    description_encodings = compute_description_encodings(model)

    if hparams['dataset'] == 'imagenet' or hparams['dataset'] == 'imagenetv2':
        num_classes = 1000
    elif hparams['dataset'] == 'nabirds':
        num_classes = 267#555
    elif hparams['dataset'] == 'cub':
        num_classes = 200
    elif hparams['dataset'] == 'places365':
        num_classes = 365
    elif hparams['dataset'] == 'inaturalist2021':
        num_classes = 425 #1486


    # num_descs = 4
    # attributes_pc = [0 for _ in range(num_descs)]
    # num_correct = 0
    # num_habitat_correct = 0

    correct_predictions = []
    incorrect_predictions = []

    for batch_number, batch in enumerate(tqdm(dataloader)):
        images, labels, paths = batch
        
        images = images.to(device)
        labels = labels.to(device)
        
        image_encodings = model.encode_image(images)
        image_encodings = F.normalize(image_encodings)
        
        image_description_similarity = [None]*n_classes
        image_description_similarity_cumulative = [None]*n_classes
        
        for i, (k, v) in enumerate(description_encodings.items()): 
            
            dot_product_matrix = image_encodings @ v.T
            
            image_description_similarity[i] = dot_product_matrix
            image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])
            
            
        # create tensor of similarity means
        cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
            
        
        descr_predictions = cumulative_tensor.argmax(dim=1)
        
        np_image = cv2.imread(paths[0])
        
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        
        label = labels[0].detach().cpu().numpy()
        prediction = descr_predictions.data[0].item()
        
        # attributes percentage
        tensor_scores = image_description_similarity[prediction].squeeze()
        scores = tensor_scores.tolist()

        if prediction == label:
            correct_predictions.append([paths[0], prediction, scores])
        else:
            incorrect_predictions.append([paths[0], prediction, scores])
        #     tensor_scores = image_description_similarity[prediction].squeeze()
        #     scores = tensor_scores.tolist()
        #     # attributes_pc[0] += scores[0]
        #     # attributes_pc[1] += scores[1]
        #     # attributes_pc[2] += scores[2]
        #     # attributes_pc[3] += scores[3]
        #     if max(scores) == scores[-1]:
        #         num_habitat_correct+=1
        #     attributes_pc = [attributes_pc[ii] + scores[ii] for ii in range(num_descs)]
        #     num_correct += 1


        # convert prediction to classname
        # if prediction == label:
        #     for j, (h, l) in enumerate(gpt_descriptions.items()):
        #         if j == prediction:
        #             filename = paths[0].split('/')[-1]
        #             plot = draw_chart(np_image, h, image_description_similarity[prediction], gpt_descriptions[h])
        #             plot = plot.save(f"correct_cub_id_figs/{filename}.jpg")
        #             break

            # for j, (h, l) in enumerate(gpt_descriptions.items()):
            #     if j == label:
            #         plot = draw_chart(np_image, h, image_description_similarity[label], gpt_descriptions[h], color='red')
            #         plot = plot.save(f"correct_cub_id_figs/gt_{batch_number}.jpg")
            #         break

        # if paths[0] in vis_paths:    
        #     for j, (h, l) in enumerate(gpt_descriptions.items()):
        #             if j == prediction:  
        #                 plot = draw_chart(np_image, h, image_description_similarity[prediction], gpt_descriptions[h])
        #                 filename = paths[0].split('/')[-1]
        #                 plot = plot.save(f"correct_cub_no_habitat_figs/{filename}.jpg")

                        # tensor_scores = image_description_similarity[prediction].squeeze()
                        # scores = tensor_scores.tolist()
                        # attributes_pc[0] += scores[0]
                        # attributes_pc[1] += scores[1]
                        # attributes_pc[2] += scores[2]
                        # attributes_pc[3] += scores[3]
                        # if max(scores[3], scores[2], scores[1], scores[0]) == scores[3]:
                        #     plot = draw_chart(np_image, h, image_description_similarity[prediction], gpt_descriptions[h])
                        #     filename = paths[0].split('/')[-1]
                        #     plot = plot.save(f"habitat_first_correct_nabirds_id_figs/{filename}.jpg")
                        # break

    # print(num_correct)
    # print(num_habitat_correct)
    # attributes_pc = [pc/num_correct for pc in attributes_pc]
    # print(attributes_pc)

    save_list_to_file(correct_predictions, f'./viz_path/correct_habitat_{hparams["dataset"]}.txt')
    save_list_to_file(incorrect_predictions, f'./viz_path/incorrect_habitat_{hparams["dataset"]}.txt')

# inference()
generate_correct(path1='./viz_path/incorrect_nohabitat_cub.txt', path2='./viz_path/correct_habitat_cub.txt')