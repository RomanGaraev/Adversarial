import torch
import numpy as np

def conditional_latent_generator_minority(distribution, class_num, batch, type):
    # types: 1 - generate 0,1,2,3,4;
    #        2 - generate 0,1,2,3,4, 6,7,8,9;
    #        2 - generate 5;
    if type == 2 or type == 3 or type == 4:
        class_labels = torch.randint(0, class_num-6, (batch,), dtype=torch.long)
    elif type == 5 or type == 6 or type == 7:
        class_labels = torch.randint(0, class_num-6, (batch//2,), dtype=torch.long)
        class_labels = torch.cat((class_labels, torch.randint(6, class_num-1, (batch//2,), dtype=torch.long)), dim=0)
    elif type == 8:
        class_labels = torch.randint(5, class_num-5, (batch,), dtype=torch.long)
    else:
        return
    generated_z = distribution[class_labels[0].item()].sample((1,))
    #generated_z = torch.from_numpy(np.asarray((distribution[class_labels[0].item()].sample((1,)), class_labels[0])))
    for c in class_labels[1:]:
        generated_z = torch.cat((generated_z, distribution[c.item()].sample((1,))), dim=0)
    return generated_z,

if __name__ == "__main__":
    distribution = torch.load('class_distribution.dt')['distribution']
    conditional_z, z_label = conditional_latent_generator_minority(distribution, config.class_num,
                                                                           config.batch_size, type)
