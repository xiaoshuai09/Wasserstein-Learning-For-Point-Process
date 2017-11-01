# PPWGAN
learning point processes by means of optimal transport and wasserstein distance
"Wasserstein Learning of Deep Generative Point Process Models"
############################################


Code accompanying the paper ["Wasserstein Learning of Deep Generative Point Process Models"](https://arxiv.org/pdf/1705.08051.pdf)

If the code helps your research, please cite our work.

    @inproceedings{xiao2017wasserstein,
    title={Wasserstein Learning of Deep Generative Point Process Models.},
    author={Xiao, Shuai and Farajtabar, Mehrdad and Ye, Xiaojing and Yan, Junchi and Xiaokang, Yang and Song, Le and Zha, Hongyuan},
    booktitle={Advances in Neural Information Processing Systems},
    year={2017}
    }

    


## Prerequisites

- Computer with Linux or OSX
- Language: TensorFlow 1.0
- GPU is strongly recommended when training.

## Notes

- For bugs and questions, contact: benjaminforever at sjtu.edu.cn

## Paper Abstract:
Point processes are becoming very popular in modeling asynchronous sequential
data due to their sound mathematical foundation and strength in modeling a variety
of real-world phenomena. Currently, they are often characterized via intensity
function which limits model’s expressiveness due to unrealistic assumptions on
its parametric form used in practice. Furthermore, they are learned via maximum
likelihood approach which is prone to failure in multi-modal distributions of
sequences. In this paper, we propose an intensity-free approach for point processes
modeling that transforms nuisance processes to a target one. Furthermore, we train
the model using a likelihood-free leveraging Wasserstein distance between point
processes. Experiments on various synthetic and real-world data substantiate the
superiority of the proposed point process model over conventional ones.
