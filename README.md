# Riemannian Diffusion Adaptation for Distributed Optimization on Manifolds

In this repository, you can find the codes to reproduce the results of the ICML paper <a href="https://xiuheng-wang.github.io/assets/pdf/wang2025riemannian.pdf">Riemannian Diffusion Adaptation for Distributed Optimization on Manifolds</a>.

Steps:

1. Run plot_topology.py to plot the graph topologies as in Figure 1 and Figure 7 (left);

2. Run main_pca.py for performing PCA on synthetic data to plot Figure 2;

3. Run main_pca_mnist.py for performing PCA on real data to plot Figure 3;

4. Run main_gmm.py for performing GMM inference on synthetic data to plot Figure 4;

5. Run main_gmm_mnist.py for performing GMM inference on real data to plot Figure 5;

6. Run main_pca_efficiency.py to compare the two algorithms to plot Figure 6;

7. Run main_pca_uniform.py for performing PCA on another network to plot Figure 7 (middle and right);

8. Run main_pca_stepsize.py to evaluate the performance with different step sizes to plot Figure 8.

For any questions, feel free to email at dr.xiuheng.wang@gmail.com.

If these codes are helpful for you, please cite our paper as follows:

    @inproceedings{wang2025riemannian,
      title={Riemannian Diffusion Adaptation for Distributed Optimization on Manifolds},
      author={Wang, Xiuheng and Borsoi, Ricardo and Richard, C{\'e}dric and Sayed, Ali H},
      booktitle={International Conference on Machine Learning (ICML)},
      year={2025},
      organization={PMLR}
    }

Note that the implementation of the following paper can also be found in ./utils/baselines.py:
    
    @inproceedings{wang2024riemannian,
      title={Riemannian diffusion adaptation over graphs with application to online distributed {PCA}},
      author={Wang, Xiuheng and Borsoi, Ricardo Augusto and Richard, C{\'e}dric},
      booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      pages={9736--9740},
      year={2024},
      organization={IEEE}
    }

Note that the copyright of the Manopt toolbox is reserved by https://pymanopt.org/.
