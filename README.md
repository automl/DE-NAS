# deNAS - Differential Evolution for Neural Architecture Search

Implementation of DE for NAS, benchmarked in the paper **Differential Evolution for Neural Architecture Search** accepted to the 1st NAS Workshop at ICLR 2020.

To access the paper:
* [updated Arxiv version of the main paper](https://arxiv.org/abs/2012.06400) 
* [ICLR 2020 version + slides + videos](https://sites.google.com/view/nas2020/accepted-papers?authuser=0)

To cite the paper or code:
```bibtex
@inproceedings{awad-iclr20,
  author    = {N. Awad and N. Mallik and F. Hutter},
  title     = {Differential Evolution for Neural Architecture Search},
  booktitle = {Proceedings of the 1st workshop on neural architecture search(@{ICLR}'20)},
  year      = {2020},
  month     = apr
}
```

To contact authors for queries reqarding the paper:
* Neeratyoy Mallik (mallik@cs.uni-freiburg.de)
* Noor Awad (awad@cs.uni-freiburg.de)

#### To run experiments from NAS ICLR Workshop 2020 paper

Refer [here](https://github.com/automl/DE-NAS/tree/nas_workshop/denas/examples) (the parallel implementation of DE is also available here).

#### To get started with DE to optimize custom functions

Refer [here](https://github.com/automl/DE-NAS/blob/master/getting_started_with_de.ipynb).

#### Differential Evolution for Neural Architecture Search

The instructions to setup the benchmarks can be found 
[here](https://github.com/automl/DE-NAS/blob/master/denas/examples/README.md).


To run DE on _NAS-Bench-101_ or _NAS-HPO-Bench_:

`PYTHONPATH=$PWD python3 denas/examples/nas101/run_de_nas101.py`


To run DE on _NAS-Bench-1shot1_:

`PYTHONPATH=$PWD python3 denas/examples/nas1shot1/run_de_nas1shot1.py`


To run DE on _NAS-Bench-201_:

`PYTHONPATH=$PWD python3 denas/examples/nas201/run_de_nas201.py`


Plots can be generated in a similar way, by passing the directory of the stored output files. For example:

`PYTHONPATH=$PWD python3 denas/utils/plot_regret.py --path denas/examples/results/cifara`

`PYTHONPATH=$PWD python3 denas/utils/plot_cdf.py --path denas/examples/results/nas101/cifara`

For the above plotting script to work with the output of NAS-Bench-1shot1, the output files need to be additionally preprocessed, for example:

`PYTHONPATH=$PWD python3 denas/utils/convert_files.py --path denas/examples/results/nas1shot1/ --ssp 1`
