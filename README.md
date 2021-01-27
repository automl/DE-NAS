# deNAS - Differential Evolution for Neural Architecture Search

Implementation of DE for NAS, benchmarked in this 
[paper](https://drive.google.com/file/d/1kxhLeXnwiAkarkXU15GrU3vyodKETWQw/view?usp=sharing), 
accepted to the 1st NAS Workshop at ICLR 2020.

To cite the paper or code:
```bibtex
@article{awad2020differential,
  title={Differential Evolution for Neural Architecture Search},
  author={Awad, Noor and Mallik, Neeratyoy and Hutter, Frank},
  journal={arXiv preprint arXiv:2012.06400},
  year={2020}
}
```



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
