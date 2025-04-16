# Multi-output Ensembles for Multi-step Forecasting

This repository contains the code to reproduce the experiments described in the paper "Multi-output Ensembles for Multi-step Forecasting" by Cerqueira and Torgo (2023).

## Overview

This work explores the effectiveness of multi-output ensembles for multi-step forecasting tasks. The experiments demonstrate how combining multiple forecasting models can improve prediction accuracy across different time horizons.

## Implementation

The codebase includes:
- Implementation of various ensemble methods for multi-step forecasting
- Experimental setup and evaluation framework
- Scripts to reproduce the results from the paper

## Running Experiments

To run the experiments:
```bash
python scripts/run_experiments.py
```

## Related Work

The methods implemented in this repository have been further improved and are now available in the [metaforecast](https://github.com/vcerqueira/metaforecast/) package. The metaforecast package provides a more comprehensive implementation of these techniques along with additional features for time series forecasting.

## Citation

If you use this code in your research, please cite:
```
@article{cerqueira2023multi,
  title={Multi-output Ensembles for Multi-step Forecasting},
  author={Cerqueira, Vitor and Torgo, Luis},
  journal={arXiv preprint arXiv:2306.14563},
  year={2023}
}
```

## Contact

Feel free to get in touch via [Twitter](https://twitter.com/vitor_cerq) or by opening an issue in this repository.