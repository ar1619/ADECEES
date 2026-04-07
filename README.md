# ADECEES: Anomaly DEtection of CO2 Emissions via Ensemble Segmentation

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Accurate and spatially explicit monitoring of anthropogenic CO2 emissions is essential for independent verification of mitigation efforts and understanding their atmospheric impact. 

**ADECEES** is a generative anomaly detection method designed for the unsupervised identification of CO2 point sources from satellite-derived XCO2 (column-averaged dry-air mole fractions of CO2) maps. By learning the distribution of healthy, background atmospheric variability, ADECEES provides the framework to globally isolate statistically significant concentration enhancements.


## Key Features

* **Unsupervised Detection:** Identifies localized emission sources directly from atmospheric concentration fields, overcoming challenges related to background variability and transport-driven spatial structures.
* **Generative Framework:** Utilizes a partial diffusion model trained exclusively on background atmospheric states.
* **High-Resolution Tracking:** Evaluated using a global daily XCO2 dataset, derived from OCO-2 observations.
* **Temporal Monitoring:** Capable of discriminating between operational and inactive periods for specific sites (e.g., coal mines) and detecting spatiotemporal changes linked to plant closures, units upgrades, or commissioning events.

## Methodology

The ADECEES framework operates through three core components:

1.  **Partial Diffusion Model**
2.  **Ensemble-Based Reconstruction:** Generates multiple reconstructed background states for a given observation to account for uncertainty and establish a robust baseline.
3.  **Residual Segmentation:** Compares the actual satellite-derived map against the reconstructed ensemble background to isolate and segment statistically significant anomalies (emissions).



## Data Requirements

This framework is built to process satellite-derived XCO2 maps. The default implementation is configured for OCO-2 observational data processed to a 0.03° × 0.04° gridded resolution. The dataset can be found [here](https://gd-xco2.dsi.ic.ac.uk/datasets).  
Link to the dataset paper: [paper](https://doi.org/10.3390/rs17091617)

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/ar1619/ADECEES.git
cd ADECEES
pip install -r requirements.txt
```

## Quick Start

Training using available configuration and sample data
```bash
python diffusion_training.py 1
```

Run anomaly detection between start_date and end_date (modify accordingly)
```bash
python compute_variance.py 1 start_date end_date positive
```

## Citation

```bibtex
@article{Rakotoharisoa2026ADECEES,
  title={Diffusion-Based Anomaly Detection for Satellite Monitoring of CO2 Point Sources},
  author={Rakotoharisoa, Andrianirina and Cenci, Simone and Arcucci, Rossella},
  journal={Available at SSRN 6368100},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
