# Lagrangian Relaxation Algorithm for Network Revenue Management

This repository provides a public implementation of the Lagrangian Relaxation Algorithm for network revenue management, originally developed by Prof. Huseyin Topaloglu (2009).

Researchers in operations research often use this algorithm and its test dataset as benchmarks, but no open implementation was available until now. This code lets researchers test, compare, and build on the algorithm easily. I have tested this implementation on all instances in Prof. Topaloglu's dataset, and the results match those in his paper.

I developed this project as part of my research with Prof. Baris Ata on solving NRM problems using deep learning-based numerical methods. I hope this implementation supports reproducible research and further work in network revenue management.

For more details, see the [documentation](documentation/documentation.pdf).

## Installation

```bash
# Clone the repository
git clone https://github.com/steve-shao/LagrangianRelaxationNRM.git
cd LagrangianRelaxationNRM

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run all instances in the dataset
python run_all_instances.py
```

## License

This project is licensed under the MIT License.

## Resources

This project is based on [Topaloglu (2009)](https://people.orie.cornell.edu/huseyin/publications/revenue_man.pdf).
The paper and its dataset are available on [Prof. Huseyin Topaloglu's website](https://people.orie.cornell.edu/huseyin).
You can access the paper directly [here](https://people.orie.cornell.edu/huseyin/publications/revenue_man.pdf).
The dataset can be downloaded from [this page](https://people.orie.cornell.edu/huseyin/research/rm_datasets/rm_datasets.html).

<br>